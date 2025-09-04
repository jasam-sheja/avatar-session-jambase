"""
This script demonstrates the use of a Thin-Plate-Spline Motion Model for animating a source image using a driving video.
It includes functions for loading models, preprocessing images, and running the animation process.
It also provides a command-line interface for users to specify input parameters and options.
"""

import logging
from argparse import ArgumentParser
from functools import partial
from typing import Any, Callable, Dict, OrderedDict, Tuple, overload
from functools import singledispatchmethod

import cv2
import face_alignment
import numpy as np
import torch
import torchvision.transforms.functional as tvF
from einops import rearrange
from scipy.spatial import ConvexHull
from torch import Tensor
from torch.nn import functional as F

from .reenactors import TPSMM_Animator
from .utils import FaceNotFoundError, draw_landmarks, MediapipeTool, crop_and_resize

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    # Ensure that CUDA is initialized properly
    # This is a workaround for a known issue with PyTorch and CUDA initialization
    # https://github.com/pytorch/pytorch/issues/90613#issuecomment-1497238767
    torch.inverse(torch.eye(3, device="cuda:0"))


def parse_args(args=None) -> ArgumentParser:
    """
    Parse command-line arguments for the demo script.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = ArgumentParser(__doc__)
    parser.add_argument("--source_image", default="", help="path to source image")
    parser.add_argument(
        "--preprocess_source",
        action="store_true",
        help="if True, preprocess source image",
    )
    parser.add_argument(
        "--preprocess_driving",
        action="store_true",
        help="if True, preprocess each frame in driving video",
    )
    parser.add_argument(
        "--img_shape",
        default="256,256",
        type=lambda x: list(map(int, x.split(","))),
        help="Shape of image, that the model was trained on.",
    )
    parser.add_argument(
        "--mode",
        default="relative-a",
        choices=["standard", "relative-0", "relative-a", "avd"],
        help="Animate mode: ['standard', 'relative-0', 'relative-a', 'avd'], when use the relative mode to animate a face, use '--find_best_frame' can get better quality result",
    )
    parser.add_argument(
        "--visualize", help="comma seperated options. visualize the result.", default=""
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="fps of the driving video",
    )

    args = parser.parse_args(args)
    if args.visualize == "all":
        args.visualize = {
            "source",
            "reference",
            "driving",
            "result",
            "frame",
            "source-lamdmarks",
            "frame-lamdmarks",
            "frame-bbox",
            "reference-lamdmarks",
            "reference-sim",
            "driving-lamdmarks",
            "driving-sim",
            "result-lamdmarks",
            "result-sim",
        }
    else:
        args.visualize = set(args.visualize.split(","))
    return args


class Tracker:
    """Tracks the position and size of the face across frames."""

    def __init__(self, history_length: int, fps=25):
        self.lmk_detector = MediapipeTool("video", fps)
        self.lmks = None
        self.transM = None
        self.history_length = history_length

    def reset(self):
        self.lmk_detector.reset()
        self.lmks = None
        self.transM = None

    def add(self, frame: np.ndarray):
        det = self.lmk_detector(frame)
        lmk = torch.tensor(det.landmarks, dtype=torch.float32, device="cuda")
        transM = torch.tensor(det.transformation_matrixes, dtype=torch.float32, device="cuda")

        if self.lmks is None:
            self.lmks = lmk.unsqueeze(0)
        elif self.lmks.shape[0] < self.history_length:
            self.lmks = torch.cat((self.lmks, lmk.unsqueeze(0)), dim=0)
        else:
            self.lmks = torch.roll(self.lmks, -1, dims=0)
            self.lmks[-1] = lmk

        if self.transM is None:
            self.transM = transM.unsqueeze(0)
        elif self.transM.shape[0] < self.history_length:
            self.transM = torch.cat((self.transM, transM.unsqueeze(0)), dim=0)
        else:
            self.transM = torch.roll(self.transM, -1, dims=0)
            self.transM[-1] = transM

        return lmk, transM

    def get_tube_xyxy(self) -> Tensor:
        """tube is a square"""
        if self.lmks is None:
            raise RuntimeError("No landmarks tracked yet.")
        min = self.lmks.amin([0, 1])[:2] # top left
        max = self.lmks.amax([0, 1])[:2] # bottom right
        size = (max - min).amax()
        center = (max + min) / 2
        min = center - size / 2
        max = center + size / 2
        return torch.cat([min, max])

    # def get_tube_xyxy(self) -> Tensor:
    #     """tube is a square"""
    #     if self.lmks is None:
    #         raise RuntimeError("No landmarks tracked yet.")
    #     # print(self.lmks.shape)
    #     size = 2*torch.norm(self.lmks[:, 0, :2] - self.lmks[:, 10, :2], dim=-1).mean()
    #     center = self.lmks[:,:,:2].mean(dim=[0, 1])
    #     center[1] = self.lmks[:,1,1].mean()
    #     min = center - size / 2
    #     max = center + size / 2
    #     # print(center.shape, size.shape, min.shape, max.shape)
    #     # exit()
    #     return torch.cat([min, max])

    def crop(self, frame: Tensor, margin: int | float = 0):
        # Crop the frame using the landmarks
        return crop_and_resize(
            frame, self.get_tube_xyxy(), 256, 256, margin=margin, round_bbox=True
        )

def rotation_dist(transM: Tensor, ref: Tensor) -> Tensor:
    """Compute the relative pose distance between the frame and a reference frame."""
    # Compute the relative pose distance using the transformation matrices
    R_self = transM[:3,:3]
    R_ref = ref[:3,:3]
    R_diff = R_self @ R_ref.T
    return torch.arccos((torch.trace(R_diff)-1) / 2)

class Demo:
    """
    Main class for running the Thin-Plate-Spline Motion Model demo.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    def __init__(self, args):
        """
        Initialize the Demo class.

        Args:
            args (argparse.Namespace): Parsed command-line arguments.
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prep_preprocessing()
        self.load_animators()
        self.reference = None  # Reference frame for relative animation modes
        if args.source_image:
            self.prep_source(args.source_image)
        else:
            self.source = None

    def is_ready(self) -> bool:
        return self.source is not None

    def reset(self):
        self.source = None
        self.reference = None
        self.tracker.reset()
        cv2.destroyAllWindows()

    def load_animators(self):
        """
        Load the model checkpoints for inpainting, keypoint detection, and motion networks.
        """
        logger.info("==> loading model")
        self.animators = OrderedDict(
            [
                # ("lia", LIA_Animator().to(self.device)),
                ("tps", TPSMM_Animator().to(self.device)),
            ]
        )

    def prep_preprocessing(self):
        """
        Prepare preprocessing functions for source and driving frames.
        """
        self.tracker = Tracker(5)

    def prep_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Preprocess a single frame and extract keypoints and landmarks.

        Args:
            frame (np.ndarray): Input frame.
            preprocess (Callable): Preprocessing function.

        Returns:
            dict: Processed frame data including image, bounding box, landmarks, and keypoints.
        """
        frame = np.ascontiguousarray(frame)
        oglm, transM = self.tracker.add(frame)
        oglm = oglm[:, :2]  # [-1 ~ 1]
        
        crop = self.tracker.crop(
            torch.from_numpy(frame)
            .cuda(non_blocking=True)
            .permute(2, 0, 1)
            .float()
            .div_(255),
            margin=0.3,
        )

        # fit lm to the crop
        orig = crop.bbox[:2]
        c_size = crop.bbox[2:] - crop.bbox[:2]
        o_size = torch.tensor(crop.crop.shape[1:], device=self.device)
        lm = (oglm - orig) * (o_size / c_size)

        data = {
            "ogframe": frame,
            "oglm": oglm,
            "image": crop.crop,
            "bbox": crop.bbox,
            "lm": lm,
            "transM": transM
        }
        for name, ani in self.animators.items():
            ani.prep_frame(data)
        return data

    def prep_source(self, file_or_img: str | np.ndarray):
        """
        Load and preprocess the source image and driving video.
        """
        if isinstance(file_or_img, str):
            source_img = cv2.imread(file_or_img)
        else:
            source_img = file_or_img
        if source_img is None:
            raise FileNotFoundError(f"Source image not found: {file_or_img}")
        source_img = cv2.flip(source_img, 1)
        ntry = 0
        while True:
            try:
                self.tracker.reset() 
                self.source = self.prep_frame(source_img[..., ::-1])
            except FaceNotFoundError as e:
                ntry += 1
                if ntry > 5:
                    raise e
                continue
            break
        self.tracker.reset() 
        # Prepare source image data
        for ani in self.animators.values():
            ani.prep_source(self.source)
        source_vis = self._show_data(self.source, "source")


        if self.reference is not None:
            # If a reference frame exists, invalidate its similarity metric so it'd be recalculated
            self.reference["sim"] = float("inf")

    def similarity_metric(self, info1, info2):
        """
        Compute a similarity metric between two sets of landmarks.

        Args:
            info1 (dict): First set of landmarks.
            info2 (dict): Second set of landmarks.

        Returns:
            float: Similarity score.
        """

        def _normalize_lms(lms):
            lms = lms - lms.mean(axis=0, keepdims=True)
            area = ConvexHull(lms[:, :2].cpu().numpy()).volume
            area = np.sqrt(area)
            lms[:, :2] = lms[:, :2] / area
            return lms

        return 100 * torch.sum(
            (_normalize_lms(info1["lm"]) - _normalize_lms(info2["lm"])) ** 2
        ) * rotation_dist(info1["transM"], info2["transM"])

    def _show_data(self, frame_data, tag, og_frame=None):
        """
        Visualize frame data with optional landmarks and bounding boxes.

        Args:
            frame_data (dict): Frame data to visualize.
            tag (str): Tag for visualization (e.g., 'source', 'driving').
            og_frame (np.ndarray, optional): Original frame for overlay. Defaults to None.

        Returns:
            np.ndarray: Visualized frame.
        """
        args = self.args
        if og_frame is None:
            if 'vis' not in frame_data:
                frame_data["vis"] = frame_data["image"].mul(255).byte().permute(1, 2, 0).cpu().numpy()
            data_vis = frame_data["vis"].copy()
            if f"{tag}-lamdmarks" in args.visualize:
                draw_landmarks(data_vis, frame_data["lm"])
        else:
            data_vis = og_frame
            if f"{tag}-lamdmarks" in args.visualize:
                lm_frame = (
                    frame_data["lm"]
                    * (frame_data["bbox"][2] - frame_data["bbox"][0])
                    / frame_data["image"].shape[1]
                ) + frame_data["bbox"][:2]
                draw_landmarks(data_vis, lm_frame)
            if f"{tag}-bbox" in args.visualize:
                _driving_bbox = tuple(frame_data["bbox"].int().tolist())
                cv2.rectangle(
                    data_vis,
                    _driving_bbox[:2],
                    _driving_bbox[2:],
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    data_vis,
                    f"height: {_driving_bbox[3] - _driving_bbox[1]}"
                    f"\nwidth: {_driving_bbox[2] - _driving_bbox[0]}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
        if f"{tag}-sim" in args.visualize:
            cv2.putText(
                data_vis,
                f"SIM {frame_data.get('sim', float('nan')):.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if f"{tag}" in args.visualize:
            cv2.imshow(f"{tag}", data_vis)
        return data_vis

    @torch.inference_mode()
    def apply(self, frame: np.ndarray) -> np.ndarray:
        source = self.source
        if source is None:
            return
        args = self.args

        try:
            driving = self.prep_frame(frame)
        except FaceNotFoundError:
            cv2.putText(
                frame,
                "no faces detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            return frame

        if self.reference is None:
            self.reference = self.reset_reference(driving)[0]

        self._show_data(driving, "frame", og_frame=frame)

        if args.mode == "relative-a":
            driving["sim"] = self.similarity_metric(source, driving)
            if driving["sim"] < self.reference["sim"]:
                self.reference, _ = self.reset_reference(driving)
        driving_vis = self._show_data(driving, "driving")

        results = []
        for name, ani in self.animators.items():
            out = ani.animate(source, self.reference, driving)
            if out.requires_grad:
                print(
                    f"Warning: {name} animator output requires grad, which is not expected."
                    " This may lead to unexpected behavior."
                    " Please check the animator implementation."
                    " If you are sure this is intended, you can ignore this warning."
                    " If you are not sure, please report this issue."
                    " If you are using a custom animator, please ensure it does not return a tensor with requires_grad=True."
                    " If you are using a pre-trained animator, please check the model implementation."
                    " If you are using a custom model, please ensure it does not return a tensor with requires_grad=True."
                )
                exit(1)
            out = rearrange(out, "c h w -> h w c")
            out = out.mul_(255).byte()
            results.append(out.cpu().numpy())
        return np.concatenate(results, axis=1)

    def reset_reference(self, driving: Dict[str, Any]):
        """
        Reset the reference frame to the current driving frame.

        Args:
            driving (dict): Current driving frame data.
        """
        reference = driving
        for ani in self.animators.values():
            ani.prep_refrence(reference)
        if self.args.mode == "relative-a" and "sim" not in reference:
            reference["sim"] = self.similarity_metric(self.source, reference)
        reference_vis = self._show_data(reference, "reference")
        return reference, reference_vis

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            vis = self.apply(frame)
            cv2.imshow("Video", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def __del__(self):
        """
        Release resources such as video capture and writer on object deletion.
        """
        cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    with torch.inference_mode():
        demo = Demo(args)
        demo.run()

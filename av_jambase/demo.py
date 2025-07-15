"""
This script demonstrates the use of a Thin-Plate-Spline Motion Model for animating a source image using a driving video.
It includes functions for loading models, preprocessing images, and running the animation process.
It also provides a command-line interface for users to specify input parameters and options.
"""

import logging
from argparse import ArgumentParser
from functools import partial
from typing import Any, Callable, Dict, OrderedDict, Tuple

import cv2
import face_alignment
import numpy as np
import torch
import torchvision.transforms.functional as tvF
from einops import rearrange
from scipy.spatial import ConvexHull
from torch import Tensor
from torch.nn import functional as F

from .reenactors import LIA_Animator, TPSMM_Animator
from .utils import (FaceNotFoundError, SingleFaceDetector, draw_landmarks,
                    preprocess)

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
    parser.add_argument(
        "--source_image", default="", help="path to source image"
    )
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
        self.fd = SingleFaceDetector(momentum=1 - 1 / self.args.fps).to(self.device)
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D,
            flip_input=True,
            device=self.device.type,
        )

        def _preprocess(
            image: np.ndarray,
            method: str,
            img_shape: Tuple[int, int],
            no_smoothing: bool = False,
        ) -> Tuple[Tensor, Tensor, Tensor]:
            if isinstance(image, np.ndarray):
                image = np.ascontiguousarray(image)
                image = torch.from_numpy(image).to(self.device)
                image = rearrange(image, "h w c -> c h w")
                image = image.float().div_(255)
            else:
                image = image.to(self.device)
            if method == "crop&resize":
                crop_image, bbox = preprocess(image[None], self.fd, img_shape)
                return image, crop_image[0], bbox[0]
            elif method == "resize":
                resize_image = tvF.resize(image[None], img_shape)[0]
                bbox = torch.tensor(
                    [0, 0, image.shape[2], image.shape[1]], device=self.device
                )
                return image, resize_image, bbox

        if self.args.preprocess_source:
            self.preprocess_source = partial(
                _preprocess,
                method="crop&resize",
                img_shape=self.args.img_shape,
                no_smoothing=True,
            )
        else:
            self.preprocess_source = partial(
                _preprocess,
                method="resize",
                img_shape=self.args.img_shape,
                no_smoothing=True,
            )
        if self.args.preprocess_driving:
            self.preprocess_driving = partial(
                _preprocess, method="crop&resize", img_shape=self.args.img_shape
            )
        else:
            self.preprocess_driving = partial(
                _preprocess, method="resize", img_shape=self.args.img_shape
            )

    def prep_frame(self, frame: np.ndarray, preprocess: Callable) -> Dict[str, Any]:
        """
        Preprocess a single frame and extract keypoints and landmarks.

        Args:
            frame (np.ndarray): Input frame.
            preprocess (Callable): Preprocessing function.

        Returns:
            dict: Processed frame data including image, bounding box, landmarks, and keypoints.
        """
        _, frame, bbox = preprocess(frame)
        lm_frame = self.fa.get_landmarks_from_batch(frame[None] * 255)[0]
        if len(lm_frame) > 68:
            lm_frame = lm_frame[:68]
        if len(lm_frame) < 1:
            raise FaceNotFoundError("no faces detected")
        frame_vis = rearrange(frame.mul(255).byte(), "c w h -> w h c").cpu().numpy()

        data = {
            "image": frame,
            "bbox": bbox,
            "vis": frame_vis,
            "lm": lm_frame,
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
        self.source = self.prep_frame(
            source_img[..., ::-1],
            self.preprocess_source,
        )
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
            area = ConvexHull(lms[:, :2]).volume
            area = np.sqrt(area)
            lms[:, :2] = lms[:, :2] / area
            return lms

        return 100 * np.sum(
            (_normalize_lms(info1["lm"]) - _normalize_lms(info2["lm"])) ** 2
        )

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
            data_vis = frame_data["vis"].copy()
            if f"{tag}-lamdmarks" in args.visualize:
                draw_landmarks(data_vis, frame_data["lm"])
        else:
            data_vis = og_frame
            if f"{tag}-lamdmarks" in args.visualize:
                lm_frame = (
                    frame_data["lm"]
                    * (frame_data["bbox"][2] - frame_data["bbox"][0]).cpu().numpy()
                    / frame_data["image"].shape[1]
                ) + frame_data["bbox"][:2].cpu().numpy()
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
        args = self.args

        try:
            driving = self.prep_frame(frame, self.preprocess_driving)
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
                print(f"Warning: {name} animator output requires grad, which is not expected."
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

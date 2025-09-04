import logging
from pathlib import Path
from typing import List, NamedTuple, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv
from face_alignment.detection.sfd.bbox import decode
from face_alignment.detection.sfd.net_s3fd import s3fd
from face_alignment.utils import load_file_from_url
from scipy.spatial import ConvexHull
from torch import Tensor
from torch.utils.model_zoo import load_url

from .base import FaceNotFoundError

__dir__ = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)

MODELS_URLS = {
    "2DFAN-4": "https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.6-c827573f02.zip",
    "3DFAN-4": "https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.6-ec5cf40a1d.zip",
    "depth": "https://www.adrianbulat.com/downloads/python-fan/depth_1.6-2aa3f18772.zip",
}


def decode(loc, priors, alpha, beta):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        alpha (float): variance of the center
        beta (float): variance of the size
    Return:
        decoded bounding box predictions
    """
    boxes = torch.cat(
        (
            priors[:, :2] + loc[:, :2] * alpha * priors[:, 2:],
            priors[:, 2:] * torch.exp(loc[:, 2:] * beta),
        ),
        1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


return_face = NamedTuple(
    "return_face", [("boxes", Tensor), ("scores", Tensor), ("index", Tensor | None)]
)


class FaceDetector(s3fd):
    def __init__(self, filter_threshold=0.5):
        super().__init__()
        model_weights = load_url(
            "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
        )
        self.load_state_dict(model_weights)
        self.eval()
        self.register_buffer(
            "_mean", torch.tensor([104, 117, 123]).float().view(1, 3, 1, 1)
        )
        self._filter_threshold = filter_threshold

    def __call__(self, *args, **kwds) -> return_face:
        return super().__call__(*args, **kwds)

    def forward(self, x: Tensor, return_all=False) -> return_face:
        """Detect faces in the input image.

        Arguments:
        - x (torch.Tensor): The input BGR images of shape (B, 3, H, W).

        Returns:
        - loc (torch.Tensor): The bounding boxes of the detected faces.
        - score (torch.Tensor): The scores of the detected faces.
        - index (torch.Tensor): The index of the image (0..B-1). if return_all is True.
        """
        # normalize the input image
        x = x.mul(255).sub_(self._mean)
        olist = super().forward(x)
        score_list = []
        loc_list = []
        index_list = []
        for i in range(len(olist) // 2):
            olist[i * 2] = F.softmax(olist[i * 2], dim=1)
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2 ** (i + 2)  # 4,8,16,32,64,128
            iindex, hindex, windex = torch.where(ocls[:, 1, :, :] > 0.05)
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            priors = torch.stack(
                [axc, ayc] + [torch.full_like(axc, stride * 4)] * 2, dim=1
            )
            score = ocls[iindex, 1, hindex, windex]
            loc = oreg[iindex, :, hindex, windex]
            boxes = decode(loc, priors, 0.1, 0.2)
            score_list.append(score)
            loc_list.append(boxes)
            index_list.append(iindex)
        score_list = torch.cat(score_list, dim=0)
        loc_list = torch.cat(loc_list, dim=0)
        index_list = torch.cat(index_list, dim=0)
        keep = tv.ops.batched_nms(loc_list, score_list, index_list, 0.3)
        keep = keep[score_list[keep] > self._filter_threshold]
        loc_list = loc_list[keep]
        score_list = score_list[keep]
        index_list = index_list[keep]
        if return_all:
            return return_face(loc_list, score_list, index_list)
        scores = torch.zeros(x.size(0), device=x.device)
        boxes = torch.zeros(x.size(0), 4, device=x.device)
        for i in range(x.size(0)):
            mask = index_list == i
            if mask.any():
                j = score_list[mask].argmax()
                scores[i] = score_list[j]
                boxes[i] = loc_list[mask][j]
        return return_face(boxes, scores, torch.arange(x.size(0), device=x.device))


class SingleFaceDetector(FaceDetector):
    def __init__(self, filter_threshold=0.5, momentum=0.9):
        super().__init__(filter_threshold)
        self.momentum = momentum
        self.reset()

    def reset(self):
        """Reset the detector to the initial state."""
        self._center_xy = None
        self._size_wh = None

    def forward(self, x: Tensor, no_smoothing=False) -> return_face:
        """Detect faces in the input image.

        Arguments:
        - x (torch.Tensor): The input BGR images of shape ([1,] 3, H, W).

        Returns:
        - loc (torch.Tensor): The bounding boxes of the detected faces.
        - score (torch.Tensor): The scores of the detected faces.
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)
        elif x.ndim == 4 and x.size(0) != 1:
            raise ValueError("Input tensor must have shape (1, 3, H, W) or (3, H, W)")
        ret = super().forward(x, return_all=False)
        _center_xy = (ret.boxes[:, :2] + ret.boxes[:, 2:]) / 2
        _size_wh = ret.boxes[:, 2:] - ret.boxes[:, :2]
        if _size_wh.prod().item() == 0:
            logger.getChild("SingleFaceDetector").error(
                "No faces detected in the image."
            )
            raise FaceNotFoundError("No faces detected in the image.")
        if no_smoothing:
            return ret
        if self._center_xy is None:
            # logger.debug(f"First frame detected: center_xy={_center_xy}, size_wh={_size_wh}")
            self._center_xy = _center_xy
            self._size_wh = _size_wh
            print("First frame detected", _size_wh.prod().item())
            return ret
        # logger.debug(f"Updated center_xy={_center_xy.tolist()}, size_wh={_size_wh.tolist()}, iou={iou((self._center_xy, self._size_wh), (_center_xy, _size_wh)).item()}")
        print(
            f"{(self.momentum * self._center_xy + (1 - self.momentum) * _center_xy).round().view(-1).tolist()} = {self.momentum:<.2f} * {self._center_xy.round().view(-1).tolist()} + {1-self.momentum:<.2f} * {_center_xy.round().view(-1).tolist()}",
            end="\t\t\t",
        )
        print(
            f"{(self.momentum * self._size_wh + (1 - self.momentum) * _size_wh).round().view(-1).tolist()} = {self.momentum:<.2f} * {self._size_wh.round().view(-1).tolist()} + {1-self.momentum:<.2f} * {_size_wh.round().view(-1).tolist()}"
        )

        self._center_xy = (
            self.momentum * self._center_xy + (1 - self.momentum) * _center_xy
        )
        self._size_wh = self.momentum * self._size_wh + (1 - self.momentum) * _size_wh
        # compute the bounding box
        bbox = torch.cat(
            (self._center_xy - self._size_wh / 2, self._center_xy + self._size_wh / 2),
            dim=1,
        )
        return return_face(bbox, ret.scores, ret.index)


class LandmarksDetector(s3fd):
    def __init__(self):
        super().__init__()
        self.face_alignment_net = torch.jit.load(
            load_file_from_url(MODELS_URLS["2DFAN-4"])
        )


def compute_aspect_preserved_bbox(bbox, increase_area):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top
    width_increase = max(
        increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width)
    )
    height_increase = max(
        increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height)
    )
    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)
    return (left, top, right, bot)


def draw_landmarks(
    image,
    lms,
    scatter=True,
    hull=True,
    scatter_params=None,
    hull_params=None,
):
    """
    Draw landmarks and convex hull on an image.

    Args:
        image (np.ndarray): Image to draw on.
        lms (np.ndarray): Landmarks to draw.
        scatter (bool): Whether to draw scatter points.
        hull (bool): Whether to draw convex hull.
        scatter_params (dict): Parameters for scatter points.
        hull_params (dict): Parameters for convex hull.
    """
    if isinstance(lms, Tensor):
        lms = lms.cpu().numpy()
    lms = lms.astype(int)
    scatter_params = scatter_params or {}
    hull_params = hull_params or {}
    # scatter points
    if scatter:
        for lm in lms:
            cv2.circle(
                image,
                tuple(lm),
                scatter_params.pop("radius", 1),
                scatter_params.pop("color", (0, 255, 0)),
                scatter_params.pop("thickness", -1),
                **scatter_params,
            )
    if hull:
        lms = np.ascontiguousarray(lms)
        hull = ConvexHull(lms)
        cv2.polylines(
            image,
            [lms[hull.vertices]],
            isClosed=True,
            color=hull_params.pop("color", (255, 255, 0)),
            thickness=hull_params.pop("thickness", 1),
            **hull_params,
        )


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


mediapipe_return_type = NamedTuple(
    "mediapipe_return_type",
    [
        ("landmarks", List[Tuple[float, float, float]]),
        ("blendshape", List[float]),
        ("transformation_matrixes", List[List[float]]),
    ],
)


def _options(
    model_asset_path: str | None = None,
    delegate: str | None = None,
    running_mode: str | None = None,
    min_face_detection_confidence: float = 0.1,
    min_face_presence_confidence: float = 0.1,
    min_tracking_confidence: float = 0.1,
    output_facial_transformation_matrixes: bool = False,
    output_face_blendshapes: bool = False,
    num_faces: int = 1,
) -> FaceLandmarkerOptions:  # pyright: ignore[reportInvalidTypeForm]
    if model_asset_path is None:
        model_asset_path = __dir__.joinpath("face_landmarker.task").as_posix()
    if delegate is None or delegate == "gpu":
        delegate = BaseOptions.Delegate.GPU
    else:
        delegate = BaseOptions.Delegate.CPU
    if running_mode is None or running_mode == "video":
        running_mode = VisionRunningMode.VIDEO
    else:
        running_mode = VisionRunningMode.IMAGE
    return FaceLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path=model_asset_path,
            delegate=delegate,
        ),
        running_mode=running_mode,
        min_face_detection_confidence=min_face_detection_confidence,
        min_face_presence_confidence=min_face_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
        output_facial_transformation_matrixes=output_facial_transformation_matrixes,
        output_face_blendshapes=output_face_blendshapes,
        num_faces=num_faces,
    )


class MediapipeTool:
    def __init__(
        self,
        running_mode: str,
        fps: int,
        model_asset_path: str | None = None,
        space: str = "image",
    ):
        try:
            self.landmarker = FaceLandmarker.create_from_options(
                _options(
                    model_asset_path=model_asset_path,
                    running_mode=running_mode,
                    delegate="gpu",
                    output_facial_transformation_matrixes=True
                )
            )
        except RuntimeError:
            logger.warning("Failed to load GPU delegate. Falling back to CPU delegate.")
            self.landmarker = FaceLandmarker.create_from_options(
                _options(
                    model_asset_path=model_asset_path,
                    running_mode=running_mode,
                    delegate="cpu",
                )
            )
        self.fps = fps
        self.running_mode = running_mode
        self.timestamp_ms = 0
        self.space = space

    def num_landmarks(self) -> int:
        return 478

    def reset(self):
        self.timestamp_ms += 10000  # to reset the internal media pipe state

    def __call__(self, frame: np.ndarray) -> mediapipe_return_type:
        self.timestamp_ms += int(1000 / self.fps) + 1
        # Convert the frame received from Numpy to a MediaPipe’s Image object.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # Perform face landmarking on the provided single image.
        if self.running_mode == "video":
            result = self.landmarker.detect_for_video(mp_image, self.timestamp_ms)
        else:
            result = self.landmarker.detect(mp_image)

        if len(result.face_landmarks) == 0:
            raise FaceNotFoundError("No face landmarks detected in the image.")

        if self.space == "normal":
            sx = 1.0
            sy = 1.0
        elif self.space == "image":
            h, w = frame.shape[:2]
            sx = w
            sy = h

        return mediapipe_return_type(
            landmarks=[
                (lm.x * sx, lm.y * sy, lm.z)
                for lm in result.face_landmarks[0]
            ],
            blendshape=(
                [blendshape.score for blendshape in result.face_blendshapes[0]]
                if result.face_blendshapes
                else []
            ),
            transformation_matrixes=(
                result.facial_transformation_matrixes[0]
                if result.facial_transformation_matrixes
                else []
            ),
        )

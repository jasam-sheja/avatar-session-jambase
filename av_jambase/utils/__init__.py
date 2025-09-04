import logging
from typing import Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
import torchvision as tv

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

from .base import FaceNotFoundError
from .face import (
    SingleFaceDetector,
    FaceDetector,
    MediapipeTool,
    draw_landmarks,
    compute_aspect_preserved_bbox,
)
from .image import crop_and_resize


def preprocess(
    vid: Tensor,
    fd: FaceDetector,
    target_shape: Tuple[int, int],
    tqdm=None,
    _margin=0.3,
    _batch_size=32,
):
    """
    Preprocess the video frames by detecting faces and cropping them.

    Args:
        vid (Tensor): Video frames as a tensor.
        fd (FaceDetector): Face detector instance.
        target_shape (Tuple[int, int]): Target shape for resizing.
        tqdm (optional): TQDM progress bar instance.
        _margin (float): Margin around the detected face bounding box.
        _batch_size (int): Batch size for face detection.

    Returns:
        Tuple[Tensor, Tensor]: Cropped video frames and bounding boxes.
    """
    vidstream = vid.split(_batch_size) if len(vid) > _batch_size else [vid]
    if tqdm is not None:
        vidstream = tqdm(vidstream, desc="Detecting faces")
    bboxes = torch.cat([fd(batch.contiguous()).boxes for batch in vidstream])
    keep = (
        bboxes != torch.tensor([0, 0, 0, 0], dtype=bboxes.dtype, device=bboxes.device)
    ).all(dim=1)
    x1, y1, x2, y2 = bboxes[keep].unbind(1)
    if x1.numel() == 0:
        logger.getChild("preprocess").error("No faces detected in the video.")
        raise FaceNotFoundError("No faces detected in the video.")
    x1 = x1.min().clamp(0, vid.size(-1) - 1)
    y1 = y1.min().clamp(0, vid.size(-2) - 1)
    x2 = x2.max().clamp(x1.item() + 1, vid.size(-1))
    y2 = y2.max().clamp(y1.item() + 1, vid.size(-2))
    bboxes = torch.stack([x1, y1, x2, y2]).unsqueeze(0)
    crops, bboxes = crop_and_resize(
        vid, bboxes, target_shape[0], target_shape[1], margin=_margin, round_bbox=True
    )
    crops = crops
    return crops, bboxes


if __name__ == "__main__":
    import face_alignment

    with torch.inference_mode():
        fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D, device="cuda"
        )
        fd = FaceDetector().cuda()
        im = tv.io.decode_image(
            "/home/alsherfawi/Downloads/biden.jpg", tv.io.image.ImageReadMode.RGB
        )
        im = im.cuda().float().unsqueeze(0).div_(255)
        # import matplotlib.pyplot as plt
        # plt.imshow(im.cpu().detach().squeeze(0).permute(1, 2, 0))
        # plt.show()
        # detection = fd(im)
        # print(detection.boxes)
        # out = crop_and_resize(im, detection.boxes, 224, 224)
        # face alignment
        det = fa.face_detector.detect_from_batch(im.mul(255))[0]
        print(det)
        exit()
        import matplotlib.pyplot as plt

        plt.imshow(out.cpu().detach().squeeze(0).permute(1, 2, 0))
        plt.show()

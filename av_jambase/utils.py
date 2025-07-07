import logging
from typing import NamedTuple, Tuple

import cv2
import torch
import torch.nn.functional as F
import torchvision as tv
from face_alignment.detection.sfd.bbox import decode
from face_alignment.detection.sfd.net_s3fd import s3fd
from face_alignment.utils import load_file_from_url
from scipy.spatial import ConvexHull
from torch import Tensor, nn
from torch.utils.model_zoo import load_url

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

MODELS_URLS = {
    "2DFAN-4": "https://www.adrianbulat.com/downloads/python-fan/2DFAN4_1.6-c827573f02.zip",
    "3DFAN-4": "https://www.adrianbulat.com/downloads/python-fan/3DFAN4_1.6-ec5cf40a1d.zip",
    "depth": "https://www.adrianbulat.com/downloads/python-fan/depth_1.6-2aa3f18772.zip",
}


def linspace(start: Tensor, stop: Tensor, num: int, outdim: int = 0):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    source(before modifications) https://github.com/pytorch/pytorch/issues/61292
    """
    # create a tensor of 'num' steps from 0 to 1
    assert start.shape == stop.shape
    assert start.device == stop.device
    assert start.dtype == stop.dtype
    if outdim < 0:
        # if outdim is negative, then it is counted from the end of the output tensor
        outdim = start.ndim + 1 + outdim
    # create the 'steps' tensor
    steps: Tensor = torch.linspace(0, 1, num, dtype=start.dtype, device=start.device)
    # reshape the 'steps' tensor to allow for broadcastings.
    # for example, if start.ndim = 3 and outdim = 1, then steps.shape = [1, num, 1, 1]
    for i in range(start.ndim):
        steps = steps.unsqueeze(-int(i >= outdim))
    # reshape the start and stop tensors to allow for broadcastings
    start = start.unsqueeze(outdim)
    stop = stop.unsqueeze(outdim)
    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start + steps * (stop - start)
    return out


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


def iou(bbox1: Tuple[Tensor, Tensor], bbox2: Tuple[Tensor, Tensor]) -> Tensor:
    cxy, shw = bbox1
    cxy2, shw2 = bbox2
    # compute the intersection
    inter_cxy = torch.max(cxy - shw / 2, cxy2 - shw2 / 2)
    inter_cxy2 = torch.min(cxy + shw / 2, cxy2 + shw2 / 2)
    inter_wh = torch.clamp(inter_cxy2 - inter_cxy, min=0)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]
    # compute the union
    area1 = shw[:, 0] * shw[:, 1]
    area2 = shw2[:, 0] * shw2[:, 1]
    union_area = area1 + area2 - inter_area
    # compute the IoU
    iou = inter_area / union_area
    return iou


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
            raise FaceNotFoundError("No faces detected in the image.")
        if no_smoothing:
            return ret
        if self._center_xy is None:
            # logger.debug(f"First frame detected: center_xy={_center_xy}, size_wh={_size_wh}")
            self._center_xy = _center_xy
            self._size_wh = _size_wh
            return ret
        # logger.debug(f"Updated center_xy={_center_xy.tolist()}, size_wh={_size_wh.tolist()}, iou={iou((self._center_xy, self._size_wh), (_center_xy, _size_wh)).item()}")
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


crop_and_resize_returntype = NamedTuple(
    "crop_and_resize_returntype", [("crop", Tensor), ("bbox", Tensor)]
)


def crop_and_resize(
    video_tensor: Tensor,
    bbox: Tensor,
    target_height: int,
    target_width: int,
    margin: int | float = 0,
    channel_last: bool = False,
    round_bbox: bool = False,
    _align_corners: bool = True,
) -> crop_and_resize_returntype:
    """
    Vectorized batch-process cropping and resizing of the blob in the video tensor using grid_sample.

    Parameters:
    - video_tensor (torch.Tensor): The video tensor of shape T x H x W x C.
    - bbox (torch.Tensor): The bounding box tensor of shape T x 4, where 4 represents the bounding box x1, y1, x2, y2.
    - target_height (int): Target height 'h' for the output tensor.
    - target_width (int): Target width 'w' for the output tensor.
    Keyword Arguments:
    - margin (Union[int, float]): Margin to add to the bounding box. If int, it is added to all sides. If float, it is
        multiplied by the bounding box half size. Default is 0.
    - round_bbox (bool): If True, the bounding box is rounded to the nearest integer. Default is False.
    - channel_last (bool): If True, the input tensor has channel last format. Default is False.

    Returns:
    - torch.Tensor: Cropped and resized video tensor of shape T x h x w x C.
    """
    if channel_last:
        video_tensor = video_tensor.permute(0, 3, 1, 2)
    T, C, H, W = video_tensor.shape
    x1, y1, x2, y2 = bbox.unbind(dim=1)
    x2, x1 = torch.maximum(x1, x2), torch.minimum(x1, x2)
    y2, y1 = torch.maximum(y1, y2), torch.minimum(y1, y2)
    cx, cy = ((x1 + x2) / 2, (y1 + y2) / 2)
    sw, sh = (x2 - x1, y2 - y1)
    aspect_compare = sw / sh > target_width / target_height
    _sh = torch.where(aspect_compare, sw * target_height / target_width, sh)
    _sw = torch.where(aspect_compare, sw, sh * target_width / target_height)
    sh, sw = _sh, _sw
    if isinstance(margin, int):
        # add margin to the bounding box by margin pixels
        x1, x2 = (cx - sw / 2 - margin, cx + sw / 2 + margin)
        y1, y2 = (cy - sh / 2 - margin, cy + sh / 2 + margin)
    elif isinstance(margin, float):
        # add margin to the bounding box by margin ratio
        x1, x2 = (cx - sw / 2 - margin * sw, cx + sw / 2 + margin * sw)
        y1, y2 = (cy - sh / 2 - margin * sh, cy + sh / 2 + margin * sh)
    else:
        # no margin
        x1, x2 = (cx - sw / 2, cx + sw / 2)
        y1, y2 = (cy - sh / 2, cy + sh / 2)

    if round_bbox:
        x1, y1, x2, y2 = (x1.round(), y1.round(), x2.round(), y2.round())

    # move to normalized coordinates [-1 1]
    nx1 = x1 / W * 2 - 1
    nx2 = x2 / W * 2 - 1
    ny1 = y1 / H * 2 - 1
    ny2 = y2 / H * 2 - 1

    # Generate the grid for grid_sample
    grid_h = linspace(ny1, ny2, target_height, outdim=1)  # [T, target_height]
    grid_w = linspace(nx1, nx2, target_width, outdim=1)  # [T, target_width]
    grid_h = grid_h.unsqueeze(-1).expand(
        -1, -1, target_width
    )  # [T, target_height, target_width]
    grid_w = grid_w.unsqueeze(-2).expand(
        -1, target_height, -1
    )  # [T, target_height, target_width]
    grid = torch.stack((grid_w, grid_h), dim=-1)  # [T, target_height, target_width, 2]
    if grid.size(0) == 1 and video_tensor.size(0) > 1:
        # expand the grid to match the batch size because grid_sample does not broadcast
        grid = grid.expand(video_tensor.size(0), -1, -1, -1)

    # grid_sample
    cropvideo_tensor = torch.nn.functional.grid_sample(
        video_tensor, grid, align_corners=_align_corners
    )
    if channel_last:
        cropvideo_tensor = cropvideo_tensor.permute(0, 2, 3, 1)
    return crop_and_resize_returntype(
        cropvideo_tensor, torch.stack((x1, y1, x2, y2), dim=1)
    )


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
        hull = ConvexHull(lms)
        cv2.polylines(
            image,
            [lms[hull.vertices]],
            isClosed=True,
            color=hull_params.pop("color", (255, 255, 0)),
            thickness=hull_params.pop("thickness", 1),
            **hull_params,
        )


class FaceNotFoundError(Exception):
    """
    Custom exception raised when no face is detected in the video.
    """

    pass

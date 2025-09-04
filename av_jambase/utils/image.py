
from typing import NamedTuple
import torch
from torch import Tensor

from ._math import linspace

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
    if (is_single_frame := video_tensor.dim() == 3):
        video_tensor = video_tensor.unsqueeze(0)  # add batch dimension
        bbox = bbox.unsqueeze(0)
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
    if is_single_frame:
        cropvideo_tensor = cropvideo_tensor.squeeze(0)
        x1 = x1.squeeze(0)
        y1 = y1.squeeze(0)
        x2 = x2.squeeze(0)
        y2 = y2.squeeze(0)
    return crop_and_resize_returntype(
        cropvideo_tensor, torch.stack((x1, y1, x2, y2), dim=-1)
    )


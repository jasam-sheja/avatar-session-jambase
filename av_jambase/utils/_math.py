from typing import Tuple
import torch
from torch import Tensor


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
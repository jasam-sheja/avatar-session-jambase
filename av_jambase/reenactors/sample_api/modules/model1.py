from typing import Any, Dict

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .util import HelperClass


class Model1(nn.Module):
    """<description>
    Args:
        config (Dict[int, Any]): <description>.
    """

    def __init__(
        self,
        config: Dict[int, Any],
    ):
        super().__init__()

    def forward(
        self,
        param1: Tensor,
        param2: Tensor,
    ):
        """
        <description>

        Args:
            param1 (Tensor): <description>.
            param2 (Tensor): <description>.

        Returns:
            Tensor: <description>.
        """
        raise NotImplementedError()

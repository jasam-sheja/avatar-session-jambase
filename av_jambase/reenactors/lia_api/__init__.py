""" """

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .networks.generator import Generator

__dir__ = Path(__file__).resolve().parent


class AnimatorAPI(nn.Module):
    def __init__(self):
        super().__init__()
        self.gen = Generator(256, 512, 20, 1)
        self.gen.load_state_dict(
            torch.load(__dir__.joinpath("assets/vox-gen.pt"), map_location="cpu")
        )
        self.gen.eval()

    def prep_source(self, source: Dict[str, Any]):
        pass

    def prep_frame(self, frame: Dict[str, Any]):
        pass

    def prep_refrence(self, reference: Dict[str, Any]):
        reference["h_start"] = self.gen.enc.enc_motion(
            reference["image"].unsqueeze(0).sub(0.5).mul_(2.0)
        )

    def animate(
        self, source: Dict[str, Any], reference: Dict[str, Any], driving: Dict[str, Any]
    ) -> Tensor:
        """
        Animate the source image using the driving video frames.

        Args:
            source (dict): Source image data.
            reference (dict): reference frame data.
            driving (dict): Driving frame data.

        Returns:
            Tensor: Animated frame.
        """
        out = (
            self.gen(
                source["image"].unsqueeze(0).sub(0.5).mul_(2.0),
                driving["image"].unsqueeze(0).sub(0.5).mul_(2.0),
                reference["h_start"],
            )[0]
            .add(1)
            .div_(2.0)
            .clamp(0, 1)
        )
        return out

    def forward(self, source: Tensor, reference: Tensor, driving: Tensor) -> Tensor:
        """
        Forward pass for the AnimatorAPI.

        Args:
            source (Tensor): Source image tensor.
            reference (Tensor): Reference frame tensor.
            driving (Tensor): Driving frame tensor.

        Returns:
            Tensor: Animated frame.
        """
        source = {"image": source}
        reference = {"image": reference}
        driving = {"image": driving}

        self.prep_frame(source)
        self.prep_frame(reference)
        self.prep_frame(driving)
        self.prep_source(source)
        self.prep_refrence(reference)

        out = self.animate(source, reference, driving)

        return out
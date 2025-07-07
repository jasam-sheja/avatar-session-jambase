""" """

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor

__dir__ = Path(__file__).resolve().parent


class AnimatorAPI(nn.Module):
    """Identity animator API.
    This is a mock API for the animator. It does not perform any animation.
    It is used for testing purposes only.
    NOTE: No arguments are sent to the constructor.
    """

    def __init__(self):
        super().__init__()
        with open(__dir__.joinpath("assets/config.yaml")) as f:
            config = yaml.full_load(f)

    def prep_source(self, source: Dict[str, Any]):
        """Prepares the source image for processing.
        Args:
            source (dict): Source image data.
                source['image'] (Tensor): Cropped image with some margin.
                source['bbox'] (Tensor): Bounding box coordinates, x1,y1,x2,y2.
                source['lm'] (Tensor): Landmark coordinates, Face_alignment.FaceAlignment 2D (68 xy points).
        """
        pass  # pass if not needed

    def prep_refrence(self, reference: Dict[str, Any]):
        """Prepares the reference image for processing.
        Args:
            reference (dict): Source image data.
                reference['image'] (Tensor): Cropped image with some margin.
                reference['bbox'] (Tensor): Bounding box coordinates, x1,y1,x2,y2.
                reference['lm'] (Tensor): Landmark coordinates, Face_alignment.FaceAlignment 2D (68 xy points).
        """
        pass  # pass if not needed

    def prep_frame(self, frame: Dict[str, Any]):
        """Prepares source and driving frames for processing.
        Args:
            frame (dict): Frame image data.
                frame['image'] (Tensor): Cropped image with some margin.
                frame['bbox'] (Tensor): Bounding box coordinates, x1,y1,x2,y2.
                frame['lm'] (Tensor): Landmark coordinates, Face_alignment.FaceAlignment 2D (68 xy points).
        """
        pass

    def animate(
        self, source: Dict[str, Any], reference: Dict[str, Any], driving: Dict[str, Any]
    ) -> Tensor:
        """
        <description>

        Args:
            source (dict): Source image data.
            reference (dict): reference frame data.
            driving (dict): Driving frame data.

        Returns:
            Tensor: Animated frame.
        """
        return driving['image']  # return the driving image as is

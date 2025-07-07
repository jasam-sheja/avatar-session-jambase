""" """

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor

from .modules import Model1, Model2
from .utils import helper_function

__dir__ = Path(__file__).resolve().parent


class AnimatorAPI(nn.Module):
    """<description>
    NOTE: No arguments are sent to the constructor.
    """

    def __init__(self):
        super().__init__()
        with open(__dir__.joinpath("assets/config.yaml")) as f:
            config = yaml.full_load(f)

        self._load(config)

    def _load(self, config):
        self.module1 = Model1(config["module1_params"])
        self.module1.load_state_dict(
            torch.load(__dir__.joinpath("assets/weights1.pt"), map_location="cpu")
        )
        self.module1.eval()

        self.module2 = Model2(config["module1_params"])
        self.module2.load_state_dict(
            torch.load(__dir__.joinpath("assets/weights2.pt"), map_location="cpu")
        )
        self.module2.eval()

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
        raise NotImplementedError()
        # example below:
        if "kp" not in frame:
            frame["kp"] = helper_function(frame["image"], frame["lm"])

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
        raise NotImplementedError()
        # example below:
        meta = self.module1(driving["image"], driving["kp"])
        animated_frame = self.module2(driving["image"], meta)
        return animated_frame

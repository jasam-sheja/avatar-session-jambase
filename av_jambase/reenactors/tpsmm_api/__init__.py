""" """

from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import Tensor

from .modules import DenseMotionNetwork, InpaintingNetwork, KPDetector
from .utils import relative_kp

__dir__ = Path(__file__).resolve().parent


class AnimatorAPI(nn.Module):
    def __init__(self):
        super().__init__()
        with open(__dir__.joinpath("assets/vox-256.yaml")) as f:
            config = yaml.full_load(f)

        self._load(config)

    def _load(self, config):
        self.kp_detector = KPDetector(**config["model_params"]["common_params"])
        self.kp_detector.load_state_dict(
            torch.load(
                __dir__.joinpath("assets/vox-256-kp_detector.pt"), map_location="cpu"
            )
        )
        self.kp_detector.eval()
        # self.kp_detector = torch.compile(self.kp_detector)

        self.dense_motion_network = DenseMotionNetwork(
            **config["model_params"]["common_params"],
            **config["model_params"]["dense_motion_params"],
        )
        self.dense_motion_network.load_state_dict(
            torch.load(
                __dir__.joinpath("assets/vox-256-dense_motion.pt"), map_location="cpu"
            )
        )
        self.dense_motion_network.eval()
        # self.dense_motion_network = torch.compile(self.dense_motion_network)

        self.inpainting = InpaintingNetwork(
            **config["model_params"]["generator_params"],
            **config["model_params"]["common_params"],
        )
        self.inpainting.load_state_dict(
            torch.load(
                __dir__.joinpath("assets/vox-256-inpainting.pt"), map_location="cpu"
            )
        )
        self.inpainting.eval()
        # self.inpainting = torch.compile(self.inpainting)

    def prep_source(self, source: Dict[str, Any]):
        pass

    def prep_frame(self, frame: Dict[str, Any]):
        frame["kp"] = self.kp_detector(frame["image"][None])

    def prep_refrence(self, reference: Dict[str, Any]):
        pass

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
        kp_driving = relative_kp(
            kp_source=source["kp"],
            kp_driving=driving["kp"],
            kp_driving_reference=reference["kp"],
            adapt_movement_scale=False,
        )
        dense_motion = self.dense_motion_network(
            source_image=source["image"].unsqueeze(0),
            kp_driving=kp_driving,
            kp_source=source["kp"],
            bg_param=None,
            dropout_flag=False,
        )
        out = self.inpainting(source["image"].unsqueeze(0), dense_motion)["prediction"][
            0
        ]
        return out

    # def forward(self, source: Tensor, reference: Tensor, driving: Tensor) -> Tensor:
    #     """
    #     Forward pass for the AnimatorAPI.

    #     Args:
    #         source (Tensor): Source image tensor.
    #         reference (Tensor): Reference frame tensor.
    #         driving (Tensor): Driving frame tensor.

    #     Returns:
    #         Tensor: Animated frame.
    #     """
    #     source = {"image": source}
    #     reference = {"image": reference}
    #     driving = {"image": driving}

    #     self.prep_frame(source)
    #     self.prep_frame(driving)
    #     self.prep_frame(reference)
    #     self.prep_source(source)
    #     self.prep_refrence(reference)

    #     return self.animate(source, reference, driving)

""" """

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import yaml
from matplotlib import pyplot as plt
from pytorch3d.renderer import (FoVPerspectiveCameras, Materials,
                                MeshRasterizer, MeshRenderer, PointLights,
                                RasterizationSettings, SoftPhongShader,
                                TexturesUV, TexturesVertex, blending,
                                look_at_view_transform)
# --------------------------------------------
from pytorch3d.structures import Meshes
from torch import Tensor
from torchvision import transforms

from .modules import networks
from .modules.AEINet import ADDGenerator, MultilevelAttributesEncoder
from .modules.bfm import ParametricFaceModel
from .modules.gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from .modules.latent_id_mappers import TransformerMapperSplit
# ----------- allam 13-5-2025--------------------
from .utils import align_img, estimate_norm_torch, inverse_align_img, load_lm3d

__dir__ = Path(__file__).resolve().parent


class AnimatorAPI(nn.Module):
    """<description>
    NOTE: No arguments are sent to the constructor.
    """

    def __init__(self):
        super().__init__()
        # with open(__dir__.joinpath("assets/config.yaml")) as f:
        #     config = yaml.full_load(f)
        # self._load(config)

        # ------------------- define the 3DMM model--------------------------------
        self.facemodel = ParametricFaceModel(
            bfm_folder=__dir__.joinpath("assets/BFM"),
            camera_distance=10.0,
            focal=1015.0,
            center=112.0,
            is_train=False,
            default_name="BFM_model_front.mat",
        )
        self.register_buffer(
            "lm3d_std", torch.from_numpy(load_lm3d(__dir__.joinpath("assets/BFM/")))
        )
        # ------------------- define the coeff encoder model-----------------------
        self.net_recon = networks.define_net_recon(
            net_recon="resnet50", use_last_fc=False, init_path=None
        )
        self.net_recon.load_state_dict(
            torch.load(__dir__.joinpath("assets/recon_model.pth"), map_location="cpu")
        )
        self.net_recon.eval()
        # ------------------- define the lmt encoder model-----------------------
        self.mlt_encoder = MultilevelAttributesEncoder()
        self.mlt_encoder.load_state_dict(
            torch.load(__dir__.joinpath("assets/mlt_encoder.pth"), map_location="cpu")
        )
        self.mlt_encoder.eval()
        # ----------------- define PAG generator model----------------------------
        self.mlt_gen = ADDGenerator(z_id_size=512)
        self.mlt_gen.load_state_dict(
            torch.load(__dir__.joinpath("assets/mlt_gen.pth"), map_location="cpu")
        )
        self.mlt_gen.eval()
        # ----------------- define gfpgan model-----------------------------------
        # channel_multiplier=2;weight=0.5;arch='clean';upscale=2;bg_upsampler='realesrgan'
        self.gfpgan = GFPGANv1Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=2,
            decoder_load_path=None,
            fix_decoder=True,
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True,
        )
        # ----------------- define IP encryptor model-----------------------------------
        self.id_encryptor = TransformerMapperSplit()  #
        self.tex_encryptor = TransformerMapperSplit()  #
        check_point = torch.load(
            __dir__.joinpath("assets/epoch_latest_ori.pth"), map_location="cpu"
        )
        self.gfpgan.load_state_dict(check_point["gfpgan"])
        self.id_encryptor.load_state_dict(check_point["id_encryptor"])
        self.tex_encryptor.load_state_dict(check_point["tex_encryptor"])
        self.gfpgan.eval()
        self.id_encryptor.eval()
        self.tex_encryptor.eval()

        self.im_rz = transforms.Resize([256, 256])
        self.im_rz_512 = transforms.Resize([512, 512])
        self.im_rz_224 = transforms.Resize([224, 224])
        self.im_rz_160 = transforms.Resize([160, 160])
        self.normz = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        self.register_buffer(
            "CPWD1", torch.randn((1, 14, 80))
        )  # same password for encrypting shape and color
        self.register_buffer(
            "CPWD2", torch.randn((1, 14, 80))
        )  # same password for encrypting shape and color
        self.tensor_to_PIL = T.ToPILImage()
        # ===================== define Pytorch3D renderer=============================
        self.fov = 2 * np.arctan(112.0 / 1015.0) * 180 / np.pi
        im_size = 224
        raster_settings = RasterizationSettings(
            image_size=(im_size, im_size), blur_radius=0.0, faces_per_pixel=1
        )

        blend_params = blending.BlendParams(background_color=[0, 0, 0])

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(raster_settings=raster_settings),
            shader=SoftPhongShader(blend_params=blend_params),
        )

    # -----------------------------------------------------------------------------
    # def _load(self, config):
    #     pass

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
                frame['image'] (Tensor): Cropped image with some margin. Shape: (3, 256, 256) of dtype float32.
                frame['bbox'] (Tensor): Bounding box coordinates, x1,y1,x2,y2. Shape (4,) of dtype float32.
                frame['lm'] (Tensor): Landmark coordinates, Face_alignment.FaceAlignment 2D (68 xy points). Shape (68, 2) of dtype float32.
        """
        im = frame["image"]
        _, W, H = im.size()
        lm = frame["lm"].clone()

        lm[:, -1] = H - 1 - lm[:, -1]

        trans_params, algn_im, algn_lm, _ = align_img(im, lm, self.lm3d_std)
        frame["algn_frame"] = algn_im
        frame["algn_lm"] = algn_lm
        frame["trans_params"] = trans_params

    # --------------- de-identification main function---------------------------------------
    def compute_visuals_test_video(
        self,
        input_img: Tensor,
        gt_lm,
        output_coeff: Tensor,
        CPWD1: Tensor,
        CPWD2: Tensor,
    ):  # global_pose=None, color_override:bool=False
        color_override = False
        global_pose = None
        if color_override:
            render_function = self.facemodel.compute_for_render_notex
        else:
            render_function = self.facemodel.compute_for_render
        # mlt_encoder.eval()
        # mlt_gen.eval()
        # id_encryptor.eval()
        # tex_encryptor.eval()
        # gfpganopt.eval()
        id_coeffs = output_coeff[:, :80]
        tex_coeffs = output_coeff[:, 144:224]
        # pred_vertex, _, pred_color, pred_lm = render_function(output_coeff)
        with torch.no_grad():
            multi_attrib = self.mlt_encoder(self.im_rz(input_img))
            # -------------------------------------------------------
            concat1_id_coeffs = torch.cat(
                [id_coeffs.unsqueeze(1).repeat(1, 14, 1), CPWD1], dim=-1
            )
            deid_S = self.id_encryptor(concat1_id_coeffs)

            concat1_tex_coeffs = torch.cat(
                [tex_coeffs.unsqueeze(1).repeat(1, 14, 1), CPWD2], dim=-1
            )
            deid_A = self.tex_encryptor(concat1_tex_coeffs)

            output_coeff_enc1 = output_coeff.clone()
            output_coeff_enc1[:, :80] = deid_S
            output_coeff_enc1[:, 144:224] = deid_A
            # ------------- prepare the 3DMM parameters for the encrypted coffs-----------
            pred_vertex_enc1, pred_tex_enc1, pred_color_enc1, pred_lm_enc1 = (
                render_function(output_coeff_enc1)
            )
            # ------------------------------------------------------
            if not color_override:
                _, pred_color_PA_enc1 = self.mlt_gen(
                    pred_vertex_enc1, multi_attrib
                )  # pred_vertex_enc is the enc id we want to swap to ...
                pred_color_enc1 += pred_color_PA_enc1
            if global_pose is not None:
                render_coeff = output_coeff_enc1.clone()
                render_coeff[:, 224:227] = global_pose
                pred_vertex_enc1, _, _, _ = render_function(render_coeff)
            # ============== relace ndiffrast with py3D=====================
            # self.facemodel.face_buf=self.facemodel.face_buf[:,[1,2,0]]
            with torch.autocast("cuda", enabled=False):
                textures_deid = TexturesVertex(
                    verts_features=pred_color_enc1.float()
                )  #
                deid_mesh = Meshes(
                    verts=pred_vertex_enc1.float(),
                    faces=self.facemodel.face_buf.unsqueeze(0)
                    .expand(pred_vertex_enc1.shape[0], -1, -1)
                    .float(),
                    textures=textures_deid,
                )
                RGBA_DEID = self.renderer(
                    deid_mesh,
                    cameras=FoVPerspectiveCameras(
                        znear=5, zfar=15, fov=self.fov, device=input_img.device
                    ),
                    lights=PointLights(
                        location=[[0, 0.0, 1e5]],
                        ambient_color=[[1, 1, 1]],
                        specular_color=[[0.0, 0.0, 0.0]],
                        diffuse_color=[[0.0, 0.0, 0.0]],
                        device=input_img.device,
                    ),
                    materials=Materials(device=input_img.device),
                )
            RGBA_DEID = torch.flip(RGBA_DEID, dims=[2]).clamp(0, 1)
            # RGBA_deid = torch.clamp(RGBA_deid, 0, 1)

            pred_face_enc1 = RGBA_DEID[..., :3]
            pred_face_enc1 = pred_face_enc1.permute(0, 3, 1, 2)
            deid_mask = (RGBA_DEID[..., 3] > 0.5).float().unsqueeze(1).clamp_(0, 1)
            # ---------------------------------------------------------------
            # pred_mask_enc1, _, pred_face_enc1 = renderer(pred_vertex_enc1, facemodel.face_buf, feat=pred_color_enc1)
            # output_vis = pred_face* pred_mask+ (1 - pred_mask) * self.input_img
            # im_rz=transforms.Resize([512, 512])
            # input_img_numpy = 255. * input_img.detach().cpu().permute(0, 2, 3, 1).numpy()
            # ***********************************************************************
            # -------------- gfpgan should be here-----------------------------------
            pred_face_enc1 = pred_face_enc1 * deid_mask + (1 - deid_mask) * input_img
            pred_face_enc1_rz = self.normz(self.im_rz_512(pred_face_enc1).clamp_(0, 1))
            retor_im = self.gfpgan(pred_face_enc1_rz, return_rgb=False, weight=0.5)[0]

            deid_face = (self.im_rz_224((retor_im.clamp(-1, 1) + 1) / 2)).clamp_(0, 1)
            # ------------- end of gfpgan--------------------------------------------

            # handle nan values
            deid_face.nan_to_num_(posinf=1.0, neginf=0.0)
            deid_mask.nan_to_num_(posinf=1.0, neginf=0.0)

        return deid_face[0], deid_mask[0]

    # ----
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
        # raise NotImplementedError()
        # # example below:
        # meta = self.module1(driving["image"], driving["kp"])
        # animated_frame = self.module2(driving["image"], meta)
        # --------------- align the input frame----------------

        test_coeffs = self.net_recon(driving["algn_frame"][None])
        deid_face, deid_mask = self.compute_visuals_test_video(
            driving["algn_frame"][None],
            driving["algn_lm"][None],
            test_coeffs,
            self.CPWD1,
            self.CPWD2,
        )
        deid_face, _ = inverse_align_img(
            driving["image"],
            deid_face,
            driving["algn_lm"],
            torch.ones(
                1, *deid_face.shape[1:], device=deid_face.device, dtype=deid_face.dtype
            ),
            driving["trans_params"],
            target_size=224.0,
        )
        # mv_avg_id+=test_coeffs[:,0:80]# the  shape coeff of the first frame
        # mv_avg_tex+=test_coeffs[:,144: 224]
        return deid_face

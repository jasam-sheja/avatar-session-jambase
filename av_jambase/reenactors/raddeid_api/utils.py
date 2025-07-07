"""This script contains the image preprocessing code for Deep3DFaceRecon_pytorch"""

import logging
import os
import warnings
from typing import Tuple

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.io import loadmat
from skimage import transform as trans
from torch import Tensor

import kornia

# try:
#     from PIL.Image import Resampling

#     RESAMPLING_METHOD = Resampling.BICUBIC
#     Resampling_NEAREST = Resampling.NEAREST
# except ImportError:
#     from PIL.Image import BICUBIC, NEAREST

#     RESAMPLING_METHOD = BICUBIC
#     Resampling_NEAREST = NEAREST


logger = logging.getLogger(__name__)

# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# calculating least square problem for image alignment
def POS(xp: Tensor, x: Tensor) -> Tuple[Tensor, Tensor]:
    npts = xp.shape[0]

    A = torch.zeros([2 * npts, 8], dtype=xp.dtype, device=xp.device)

    A[0 : 2 * npts - 1 : 2, 0:3] = x
    A[0 : 2 * npts - 1 : 2, 3] = 1

    A[1 : 2 * npts : 2, 4:7] = x
    A[1 : 2 * npts : 2, 7] = 1

    b = xp.reshape([2 * npts, 1])

    k = torch.linalg.lstsq(A, b).solution

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (torch.linalg.norm(R1) + torch.linalg.norm(R2)) / 2
    t = torch.stack([sTx, sTy], axis=0)
    return t, s


# bounding box for 68 landmark detection
def BBRegression(points, params):

    w1 = params["W1"]
    b1 = params["B1"]
    w2 = params["W2"]
    b2 = params["B2"]
    data = points.copy()
    data = data.reshape([5, 2])
    data_mean = np.mean(data, axis=0)
    x_mean = data_mean[0]
    y_mean = data_mean[1]
    data[:, 0] = data[:, 0] - x_mean
    data[:, 1] = data[:, 1] - y_mean

    rms = np.sqrt(np.sum(data**2) / 5)
    data = data / rms
    data = data.reshape([1, 10])
    data = np.transpose(data)
    inputs = np.matmul(w1, data) + b1
    inputs = 2 / (1 + np.exp(-2 * inputs)) - 1
    inputs = np.matmul(w2, inputs) + b2
    inputs = np.transpose(inputs)
    x = inputs[:, 0] * rms + x_mean
    y = inputs[:, 1] * rms + y_mean
    w = 224 / inputs[:, 2] * rms
    rects = [x, y, w, w]
    return np.array(rects).reshape([4])


# utils for landmark detection
def img_padding(img, box):
    success = True
    bbox = box.copy()
    res = np.zeros([2 * img.shape[0], 2 * img.shape[1], 3])
    res[
        img.shape[0] // 2 : img.shape[0] + img.shape[0] // 2,
        img.shape[1] // 2 : img.shape[1] + img.shape[1] // 2,
    ] = img

    bbox[0] = bbox[0] + img.shape[1] // 2
    bbox[1] = bbox[1] + img.shape[0] // 2
    if bbox[0] < 0 or bbox[1] < 0:
        success = False
    return res, bbox, success


# utils for landmark detection
def crop(img, bbox):
    padded_img, padded_bbox, flag = img_padding(img, bbox)
    if flag:
        crop_img = padded_img[
            padded_bbox[1] : padded_bbox[1] + padded_bbox[3],
            padded_bbox[0] : padded_bbox[0] + padded_bbox[2],
        ]
        crop_img = cv2.resize(
            crop_img.astype(np.uint8), (224, 224), interpolation=cv2.INTER_CUBIC
        )
        scale = 224 / padded_bbox[3]
        return crop_img, scale
    else:
        return padded_img, 0


# utils for landmark detection
def scale_trans(img, lm, t, s):
    imgw = img.shape[1]
    imgh = img.shape[0]
    M_s = np.array(
        [[1, 0, -t[0] + imgw // 2 + 0.5], [0, 1, -imgh // 2 + t[1]]], dtype=np.float32
    )
    img = cv2.warpAffine(img, M_s, (imgw, imgh))
    w = int(imgw / s * 100)
    h = int(imgh / s * 100)
    img = cv2.resize(img, (w, h))
    lm = (
        np.stack([lm[:, 0] - t[0] + imgw // 2, lm[:, 1] - t[1] + imgh // 2], axis=1)
        / s
        * 100
    )

    left = w // 2 - 112
    up = h // 2 - 112
    bbox = [left, up, 224, 224]
    cropped_img, scale2 = crop(img, bbox)
    assert scale2 != 0
    t1 = np.array([bbox[0], bbox[1]])

    # back to raw img s * crop + s * t1 + t2
    t1 = np.array([w // 2 - 112, h // 2 - 112])
    scale = s / 100
    t2 = np.array([t[0] - imgw / 2, t[1] - imgh / 2])
    inv = (scale / scale2, scale * t1 + t2.reshape([2]))
    return cropped_img, inv


# utils for landmark detection
def align_for_lm(img, five_points):
    five_points = np.array(five_points).reshape([1, 10])
    params = loadmat("util/BBRegressorParam_r.mat")
    bbox = BBRegression(five_points, params)
    assert bbox[2] != 0
    bbox = np.round(bbox).astype(np.int32)
    crop_img, scale = crop(img, bbox)
    return crop_img, scale, bbox


def general_crop(
    img: Tensor, left: int, up: int, right: int, down: int, output: Tensor = None
) -> Tensor:
    """
    Crop a PyTorch tensor image to mimic PIL.Image.crop with negative coordinates.
    Args:
        img (torch.Tensor): Input image tensor of shape (C, H, W).
        left (int): Left crop coordinate.
        up (int): Upper crop coordinate.
        right (int): Right crop coordinate.
        down (int): Lower crop coordinate.
    """
    Hi, Wi = img.size(1), img.size(2)
    Hc, Wc = down - up, right - left
    if output is not None:
        assert (
            output.dtype == img.dtype
        ), f"Output dtype {output.dtype} does not match input dtype {img.dtype}"
        assert (
            output.device == img.device
        ), f"Output device {output.device} does not match input device {img.device}"
        assert (
            output.size(1) == Hc
        ), f"Output height {output.size(1)} does not match expected height {Hc}"
        assert (
            output.size(2) == Wc
        ), f"Output width {output.size(2)} does not match expected width {Wc}"
    else:
        output = torch.zeros(img.size(0), Hc, Wc, dtype=img.dtype, device=img.device)
    # Calculate the crop coordinates
    left_i = max(min(left, Wi), 0)
    left_c = left_i - left  # should be positive
    right_i = max(min(right, Wi), 0)
    right_c = right_i - left  # should be positive
    up_i = max(min(up, Hi), 0)
    up_c = up_i - up  # should be positive
    down_i = max(min(down, Hi), 0)
    down_c = down_i - up  # should be positive
    if (
        left_c < 0
        or up_c < 0
        or right_c < 0
        or down_c < 0
        or left_c > Wc
        or up_c > Hc
        or right_c > Wc
        or down_c > Hc
    ):
        return output
    output[:, up_c:down_c, left_c:right_c] = img[
        :, up_i:down_i, left_i:right_i
    ]  # Copy the cropped region
    return output


# resize and crop images for face reconstruction
def resize_n_crop_img(
    img: Tensor,
    lm: Tensor,
    t: Tensor,
    s: Tensor,
    target_size: float = 224.0,
    mask: Tensor | None = None,
):
    _, w0, h0 = img.size()
    w = (w0 * s).round()
    h = (h0 * s).round()
    left = (w / 2 - target_size / 2 + (t[0] - w0 / 2) * s).round()
    right = left + target_size
    up = (h / 2 - target_size / 2 + (h0 / 2 - t[1]) * s).round()
    below = up + target_size

    w = w.int().item()
    h = h.int().item()
    left = left.int().item()
    right = right.int().item()
    up = up.int().item()
    below = below.int().item()

    img = F.interpolate(img[None], size=(w, h), mode="bicubic", align_corners=False)[0]
    img = general_crop(img, left, up, right, below).clip_(0, 1)

    if mask is not None:
        mask = F.interpolate(mask[None], size=(w, h), mode="bicubic")[0]
        mask = general_crop(mask, left, up, right, below).clip_(0, 1)

    lm = torch.stack([lm[:, 0] - t[0] + w0 / 2, lm[:, 1] - t[1] + h0 / 2], axis=1) * s
    lm = lm - torch.tensor(
        [(w / 2 - target_size / 2), (h / 2 - target_size / 2)],
        dtype=lm.dtype,
        device=lm.device,
    ).reshape(1, 2)

    return img, lm, mask


def pad_torch(
    img: torch.Tensor, left: int, up: int, right: int, down: int
) -> torch.Tensor:
    """
    Pad a PyTorch tensor image to mimic PIL.Image.crop with negative coordinates.

    Args:
        img (torch.Tensor): Input image tensor of shape (C, H, W).
        left (int): Padding on the left.
        up (int): Padding on the top.
        right (int): Padding on the right.
        down (int): Padding on the bottom.

    Returns:
        torch.Tensor: Padded image tensor.
    """
    print(f"input: {img.shape}, left: {left}, up: {up}, right: {right}, down: {down}")
    # Calculate padding values for F.pad (order: (left, right, top, bottom))
    padding = (max(left, 0), max(right, 0), max(up, 0), max(down, 0))
    print(f"padding: {padding}")
    # Apply padding
    img_padded = F.pad(img, padding, mode="constant", value=0)
    print(f"padded: {img_padded.shape}")
    # Crop if any padding values are negative
    if left < 0 or up < 0 or right < 0 or down < 0:
        h, w = img_padded.shape[1:]  # Get height and width
        left_crop = max(-left, 0)
        up_crop = max(-up, 0)
        right_crop = w - max(-right, 0)
        down_crop = h - max(-down, 0)
        img_padded = img_padded[:, up_crop:down_crop, left_crop:right_crop]

    return img_padded


def inverse_resize_n_crop_img(
    img: Tensor,
    lm: Tensor,
    t: Tensor,
    s: Tensor,
    background: Tensor,
    target_size: float = 224.0,
    mask: Tensor | None = None,
):
    # original width and height before resizing and cropping
    w0, h0 = background.size()[-2:]
    w = (w0 * s).round()
    h = (h0 * s).round()

    # Calculate the crop coordinates used in the original function
    left = (w / 2 - target_size / 2 + (t[0] - w0 / 2) * s).round()
    # right = left + target_size
    up = (h / 2 - target_size / 2 + (h0 / 2 - t[1]) * s).round()
    # below = up + target_size

    # Convert to integers
    w = w.int().item()
    h = h.int().item()
    left = left.int().item()
    # right = right.int().item()
    up = up.int().item()
    # below = below.int().item()

    # Step 1: Reverse the crop by padding the image back to (w, h)
    img = general_crop(img, -left, -up, w - left, h - up)  # Padding
    img = F.interpolate(img[None], size=(w0, h0), mode="bicubic", align_corners=False)[
        0
    ].clip_(
        0, 1
    )  # Resize to original size

    # If mask is provided, reverse crop and resize in the same way
    if mask is not None:
        mask = general_crop(mask, -left, -up, w - left, h - up)
        mask = F.interpolate(mask[None], size=(w0, h0), mode="bicubic")[0].clip_(0, 1)

    # Step 2: Reverse transformations on landmarks
    lm = (
        lm
        + torch.tensor(
            [(w / 2 - target_size / 2), (h / 2 - target_size / 2)],
            dtype=lm.dtype,
            device=lm.device,
        ).reshape(1, 2)
    ) / s
    lm = torch.stack([lm[:, 0] + t[0] - w0 / 2, lm[:, 1] + t[1] - h0 / 2], axis=1)

    return img, lm, mask


# utils for face reconstruction
def extract_5p(lm: Tensor) -> Tensor:
    return torch.stack(
        [
            (lm[36] + lm[39]) / 2,
            (lm[42] + lm[45]) / 2,
            lm[30],
            lm[48],
            lm[54],
        ],
        axis=0,
    )


# utils for face reconstruction
def align_img(
    img: Tensor,
    lm: Tensor,
    lm3D: Tensor,
    mask=None,
    target_size=224.0,
    rescale_factor=102.0,
):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --PIL.Image  (target_size, target_size)

    Parameters:
        img                --Tensor  (3, raw_H, raw_W)
        lm                 --Tensor  (68, 2), y direction is opposite to v direction
        lm3D               --Tensor  (5, 3)
        mask               --Tensor (3, raw_H, raw_W)
    """

    _, w0, h0 = img.size()
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p, lm3D)
    s = rescale_factor / s
    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(
        img, lm, t, s, target_size=target_size, mask=mask
    )
    trans_params = torch.tensor(
        [w0, h0, s, t.squeeze()[0], t.squeeze()[1]], device=img.device
    )

    return trans_params, img_new, lm_new, mask_new


import torch


def composite_torch(image1: Tensor, image2: Tensor, mask: Tensor) -> Tensor:
    """
    Mimic PIL.Image.composite using PyTorch.

    Args:
        image1 (Tensor): The first image tensor of shape (C, H, W).
        image2 (Tensor): The second image tensor of shape (C, H, W).
        mask (Tensor): The mask tensor of shape (1, H, W) or (H, W), with values in [0, 1].

    Returns:
        Tensor: The composite image tensor of shape (C, H, W).
    """
    # Ensure the mask has the same shape as the images
    if mask.dim() == 2:  # If mask is (H, W), add a channel dimension
        mask = mask.unsqueeze(0)
    elif mask.dim() == 3 and mask.shape[0] != 1:
        raise ValueError("Mask must have shape (1, H, W) or (H, W).")

    # Ensure the mask is broadcastable to the image shape
    if mask.shape[0] == 1:
        mask = mask.expand_as(image1)

    # Perform the composite operation

    layer1 = image1 * mask
    layer2 = image2 * (1 - mask)
    composite_image = layer1 + layer2

    return composite_image


def inverse_align_img(
    background: Tensor,
    img: Tensor,
    lm: Tensor,
    mask: Tensor,
    trans_params,
    target_size=224.0,
):
    # return mask, lm
    # return composite_torch(img, torch.ones_like(img), mask), lm
    w0, h0, s = trans_params[:3]
    if (w0, h0) != background.size()[-2:]:
        logger.warning("background size is different from the original image size")
    t = trans_params[3:][:, np.newaxis]
    im_back, lm_back, back_mask = inverse_resize_n_crop_img(
        img, lm, t, s, background, target_size=target_size, mask=mask
    )  # mask=Image.new('L', img.size, 255)
    if back_mask is not None:
        # erode the mask to avoid artifacts
        back_mask = kornia.morphology.erosion(
            back_mask[None], torch.ones(5, 5, device=back_mask.device)
        )[0]
        back_mask = kornia.morphology.erosion(
            back_mask[None], torch.ones(5, 5, device=back_mask.device)
        )[0]
        im_back = composite_torch(im_back, background, back_mask)
    return im_back, lm_back


# utils for face recognition model
def estimate_norm(lm_68p: Tensor, H):
    # from https://github.com/deepinsight/insightface/blob/c61d3cd208a603dfa4a338bd743b320ce3e94730/recognition/common/face_align.py#L68
    """
    Return:
        trans_m            --numpy.array  (2, 3)
    Parameters:
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        H                  --int/float , image height
    """
    lm = extract_5p(lm_68p)
    lm[:, -1] = H - 1 - lm[:, -1]
    # TODO: use another lib that supports pytorch
    tform = trans.SimilarityTransform()
    src = np.array(
        [
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    tform.estimate(lm.cpu().numpy(), src)
    M = tform.params
    if np.linalg.det(M) == 0:
        M = np.eye(3)

    return M[0:2, :]


def estimate_norm_torch(lm_68p, H):
    lm_68p_ = lm_68p
    M = []
    for i in range(lm_68p_.shape[0]):
        M.append(estimate_norm(lm_68p_[i], H))
    # TODO: Fix devices and numpy torch confilicts
    M = torch.tensor(np.array(M), dtype=torch.float32).to(lm_68p.device)
    return M


"""This script is to load 3D face model for Deep3DFaceRecon_pytorch
"""

import os.path as osp
from array import array

import numpy as np
from PIL import Image
from scipy.io import loadmat, savemat


# load expression basis
def LoadExpBasis(bfm_folder="BFM"):
    n_vertex = 53215
    Expbin = open(osp.join(bfm_folder, "Exp_Pca.bin"), "rb")
    exp_dim = array("i")
    exp_dim.fromfile(Expbin, 1)
    expMU = array("f")
    expPC = array("f")
    expMU.fromfile(Expbin, 3 * n_vertex)
    expPC.fromfile(Expbin, 3 * exp_dim[0] * n_vertex)
    Expbin.close()

    expPC = np.array(expPC)
    expPC = np.reshape(expPC, [exp_dim[0], -1])
    expPC = np.transpose(expPC)

    expEV = np.loadtxt(osp.join(bfm_folder, "std_exp.txt"))

    return expPC, expEV


# transfer original BFM09 to our face model
def transferBFM09(bfm_folder="BFM"):
    print("Transfer BFM09 to BFM_model_front......")
    original_BFM = loadmat(osp.join(bfm_folder, "01_MorphableModel.mat"))
    shapePC = original_BFM["shapePC"]  # shape basis
    shapeEV = original_BFM["shapeEV"]  # corresponding eigen value
    shapeMU = original_BFM["shapeMU"]  # mean face
    texPC = original_BFM["texPC"]  # texture basis
    texEV = original_BFM["texEV"]  # eigen value
    texMU = original_BFM["texMU"]  # mean texture

    expPC, expEV = LoadExpBasis()

    # transfer BFM09 to our face model

    idBase = shapePC * np.reshape(shapeEV, [-1, 199])
    idBase = idBase / 1e5  # unify the scale to decimeter
    idBase = idBase[:, :80]  # use only first 80 basis

    exBase = expPC * np.reshape(expEV, [-1, 79])
    exBase = exBase / 1e5  # unify the scale to decimeter
    exBase = exBase[:, :64]  # use only first 64 basis

    texBase = texPC * np.reshape(texEV, [-1, 199])
    texBase = texBase[:, :80]  # use only first 80 basis

    # our face model is cropped along face landmarks and contains only 35709 vertex.
    # original BFM09 contains 53490 vertex, and expression basis provided by Guo et al. contains 53215 vertex.
    # thus we select corresponding vertex to get our face model.

    index_exp = loadmat(osp.join(bfm_folder, "BFM_front_idx.mat"))
    index_exp = index_exp["idx"].astype(np.int32) - 1  # starts from 0 (to 53215)

    index_shape = loadmat(osp.join(bfm_folder, "BFM_exp_idx.mat"))
    index_shape = (
        index_shape["trimIndex"].astype(np.int32) - 1
    )  # starts from 0 (to 53490)
    index_shape = index_shape[index_exp]

    idBase = np.reshape(idBase, [-1, 3, 80])
    idBase = idBase[index_shape, :, :]
    idBase = np.reshape(idBase, [-1, 80])

    texBase = np.reshape(texBase, [-1, 3, 80])
    texBase = texBase[index_shape, :, :]
    texBase = np.reshape(texBase, [-1, 80])

    exBase = np.reshape(exBase, [-1, 3, 64])
    exBase = exBase[index_exp, :, :]
    exBase = np.reshape(exBase, [-1, 64])

    meanshape = np.reshape(shapeMU, [-1, 3]) / 1e5
    meanshape = meanshape[index_shape, :]
    meanshape = np.reshape(meanshape, [1, -1])

    meantex = np.reshape(texMU, [-1, 3])
    meantex = meantex[index_shape, :]
    meantex = np.reshape(meantex, [1, -1])

    # other info contains triangles, region used for computing photometric loss,
    # region used for skin texture regularization, and 68 landmarks index etc.
    other_info = loadmat(osp.join(bfm_folder, "facemodel_info.mat"))
    frontmask2_idx = other_info["frontmask2_idx"]
    skinmask = other_info["skinmask"]
    keypoints = other_info["keypoints"]
    point_buf = other_info["point_buf"]
    tri = other_info["tri"]
    tri_mask2 = other_info["tri_mask2"]

    # save our face model
    savemat(
        osp.join(bfm_folder, "BFM_model_front.mat"),
        {
            "meanshape": meanshape,
            "meantex": meantex,
            "idBase": idBase,
            "exBase": exBase,
            "texBase": texBase,
            "tri": tri,
            "point_buf": point_buf,
            "tri_mask2": tri_mask2,
            "keypoints": keypoints,
            "frontmask2_idx": frontmask2_idx,
            "skinmask": skinmask,
        },
    )


# load landmarks for standard face, which is used for image preprocessing
def load_lm3d(bfm_folder):

    Lm3D = loadmat(osp.join(bfm_folder, "similarity_Lm3D_all.mat"))
    Lm3D = Lm3D["lm"]

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack(
        [
            Lm3D[lm_idx[0], :],
            np.mean(Lm3D[lm_idx[[1, 2]], :], 0),
            np.mean(Lm3D[lm_idx[[3, 4]], :], 0),
            Lm3D[lm_idx[5], :],
            Lm3D[lm_idx[6], :],
        ],
        axis=0,
    )
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D

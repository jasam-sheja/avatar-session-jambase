import numpy as np
import torch
import torch.nn as nn
from kornia.geometry import warp_affine
import torch.nn.functional as F
from models.encoders.model_irse import Backbone
from  torchvision.utils import save_image


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

def resize_n_crop(image, M, dsize=112):
    # image: (b, c, h, w)
    # M   :  (b, 2, 3)
    return warp_affine(image, M, dsize=(dsize, dsize))
#----------------------------------------------------------------
# a class for calculating all kinds of id loss
class IDLossExtractor(torch.nn.Module):
    def __init__(self,Backbone,requires_grad=False):
        #requirement input 256*256 cropped and normalized face image tensor
        #self.opts=opts
        super(IDLossExtractor, self).__init__()
        Backbone.eval()
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))

        self.margin=0 #self.opts.id_cos_margin if self.opts.id_cos_margin is not None  else 0.1

        self.output_layer=Backbone.output_layer
        self.input_layer=Backbone.input_layer
        body = Backbone.body
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(3):
            self.slice1.add_module(str(x), body[x])
        for x in range(3, 7):
            self.slice2.add_module(str(x), body[x])
        for x in range(7, 21):
            self.slice3.add_module(str(x), body[x])
        for x in range(21, 24):
            self.slice4.add_module(str(x), body[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def extract_feats(self, X):
        #preprocessing
        if X.shape[-1] != 256:
            X = self.pool(X)
        h=X[:, :, 35:223, 32:220]
        h=self.face_pool(h)
        #feed into to network
        h=self.input_layer(h)
        h0=h
        h = self.slice1(h)
        h1 = h
        h = self.slice2(h)
        h2 = h
        h = self.slice3(h)
        h3 = h
        h = self.slice4(h)
        h4 = h
        h = self.output_layer(h)
        h = l2_norm(h)
        return [h0,h1,h2,h3,h4,h]


    def calculate_loss(self,features1,features2):

        assert len(features1)==len(features2)

        #percept_loss,id_loss=0,0
        #or i in range(len(features1)-1):
            #print(features1[i].shape,features2[i].shape)
           # percept_loss+=F.l1_loss(features1[i],features2[i])
        def random_orthogonal_matrix(n):
            # 生成随机矩阵
            random_matrix = torch.randn(n, n)
            # 计算 QR 分解
            q, _ = torch.qr(random_matrix)
            return q
        # 生成一个512维度的随机正交矩阵
        matrix_512d = random_orthogonal_matrix(512).cuda()
        EM_loss = 1-torch.cosine_similarity(torch.matmul(features2[-1], matrix_512d),features1[-1]).clamp(min=self.margin).mean()

        #avg_deid_feat=features1[-1].mean(dim=0,keepdims=True).expand(features1[-1].shape[0],-1,-1)

        id_loss=1-torch.cosine_similarity(features1[-1],features2[-1]).clamp(min=self.margin).mean()
        rev_id_loss=1+torch.cosine_similarity(features1[-1],features2[-1]).mean() # # no mean for testing
        #deid_avg_loss=(1-torch.cosine_similarity(avg_deid_feat,features2[-1]).clamp(min=self.margin)).mean()


        return id_loss,rev_id_loss+EM_loss #percept_loss


    def forward(self,img1,img2):

        features1=self.extract_feats(img1)
        features2=self.extract_feats(img2)

        return self.calculate_loss(features1,features2)
### perceptual level loss
class GANLoss(nn.Module):
    def __init__(self, gan_mode='hinge', target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            return torch.ones_like(input).detach()
        else:
            return torch.zeros_like(input).detach()

    def get_zero_tensor(self, input):
        return torch.zeros_like(input).detach()

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = F.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return F.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = torch.min(input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
                else:
                    minval = torch.min(-input - 1, self.get_zero_tensor(input))
                    loss = -torch.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -torch.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)

class AEI_Loss(nn.Module):
    def __init__(self):
        super(AEI_Loss, self).__init__()

        self.att_weight = 10
        self.id_weight = 5
        self.rec_weight = 10

        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def att_loss(self, z_att_X, z_att_Y):
        loss = 0
        for i in range(8):
            loss += self.l2(z_att_X[i], z_att_Y[i])
        return 0.5*loss

    def id_loss(self, z_id_X, z_id_Y):
        inner_product = (torch.bmm(z_id_X.unsqueeze(1), z_id_Y.unsqueeze(2)).squeeze())
        return self.l1(torch.ones_like(inner_product), inner_product)

    def rec_loss(self, X, Y, same):
        same = same.unsqueeze(-1).unsqueeze(-1)
        same = same.expand(X.shape)
        X = torch.mul(X, same)
        Y = torch.mul(Y, same)
        return 0.5*self.l2(X, Y)

    def forward(self, X, Y, z_att_X, z_att_Y, z_id_X, z_id_Y, same):

        att_loss = self.att_loss(z_att_X, z_att_Y)
        id_loss = self.id_loss(z_id_X, z_id_Y)
        rec_loss = self.rec_loss(X, Y, same)

        return self.att_weight*att_loss + self.id_weight*id_loss + self.rec_weight*rec_loss, att_loss, id_loss, rec_loss


class PerceptualLoss(nn.Module):
    def __init__(self, recog_net, input_size=112):
        super(PerceptualLoss, self).__init__()
        self.recog_net = recog_net
        self.preprocess = lambda x: 2 * x - 1
        self.input_size=input_size
    def forward(self, imageA, imageB, M):
        """
        1 - cosine distance
        Parameters:
            imageA       --torch.tensor (B, 3, H, W), range (0, 1) , RGB order
            imageB       --same as imageA
        """

        imageA = self.preprocess(resize_n_crop(imageA, M, self.input_size))
        imageB = self.preprocess(resize_n_crop(imageB, M, self.input_size))

        # freeze bn
        self.recog_net.eval()

        id_featureA = F.normalize(self.recog_net(imageA), dim=-1, p=2)
        id_featureB = F.normalize(self.recog_net(imageB), dim=-1, p=2)
        cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
        # assert torch.sum((cosine_d > 1).float()) == 0
        return torch.sum(1 - cosine_d) / cosine_d.shape[0]

def perceptual_loss(id_featureA, id_featureB):
    cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
        # assert torch.sum((cosine_d > 1).float()) == 0
    return torch.sum(1 - cosine_d) / cosine_d.shape[0]

def identity_loss(id_featureA, id_featureB,deid=False):
    if deid:
        cosine_d = torch.cosine_similarity(id_featureA, id_featureB).mean()
    else:
        cosine_d = 1-torch.cosine_similarity(id_featureA, id_featureB).clamp(0).mean()
        # assert torch.sum((cosine_d > 1).float()) == 0
    return cosine_d

def perceptual_loss_enc(id_featureA, id_featureB):
    cosine_d = torch.sum(id_featureA * id_featureB, dim=-1)
        # assert torch.sum((cosine_d > 1).float()) == 0
    return torch.sum(cosine_d) / cosine_d.shape[0]

### image level loss
#-------------------------------------------------------------------------------
landmark_groups = {
    "jaw": list(range(0, 17)),       # Contour
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose": list(range(27, 36)),     # Includes bridge and tip
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "mouth": list(range(48, 68)),    # Outer + inner lips
}

class FacialComponentLoss(nn.Module):
    def __init__(self, landmark_groups=landmark_groups, weights=None, loss_type="l1"):
        super().__init__()
        self.landmark_groups = landmark_groups
        self.weights = weights or {k: 1.0 for k in landmark_groups}  # Equal weights by default
        self.loss = nn.L1Loss() if loss_type == "l1" else nn.MSELoss()

    def forward(self, pred_landmarks, target_landmarks):
        """
        Args:
            pred_landmarks: (B, 68, 2) - Predicted 2D landmarks
            target_landmarks: (B, 68, 2) - Ground truth from face_alignment
        """
        total_loss = 0.0
        for region, indices in self.landmark_groups.items():
            pred_region = pred_landmarks[:, indices]
            target_region = target_landmarks[:, indices]
            total_loss += self.weights[region] * self.loss(pred_region, target_region)
        return total_loss / len(self.landmark_groups)
#===============================================================================
import torch
import cv2
import numpy as np
from torchvision.models import vgg16
from torchvision import transforms #import Normalize
im_rz=transforms.Resize([64, 64])
im_rz_224=transforms.Resize([224, 224])
#-------------------------------------------------------------------------------
import torch
import torchvision.transforms.functional as TF

def crop_eye_batch(batch_images, batch_landmarks, eye_indices, padding=10):
    """
    Crops the eye regions from a batch of images based on facial landmarks.

    :param batch_images: Tensor of shape (B, 3, 256, 256), batch of images.
    :param batch_landmarks: Tensor of shape (B, 68, 2), batch of facial landmarks.
    :param eye_indices: List of landmark indices for the eye.
    :param padding: Extra padding around the eye.
    :return: Tensor of cropped eyes (B, 3, H, W), where H and W vary.
    """
    cropped_eyes = []

    for i in range(batch_images.shape[0]):  # Loop over batch
        image = batch_images[i]  # Shape: (3, 256, 256)
        landmarks = batch_landmarks[i]  # Shape: (68, 2)

        # Extract eye landmarks
        eye_points = landmarks[eye_indices]  # Shape: (6, 2)

        # Get bounding box
        eye_min = torch.min(eye_points, dim=0).values - padding
        eye_max = torch.max(eye_points, dim=0).values + padding

        # Convert to int and ensure bounds are within the image size
        x_min, y_min = max(0, int(eye_min[0])), max(0, int(eye_min[1]))
        x_max, y_max = min(256, int(eye_max[0])), min(256, int(eye_max[1]))

        # Crop using torchvision
        cropped_eye = TF.crop(image, top=y_min, left=x_min, height=y_max - y_min, width=x_max - x_min)
        cropped_eyes.append(im_rz(cropped_eye))

    # Stack results into a batch tensor
    return torch.stack(cropped_eyes)  # Shape: (B, 3, H, W)
# def get_eyes_region(img, landmarks, expand_ratio=1.4):
#     """
#     Robust eye region cropper with coordinate validation
#     Args:
#         image: (3, H, W) tensor in [0,1] or (H, W, 3) numpy array
#         landmarks: (68, 2) numpy array or tensor
#         expand_ratio: bbox expansion factor
#         output_size: (width, height) of output
#     Returns:
#         (1, 3, h, w) tensor in [0,1]
#     """

#         # 1. Convert and validate landmarks
#     #if isinstance(landmarks, torch.Tensor):
#     landmarks = landmarks.detach().cpu().numpy()
# #======= borrowed from gfpgan===================================================
#         # get landmarks for each component
#     map_left_eye = list(range(36, 42))
#     map_right_eye = list(range(42, 48))
#     map_mouth = list(range(27, 35))

#     # eye_left
#     mean_left_eye = np.mean(landmarks[map_left_eye], 0)  # (x, y)
#     half_len_left_eye = np.max((np.max(np.max(landmarks[map_left_eye], 0) - np.min(landmarks[map_left_eye], 0)) / 2, 16))
#     #item_dict['left_eye'] = [mean_left_eye[0], mean_left_eye[1], half_len_left_eye]
#     # mean_left_eye[0] = 512 - mean_left_eye[0]  # for testing flip
#     half_len_left_eye *= expand_ratio
#     loc_left_eye = np.hstack((mean_left_eye - half_len_left_eye + 1, mean_left_eye + half_len_left_eye)).astype(int)

#     eye_left_img = img[:, loc_left_eye[0]:loc_left_eye[2],loc_left_eye[1]:loc_left_eye[3]]
#     eye_left_img=im_rz(torch.from_numpy(eye_left_img))
#     save_image(eye_left_img[0],'/home/allam/GFPGAN/eyes/eye_left_img.png')

#     # eye_right
#     mean_right_eye = np.mean(landmarks[map_right_eye], 0)
#     half_len_right_eye = np.max((np.max(np.max(landmarks[map_right_eye], 0) - np.min(landmarks[map_right_eye], 0)) / 2, 16))
#     #item_dict['right_eye'] = [mean_right_eye[0], mean_right_eye[1], half_len_right_eye]
#     # mean_right_eye[0] = 512 - mean_right_eye[0]  # # for testing flip
#     half_len_right_eye *= expand_ratio
#     loc_right_eye = np.hstack((mean_right_eye - half_len_right_eye + 1, mean_right_eye + half_len_right_eye)).astype(int)

#     eye_right_img = img[:, loc_right_eye[0]:loc_right_eye[2],loc_right_eye[1]:loc_right_eye[3]]
#     #cv2.imwrite(f'tmp/{item_idx:08d}_eye_right.png', eye_right_img * 255)
#     eye_right_img=im_rz(torch.from_numpy(eye_right_img))
#     save_image(eye_right_img[0],'/home/allam/GFPGAN/eyes/eye_right_img.png')

#     # mouth
#     mean_mouth = np.mean(landmarks[map_mouth], 0)
#     half_len_mouth = np.max((np.max(np.max(landmarks[map_mouth], 0) - np.min(landmarks[map_mouth], 0)) / 2, 16))
#     #item_dict['mouth'] = [mean_mouth[0], mean_mouth[1], half_len_mouth]
#     # mean_mouth[0] = 512 - mean_mouth[0]  # for testing flip
#     loc_mouth = np.hstack((mean_mouth - half_len_mouth + 1, mean_mouth + half_len_mouth)).astype(int)

#     nose_img = img[:, loc_mouth[0]:loc_mouth[2],loc_mouth[1]:loc_mouth[3]]
#     #cv2.imwrite(f'tmp/{item_idx:08d}_mouth.png', mouth_img * 255)
#     nose_img=im_rz(torch.from_numpy(nose_img))
#     save_image(nose_img[0],'/home/allam/GFPGAN/eyes/nose_img.png')


#     #print(landmarks.shape)
#     #eyes_landmarks = landmarks[36:48,:]  # Both eyes
#    # nose_lm = landmarks[27:35,:]  # Both eyes
#     #

#     #         # eye_left
#     # mean_left_eye = np.mean(eyes_landmarks[0:6], 0)  # (x, y)
#     # mean_right_eye = np.mean(eyes_landmarks[6:12], 0)  # (x, y)
#     # nose_lm_mean =np.mean(landmarks[27:35,:] , 0)  # (x, y)
#     # #----------------------------------left eye--------------------------------
#     # print(eyes_landmarks[0:6])
#     # exit()
#     # half_len_left_eye = np.max((np.max(np.max(eyes_landmarks[0:6], 0) - np.min(eyes_landmarks[0:6], 0)) / 2, 16))
#     # #item_dict['left_eye'] = [mean_left_eye[0], mean_left_eye[1], half_len_left_eye]
#     # # mean_left_eye[0] = 512 - mean_left_eye[0]  # for testing flip
#     # half_len_left_eye *= expand_ratio
#     # loc_left_eye = np.hstack((mean_left_eye - half_len_left_eye + 1, mean_left_eye + half_len_left_eye)).astype(int)

#     # eye_left_img = image[:,loc_left_eye[1]:loc_left_eye[3], loc_left_eye[0]:loc_left_eye[2]]

#     # eye_left_img=im_rz(torch.from_numpy(eye_left_img))
#     # save_image(eye_left_img[0],'/home/allam/GFPGAN/eyes/eye_left_img.png')
#     # #----------------------------------right eye--------------------------------
#     # half_len_right_eye = np.max((np.max(np.max(eyes_landmarks[6:12], 0) - np.min(eyes_landmarks[6:12], 0)) / 2, 16))
#     # #item_dict['left_eye'] = [mean_left_eye[0], mean_left_eye[1], half_len_left_eye]
#     # # mean_left_eye[0] = 512 - mean_left_eye[0]  # for testing flip
#     # half_len_right_eye *= expand_ratio
#     # loc_right_eye = np.hstack((mean_right_eye - half_len_right_eye + 1, mean_right_eye + half_len_right_eye)).astype(int)

#     # eye_right_img = image[:,loc_right_eye[1]:loc_right_eye[3], loc_right_eye[0]:loc_right_eye[2]]
#     # eye_right_img=im_rz(torch.from_numpy(eye_right_img))
#     # save_image(eye_right_img[0],'/home/allam/GFPGAN/eyes/eye_right_img.png')
#     # #------------------------------------nose---------------------------------------------------
#     # nose_lm_max = np.max((np.max(np.max(nose_lm, 0) - np.min(nose_lm, 0)) / 2, 16))
#     # nose_lm_max *= expand_ratio
#     # nose_loc = np.hstack((nose_lm_mean - nose_lm_max + 1, nose_lm_mean + nose_lm_max)).astype(int)

#     # nose_img = image[:,nose_loc[1]:nose_loc[3], nose_loc[0]:nose_loc[2]]

#     # nose_img=im_rz(torch.from_numpy(nose_img))
#     # save_image(nose_img[0],'/home/allam/GFPGAN/eyes/nose_img.png')
#     # # print(eye_left_img.shape)
#     # # print(eye_right_img.shape)
#     # # exit()
#     return eye_left_img,eye_right_img,nose_img
#-------------------------------------------------------------------------------

# Initialize VGG (up to relu3_3)
vgg = vgg16(pretrained=True).features[:16].eval().cuda()

# Normalize input (VGG expects ImageNet stats)
vgg_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda()
vgg_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda()

#--------------------------------------------------------------------------------
import torch
import numpy as np
import cv2

def extract_eyes_batch(image_batch, landmarks_batch, padding_ratio=0.3, output_size=(64, 64)):
    """
    Extract left and right eye regions from a batch of images and landmarks

    Args:
        image_batch: Tensor of shape (B, 3, H, W) in range [0,1] or [0,255]
        landmarks_batch: Tensor of shape (B, 68, 2) in pixel coordinates
        padding_ratio: Additional padding around eyes (0.0-1.0)
        output_size: Tuple (width, height) for resizing eye regions

    Returns:
        Tuple of (left_eyes, right_eyes) where each is a tensor of shape (B, 3, h, w)
        Non-visible eyes are zero-filled tensors
    """
    # Convert to numpy if needed and ensure correct range
    if torch.is_tensor(image_batch):
        device = image_batch.device
        images_np = image_batch.permute(0, 2, 3, 1).detach().cpu().numpy()  # BxHxWxC
        if image_batch.max() <= 1.0:
            images_np = (images_np * 255).astype(np.uint8)
        else:
            images_np = images_np.astype(np.uint8)
    else:
        raise ValueError("Input must be a tensor")

    if torch.is_tensor(landmarks_batch):
        landmarks_np = landmarks_batch.detach().cpu().numpy()
    else:
        landmarks_np = np.array(landmarks_batch)

    batch_size = images_np.shape[0]
    left_eyes = []
    right_eyes = []

    # Define eye region points (indices for 68-point model)
    LEFT_EYE_POINTS = list(range(36, 42))
    RIGHT_EYE_POINTS = list(range(42, 48))

    for i in range(batch_size):
        img = images_np[i]
        landmarks = landmarks_np[i]

        # Process left eye
        left_eye_pts = landmarks[LEFT_EYE_POINTS]
        left_eye = extract_single_eye(img, left_eye_pts, padding_ratio, output_size)

        # Process right eye
        right_eye_pts = landmarks[RIGHT_EYE_POINTS]
        right_eye = extract_single_eye(img, right_eye_pts, padding_ratio, output_size)

        left_eyes.append(left_eye)
        right_eyes.append(right_eye)

    # Convert lists to tensors and handle non-visible eyes
    def process_eye_list(eye_list):
        # Replace None with zero arrays
        eye_tensors = []
        for eye in eye_list:
            if eye is None:
                eye = np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)
            eye_tensors.append(eye)

        # Stack and convert to tensor
        eyes_np = np.stack(eye_tensors)  # BxHxWxC
        eyes_tensor = torch.from_numpy(eyes_np).permute(0, 3, 1, 2).float()  # BxCxHxW

        # Normalize if original was in [0,1]
        if image_batch.max() <= 1.0:
            eyes_tensor = eyes_tensor / 255.0

        return eyes_tensor.to(device)

    left_eye_tensors = process_eye_list(left_eyes)
    right_eye_tensors = process_eye_list(right_eyes)


    #save_image(left_eye_tensors[0],'/home/allam/GFPGAN/eyes/left_eye_tensors.png')
    #save_image(right_eye_tensors[0],'/home/allam/GFPGAN/eyes/right_eye_tensors.png')

    return left_eye_tensors, right_eye_tensors

def extract_single_eye(image, eye_points, padding_ratio, output_size):
    """Helper function to extract single eye region from image"""
    # Get bounding box
    x_min, y_min = np.min(eye_points, axis=0)
    x_max, y_max = np.max(eye_points, axis=0)

    # Calculate padding
    eye_width = x_max - x_min
    eye_height = y_max - y_min
    pad_x = int(eye_width * padding_ratio)
    pad_y = int(eye_height * padding_ratio)

    # Calculate coordinates with boundary checks
    h, w = image.shape[:2]
    x_start = max(0, int(x_min) - pad_x)
    x_end = min(w, int(x_max) + pad_x)
    y_start = max(0, int(y_min) - pad_y)
    y_end = min(h, int(y_max) + pad_y)

    # Check if region is valid
    if x_start >= x_end or y_start >= y_end:
        return None

    # Extract and resize eye region
    eye_region = image[y_start:y_end, x_start:x_end]
    if eye_region.size == 0:
        return None

    eye_region = cv2.resize(eye_region, output_size)
    return eye_region
import torch
import numpy as np
import cv2

def extract_eyes_convex_hull(image_batch, landmarks_batch, padding=5, output_size=(128, 64)):
    """
    Extract eye regions using convex hull of eye landmarks

    Args:
        image_batch: Tensor (B, 3, H, W) in [0,1] or [0,255]
        landmarks_batch: Tensor (B, 68, 2) in pixel coordinates
        padding: Additional pixels around convex hull
        output_size: Target output size (width, height)

    Returns:
        (left_eyes, right_eyes) tensors of shape (B, 3, output_h, output_w)
    """
    device = image_batch.device
    images_np = image_batch.permute(0, 2, 3, 1).detach().cpu().numpy()
    if image_batch.max() <= 1.0:
        images_np = (images_np * 255).astype(np.uint8)
    landmarks_np = landmarks_batch.detach().cpu().numpy()

    batch_size = images_np.shape[0]
    left_eyes = []
    right_eyes = []

    # Landmark indices (0-based)
    LEFT_EYE_POINTS = list(range(36, 42))
    RIGHT_EYE_POINTS = list(range(42, 48))

    for i in range(batch_size):
        img = images_np[i]
        landmarks = landmarks_np[i]
        print(img)
        print(landmarks)
        exit()

        # Process left eye
        left_eye = crop_convex_hull(img, landmarks[LEFT_EYE_POINTS], padding, output_size)
        print(left_eye)

        # Process right eye
        right_eye = crop_convex_hull(img, landmarks[RIGHT_EYE_POINTS], padding, output_size)
        print(left_eye)
        exit()

        left_eyes.append(left_eye if left_eye is not None else np.zeros((*output_size[::-1], 3), dtype=np.uint8))
        right_eyes.append(right_eye if right_eye is not None else np.zeros((*output_size[::-1], 3), dtype=np.uint8))
        print(np.stack(left_eyes).shape,np.stack(left_eyes).min(),np.stack(left_eyes).max())
        print(np.stack(right_eyes).shape,np.stack(right_eyes).min(),np.stack(right_eyes).max())
        exit()

    # Convert to output tensors
    def to_tensor(arr_list):
        arr = np.stack(arr_list)
        tensor = torch.from_numpy(arr).permute(0, 3, 1, 2).float()
        if image_batch.max() <= 1.0:
            tensor = tensor / 255.0
        return tensor.to(device)

    #save_image(to_tensor(left_eyes)[0],'/home/allam/GFPGAN/eyes/left_eye_tensors.png')
    #save_image(to_tensor(right_eyes)[0],'/home/allam/GFPGAN/eyes/right_eye_tensors.png')

    return to_tensor(left_eyes), to_tensor(right_eyes)

def crop_convex_hull(image, points, padding, output_size):
    """Crop image to convex hull of points with padding"""
    # Convert points to integer and ensure proper shape
    points = points.astype(np.int32).reshape(-1, 1, 2)

    # Calculate convex hull
    hull = cv2.convexHull(points)

    # Get bounding rectangle of convex hull
    x, y, w, h = cv2.boundingRect(hull)

    # Apply padding
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(image.shape[1], x + w + padding)
    y_end = min(image.shape[0], y + h + padding)

    # Check for valid region
    if x_start >= x_end or y_start >= y_end:
        return None

    # Create mask for convex hull
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [hull], -1, 255, -1)

    # Crop region
    cropped = image[y_start:y_end, x_start:x_end]
    #mask_cropped = mask[y_start:y_end, x_start:x_end]

    # Apply mask (optional - removes non-eye pixels)
    # cropped = cv2.bitwise_and(cropped, cropped, mask=mask_cropped)

    # Resize to output size
    if cropped.size == 0:
        return None

    resized = cv2.resize(cropped, output_size)
    return resized

def vgg_eyes_loss(pred_img, target_img, pred_landmarks, target_landmarks):
    """
    Args:
        pred_img: (B, 3, H, W) - Predicted face
        target_img: (B, 3, H, W) - Ground truth face
        pred_landmarks: (B, 68, 2) - Predicted landmarks
        target_landmarks: (B, 68, 2) - GT landmarks
    """

    # Get eyes crops
    #pred_lft_eyes=[]
    #pred_rt_eyes=[]
    #pred_nose_batch=[]
    #trg_lft_eyes=[]
    #trg_rt_eyes=[]
    #gt_nose_batch=[]
    #right_eye_indices = torch.tensor([36, 37, 38, 39, 40, 41])
    #left_eye_indices = torch.tensor([42, 43, 44, 45, 46, 47])
     # Crop both eyes
        # get landmarks for each component

    # pred_img=pred_img.permute(0,2,3,1) # reshape to be (B,256,256,3)
    # target_img=target_img.permute(0,2,3,1)
    map_left_eye = list(range(36, 42))
    map_right_eye = list(range(42, 48))
    map_nose = list(range(27, 36))
    enlarge_ratio = 1.4  # only for eyes

    # eye_left
    pred_lft_eyes =[]
    pred_rt_eyes = []
    trg_lft_eyes = []
    trg_rt_eyes =[]

    trg_nose=[]
    pred_nose=[]
    target_img=target_img.permute(0,2,3,1)
    for pr_im,pr_lm,tr_im,tr_lm in zip(pred_img,pred_landmarks, target_img, target_landmarks):
        #print(pr_im.shape, pr_im.min(),pr_im.max())
        #pr_lm=pr_lm[0]
        pr_lm=pr_lm.detach().cpu().numpy()
        tr_lm=tr_lm.detach().cpu().numpy()

        #-------------------------predicted eyes--------------------------------
        mean_left_eye = np.mean(pr_lm[map_left_eye], 0)  # (x, y)
        half_len_left_eye = np.max((np.max(np.max(pr_lm[map_left_eye], 0) - np.min(pr_lm[map_left_eye], 0)) / 2, 16))
        half_len_left_eye *= enlarge_ratio
        loc_left_eye = np.hstack((mean_left_eye - half_len_left_eye + 1, mean_left_eye + half_len_left_eye)).astype(int)

        mean_right_eye = np.mean(pr_lm[map_right_eye], 0)  # (x, y)
        half_len_right_eye = np.max((np.max(np.max(pr_lm[map_right_eye], 0) - np.min(pr_lm[map_right_eye], 0)) / 2, 16))
        half_len_right_eye *= enlarge_ratio
        loc_right_eye = np.hstack((mean_right_eye - half_len_right_eye + 1, mean_right_eye + half_len_right_eye)).astype(int)

        eye_left_img = pr_im[loc_left_eye[1]:loc_left_eye[3], loc_left_eye[0]:loc_left_eye[2], :]
        eye_right_img = pr_im[loc_right_eye[1]:loc_right_eye[3], loc_right_eye[0]:loc_right_eye[2], :]

        if eye_right_img.shape[0]==0 or eye_right_img.shape[1]==0 or eye_left_img.shape[0]==0 or eye_left_img.shape[1]==0 :
            continue
        eye_left_img=im_rz_224(eye_left_img.permute(2,0,1))/eye_left_img.max()
        eye_right_img=im_rz_224(eye_right_img.permute(2,0,1))/eye_right_img.max()

        pred_lft_eyes.append(eye_left_img)
        pred_rt_eyes.append(eye_right_img)
        # print(eye_right_img.shape, eye_right_img.min(),eye_right_img.max())
        # print(eye_left_img.shape, eye_left_img.min(),eye_left_img.max())

        # save_image(eye_right_img.unsqueeze(0),'/home/allam/GFPGAN/eyes/1pred_right_eyes.png')
        # save_image(eye_left_img.unsqueeze(0),'/home/allam/GFPGAN/eyes/1pred_left_eyes.png')
        # #exit()
        # #---------------------------GT eyes-------------------------------------
        mean_left_eye = np.mean(tr_lm[map_left_eye], 0)  # (x, y)
        half_len_left_eye = np.max((np.max(np.max(tr_lm[map_left_eye], 0) - np.min(tr_lm[map_left_eye], 0)) / 2, 16))
        half_len_left_eye *= enlarge_ratio
        loc_left_eye = np.hstack((mean_left_eye - half_len_left_eye + 1, mean_left_eye + half_len_left_eye)).astype(int)

        mean_right_eye = np.mean(tr_lm[map_right_eye], 0)  # (x, y)
        half_len_right_eye = np.max((np.max(np.max(tr_lm[map_right_eye], 0) - np.min(tr_lm[map_right_eye], 0)) / 2, 16))
        half_len_right_eye *= enlarge_ratio
        loc_right_eye = np.hstack((mean_right_eye - half_len_right_eye + 1, mean_right_eye + half_len_right_eye)).astype(int)

        eye_left_img = tr_im[loc_left_eye[1]:loc_left_eye[3], loc_left_eye[0]:loc_left_eye[2], :]
        eye_right_img = tr_im[loc_right_eye[1]:loc_right_eye[3], loc_right_eye[0]:loc_right_eye[2], :]
        #print(eye_left_img.shape)
        #print(eye_right_img.shape)
        #exit()
        if eye_right_img.shape[0]==0 or eye_right_img.shape[1]==0 or eye_left_img.shape[0]==0 or eye_left_img.shape[1]==0 :
            continue
        eye_left_img=im_rz_224(eye_left_img.permute(2,0,1))/eye_left_img.max()
        eye_right_img=im_rz_224(eye_right_img.permute(2,0,1))/eye_right_img.max()

        #exit()
        trg_lft_eyes.append(eye_left_img)
        trg_rt_eyes.append(eye_right_img)
        # eye_left_img=im_rz_224(eye_left_img.permute(2,0,1))/eye_left_img.max()
        # eye_right_img=im_rz_224(eye_right_img.permute(2,0,1))/eye_right_img.max()

        # #exit()
        # trg_lft_eyes.append(eye_left_img)
        # trg_rt_eyes.append(eye_right_img)
        # save_image(eye_right_img.unsqueeze(0),'/home/allam/GFPGAN/eyes/0gt_right_eyes.png')
        # save_image(eye_left_img.unsqueeze(0),'/home/allam/GFPGAN/eyes/0gt_left_eyes.png')
        #-----------------------------------------------------------------------
        #-------------------------- pred nose-----------------------------------
        mean_nose = np.mean(pr_lm[map_nose], 0)  # (x, y)
        half_len_nose= np.max((np.max(np.max(pr_lm[map_nose], 0) - np.min(pr_lm[map_nose], 0)) / 2, 16))
        half_len_nose *= enlarge_ratio
        loc_nose = np.hstack((mean_nose - half_len_nose + 1, mean_nose + half_len_nose)).astype(int)
        pred_nose_img = pr_im[loc_nose[1]:loc_nose[3], loc_nose[0]:loc_nose[2], :]
        if pred_nose_img.shape[0]==0 or pred_nose_img.shape[1]==0:
            continue
        pred_nose_img=im_rz_224(pred_nose_img.permute(2,0,1))/pred_nose_img.max()
        #print(pred_nose_img.shape,pred_nose_img.min(),pred_nose_img.max())
        #save_image(pred_nose_img.unsqueeze(0),'/home/allam/GFPGAN/eyes/2pred_nose.png')
        pred_nose.append(pred_nose_img)
         #-------------------- gt nose--------------------------------
        mean_nose = np.mean(tr_lm[map_nose], 0)  # (x, y)
        half_len_nose= np.max((np.max(np.max(tr_lm[map_nose], 0) - np.min(tr_lm[map_nose], 0)) / 2, 16))
        half_len_nose *= enlarge_ratio
        loc_nose = np.hstack((mean_nose - half_len_nose + 1, mean_nose + half_len_nose)).astype(int)
        trg_nose_img = tr_im[loc_nose[1]:loc_nose[3], loc_nose[0]:loc_nose[2], :]
        if trg_nose_img.shape[0]==0 or trg_nose_img.shape[1]==0:
            continue

        trg_nose_img=im_rz_224(trg_nose_img.permute(2,0,1))/trg_nose_img.max()
        #print(trg_nose_img.shape,trg_nose_img.min(),trg_nose_img.max())
        #save_image(trg_nose_img.unsqueeze(0),'/home/allam/GFPGAN/eyes/2gt_nose.png')
        trg_nose.append(trg_nose_img)
        #exit()

    pred_lft_eyes=torch.stack(pred_lft_eyes)
    pred_rt_eyes=torch.stack(pred_rt_eyes)

    trg_lft_eyes=torch.stack(trg_lft_eyes)
    trg_rt_eyes=torch.stack(trg_rt_eyes)

    PRD_nose=torch.stack(pred_nose)
    TRG_nose=torch.stack(trg_nose)

    #pred_lft_eyes = (pred_lft_eyes.cuda()*255 - 127) / 128
    pred_lft_eyes = (pred_lft_eyes.cuda()-vgg_mean)/vgg_std
    #pred_rt_eyes = (pred_rt_eyes.cuda()*255 - 127) / 128
    pred_rt_eyes = (pred_rt_eyes.cuda()-vgg_mean)/vgg_std

    #pred_nose_batch = (pred_nose_batch.cuda()*255 - 127) / 128

    #trg_lft_eyes = (trg_lft_eyes.cuda()*255 - 127) / 128
    trg_lft_eyes = (trg_lft_eyes.cuda()-vgg_mean)/vgg_std

    #trg_rt_eyes = (trg_rt_eyes.cuda()*255 - 127) / 128
    trg_rt_eyes = (trg_rt_eyes.cuda()-vgg_mean)/vgg_std

    # PRD_nose = (PRD_nose.cuda()*255 - 127) / 128
    # TRG_nose = (TRG_nose.cuda()*255 - 127) / 128

    PRD_nose = (PRD_nose.cuda()-vgg_mean)/vgg_std
    TRG_nose = (TRG_nose.cuda()-vgg_mean)/vgg_std

    #gt_nose_batch = (gt_nose_batch.cuda()*255 - 127) / 128

    # Extract VGG features
    with torch.no_grad():
        pred_lft_eyes_features = vgg(pred_lft_eyes)
        pred_rt_eyes_features = vgg(pred_rt_eyes)

        #pred_nose_batch_feat = vgg(pred_nose_batch)

        trg_lft_eyes_features = vgg(trg_lft_eyes)
        trg_rt_eyes_features = vgg(trg_rt_eyes)

        pred_nose_features = vgg(PRD_nose)
        trg_nose_features = vgg(TRG_nose)

        #gt_nose_batch_feat = vgg(gt_nose_batch)

        loss=0
        loss+=torch.nn.functional.l1_loss(pred_lft_eyes_features, trg_lft_eyes_features)
        loss+=torch.nn.functional.l1_loss(pred_rt_eyes_features, trg_rt_eyes_features)
        loss+=torch.nn.functional.l1_loss(pred_nose_features, trg_nose_features)
        #print(loss)
        #exit()
        #loss+=10*torch.nn.functional.mse_loss(pred_nose_batch_feat, gt_nose_batch_feat)
    # L1/L2 loss on features
    return loss
#-------------------------------------------------------------------------------
# def vgg_eyes_loss(pred_img, target_img, pred_landmarks, target_landmarks):
#     # Landmark loss (only for eyes region)

#     # VGG eyes loss
#     perceptual_loss = vgg_eyes_loss(pred_img, target_img, pred_landmarks, target_landmarks)

#     return  perceptual_loss  # Weighted sum
#===============================================================================
def photo_loss(imageA, imageB, mask, eps=1e-6):
    """
    l2 norm (with sqrt, to ensure backward stabililty, use eps, otherwise Nan may occur)
    Parameters:
        imageA       --torch.tensor (B, 3, H, W), range (0, 1), RGB order
        imageB       --same as imageA
    """
    loss = torch.sqrt(eps + torch.sum((imageA - imageB) ** 2, dim=1, keepdims=True)) * mask
    loss = torch.sum(loss) / torch.max(torch.sum(mask), torch.tensor(1.0).to(mask.device))
    return loss

def landmark_loss(predict_lm, gt_lm, weight=None):
    """
    weighted mse loss
    Parameters:
        predict_lm    --torch.tensor (B, 68, 2)
        gt_lm         --torch.tensor (B, 68, 2)
        weight        --numpy.array (1, 68)
    """
    if not weight:
        weight = np.ones([68])
        weight[28:31] = 20
        weight[-8:] = 20
        weight = np.expand_dims(weight, 0)
        weight = torch.tensor(weight).to(predict_lm.device)
    loss = torch.sum((predict_lm - gt_lm)**2, dim=-1) * weight
    loss = torch.sum(loss) / (predict_lm.shape[0] * predict_lm.shape[1])
    return loss


### regulization
def reg_loss(coeffs_dict, opt=None):
    """
    l2 norm without the sqrt, from yu's implementation (mse)
    tf.nn.l2_loss https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss
    Parameters:
        coeffs_dict     -- a  dict of torch.tensors , keys: id, exp, tex, angle, gamma, trans

    """
    # coefficient regularization to ensure plausible 3d faces
    if opt:
        w_id, w_exp, w_tex = opt.w_id, opt.w_exp, opt.w_tex
    else:
        w_id, w_exp, w_tex = 1, 1, 1, 1
    creg_loss = w_id * torch.sum(coeffs_dict['id'] ** 2) +  \
           w_exp * torch.sum(coeffs_dict['exp'] ** 2) + \
           w_tex * torch.sum(coeffs_dict['tex'] ** 2)
    creg_loss = creg_loss / coeffs_dict['id'].shape[0]

    # gamma regularization to ensure a nearly-monochromatic light
    gamma = coeffs_dict['gamma'].reshape([-1, 3, 9])
    gamma_mean = torch.mean(gamma, dim=1, keepdims=True)
    gamma_loss = torch.mean((gamma - gamma_mean) ** 2)

    return creg_loss, gamma_loss

def reflectance_loss(texture, mask):
    """
    minimize texture variance (mse), albedo regularization to ensure an uniform skin albedo
    Parameters:
        texture       --torch.tensor, (B, N, 3)
        mask          --torch.tensor, (N), 1 or 0

    """
    mask = mask.reshape([1, mask.shape[0], 1])
    texture_mean = torch.sum(mask * texture, dim=1, keepdims=True) / torch.sum(mask)
    loss = torch.sum(((texture - texture_mean) * mask)**2) / (texture.shape[0] * torch.sum(mask))
    return loss


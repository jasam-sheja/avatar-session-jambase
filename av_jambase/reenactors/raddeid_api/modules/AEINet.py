import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#from blocks import ResBlocks, UpConv2dBlock, Conv2dBlock
class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for _ in range(num_blocks):
            self.model += [ResBlock(dim,
                                    norm=norm,
                                    activation=activation,
                                    pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class UpConv2dBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='reflect'):
        super(UpConv2dBlock, self).__init__()
        model = [nn.Upsample(scale_factor=2),
                 Conv2dBlock(dim, dim // 2, 5, 1, 2,
                             norm=norm,
                             activation=activation,
                             pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        out = self.model(x)
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                 norm='none', activation='relu', pad_type='zero',
                 use_bias=True, activation_first=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        # elif norm == 'adain':
        #     self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm in ['none', 'spectral']:
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if norm == 'spectral':
            self.conv = nn.utils.spectral_norm(
                nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x
#===============================================================================================
class Transfer(nn.Module):
    def __init__(self, c_up,down):#args
        super(Transfer, self).__init__()
        c_up = c_up#args.c_up  # 32
        down =down# args.down  # 2
        self.model = [
            Conv2dBlock(7, c_up, 7, 1, 3, norm='in', pad_type='reflect'),  # RGB + mask + target
        ]
        for i in range(down):
            self.model.append(Conv2dBlock(c_up, 2 * c_up, 4, 2, 1, norm='in', pad_type='reflect'))
            c_up *= 2
        self.model.append(ResBlocks(5, c_up, norm='in', activation='relu', pad_type='reflect'))
        for i in range(down):
            self.model.append(UpConv2dBlock(c_up, norm='in', activation='relu', pad_type='reflect'))
            c_up //= 2
        self.model.append(Conv2dBlock(c_up, 3, 7, 1, padding=3, norm='none', activation='none', pad_type='reflect'))
        self.model = nn.Sequential(*self.model)

    def to_rgb(self, x):
        return 127.5 * (x + 1)

    def forward(self, x):
        """
        Args:
            x: (B, 7, ts, ts): RGB + mask + target

        Returns: (B, 3, ts, ts): RGB

        """
        return self.model(x)

#======================================================================
# simple style transfer
class Refiner(nn.Module):
    def __init__(self, args):
        super(Refiner, self).__init__()
        c_up = args.c_up // 2  # 16
        down = args.down  # 2
        self.model = [
            Conv2dBlock(3, c_up, 7, 1, 3, norm='in', pad_type='reflect'),  # RGB
        ]
        for i in range(down):
            self.model.append(Conv2dBlock(c_up, 2 * c_up, 4, 2, 1, norm='in', pad_type='reflect'))
            c_up *= 2
        self.model.append(ResBlocks(5, c_up, norm='in', activation='relu', pad_type='reflect'))
        for i in range(down):
            self.model.append(UpConv2dBlock(c_up, norm='in', activation='relu', pad_type='reflect'))
            c_up //= 2
        self.model.append(Conv2dBlock(c_up, 3, 7, 1, padding=3, norm='none', activation='none', pad_type='reflect'))
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        """
        Args:
            x: (B, 3, ts, ts): RGB

        Returns: (B, 3, ts, ts): RGB

        """
        return self.model(x)

#=================================================
# Image gradient
class LaplacianFilter(nn.Module):
    def __init__(self):
        super(LaplacianFilter, self).__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        kernel = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]).float().reshape_as(self.conv.weight)
        self.conv.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        # x: (B, 3, H, W) RGB image, [-1, 1]
        red_img_tensor = x[:, 0, :, :].unsqueeze(1)
        green_img_tensor = x[:, 1, :, :].unsqueeze(1)
        blue_img_tensor = x[:, 2, :, :].unsqueeze(1)

        red_gradient_tensor = self.conv(red_img_tensor).squeeze(1)
        green_gradient_tensor = self.conv(green_img_tensor).squeeze(1)
        blue_gradient_tensor = self.conv(blue_img_tensor).squeeze(1)

        return red_gradient_tensor, green_gradient_tensor, blue_gradient_tensor  # (B, H, W)
#=====================================================
class MultilevelAttributesEncoder(nn.Module):
    def __init__(self):
        super(MultilevelAttributesEncoder, self).__init__()
        self.Encoder_channel = [3, 32, 64, 128, 256, 512, 1024, 1024]
        self.Encoder = nn.ModuleDict({f'layer_{i}' : nn.Sequential(
                nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i+1], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.Encoder_channel[i+1]),
                nn.LeakyReLU(0.1)
            )for i in range(7)})

        self.Decoder_inchannel = [1024, 2048, 1024, 512, 256, 128]
        self.Decoder_outchannel = [1024, 512, 256, 128, 64, 32]
        self.Decoder = nn.ModuleDict({f'layer_{i}' : nn.Sequential(
                nn.ConvTranspose2d(self.Decoder_inchannel[i], self.Decoder_outchannel[i], kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(self.Decoder_outchannel[i]),
                nn.LeakyReLU(0.1)
            )for i in range(6)})

        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x):
        arr_x = []
        for i in range(7):
           # print('i',i)
            x = self.Encoder[f'layer_{i}'](x)
            #print(x.shape)
            arr_x.append(x)


        arr_y = []
        arr_y.append(arr_x[6])
        y = arr_x[6]
        '''torch.Size([32, 32, 112, 112])
        torch.Size([32, 64, 56, 56])
        torch.Size([32, 128, 28, 28])
        torch.Size([32, 256, 14, 14])
        torch.Size([32, 512, 7, 7])
        torch.Size([32, 1024, 3, 3])
        torch.Size([32, 1024, 1, 1])
        torch.Size([32, 1024, 1, 1])'''

        for i in range(6):
            #print('i',i)
            y = self.Decoder[f'layer_{i}'](y)
           # print('y',y.shape)
            y = torch.cat((y, arr_x[5-i]), 1)
            arr_y.append(y)
       # exit()
        arr_y.append(self.Upsample(y))
        # for i in range(len(arr_y)):
        #     print(arr_y[i].shape)
        return arr_y
# torch.Size([8, 1024, 2, 2])
# torch.Size([8, 2048, 4, 4])
# torch.Size([8, 1024, 8, 8])
# torch.Size([8, 512, 16, 16])
# torch.Size([8, 256, 32, 32])
# torch.Size([8, 128, 64, 64])
# torch.Size([8, 64, 128, 128])
# torch.Size([8, 64, 256, 256])
#===============================================================================
# class MultilevelAttributesEncoder(nn.Module):
#     def __init__(self):
#         super(MultilevelAttributesEncoder, self).__init__()
#         self.Encoder_channel = [3, 32, 64, 128, 256, 512, 1024, 1024]
#         self.Encoder = nn.ModuleDict({f'layer_{i}' : nn.Sequential(
#                 nn.Conv2d(self.Encoder_channel[i], self.Encoder_channel[i+1], kernel_size=4, stride=2, padding=1),
#                 nn.BatchNorm2d(self.Encoder_channel[i+1]),
#                 nn.LeakyReLU(0.1)
#             )for i in range(7)})

#         self.Decoder_inchannel = [1024, 2048, 1024, 512, 256, 128]
#         self.Decoder_outchannel = [1024, 512, 256, 128, 64, 32]
#         self.Decoder = nn.ModuleDict({f'layer_{i}' : nn.Sequential(
#                 nn.ConvTranspose2d(self.Decoder_inchannel[i], self.Decoder_outchannel[i], kernel_size=4, stride=2, padding=1),
#                 nn.BatchNorm2d(self.Decoder_outchannel[i]),
#                 nn.LeakyReLU(0.1)
#             )for i in range(6)})

#         self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)

#     def forward(self, x):
#         arr_x = []
#         for i in range(7):
#             x = self.Encoder[f'layer_{i}'](x)
#             arr_x.append(x)


#         arr_y = []
#         arr_y.append(arr_x[6])
#         y = arr_x[6]
#         for i in range(6):
#             y = self.Decoder[f'layer_{i}'](y)
#             y = torch.cat((y, arr_x[5-i]), 1)
#             arr_y.append(y)

#         arr_y.append(self.Upsample(y))

#         return arr_y
#===============================================================================
'''y torch.Size([8, 1024, 4, 4])
y torch.Size([8, 512, 8, 8])
y torch.Size([8, 256, 16, 16])
y torch.Size([8, 128, 32, 32])
y torch.Size([8, 64, 64, 64])
y torch.Size([8, 32, 128, 128])''' # 524,288
class ADD_PAG(nn.Module):
    def __init__(self, h_inchannel, z_inchannel):
        super(ADD_PAG, self).__init__()

        self.BNorm = nn.BatchNorm2d(h_inchannel)
        #self.conv1= nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

        #self.sigmoid = nn.Sigmoid()

        #self.fc_1 = nn.Linear(z_id_size, h_inchannel)
        #self.fc_2 = nn.Linear(z_id_size, h_inchannel)

        self.conv1 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        #self.upsamp=nn.Upsample(scale_factor=1,mode='bilinear',align_corners=False)
        #self.conv2 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

    def forward(self,h_in, z_att):
        # print(h_in.shape)
        # print(z_att.shape)
        h_bar = self.BNorm(h_in)
        r_att = self.conv1(z_att)

        #=================
        # print('h_in',h_bar.shape)
        # print("r_att",r_att.shape)
        # exit()
        #beta_att = self.conv2(z_att)
        if r_att.size(2)!=h_bar.size(2)  or  r_att.size(3)!=h_bar.size(3):#, f"Dim missmatch:{r_att.size()}"
            r_att= F.interpolate(r_att,size=h_bar.size()[2:],mode='bilinear',align_corners=False)
            #r_att=self.upsamp(r_att,size=h_bar.size()[2:])

        h_out = r_att * h_bar

        #h_out = a#(1-m)*a + m*i

        return h_out
#-------------------------------------------------------------------------------
class ADDResBlock_PAG(nn.Module):
    def __init__(self, h_inchannel, z_inchannel, h_outchannel):
        super(ADDResBlock_PAG, self).__init__()

        self.h_inchannel = h_inchannel
        self.z_inchannel = z_inchannel
        self.h_outchannel = h_outchannel

        self.add1 = ADD_PAG(h_inchannel, z_inchannel)
        self.add2 = ADD_PAG(h_inchannel, z_inchannel)

        self.conv1 = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(h_inchannel, h_outchannel, kernel_size=3, stride=1, padding=1)

        if not self.h_inchannel == self.h_outchannel:
            self.add3 = ADD_PAG(h_inchannel, z_inchannel)
            self.conv3 = nn.Conv2d(h_inchannel, h_outchannel, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

    def forward(self, h_in, z_att):
        x1 = self.activation(self.add1(h_in, z_att))
        x1 = self.conv1(x1)
        x1 = self.activation(self.add2(x1, z_att))
        x1 = self.conv2(x1)

        x2 = h_in
        if not self.h_inchannel == self.h_outchannel:
            x2 = self.activation(self.add3(h_in, z_att))
            x2 = self.conv3(x2)

        return x1 + x2
#-------------------------------------------------------------------------------
#===============================================================================
class ADDGenerator_PAG(nn.Module):
    def __init__(self, z_id_size=1024):
        super(ADDGenerator_PAG, self).__init__()

        self.convt = nn.ConvTranspose2d(z_id_size, 1024, kernel_size=2, stride=1, padding=0)
        self.convt2 = nn.ConvTranspose2d(384, 32, kernel_size=1, stride=1, padding=0)
        #self.linear = nn.Linear(35709*3, 512)
        self.linear2 = nn.Linear(3072, 35709*3)
        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.sigmoid=nn.Sigmoid()
        self.h_inchannel = [1024, 1024, 1024, 1024, 512, 256, 128, 64]
        self.z_inchannel = [1024, 2048, 1024, 512, 256, 128, 64, 64]
        self.h_outchannel = [1024, 1024, 1024, 512, 256, 128, 64, 3]


        self.model = nn.ModuleDict(
            {f"layer_{i}" : ADDResBlock_PAG(self.h_inchannel[i], self.z_inchannel[i],self.h_outchannel[i])
        for i in range(8)})

    def forward(self, z_att):

        x=self.convt(z_att[0])#.unsqueeze(-1).unsqueeze(-1))
        for i in range(7):
            x = self.Upsample(self.model[f"layer_{i}"](x,z_att[i]))
            #print(z_att[i].shape)
        x = self.model["layer_7"](x, z_att[7])
        #x=PersonAlbedo_est(x)
        x=self.convt2(x.permute(0,2,3,1))
        x=self.convt2(x.permute(0,2,3,1))

        x=self.linear2(x.reshape(x.shape[0],32*32*3))
        x=x.reshape(x.shape[0],35709,3)
        # print(x.shape)
        # exit()

        return self.sigmoid(x)
#===============================================================================
class ADD(nn.Module):
    def __init__(self, h_inchannel, z_inchannel, z_id_size=512):
        super(ADD, self).__init__()

        self.BNorm = nn.BatchNorm2d(h_inchannel)
        self.conv_f = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

        self.fc_1 = nn.Linear(z_id_size, h_inchannel)
        self.fc_2 = nn.Linear(z_id_size, h_inchannel)

        self.conv1 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(z_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)

    def forward(self, h_in, z_att, z_id):
        h_bar = self.BNorm(h_in)
        m = self.sigmoid(self.conv_f(h_bar))

        r_id = self.fc_1(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)
        beta_id = self.fc_2(z_id).unsqueeze(-1).unsqueeze(-1).expand_as(h_in)

        i = r_id*h_bar + beta_id

        r_att = self.conv1(z_att)
        beta_att = self.conv2(z_att)
        a = r_att * h_bar + beta_att

        h_out = (1-m)*a + m*i

        return h_out

#--------------------------------------------------------------------------------
class ADDResBlock(nn.Module):
    def __init__(self, h_inchannel, z_inchannel, h_outchannel):
        super(ADDResBlock, self).__init__()

        self.h_inchannel = h_inchannel
        self.z_inchannel = z_inchannel
        self.h_outchannel = h_outchannel

        self.add1 = ADD(h_inchannel, z_inchannel)
        self.add2 = ADD(h_inchannel, z_inchannel)

        self.conv1 = nn.Conv2d(h_inchannel, h_inchannel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(h_inchannel, h_outchannel, kernel_size=3, stride=1, padding=1)

        if not self.h_inchannel == self.h_outchannel:
            self.add3 = ADD(h_inchannel, z_inchannel)
            self.conv3 = nn.Conv2d(h_inchannel, h_outchannel, kernel_size=3, stride=1, padding=1)

        self.activation = nn.ReLU()

    def forward(self, h_in, z_att, z_id):
        x1 = self.activation(self.add1(h_in, z_att, z_id))
        x1 = self.conv1(x1)
        x1 = self.activation(self.add2(x1, z_att, z_id))
        x1 = self.conv2(x1)

        x2 = h_in
        if not self.h_inchannel == self.h_outchannel:
            x2 = self.activation(self.add3(h_in, z_att, z_id))
            x2 = self.conv3(x2)

        return x1 + x2

#=================================================================
class ADDGenerator(nn.Module):
    def __init__(self, z_id_size=512):
        super(ADDGenerator, self).__init__()

        self.convt = nn.ConvTranspose2d(z_id_size, 1024, kernel_size=2, stride=1, padding=0)
        self.convt2 = nn.ConvTranspose2d(256, 32, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(35709*3, 512)
        self.linear2 = nn.Linear(3072, 35709*3)
        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.sigmoid=nn.Sigmoid()
        self.h_inchannel = [1024, 1024, 1024, 1024, 512, 256, 128, 64]
        self.z_inchannel = [1024, 2048, 1024, 512, 256, 128, 64, 64]
        self.h_outchannel = [1024, 1024, 1024, 512, 256, 128, 64, 3]

        self.model = nn.ModuleDict(
            {f"layer_{i}" : ADDResBlock(self.h_inchannel[i], self.z_inchannel[i], self.h_outchannel[i])
        for i in range(8)})

    def forward(self, z_id, z_att):
        #print(z_id.unsqueeze(-1).shape)
        z_id = self.linear(z_id.reshape(z_id.shape[0],35709*3))
        #z_tex = self.Linear(z_tex.reshape(z_tex.shape[0],35709*3))
        #print(z_id.shape)
        #z_id=(z_id+z_tex)
        z_id = F.normalize(z_id)
        x = self.convt(z_id.unsqueeze(-1).unsqueeze(-1))
       # print(x.shape)
        for i in range(7):
            x = self.Upsample(self.model[f"layer_{i}"](x, z_att[i], z_id))
            #print(z_att[i].shape)
        x = self.model["layer_7"](x, z_att[7], z_id)
        x=self.convt2(x.permute(0,2,3,1))
        x=self.convt2(x.permute(0,2,3,1))
        x=self.linear2(x.reshape(x.shape[0],32*32*3))
        x=x.reshape(x.shape[0],35709,3)

        return x,self.sigmoid(x)
#===============================================================================
class ADDGenerator_oceff(nn.Module):#160
    def __init__(self, z_id_size=512):
        super(ADDGenerator_oceff, self).__init__()

        self.convt = nn.ConvTranspose2d(z_id_size, 1024, kernel_size=2, stride=1, padding=0)
        self.convt2 = nn.ConvTranspose2d(256, 32, kernel_size=1, stride=1, padding=0)
        self.linear = nn.Linear(80, 512)
        self.linear2 = nn.Linear(3072, 35709*3)
        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.sigmoid=nn.Sigmoid()
        self.h_inchannel = [1024, 1024, 1024, 1024, 512, 256, 128, 64]
        self.z_inchannel = [1024, 2048, 1024, 512, 256, 128, 64, 64]
        self.h_outchannel = [1024, 1024, 1024, 512, 256, 128, 64, 3]

        self.model = nn.ModuleDict(
            {f"layer_{i}" : ADDResBlock(self.h_inchannel[i], self.z_inchannel[i], self.h_outchannel[i])
        for i in range(8)})

    def forward(self, z_id, z_att):
        #print(z_id.unsqueeze(-1).shape)
        z_id = self.linear(z_id)
        #z_tex = self.Linear(z_tex.reshape(z_tex.shape[0],35709*3))
        #print(z_id.shape)
        #z_id=(z_id+z_tex)
        z_id = F.normalize(z_id)
        x = self.convt(z_id.unsqueeze(-1).unsqueeze(-1))
       # print(x.shape)
        for i in range(7):
            x = self.Upsample(self.model[f"layer_{i}"](x, z_att[i], z_id))
            #print(z_att[i].shape)
        x = self.model["layer_7"](x, z_att[7], z_id)
        x=self.convt2(x.permute(0,2,3,1))
        x=self.convt2(x.permute(0,2,3,1))
        x=self.linear2(x.reshape(x.shape[0],32*32*3))
        x=x.reshape(x.shape[0],35709,3)

        return self.sigmoid(x)
#===============================================================================
class ADDGenerator_org(nn.Module):
    def __init__(self, z_id_size=512):
        super(ADDGenerator_org, self).__init__()
        #self.linear = nn.Linear(80, 512)
        self.convt = nn.ConvTranspose2d(z_id_size, 1024, kernel_size=2, stride=1, padding=0)
        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.h_inchannel = [1024, 1024, 1024, 1024, 512, 256, 128, 64]
        self.z_inchannel = [1024, 2048, 1024, 512, 256, 128, 64, 64]
        self.h_outchannel = [1024, 1024, 1024, 512, 256, 128, 64, 3]

        self.model = nn.ModuleDict(
            {f"layer_{i}" : ADDResBlock(self.h_inchannel[i], self.z_inchannel[i], self.h_outchannel[i])
        for i in range(8)})


    def forward(self, z_id, z_att):
        #z_id = self.linear(z_id.reshape(z_id.shape[0],512))
        #z_id = F.normalize(z_id)
        x = self.convt(z_id.unsqueeze(-1).unsqueeze(-1))

        for i in range(7):
            x = self.Upsample(self.model[f"layer_{i}"](x, z_att[i], z_id))
        x = self.model["layer_7"](x, z_att[7], z_id)

        return nn.Sigmoid()(x)

#-------------------MultiscaleDiscriminator-----------------------
class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result

#class StyleCodeGenerator(nn.Module):
class AttentionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv1d(channels // 8, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.attention(x)
        return x * attn
#===============================Original one================================================
# class StyleCodeGenerator(nn.Module):
#     def __init__(self):
#         super(StyleCodeGenerator, self).__init__()

#         # Define separate convolutional branches for each input
#         self.branch1_conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
#         self.branch1_bn1 = nn.BatchNorm1d(32)
#         self.branch1_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.branch1_bn2 = nn.BatchNorm1d(64)
#         self.branch1_conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.branch1_bn3 = nn.BatchNorm1d(128)
#         self.attention1 = AttentionModule(128)

#         self.branch2_conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
#         self.branch2_bn1 = nn.BatchNorm1d(32)
#         self.branch2_conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
#         self.branch2_bn2 = nn.BatchNorm1d(64)
#         self.branch2_conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
#         self.branch2_bn3 = nn.BatchNorm1d(128)
#         self.attention2 = AttentionModule(128)

#         # Convolutional layers for generating 32 channels, 512x512 output
#         self.final_conv = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=3, padding=1)
#         self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)

#     def forward(self, input1, input2):
#         # Process first input through its branch
#         x1 = input1.permute(0, 2, 1)  # Shape: BN x 3 x 35709
#         x1 = torch.relu(self.branch1_bn1(self.branch1_conv1(x1)))
#         x1 = torch.relu(self.branch1_bn2(self.branch1_conv2(x1)))
#         x1 = torch.relu(self.branch1_bn3(self.branch1_conv3(x1)))
#         x1 = self.attention1(x1)

#         # Process second input through its branch
#         x2 = input2.permute(0, 2, 1)  # Shape: BN x 3 x 35709
#         x2 = torch.relu(self.branch2_bn1(self.branch2_conv1(x2)))
#         x2 = torch.relu(self.branch2_bn2(self.branch2_conv2(x2)))
#         x2 = torch.relu(self.branch2_bn3(self.branch2_conv3(x2)))
#         x2 = self.attention2(x2)

#         # Fuse the two branches
#         x = torch.cat((x1, x2), dim=1)  # Shape: BN x 256 x 35709

#         # Reshape for convolutional processing (Add a spatial dimension for Conv2D)
#         x = x.unsqueeze(2)  # Shape: BN x 256 x 1 x 35709

#         # Apply the final convolution to adjust the channel size to 32
#         x = self.final_conv(x)  # Shape: BN x 32 x 1 x 35709
#         x=x.view(x.size(0),32,1,-1)

#         # Upsample the output to the desired size: BN x 32 x 512 x 512
#         x = self.upsample(x)  # Squeeze the 1 in the third dimension
#         #print(x.shape)
#         #exit()

#         return x
#===================== more complex style generator=============================
class StyleCodeGenerator(nn.Module):
    def __init__(self):
        super(StyleCodeGenerator, self).__init__()

        # Define separate convolutional branches for each input
        self.branch1_conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.branch1_bn1 = nn.BatchNorm1d(64)
        self.branch1_conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.branch1_bn2 = nn.BatchNorm1d(128)
        self.branch1_conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.branch1_bn3 = nn.BatchNorm1d(256)
        self.attention1 = AttentionModule(256)

        # Residual connection
        self.residual1 = nn.Conv1d(in_channels=3, out_channels=256, kernel_size=1)

        self.branch2_conv1 = nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.branch2_bn1 = nn.BatchNorm1d(64)
        self.branch2_conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.branch2_bn2 = nn.BatchNorm1d(128)
        self.branch2_conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.branch2_bn3 = nn.BatchNorm1d(256)
        self.attention2 = AttentionModule(256)

        # Residual connection
        self.residual2 = nn.Conv1d(in_channels=3, out_channels=256, kernel_size=1)

        # Convolutional layers for generating 32 channels, 512x512 output
        self.final_conv = nn.Conv2d(in_channels=512, out_channels=32, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)

    def forward(self, input1, input2):
        input1=input1/input1
        input1=input1
        input1=F.leaky_relu_(input1, negative_slope=0.2)
        input2=F.leaky_relu_(input2, negative_slope=0.2)
        # Process first input through its branch
        # mean1=input1.mean(dim=1,keepdims=True)
        # mean2=input2.mean(dim=1,keepdims=True)

        # std1=input1.std(dim=1,keepdims=True)
        # std2=input1.std(dim=1,keepdims=True)
        # input1=(input1-mean1)/std1
        # input2=(input2-mean2)/std2
        x1 = input1.permute(0, 2, 1)  # Shape: BN x 3 x 35709
        x1 = torch.relu(self.branch1_bn1(self.branch1_conv1(x1)))
        x1 = torch.relu(self.branch1_bn2(self.branch1_conv2(x1)))
        x1 = torch.relu(self.branch1_bn3(self.branch1_conv3(x1)))
        x1 = self.attention1(x1)
        x1_res = self.residual1(input1.permute(0, 2, 1))  # Residual for branch 1

        # Process second input through its branch
        x2 = input2.permute(0, 2, 1)  # Shape: BN x 3 x 35709
        x2 = torch.relu(self.branch2_bn1(self.branch2_conv1(x2)))
        x2 = torch.relu(self.branch2_bn2(self.branch2_conv2(x2)))
        x2 = torch.relu(self.branch2_bn3(self.branch2_conv3(x2)))
        x2 = self.attention2(x2)
        x2_res = self.residual2(input2.permute(0, 2, 1))  # Residual for branch 2

        # Fuse the two branches
        x = torch.cat((x1 + x1_res, x2 + x2_res), dim=1)  # Shape: BN x 512 x 35709

        # Reshape for convolutional processing (Add a spatial dimension for Conv2D)
        x = x.unsqueeze(2)  # Shape: BN x 512 x 1 x 35709

        # Apply the final convolution to adjust the channel size to 32
        x = self.final_conv(x)  # Shape: BN x 32 x 1 x 35709

        # Upsample to the desired output size of 512x512
        x = x.view(x.size(0), 32, 1, -1)  # Shape: BN x 32 x 1 x 35709
        x = self.upsample(x)  # Apply upsampling to shape (BN x 32 x 512 x 512)

        return x  # Final output: BN x 32 x 512 x 512


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)
#===============================================================================
class id_encryptor(nn.Module):
    def __init__(self,first_dim=160): # two files running with 300, 1-Feb-2024
        super(id_encryptor,self).__init__()
        slope = 0.5
        # self.linear1 = nn.Linear(35709*6, 1024)
        # self.linear2 = nn.Linear(1024, 512)
        # self.linear3 = nn.Linear(512, 256)


        # self.linear4 = nn.Linear(256, 512)
        # self.linear5 = nn.Linear(512, 1024)
        # self.linear6 = nn.Linear(1024, 35709*3)

        self.leaky=nn.LeakyReLU(negative_slope=0.5)#nn.ReLU()


        self.encryptor = nn.Sequential(
           nn.Linear(160, 256),
           nn.LeakyReLU(negative_slope=slope),
           nn.Linear(256, 512),
           nn.LeakyReLU(negative_slope=slope),
           nn.Linear(512, 1024),
           nn.LeakyReLU(negative_slope=slope),
           nn.Linear(1024, 512),
           nn.LeakyReLU(negative_slope=slope),
           nn.Linear(512, 256),
           nn.LeakyReLU(negative_slope=slope),
           nn.Linear(256, 80),

        )
        for m in self.encryptor:
            if isinstance(m, nn.Linear):
               # nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
                nn.init.kaiming_normal_(m.weight, a= 0.002)
                nn.init.constant_(m.bias, 0)


    def forward(self, x_id):
        #print('x_id',x_id.shape)
        x_id=x_id.reshape(x_id.shape[0],160)
        x=self.encryptor(x_id)
        # x = self.leaky(self.linear1((x_id.reshape(x_id.shape[0],35709*6))))
        # x = self.leaky(self.linear2(x))
        # x = self.leaky(self.linear3(x))
        # x = self.leaky(self.linear4(x))
        # x = self.leaky(self.linear5(x))
        # x = self.linear6(x)
        #x=x.reshape(x.shape[0],80)

        #z_id=self.encryptor(x_id) # encrypted identity embedding bzx512
        return x
#-------------------------------------------
class tex_encryptor(nn.Module):
    def __init__(self,first_dim=160): # two files running with 300, 1-Feb-2024
        super(tex_encryptor,self).__init__()
        slope = 0.5
        # self.linear1 = nn.Linear(35709*6, 1024)
        # self.linear2 = nn.Linear(1024, 512)
        # self.linear3 = nn.Linear(512, 256)


        # self.linear4 = nn.Linear(256, 512)
        # self.linear5 = nn.Linear(512, 1024)
        # self.linear6 = nn.Linear(1024, 35709*3)

        self.leaky=nn.LeakyReLU(negative_slope=0.5)#nn.ReLU()
        self.sigmoid=nn.Sigmoid()

        self.encryptor = nn.Sequential(
           nn.Linear(160, 256),
           nn.LeakyReLU(negative_slope=slope),
           nn.Linear(256, 512),
           nn.LeakyReLU(negative_slope=slope),
           nn.Linear(512, 1024),
           nn.LeakyReLU(negative_slope=slope),
           nn.Linear(1024, 512),
           nn.LeakyReLU(negative_slope=slope),
           nn.Linear(512, 256),
           nn.LeakyReLU(negative_slope=slope),
           nn.Linear(256, 80),
        )
        for m in self.encryptor:
            if isinstance(m, nn.Linear):
               # nn.init.xavier_normal_(m.weight)
                #nn.init.kaiming_normal_(m.weight.data, a=0.1, mode='fan_in')
                nn.init.kaiming_normal_(m.weight, a= 0.002)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_id):
        #print('x_id',x_id.shape)
        x_id=x_id.reshape(x_id.shape[0],160)
        x=self.encryptor(x_id)
        # x = self.leaky(self.linear1((x_id.reshape(x_id.shape[0],35709*6))))
        # x = self.leaky(self.linear2(x))
        # x = self.leaky(self.linear3(x))
        # x = self.leaky(self.linear4(x))
        # x = self.leaky(self.linear5(x))
        # x = self.linear6(x)
        #z_id=self.encryptor(x_id) # encrypted identity embedding bzx512
        x=self.sigmoid(x)
        return x
'''class Merge_id_att(nn.Module):
    def __init__(self, z_id_size=512):
        super(Merge_id_att, self).__init__()

        self.convt = nn.ConvTranspose2d(z_id_size, 1024, kernel_size=2, stride=1, padding=0)
        self.Upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1 = nn.Conv2d(256, 64, kernel_size=4, stride=3, padding=1)
        self.fc=nn.Linear(7744,7168)
        self.bn=nn.InstanceNorm1d(7168)
        self.h_inchannel = [1024, 1024, 1024, 1024, 512]
        self.z_inchannel = [1024, 2048, 1024, 512, 256]
        self.h_outchannel = [1024, 1024, 1024, 512, 256]

        self.model = nn.ModuleDict(
            {f"layer_{i}" : ADDResBlock(self.h_inchannel[i], self.z_inchannel[i], self.h_outchannel[i])
        for i in range(5)})


    def forward(self, z_id, z_att):
        x = self.convt(z_id.unsqueeze(-1).unsqueeze(-1))

        for i in range(4):
            x = self.Upsample(self.model[f"layer_{i}"](x, z_att[i], z_id))
        x = self.model["layer_4"](x, z_att[4], z_id)
        x=self.conv1(x)
        x=x.view(-1,7744)
        out= self.bn(self.fc(x))
        out=out.view(-1,14,512)
        #out=out-out.mean()/out.std()
       # print(out.shape)
        #exit()
        return out'''

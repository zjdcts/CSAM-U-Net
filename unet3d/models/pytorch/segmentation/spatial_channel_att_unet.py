import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
from functools import partial
from ..classification.myronenko import MyronenkoEncoder, MyronenkoLayer, MyronenkoResidualBlock
from ..classification.decoder import MirroredDecoder
from ..autoencoder.variational import ConvolutionalAutoEncoder
from ..classification.resnet import conv3x3x3, conv1x1x1
from ..classification import resnet


class UNetEncoder(MyronenkoEncoder):
    def __init__(self, n_features, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 feature_dilation=2, downsampling_stride=2, dropout=0.2, layer_widths=None, kernel_size=3):
        super(MyronenkoEncoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 4]
        self.scale = [2, 1, 1, 1, 1]
        self.window = [8, 8, 4, 2, 1]
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
        self.scam = nn.ModuleList()
        in_width = n_features
        for i, n_blocks in enumerate(layer_blocks):
            if layer_widths is not None:
                out_width = layer_widths[i]
            else:
                out_width = base_width * (feature_dilation ** i)
            if dropout and i == 0:
                layer_dropout = dropout
            else:
                layer_dropout = None
            self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                     dropout=layer_dropout, kernel_size=kernel_size))
            if i != len(layer_blocks) - 1:
                # self.scam.append(SCAM2(out_width, self.scale[i], self.window[i]))
                self.scam.append(SCAM(out_width))
                self.downsampling_convolutions.append(conv3x3x3(out_width, out_width, stride=downsampling_stride,
                                                                kernel_size=kernel_size))
            print("Encoder {}:".format(i), in_width, out_width)
            in_width = out_width

    def forward(self, x, y=None):
        outputs = list()
        for layer, downsampling, scam in zip(self.layers[:-1], self.downsampling_convolutions, self.scam):
            x = layer(x)
            x = scam(x)
            outputs.insert(0, x)
            x = downsampling(x)
        x = self.layers[-1](x)
        # print("encoder output:", x.size())
        outputs.insert(0, x)
        return outputs


class UNetDecoder(MirroredDecoder):
    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 upsampling_scale=2, feature_reduction_scale=2, upsampling_mode="trilinear", align_corners=False,
                 layer_widths=None, use_transposed_convolutions=False, kernel_size=2):
        super(MirroredDecoder, self).__init__()
        self.use_transposed_convolutions = use_transposed_convolutions
        print("use_transposed_convolutions:", use_transposed_convolutions)
        if layer_blocks is None:
            self.layer_blocks = [1, 1, 1, 1]
        else:
            self.layer_blocks = layer_blocks
        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        # self.scam = nn.ModuleList()
        if use_transposed_convolutions:
            self.upsampling_blocks = nn.ModuleList()
        else:
            self.upsampling_blocks = list()
        self.base_width = base_width
        self.feature_reduction_scale = feature_reduction_scale
        self.layer_widths = layer_widths
        for i, n_blocks in enumerate(self.layer_blocks):
            depth = len(self.layer_blocks) - (i + 1)
            in_width, out_width = self.calculate_layer_widths(depth)

            if depth != 0:
                self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=in_width,
                                         kernel_size=kernel_size))
                # self.scam.append(SCAM(out_width))
                if self.use_transposed_convolutions:
                    self.pre_upsampling_blocks.append(nn.Sequential())
                    # print(kernel_size, upsampling_scale)
                    self.upsampling_blocks.append(nn.ConvTranspose3d(in_width, out_width, kernel_size=upsampling_scale,
                                                                     stride=upsampling_scale))
                else:
                    self.pre_upsampling_blocks.append(resnet.conv1x1x1(in_width, out_width, stride=1))
                    self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                                          mode=upsampling_mode, align_corners=align_corners))
            else:
                self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                         kernel_size=kernel_size))

    def calculate_layer_widths(self, depth):
        in_width, out_width = super().calculate_layer_widths(depth=depth)
        if depth != len(self.layer_blocks) - 1:
            in_width *= 2
        print("Decoder {}:".format(depth), in_width, out_width)
        return in_width, out_width

    def forward(self, inputs, y=None):
        x = inputs[0]
        # print("x:", x.size())
        for i, (pre, up, lay) in enumerate(
                zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1])):
            x = lay(x)
            # x = kqv(k, q, x)
            # print("lay(x):", x.size())
            x = pre(x)
            x = up(x)
            # print("up(x):", x.size())
            x = torch.cat((x, inputs[i + 1]), 1)
            # print("cat(x):", x.size())
        x = self.layers[-1](x)
        return x


class SCAM(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ch_maxpool = nn.AdaptiveMaxPool3d(1)
        self.ch_avgpool = nn.AdaptiveAvgPool3d(1)
        self.ch_att = nn.TransformerEncoderLayer(in_channels, 8, in_channels // 4)
        self.ch_conv = nn.Conv1d(2, 1, 1)
        self.ch_sigmoid = nn.Sigmoid()

        self.sp_conv = conv3x3x3(2, 1, stride=2, padding=0, kernel_size=2)
        self.sp_rev_conv = nn.ConvTranspose3d(1, 1, stride=2, kernel_size=2)
        self.sp_att = nn.TransformerEncoderLayer(in_channels, 8, in_channels // 4)
        self.sp_sigmoid = nn.Sigmoid()

    def forward(self, x):
        # identity = x
        ch_maxpool = self.ch_maxpool(x)
        ch_avgpool = self.ch_avgpool(x)
        # print(ch_maxpool.size(), ch_avgpool.size())
        ch_maxpool = ch_maxpool.squeeze(dim=4).squeeze(dim=3).squeeze(dim=2).unsqueeze(dim=1)
        ch_avgpool = ch_avgpool.squeeze(dim=4).squeeze(dim=3).squeeze(dim=2).unsqueeze(dim=1)
        # print(ch_maxpool.size(), ch_avgpool.size())
        # ch_att = self.ch_att(torch.cat([ch_maxpool, ch_avgpool], dim=1))
        ch_att = self.ch_att(torch.cat([ch_maxpool, ch_maxpool], dim=1))
        # print(ch_att.size())
        ch_conv = self.ch_conv(ch_att)
        # print(ch_conv.size())
        ch_out = self.ch_sigmoid(ch_conv.squeeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4))
        x = x + x * ch_out

        sp_maxpool, _ = torch.max(x, dim=1, keepdim=True)
        sp_avgpool = torch.mean(x, dim=1, keepdim=True)
        # print(sp_maxpool.size(), sp_avgpool.size())
        sp_out = torch.cat([sp_maxpool, sp_avgpool], dim=1)
        sp_out = self.sp_conv(sp_out)
        sp_out = self.sp_rev_conv(sp_out)
        sp_out = self.sp_sigmoid(sp_out)
        x = x + x * sp_out

        # print('sp_out', sp_out.squeeze().size())
        # affine = np.array([[-1., 0., 0., 0.],
        #                    [0., -1., 0., 239.],
        #                    [0., 0., 1., 0.],
        #                    [0., 0., 0., 1.]])
        #
        # y = sp_out
        # if y.size()[2] == 112:
        #     y = F.interpolate(y, size=[240, 240, 155], mode='trilinear', align_corners=False).squeeze().cpu().numpy()
        #     img = nib.load('/home/cjp/hezhe/3DUnetCNN/examples/brats2020/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_057/BraTS20_Validation_057_t1ce.nii.gz')
        #     y = y[slice(None, None, -1), slice(None, None, -1), slice(None, None, None)]
        #     new_img = nib.Nifti1Image(y, img.affine)
        #     nib.save(new_img, '/home/cjp/hezhe/3DUnetCNN/examples/brats2020/vision/12_70.nii.gz')
        # elif y.size()[2] == 56:
        #     y = F.interpolate(y, size=[240, 240, 155], mode='trilinear', align_corners=False).squeeze().cpu().numpy()
        #     img = nib.load('/home/cjp/hezhe/3DUnetCNN/examples/brats2020/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_057/BraTS20_Validation_057_t1ce.nii.gz')
        #     y = y[slice(None, None, -1), slice(None, None, -1), slice(None, None, None)]
        #     new_img = nib.Nifti1Image(y, img.affine)
        #     nib.save(new_img, '/home/cjp/hezhe/3DUnetCNN/examples/brats2020/vision/057_2_70.nii.gz')
        # elif y.size()[2] == 28:
        #     y = F.interpolate(y, size=[240, 240, 155], mode='trilinear', align_corners=False).squeeze().cpu().numpy()
        #     img = nib.load('/home/cjp/hezhe/3DUnetCNN/examples/brats2020/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_057/BraTS20_Validation_057_t1ce.nii.gz')
        #     y = y[slice(None, None, -1), slice(None, None, -1), slice(None, None, None)]
        #     new_img = nib.Nifti1Image(y, img.affine)
        #     nib.save(new_img, '/home/cjp/hezhe/3DUnetCNN/examples/brats2020/vision/057_3.nii.gz')
        # else:
        #     y = F.interpolate(y, size=[240, 240, 155], mode='trilinear', align_corners=False).squeeze().cpu().numpy()
        #     img = nib.load('/home/cjp/hezhe/3DUnetCNN/examples/brats2020/MICCAI_BraTS2020_ValidationData/BraTS20_Validation_057/BraTS20_Validation_057_t1ce.nii.gz')
        #     y = y[slice(None, None, -1), slice(None, None, -1), slice(None, None, None)]
        #     new_img = nib.Nifti1Image(y, img.affine)
        #     nib.save(new_img, '/home/cjp/hezhe/3DUnetCNN/examples/brats2020/vision/057_4.nii.gz')
        # return x + identity
        return x


class SCAM2(nn.Module):
    def __init__(self, in_channels, scale, window):
        super().__init__()
        self.ch_maxpool = nn.AdaptiveMaxPool3d(1)
        self.ch_avgpool = nn.AdaptiveAvgPool3d(1)
        self.ch_att = nn.TransformerEncoder(nn.TransformerEncoderLayer(in_channels, 4, in_channels * 2), 4)
        self.ch_conv = nn.Conv3d(in_channels * 2, in_channels, 1)

        self.window = window
        # self.flatten_dim = 378
        self.flatten_dim = window * window * window
        self.embedding_dim = self.flatten_dim * 2
        self.sp_conv = conv3x3x3(2, 1, stride=scale, padding=0, kernel_size=scale)
        self.sp_rev_conv = nn.ConvTranspose3d(1, 1, stride=scale, kernel_size=scale)
        self.sp_att = nn.TransformerEncoder(nn.TransformerEncoderLayer(self.flatten_dim, self.window, self.embedding_dim), 4)
        self.patch_x = None
        self.patch_y = None
        self.patch_z = None

    def forward(self, x):
        # print("!!")
        ch_maxpool = self.ch_maxpool(x)
        ch_avgpool = self.ch_avgpool(x)
        # print(ch_maxpool.size(), ch_avgpool.size())
        ch_conv = self.ch_conv(torch.cat([ch_maxpool, ch_avgpool], dim=1))

        ch_conv = ch_conv.squeeze(dim=4).squeeze(dim=3).squeeze(dim=2).unsqueeze(dim=1)
        # print(ch_maxpool.size(), ch_avgpool.size())
        ch_att = self.ch_att(ch_conv)
        # print(ch_att.size())
        # print(ch_conv.size())
        # ch_out = self.ch_sigmoid(ch_conv.squeeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4))
        ch_att = ch_att.squeeze(dim=1).unsqueeze(dim=2).unsqueeze(dim=3).unsqueeze(dim=4)
        x = x + ch_att

        sp_maxpool, _ = torch.max(x, dim=1, keepdim=True)
        sp_avgpool = torch.mean(x, dim=1, keepdim=True)
        # print(sp_maxpool.size(), sp_avgpool.size())
        sp_out = self.sp_conv(torch.cat([sp_maxpool, sp_avgpool], dim=1))
        sp_out = (sp_out.unfold(2, self.window, self.window).unfold(3, self.window, self.window).unfold(4, self.window, self.window).contiguous())
        self.patch_x = sp_out.size(2)
        self.patch_y = sp_out.size(3)
        self.patch_z = sp_out.size(4)
        # print(sp_out.size())
        sp_out = sp_out.view(sp_out.size(0), sp_out.size(1), -1, self.flatten_dim)
        sp_out = sp_out.permute(0, 2, 3, 1).contiguous()
        sp_out = sp_out.view(sp_out.size(0), -1, self.flatten_dim)
        sp_out = self.sp_att(sp_out)
        sp_out = sp_out.view(sp_out.size(0), 1, self.patch_x, self.patch_y, self.patch_z, self.window, self.window, self.window).contiguous()
        sp_out = sp_out.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous().view(sp_out.size(0), sp_out.size(1),
                                                                          self.patch_x * self.window,
                                                                          self.patch_y * self.window,
                                                                          self.patch_z * self.window)
        sp_out = self.sp_rev_conv(sp_out)
        return x + sp_out


class SCAM_UNet(ConvolutionalAutoEncoder):
    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetDecoder, n_outputs=1, is_training, is_dsv,
                 is_half, **kwargs):
        super().__init__(*args, encoder_class=encoder_class, decoder_class=decoder_class, n_outputs=n_outputs, **kwargs)
        self.set_final_convolution(n_outputs=n_outputs)


class AutoImplantUNet(SCAM_UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y = super(AutoImplantUNet, self).forward(x)
        return y - x

    def test(self, x):
        return super(AutoImplantUNet, self).forward(x)

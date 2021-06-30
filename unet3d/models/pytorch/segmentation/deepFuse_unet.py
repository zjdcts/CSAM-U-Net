import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .. import conv3x3x3
from ..classification.myronenko import MyronenkoEncoder, MyronenkoLayer, MyronenkoResidualBlock
from ..classification.decoder import MirroredDecoder
from ..autoencoder.variational import DeepFuseConvolutionalAutoEncoder
from ..classification import resnet


class UNetEncoder(MyronenkoEncoder):
    def __init__(self, n_features, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 feature_dilation=2, downsampling_stride=2, dropout=0.2, layer_widths=None, kernel_size=3):
        super(MyronenkoEncoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 4]
        in_width = n_features
        self.conv1 = layer(n_blocks=layer_blocks[0], block=block, in_planes=in_width, planes=base_width,
                           dropout=dropout, kernel_size=kernel_size)
        self.downsample1 = conv3x3x3(base_width, base_width, stride=downsampling_stride, kernel_size=kernel_size)
        self.conv2 = layer(n_blocks=layer_blocks[1], block=block, in_planes=base_width, planes=base_width * 2,
                           dropout=None, kernel_size=kernel_size)
        self.downsample2 = conv3x3x3(base_width * 2, base_width * 2, stride=downsampling_stride,
                                     kernel_size=kernel_size)
        self.conv3 = layer(n_blocks=layer_blocks[2], block=block, in_planes=base_width * 2, planes=base_width * 4,
                           dropout=None, kernel_size=kernel_size)
        self.downsample3 = conv3x3x3(base_width * 4, base_width * 4, stride=downsampling_stride,
                                     kernel_size=kernel_size)
        self.conv4 = layer(n_blocks=layer_blocks[3], block=block, in_planes=base_width * 6, planes=base_width * 8,
                           dropout=None, kernel_size=kernel_size)
        # self.downsample4 = conv3x3x3(base_width * 8, base_width * 8, stride=downsampling_stride,
        #                              kernel_size=kernel_size)
        # self.conv5 = layer(n_blocks=layer_blocks[4], block=block, in_planes=base_width * 8 + base_width * 4,
        #                    planes=base_width * 16,
        #                    dropout=None, kernel_size=kernel_size)

    def forward(self, x, y=None):
        outputs = list()
        x = self.conv1(x)
        outputs.insert(0, x)
        x = self.conv2(self.downsample1(x))
        outputs.insert(0, x)
        x2_4 = F.interpolate(x, scale_factor=0.25, mode='trilinear', align_corners=False)
        x = self.conv3(self.downsample2(x))
        outputs.insert(0, x)
        x = self.conv4(torch.cat([x2_4, self.downsample3(x)], dim=1))
        outputs.insert(0, x)
        # x = self.conv5(torch.cat([x3_5, self.downsample4(x)], dim=1))
        # outputs.insert(0, x)
        return outputs


class UNetEncoder2(MyronenkoEncoder):
    def __init__(self, n_features, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 feature_dilation=2, downsampling_stride=2, dropout=0.2, layer_widths=None, kernel_size=3):
        super(MyronenkoEncoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 4]
        in_width = n_features
        # self.conv1 = layer(n_blocks=layer_blocks[0], block=block, in_planes=in_width + base_width * 3, planes=base_width,
        #                    dropout=dropout, kernel_size=kernel_size) v2
        self.conv1 = layer(n_blocks=layer_blocks[0], block=block, in_planes=in_width,
                           planes=base_width, dropout=dropout, kernel_size=kernel_size)
        self.downsample1 = conv3x3x3(base_width, base_width, stride=downsampling_stride, kernel_size=kernel_size)
        # self.conv2 = layer(n_blocks=layer_blocks[1], block=block, in_planes=base_width * 4, planes=base_width * 2,
        #                    dropout=None, kernel_size=kernel_size) v2
        self.conv2 = layer(n_blocks=layer_blocks[1], block=block, in_planes=base_width, planes=base_width * 2,
                           dropout=None, kernel_size=kernel_size)
        self.downsample2 = conv3x3x3(base_width * 2, base_width * 2, stride=downsampling_stride,
                                     kernel_size=kernel_size)
        # self.conv3 = layer(n_blocks=layer_blocks[2], block=block, in_planes=base_width * 5, planes=base_width * 4,
        #                    dropout=None, kernel_size=kernel_size) v2
        self.conv3 = layer(n_blocks=layer_blocks[2], block=block, in_planes=base_width * 2, planes=base_width * 4,
                           dropout=None, kernel_size=kernel_size)
        self.downsample3 = conv3x3x3(base_width * 4, base_width * 4, stride=downsampling_stride,
                                     kernel_size=kernel_size)
        # self.conv4 = layer(n_blocks=layer_blocks[3], block=block, in_planes=base_width * 7, planes=base_width * 8,
        #                    dropout=None, kernel_size=kernel_size) v2
        # self.conv4 = layer(n_blocks=layer_blocks[3], block=block, in_planes=base_width * 6, planes=base_width * 8,
        #                    dropout=None, kernel_size=kernel_size) BTS-Net
        self.conv4 = layer(n_blocks=layer_blocks[3], block=block, in_planes=base_width * 4, planes=base_width * 8,
                           dropout=None, kernel_size=kernel_size)
        self.downsample4 = conv3x3x3(base_width * 8, base_width * 8, stride=downsampling_stride,
                                     kernel_size=kernel_size)
        # self.conv5 = layer(n_blocks=layer_blocks[4], block=block, in_planes=base_width * 8 + base_width * 4,
        #                    planes=base_width * 16,
        #                    dropout=None, kernel_size=kernel_size) v2
        # self.conv5 = layer(n_blocks=layer_blocks[4], block=block, in_planes=base_width * 10,
        #                    planes=base_width * 16,
        #                    dropout=None, kernel_size=kernel_size) BTS-Net
        self.conv5 = layer(n_blocks=layer_blocks[4], block=block, in_planes=base_width * 8,
                           planes=base_width * 16,
                           dropout=None, kernel_size=kernel_size)

    def forward(self, x, y=None):
        outputs = list()
        # x_l = F.interpolate(x[0], scale_factor=2, mode='trilinear', align_corners=False)
        # z = self.conv1(torch.cat([y, x[1], x_l], dim=1))
        z = self.conv1(x)
        outputs.insert(0, z)
        # x_g = F.interpolate(x[1], scale_factor=0.5, mode='trilinear', align_corners=False)
        # z = self.conv2(torch.cat([self.downsample1(z), x_g, x[0]], dim=1))
        z = self.conv2(self.downsample1(z))
        outputs.insert(0, z)
        # x_g = F.interpolate(z, scale_factor=0.25, mode='trilinear', align_corners=False) BTS-Net
        # x_l = F.interpolate(z, scale_factor=0.125, mode='trilinear', align_corners=False) BTS-Net
        # z = self.conv3(torch.cat([self.downsample2(z), x_g, x_l], dim=1))
        z = self.conv3(self.downsample2(z))
        outputs.insert(0, z)
        # x_g = F.interpolate(x[1], scale_factor=0.125, mode='trilinear', align_corners=False)
        # x3_5 = F.interpolate(z, scale_factor=0.25, mode='trilinear', align_corners=False)
        # x_l = F.interpolate(x[0], scale_factor=0.25, mode='trilinear', align_corners=False)
        # z = self.conv4(torch.cat([self.downsample3(z), x_g, x_l], dim=1))
        # z = self.conv4(torch.cat([self.downsample3(z), x_g], dim=1)) BTS-Net
        z = self.conv4(self.downsample3(z))
        outputs.insert(0, z)
        # z = self.conv5(torch.cat([x3_5, self.downsample4(z)], dim=1))
        # z = self.conv5(torch.cat([self.downsample4(z), x_l], dim=1)) BTS-Net
        z = self.conv5(self.downsample4(z))
        outputs.insert(0, z)
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
        # self.conv1 = layer(n_blocks=layer_blocks[0], block=block,
        #                    in_planes=base_width * 12, planes=base_width * 4,
        #                    dropout=None, kernel_size=kernel_size) v2
        self.conv1 = layer(n_blocks=layer_blocks[0], block=block,
                           in_planes=base_width * 14, planes=base_width * 4,
                           dropout=None, kernel_size=kernel_size)
        # self.pre_upsample1 = resnet.conv1x1x1(base_width * 28, base_width * 8, 1)
        self.upsample1 = partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                 mode=upsampling_mode, align_corners=align_corners)
        # self.conv2 = layer(n_blocks=layer_blocks[1], block=block,
        #                    in_planes=base_width * 16, planes=base_width * 2,
        #                    dropout=None, kernel_size=kernel_size) v2
        self.conv2 = layer(n_blocks=layer_blocks[1], block=block,
                           in_planes=base_width * 10, planes=base_width * 2,
                           dropout=None, kernel_size=kernel_size)
        # self.pre_upsample2 = resnet.conv1x1x1(base_width * 20, base_width * 4, 1)
        self.upsample2 = partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                 mode=upsampling_mode, align_corners=align_corners)
        self.conv3 = layer(n_blocks=layer_blocks[2], block=block,
                           in_planes=base_width * 4, planes=base_width,
                           dropout=None, kernel_size=kernel_size)
        # self.pre_upsample3 = resnet.conv1x1x1(base_width * 24, base_width * 2, 1)
        self.upsample3 = partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                 mode=upsampling_mode, align_corners=align_corners)
        self.conv4 = layer(n_blocks=layer_blocks[3], block=block,
                           in_planes=base_width * 2, planes=base_width,
                           dropout=None, kernel_size=kernel_size)
        # self.pre_upsample4 = resnet.conv1x1x1(base_width * 4, base_width, 1)
        # self.upsample4 = partial(nn.functional.interpolate, scale_factor=upsampling_scale,
        #                          mode=upsampling_mode, align_corners=align_corners)
        # self.conv5 = layer(n_blocks=layer_blocks[4], block=block,
        #                    in_planes=base_width * 2, planes=base_width * 2,
        #                    dropout=None, kernel_size=kernel_size)
        # self.pre_upsample5 = resnet.conv1x1x1(base_width * 2, base_width, 1)

    def forward(self, inputs, y=None):
        outputs = list()
        x = inputs[0]
        # print(inputs[0].size(), inputs[1].size(), inputs[2].size(), inputs[3].size())
        x1_1 = F.interpolate(inputs[1], scale_factor=0.5, mode='trilinear', align_corners=False)
        x1_2 = F.interpolate(inputs[2], scale_factor=0.25, mode='trilinear', align_corners=False)
        # print(torch.cat([x1_2, x1_1, x], dim=1).size())
        # x = self.conv1(torch.cat([x1_1, x], dim=1)) v2
        x = self.conv1(torch.cat([x1_2, x1_1, x], dim=1))
        # outputs.insert(1, x)
        x = self.upsample1(x)
        x2_1 = F.interpolate(inputs[2], scale_factor=0.5, mode='trilinear', align_corners=False)
        # x = self.conv2(torch.cat([x2_1, inputs[1], x], dim=1)) v2
        x = self.conv2(torch.cat([x2_1, inputs[1], x], dim=1))
        # outputs.insert(1, x)
        x = self.upsample2(x)
        x = self.conv3(torch.cat([inputs[2], x], dim=1))
        # outputs.insert(1, x)
        x = self.upsample3(x)
        # print(x.size())
        x = self.conv4(torch.cat([inputs[3], x], dim=1))
        # outputs.insert(1, x)
        # x = self.pre_upsample5(self.conv5(torch.cat([inputs[4], x], dim=1)))
        return x


class UNetDecoder2(MirroredDecoder):
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
        self.conv1 = layer(n_blocks=layer_blocks[0], block=block,
                           in_planes=base_width * 28, planes=base_width * 28,
                           dropout=None, kernel_size=kernel_size)
        self.pre_upsample1 = resnet.conv1x1x1(base_width * 28, base_width * 8, 1)
        self.upsample1 = partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                 mode=upsampling_mode, align_corners=align_corners)
        self.conv2 = layer(n_blocks=layer_blocks[1], block=block,
                           in_planes=base_width * 20, planes=base_width * 20,
                           dropout=None, kernel_size=kernel_size)
        self.pre_upsample2 = resnet.conv1x1x1(base_width * 20, base_width * 4, 1)
        self.upsample2 = partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                 mode=upsampling_mode, align_corners=align_corners)
        # self.conv3 = layer(n_blocks=layer_blocks[2], block=block,
        #                    in_planes=base_width * 24, planes=base_width * 24,
        #                    dropout=None, kernel_size=kernel_size)
        # self.sup3 = resnet.conv1x1x1(base_width * 24, base_width, 1)
        # self.pre_upsample3 = resnet.conv1x1x1(base_width * 24, base_width * 2, 1)
        self.conv3 = layer(n_blocks=layer_blocks[2], block=block,
                           in_planes=base_width * 8, planes=base_width * 8,
                           dropout=None, kernel_size=kernel_size)
        # self.sup3 = resnet.conv1x1x1(base_width * 8, base_width, 1) BTS-Net
        self.pre_upsample3 = resnet.conv1x1x1(base_width * 8, base_width * 2, 1)
        self.upsample3 = partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                 mode=upsampling_mode, align_corners=align_corners)
        self.conv4 = layer(n_blocks=layer_blocks[3], block=block,
                           in_planes=base_width * 4, planes=base_width * 4,
                           dropout=None, kernel_size=kernel_size)
        self.pre_upsample4 = resnet.conv1x1x1(base_width * 4, base_width, 1)
        # self.sup4 = resnet.conv1x1x1(base_width * 4, base_width, 1) BTS-Net
        self.upsample4 = partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                 mode=upsampling_mode, align_corners=align_corners)
        self.conv5 = layer(n_blocks=layer_blocks[4], block=block,
                           in_planes=base_width * 2, planes=base_width * 2,
                           dropout=None, kernel_size=kernel_size)
        self.pre_upsample5 = resnet.conv1x1x1(base_width * 2, base_width, 1)

        self.conv1_p = layer(n_blocks=layer_blocks[0], block=block,
                             in_planes=base_width * 19, planes=base_width * 8,
                             dropout=None, kernel_size=kernel_size)
        # self.sup1 = resnet.conv1x1x1(base_width * 8, base_width, 1)
        self.conv2_p = layer(n_blocks=layer_blocks[1], block=block,
                             in_planes=base_width * 19, planes=base_width * 4,
                             dropout=None, kernel_size=kernel_size)
        # self.sup2 = resnet.conv1x1x1(base_width * 4, base_width, 1)
        self.conv3_p = layer(n_blocks=layer_blocks[2], block=block,
                             in_planes=base_width * 11, planes=base_width * 2,
                             dropout=None, kernel_size=kernel_size)
        self.sup3 = resnet.conv1x1x1(base_width * 2, base_width, 1)
        self.conv4_p = layer(n_blocks=layer_blocks[3], block=block,
                             in_planes=base_width * 6, planes=base_width,
                             dropout=None, kernel_size=kernel_size)
        self.conv5_p = layer(n_blocks=layer_blocks[4], block=block,
                             in_planes=base_width * 3, planes=base_width,
                             dropout=None, kernel_size=kernel_size)

    def forward(self, inputs, y=None):
        outputs = list()
        x = inputs[0]
        # print(inputs[0].size(), inputs[1].size(), inputs[2].size(), inputs[3].size(), inputs[4].size())
        x1_1 = F.interpolate(inputs[1], scale_factor=0.5, mode='trilinear', align_corners=False)
        x1_2 = F.interpolate(inputs[2], scale_factor=0.25, mode='trilinear', align_corners=False)
        # print(torch.cat([x1_2, x1_1, x], dim=1).size())
        x = self.upsample1(self.pre_upsample1(self.conv1(torch.cat([x1_2, x1_1, x], dim=1))))
        x2_1 = F.interpolate(inputs[2], scale_factor=0.5, mode='trilinear', align_corners=False)
        x = self.upsample2(self.pre_upsample2(self.conv2(torch.cat([x2_1, inputs[1], x], dim=1))))
        # outputs.append(F.interpolate(x, scale_factor=8, mode='trilinear', align_corners=False))
        # x3_1 = F.interpolate(inputs[0], scale_factor=4, mode='trilinear', align_corners=False) v2
        # x = self.conv3(torch.cat([inputs[2], x, x3_1], dim=1)) v2
        x = self.conv3(torch.cat([inputs[2], x], dim=1))
        # outputs.append(F.interpolate(self.sup3(x), scale_factor=4, mode='trilinear', align_corners=False)) BTS-Net
        x = self.upsample3(self.pre_upsample3(x))
        y_1 = self.conv4(torch.cat([inputs[3], x], dim=1))
        # outputs.append(F.interpolate(self.sup4(x), scale_factor=2, mode='trilinear', align_corners=False)) BTS-Net
        x = self.upsample4(self.pre_upsample4(y_1))
        y_2 = self.conv5(torch.cat([inputs[4], x], dim=1))
        x = self.pre_upsample5(y_2)

        outputs.append(x)
        x_1 = F.interpolate(x, scale_factor=0.0625, mode='trilinear', align_corners=False)
        x_2 = F.interpolate(inputs[3], scale_factor=0.125, mode='trilinear', align_corners=False)
        x1_p = self.conv1_p(torch.cat([inputs[0], x_1, x_2], dim=1))
        # outputs.append(F.interpolate(self.sup1(x1_p), scale_factor=16, mode='trilinear', align_corners=False))
        x_1 = F.interpolate(x, scale_factor=0.125, mode='trilinear', align_corners=False)
        x_2 = F.interpolate(inputs[3], scale_factor=0.25, mode='trilinear', align_corners=False)
        x1_p = F.interpolate(x1_p, scale_factor=2, mode='trilinear', align_corners=False)
        # print(inputs[1].size(), x1_p.size(), x_1.size(), x_2.size())
        x2_p = self.conv2_p(torch.cat([inputs[1], x1_p, x_1, x_2], dim=1))
        # outputs.append(F.interpolate(self.sup2(x2_p), scale_factor=8, mode='trilinear', align_corners=False))
        x_1 = F.interpolate(x, scale_factor=0.25, mode='trilinear', align_corners=False)
        x_2 = F.interpolate(inputs[3], scale_factor=0.5, mode='trilinear', align_corners=False)
        x2_p = F.interpolate(x2_p, scale_factor=2, mode='trilinear', align_corners=False)
        # print(inputs[2].size(), x2_p.size(), x_1.size(), x_2.size())
        x3_p = self.conv3_p(torch.cat([inputs[2], x2_p, x_1, x_2], dim=1))
        outputs.append(F.interpolate(self.sup3(x3_p), scale_factor=4, mode='trilinear', align_corners=False))
        x3_p = F.interpolate(x3_p, scale_factor=2, mode='trilinear', align_corners=False)
        # print(y_1.size(), x3_p.size())
        x = self.conv4_p(torch.cat([y_1, x3_p], dim=1))
        x = F.interpolate(x, scale_factor=2, mode='trilinear', align_corners=False)
        # print(y_2.size(), x.size())
        x = self.conv5_p(torch.cat([y_2, x], dim=1))
        outputs.append(x)
        return outputs


class DeepFuseUNet(DeepFuseConvolutionalAutoEncoder):
    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetDecoder, n_outputs=1, is_training, is_dsv,
                 is_half, **kwargs):
        super().__init__(*args, encoder_class=encoder_class, encoder_class2=UNetEncoder2, decoder_class=decoder_class,
                         decoder_class2=UNetDecoder2, n_outputs=n_outputs, **kwargs)
        self.set_final_convolution(n_outputs=n_outputs)


class AutoImplantUNet(DeepFuseUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y = super(AutoImplantUNet, self).forward(x)
        return y - x

    def test(self, x):
        return super(AutoImplantUNet, self).forward(x)

import torch
import torch.nn.functional as F
import torch.nn as nn

from .. import conv3x3x3
from ..classification.myronenko import MyronenkoEncoder, MyronenkoLayer, MyronenkoResidualBlock
from ..classification.decoder import MirroredDecoder
from ..autoencoder.variational import ConvolutionalAutoEncoder


class UNetEncoder(MyronenkoEncoder):
    def __init__(self, n_features, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 feature_dilation=2, downsampling_stride=2, dropout=0.2, layer_widths=None, kernel_size=3):
        super(MyronenkoEncoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 4]
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
        self.ua = nn.ModuleList()
        self.conv = nn.ModuleList()
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
                self.downsampling_convolutions.append(conv3x3x3(out_width, out_width, stride=downsampling_stride,
                                                                kernel_size=kernel_size))
                self.ua.append(UNetAttention(len(layer_blocks) - 1 - i, out_width, scale=1, inter_planes=out_width))
                self.conv.append(nn.Conv3d(2 * out_width, out_width, 1))

            print("Encoder {}:".format(i), in_width, out_width)
            in_width = out_width

    def forward(self, x, y=None):
        outputs = list()
        for layer, downsampling, ua, conv in zip(self.layers[:-1], self.downsampling_convolutions, self.ua, self.conv):
            x = layer(x)
            y = x
            z = x
            y = ua(y)
            x = (x + 1) * y
            outputs.insert(0, conv(torch.cat([x, z], dim=1)))
            x = downsampling(x)

        x = self.layers[-1](x)
        # print("encoder output:", x.size())
        outputs.insert(0, x)
        return outputs


class UNetDecoder(MirroredDecoder):
    def calculate_layer_widths(self, depth):
        in_width, out_width = super().calculate_layer_widths(depth=depth)
        if depth != len(self.layer_blocks) - 1:
            in_width *= 2
        print("Decoder {}:".format(depth), in_width, out_width)
        return in_width, out_width

    def forward(self, inputs, y=None):
        x = inputs[0]
        # print("x:", x.size())
        for i, (pre, up, lay) in enumerate(zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1])):
            x = lay(x)
            # print("lay(x):", x.size())
            x = pre(x)
            x = up(x)
            # print("up(x):", x.size())
            x = torch.cat((x, inputs[i + 1]), 1)
            # print("cat(x):", x.size())
        x = self.layers[-1](x)
        return x


class UNetAttention(nn.Module):
    def __init__(self, block_num, in_planes, h=7, w=7, d=7, scale=8, inter_planes=16):
        super(UNetAttention, self).__init__()
        self.conv = nn.Conv3d(in_planes, inter_planes, 1)
        self.re_conv = nn.Conv3d(inter_planes, in_planes, 1)
        self.downsample = nn.ModuleList()
        self.upsample = nn.ModuleList()
        self.block_num = block_num
        self.h = h
        self.w = w
        self.d = d
        self.scale = scale
        self.dim_s = h * w * d
        self.dim_c = inter_planes * scale

        for i in range(block_num):
            self.downsample.append(nn.Conv3d(inter_planes, inter_planes, 2, 2))
            self.upsample.append(nn.ConvTranspose3d(2 * inter_planes, inter_planes, 2, 2))

        self.trans_c = nn.TransformerEncoderLayer(d_model=self.dim_s, nhead=self.h)
        self.t_c = nn.TransformerEncoder(self.trans_c, 8)
        self.trans_s = nn.TransformerEncoderLayer(d_model=self.dim_c, nhead=8)
        self.t_s = nn.TransformerEncoder(self.trans_s, 8)
        self.bottle = nn.Conv3d(self.dim_c, inter_planes, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        sk = list()
        for i in range(self.block_num):
            x = self.downsample[i](x)
            sk.insert(0, x)
        x = x.unfold(2, self.h, self.h).unfold(3, self.w, self.w).unfold(4, self.d, self.d).contiguous()
        x = x.view(-1, self.dim_c, self.dim_s).contiguous()
        x = self.t_c(x)
        x = x.permute(0, 2, 1).contiguous()
        x = self.t_s(x)
        x = x.permute(0, 2, 1).contiguous().view(-1, self.dim_c, self.h, self.w, self.d).contiguous()
        # print(x.size())
        # x = F.interpolate(self.bottle(x), scale_factor=2, mode='trilinear', align_corners=False)
        for i in range(self.block_num):
            x = self.upsample[i](torch.cat([x, sk[i]], dim=1))
        return self.sigmoid(self.re_conv(x))


class UAUNet(ConvolutionalAutoEncoder):
    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetDecoder, n_outputs=1, is_training, is_dsv,
                 is_half, **kwargs):
        super().__init__(*args, encoder_class=encoder_class, decoder_class=decoder_class, n_outputs=n_outputs, **kwargs)
        self.set_final_convolution(n_outputs=n_outputs)


class AutoImplantUNet(UAUNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y = super(AutoImplantUNet, self).forward(x)
        return y - x

    def test(self, x):
        return super(AutoImplantUNet, self).forward(x)

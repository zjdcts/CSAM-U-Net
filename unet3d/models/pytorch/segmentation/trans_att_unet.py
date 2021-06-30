import torch
import torch.nn as nn
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
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
        self.trans_att = nn.ModuleList()
        self.len = len(layer_blocks)
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
                self.trans_att.append(Trans_Att(out_width, 2 ** (self.len - i - 1)))
                self.downsampling_convolutions.append(conv3x3x3(out_width, out_width, stride=downsampling_stride,
                                                                kernel_size=kernel_size))
            print("Encoder {}:".format(i), in_width, out_width)
            in_width = out_width

    def forward(self, x, y=None):
        outputs = list()
        for layer, downsampling, trans_att in zip(self.layers[:-1], self.downsampling_convolutions, self.trans_att):
            x = layer(x)
            x = trans_att(x)
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


class Trans_Att(nn.Module):
    def __init__(self, in_channels, down_scale):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, down_scale, down_scale)
        self.trans_ch = nn.TransformerEncoderLayer(in_channels, 8, in_channels // 2)
        self.trans_sp = nn.TransformerEncoderLayer(378, 9, 189)
        self.re_conv = nn.ConvTranspose3d(in_channels, in_channels, down_scale, down_scale)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        identity = x
        x = self.conv(x)
        N, C, H, W, D = x.size()
        trans_ch = self.trans_ch(x.reshape(N, C, -1).permute(0, 2, 1))
        trans_sp = self.trans_sp(trans_ch.permute(0, 2, 1))
        x = trans_sp.permute(0, 2, 1).reshape(N, C, H, W, D)
        return identity + identity * self.sig(self.re_conv(x))


class Trans_Att_UNet(ConvolutionalAutoEncoder):
    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetDecoder, n_outputs=1, is_training, is_dsv,
                 is_half, **kwargs):
        super().__init__(*args, encoder_class=encoder_class, decoder_class=decoder_class, n_outputs=n_outputs, **kwargs)
        self.set_final_convolution(n_outputs=n_outputs)


class AutoImplantUNet(Trans_Att_UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y = super(AutoImplantUNet, self).forward(x)
        return y - x

    def test(self, x):
        return super(AutoImplantUNet, self).forward(x)

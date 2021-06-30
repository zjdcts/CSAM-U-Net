import torch
import torch.nn as nn
from ..classification.myronenko import MyronenkoEncoder, MyronenkoLayer, MyronenkoResidualBlock
from ..classification.decoder import MirroredDecoder
from ..autoencoder.variational import CascadeKQVConvolutionalAutoEncoder
from ..classification.resnet import conv3x3x3, conv1x1x1


class UNetEncoder(MyronenkoEncoder):
    def forward(self, x, q=None):
        outputs = list()
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            x = layer(x)
            outputs.insert(0, x)
            x = downsampling(x)
        x = self.layers[-1](x)
        # print("encoder output:", x.size())
        outputs.insert(0, x)
        return outputs


class UNetEncoder2(MyronenkoEncoder):
    def __init__(self, n_features, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 feature_dilation=2, downsampling_stride=2, dropout=0.2, layer_widths=None, kernel_size=3):
        super(MyronenkoEncoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 4]
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
        self.kqv = nn.ModuleList()
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
            if i == 0:
                self.kqv.append(nn.Sequential())
            else:
                self.kqv.append(kqv_unit(in_width, out_width))
            self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                     dropout=layer_dropout, kernel_size=kernel_size))
            if i != len(layer_blocks) - 1:
                self.downsampling_convolutions.append(conv3x3x3(out_width, out_width, stride=downsampling_stride,
                                                                kernel_size=kernel_size))
            print("Encoder {}:".format(i), in_width, out_width)
            in_width = out_width

    def forward(self, x, q=None):
        k = None
        outputs = list()
        for i, (layer, downsampling, kqv) in enumerate(zip(self.layers[:-1], self.downsampling_convolutions, self.kqv[:-1])):
            x = layer(x)
            if i > 0:
                x = kqv(k, q[i], x)
            outputs.insert(0, x)
            x = downsampling(x)
            k = x
        x = self.layers[-1](x)
        # print("encoder output:", x.size())
        outputs.insert(0, x)
        return outputs


class kqv_unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(kqv_unit, self).__init__()

        self.alpha = conv1x1x1(in_channels, out_channels, 1)
        self.beta = conv1x1x1(in_channels * 4, out_channels, 1)
        self.gamma = conv1x1x1(out_channels, out_channels, 1)

    def forward(self, k, q, v):
        # print(k.size(), q.size(), v.size())
        k = self.alpha(k)
        q = self.beta(q)
        # print(k.size(), q.size())
        kq = torch.sigmoid(k + q)
        return v + self.gamma(v) * kq


class UNetDecoder1(MirroredDecoder):
    def calculate_layer_widths(self, depth):
        in_width, out_width = super().calculate_layer_widths(depth=depth)
        if depth != len(self.layer_blocks) - 1:
            in_width *= 2
        print("Decoder {}:".format(depth), in_width, out_width)
        return in_width, out_width

    def forward(self, inputs, pre_inputs=None):
        x = inputs[0]
        outputs = list()
        # print("x:", x.size())
        for i, (pre, up, lay) in enumerate(zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1])):
            # print("!!!", x.size())
            x = lay(x)
            outputs.insert(0, x)
            # print("???", x.size())
            # print("lay(x):", x.size())
            x = pre(x)
            x = up(x)
            # print("up(x):", x.size())
            x = torch.cat((x, inputs[i + 1]), 1)
            # print("cat(x):", x.size())
        x = self.layers[-1](x)
        outputs.insert(0, x)
        return x, outputs


class UNetDecoder2(MirroredDecoder):
    def calculate_layer_widths(self, depth):
        in_width, out_width = super().calculate_layer_widths(depth=depth)
        if depth != len(self.layer_blocks) - 1:
            in_width *= 3
        print("Decoder {}:".format(depth), in_width, out_width)
        return in_width, out_width

    def forward(self, inputs, pre_inputs):
        # print("111")
        x = inputs[0]
        # print("x:", x.size())
        for i, (pre, up, lay) in enumerate(zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1])):
            x = lay(x)
            # print("lay(x):", x.size())
            x = pre(x)
            x = up(x)
            # print("up(x):", x.size())
            # print(x.size(), inputs[i + 1].size(), pre_inputs[i + 1].size())
            x = torch.cat((x, inputs[i + 1], pre_inputs[i + 1]), 1)
            # print("cat(x):", x.size())
        x = self.layers[-1](x)
        return x


class Cascade_kqv_UNet(CascadeKQVConvolutionalAutoEncoder):
    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetDecoder1, n_outputs=1, is_training, is_dsv, is_half, **kwargs):
        super().__init__(*args, encoder_class=encoder_class, encoder_class2=UNetEncoder2, decoder_class=decoder_class, decoder_class2=UNetDecoder2,
                         n_outputs=n_outputs, is_training=is_training, **kwargs)
        # self.set_final_convolution(n_outputs=n_outputs)


class AutoImplantUNet(Cascade_kqv_UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y = super(AutoImplantUNet, self).forward(x)
        return y - x

    def test(self, x):
        return super(AutoImplantUNet, self).forward(x)

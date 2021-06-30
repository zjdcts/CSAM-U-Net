import torch
import torch.nn as nn
from ..classification.myronenko import MyronenkoEncoder, MyronenkoLayer, MyronenkoResidualBlock
from ..classification.decoder import MirroredDecoder
from ..autoencoder.variational import CascadeKQVConvolutionalAutoEncoder, CascadeConvolutionalAutoEncoder
from ..classification.resnet import conv3x3x3, conv1x1x1


class UNetEncoder(MyronenkoEncoder):
    def __init__(self, n_features, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 feature_dilation=2, downsampling_stride=2, dropout=0.2, layer_widths=None, kernel_size=3):
        super(MyronenkoEncoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 4]
        self.layers = nn.ModuleList()
        self.spatial_feature = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
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
                self.spatial_feature.append(SpatialFeatureAttention(out_width))
                self.downsampling_convolutions.append(conv3x3x3(out_width, out_width, stride=downsampling_stride,
                                                                kernel_size=kernel_size))
            print("Encoder {}:".format(i), in_width, out_width)
            in_width = out_width

    def forward(self, x, q=None):
        outputs = list()
        for layer, downsampling, spAtt in zip(self.layers[:-1], self.downsampling_convolutions, self.spatial_feature):
            x = layer(x)
            x = spAtt(x)
            outputs.insert(0, x)
            x = downsampling(x)
        x = self.layers[-1](x)
        # print("encoder output:", x.size())
        outputs.insert(0, x)
        return outputs


class UNetDecoder1(MirroredDecoder):
    def calculate_layer_widths(self, depth):
        in_width, out_width = super().calculate_layer_widths(depth=depth)
        if depth != len(self.layer_blocks) - 1:
            in_width *= 2
        print("Decoder {}:".format(depth), in_width, out_width)
        return in_width, out_width

    def forward(self, inputs, pre_inputs=None):
        x = inputs[0]
        # print("x:", x.size())
        for i, (pre, up, lay) in enumerate(zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers[:-1])):
            # print("!!!", x.size())
            x = lay(x)
            # print("???", x.size())
            # print("lay(x):", x.size())
            x = pre(x)
            x = up(x)
            # print("up(x):", x.size())
            x = torch.cat((x, inputs[i + 1]), 1)
            # print("cat(x):", x.size())
        x = self.layers[-1](x)
        return x


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


class SpatialFeatureAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialFeatureAttention, self).__init__()
        assert in_channels % 2 == 0
        self.chAtt1 = conv1x1x1(in_channels, in_channels // 2, 1)
        self.chAtt2 = conv1x1x1(in_channels, in_channels // 2, 1)
        self.chOut = conv1x1x1(in_channels // 2, in_channels, 1)
        self.spAtt1 = conv3x3x3(in_channels, in_channels, stride=2, padding=0, kernel_size=2)
        self.spAtt2 = conv3x3x3(in_channels, in_channels, stride=2, padding=0, kernel_size=2)
        self.spOut = nn.ConvTranspose3d(in_channels, in_channels, stride=2, kernel_size=2)

    def forward(self, x):
        y = x
        chAtt1 = self.chAtt1(x)
        chAtt2 = self.chAtt2(x)
        chOut = self.chOut(torch.relu(chAtt1 + chAtt2))
        chOut = torch.sigmoid(chOut)
        spIn = y * chOut
        spAtt1 = self.spAtt1(spIn)
        spAtt2 = self.spAtt2(spIn)
        spOut = self.spOut(torch.relu(spAtt1 + spAtt2))
        spOut = torch.sigmoid(spOut)
        spOut = spIn * spOut
        return y + spOut



class SpatialFeatureCascadeUnet(CascadeConvolutionalAutoEncoder):
    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetDecoder1, n_outputs=1, is_training, is_dsv,
                 is_half, **kwargs):
        super().__init__(*args, encoder_class=encoder_class, decoder_class=decoder_class, decoder_class2=UNetDecoder2,
                         n_outputs=n_outputs, is_training=is_training, **kwargs)
        # self.set_final_convolution(n_outputs=n_outputs)


class AutoImplantUNet(SpatialFeatureCascadeUnet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y = super(AutoImplantUNet, self).forward(x)
        return y - x

    def test(self, x):
        return super(AutoImplantUNet, self).forward(x)

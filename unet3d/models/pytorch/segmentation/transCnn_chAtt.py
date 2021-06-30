import torch
from torch import nn

from ..classification.resnet import conv3x3x3, conv1x1x1
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
        self.ch_sp_att = nn.ModuleList()
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
            self.ch_sp_att.append(ChSpAtt(out_width, 2 ** (self.len - i - 1)))
            if i != len(layer_blocks) - 1:
                self.downsampling_convolutions.append(conv3x3x3(out_width, out_width, stride=downsampling_stride,
                                                                kernel_size=kernel_size))
            print("Encoder {}:".format(i), in_width, out_width)
            in_width = out_width

    def forward(self, x, y=None):
        outputs = list()
        for layer, downsampling, chsp in zip(self.layers[:-1], self.downsampling_convolutions, self.ch_sp_att[:-1]):
            x = layer(x)
            x = chsp(x)
            outputs.insert(0, x)
            x = downsampling(x)
        x = self.layers[-1](x)
        x = self.ch_sp_att[-1](x)
        # print("encoder output:", x.size())
        outputs.insert(0, x)
        return outputs


class UNetDecoder(MirroredDecoder):
    def calculate_layer_widths(self, depth):
        in_width, out_width = super().calculate_layer_widths(depth=depth)
        # if depth != len(self.layer_blocks) - 1:
        #     in_width *= 2
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


class ChAtt(nn.Module):
    def __init__(self, in_channels):
        super(ChAtt, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool3d(1)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv3d(2 * in_channels, in_channels, 1)
        # self.ch_att_layer = nn.TransformerEncoderLayer(in_channels, 8, in_channels // 2)
        self.ch_att = nn.TransformerEncoderLayer(in_channels, 8, in_channels // 4)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        N, C = x.shape[0], x.shape[1]
        maxpool = self.maxpool(x)
        avgpool = self.avgpool(x)
        conv_out = self.conv(torch.cat([maxpool, avgpool], dim=1))
        ch_att_out = self.ch_att(conv_out.reshape(N, C, -1).permute(0, 2, 1).contiguous())
        return ch_att_out.permute(0, 2, 1).contiguous().reshape(N, C, 1, 1, -1)


class SpAtt(nn.Module):
    def __init__(self, in_channels, scale):
        super(SpAtt, self).__init__()
        self.conv = nn.Conv3d(in_channels, in_channels, (7, 9, 6), (7, 9, 6))
        self.h_att = nn.TransformerEncoderLayer(in_channels * scale, 8, in_channels * scale // 4)
        self.w_att = nn.TransformerEncoderLayer(in_channels * scale, 8, in_channels * scale // 4)
        self.d_att = nn.TransformerEncoderLayer(in_channels * scale, 8, in_channels * scale // 4)
        self.rev_conv = nn.ConvTranspose3d(in_channels, in_channels, (7, 9, 6), (7, 9, 6))

    def forward(self, x):
        x = self.conv(x)
        N, C, H, W, D = x.shape
        x = x.reshape(N, -1, W, D).permute(0, 2, 3, 1).contiguous().reshape(N, W * D, -1)
        x = self.h_att(x)
        x = x.reshape(N, W, D, H, C).permute(0, 3, 2, 1, 4).contiguous().reshape(N, H * D, -1)
        x = self.h_att(x)
        x = x.reshape(N, H, D, W, C).permute(0, 1, 3, 2, 4).contiguous().reshape(N, H * W, -1)
        x = self.h_att(x)
        x = x.reshape(N, H, W, D, C).permute(0, 4, 1, 2, 3).contiguous()
        x = self.rev_conv(x)
        return x


class ChSpAtt(nn.Module):
    def __init__(self, in_channels, scale):
        super(ChSpAtt, self).__init__()
        self.ch_att = ChAtt(in_channels)
        self.sp_att = SpAtt(in_channels, scale)

    def forward(self, x):
        identity = x
        x = self.ch_att(x)
        identity = identity + identity * x
        return identity + self.sp_att(identity)


class TransSpChAtt(ConvolutionalAutoEncoder):
    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetDecoder, n_outputs=1, is_training, is_dsv,
                 is_half, **kwargs):
        super().__init__(*args, encoder_class=encoder_class, decoder_class=decoder_class, n_outputs=n_outputs, **kwargs)
        self.set_final_convolution(n_outputs=n_outputs)


class AutoImplantUNet(TransSpChAtt):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y = super(AutoImplantUNet, self).forward(x)
        return y - x

    def test(self, x):
        return super(AutoImplantUNet, self).forward(x)

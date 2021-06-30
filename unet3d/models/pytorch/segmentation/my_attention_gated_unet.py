import torch
import torch.nn as nn

from torch.nn import functional as F
from functools import partial
from ..classification.myronenko import MyronenkoEncoder, MyronenkoLayer, MyronenkoResidualBlock
from ..classification.decoder import MirroredDecoder
from ..autoencoder.variational import AttentionGatedConvolutionalAutoEncoder, AttentionCascadeConvolutionalAutoEncoder
from ..classification import resnet


class UNetEncoder(MyronenkoEncoder):
    def forward(self, x):
        # print(x.size())
        outputs = list()
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            x = layer(x)
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
        self.filters = []
        self.filters.append(base_width)
        for i in range(1, len(self.layer_blocks) + 1):
            self.filters.append(self.filters[i - 1] * 2)
        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        if use_transposed_convolutions:
            self.upsampling_blocks = nn.ModuleList()
        else:
            self.upsampling_blocks = list()
        self.base_width = base_width
        self.feature_reduction_scale = feature_reduction_scale
        self.layer_widths = layer_widths
        in_width = self.filters[len(self.layer_blocks)]
        self.gating = UnetGridGatingSignal3(in_width, in_width, kernel_size=(1, 1, 1), is_batchnorm=True)
        # self.attentionblocks = nn.ModuleList()
        for i, n_blocks in enumerate(self.layer_blocks):
            depth = len(self.layer_blocks) - i
            in_width, out_width = self.calculate_layer_widths(depth)
            # if i != len(self.layer_blocks) - 1:
            #     # print(out_width, in_width, '!')
            #     self.attentionblocks.append(
            #         MultiAttentionBlock(in_size=out_width, gate_size=in_width, inter_size=out_width,
            #                             nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2),
            #                             align_corners=align_corners)
            #     )

            if self.use_transposed_convolutions:
                self.pre_upsampling_blocks.append(nn.Sequential())
                # print(kernel_size, upsampling_scale)
                self.upsampling_blocks.append(nn.ConvTranspose3d(in_width, out_width, kernel_size=upsampling_scale,
                                                                 stride=upsampling_scale))
            else:
                self.pre_upsampling_blocks.append(resnet.conv1x1x1(in_width, out_width, stride=1))
                self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                                      mode=upsampling_mode, align_corners=align_corners))

            self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                     kernel_size=kernel_size))

    def calculate_layer_widths(self, depth):
        in_width, out_width = super().calculate_layer_widths(depth=depth)
        # if depth != len(self.layer_blocks) - 1:
        #     in_width *= 2
        print("Decoder {}:".format(depth), in_width, out_width)
        return in_width, out_width

    def forward(self, inputs):
        outputs = []
        x = inputs[0]
        # print(x.size())
        # x = self.gating(x)
        # print("x:", x.size())
        gating = self.gating(x)
        for i, (pre, up, lay) in enumerate(zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers)):
            # print("lay(x):", x.size())
            # if i == 0:
            #     # print(inputs[i + 1].size(), gating.size())
            #     g_conv, atti = att(inputs[i + 1], gating)
            # else:
            #     # print(inputs[i + 1].size(), x.size())
            #     g_conv, atti = att(inputs[i + 1], x)
            x = pre(x)
            x = up(x)
            # print("up(x):", x.size())
            # print(x.size(), g_conv.size())
            x = torch.cat((x, inputs[i + 1]), 1)
            x = lay(x)
            outputs.append(x)
            # print("cat(x):", x.size())
        # x = self.pre_upsampling_blocks[-1](x)
        # x = self.upsampling_blocks[-1](x)
        # x = torch.cat((x, inputs[-1]), 1)
        # x = self.layers[-1](x)
        # outputs.append(x)
        return outputs


class UNetAttentionDecoder(MirroredDecoder):
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
        self.filters = []
        self.filters.append(base_width)
        for i in range(1, len(self.layer_blocks) + 1):
            self.filters.append(self.filters[i - 1] * 2)
        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        if use_transposed_convolutions:
            self.upsampling_blocks = nn.ModuleList()
        else:
            self.upsampling_blocks = list()
        self.base_width = base_width
        self.feature_reduction_scale = feature_reduction_scale
        self.layer_widths = layer_widths
        in_width = self.filters[len(self.layer_blocks)]
        self.gating = UnetGridGatingSignal3(in_width, in_width, kernel_size=(1, 1, 1),
                                            is_batchnorm=True)
        self.attentionblocks = nn.ModuleList()
        for i, n_blocks in enumerate(self.layer_blocks):
            depth = len(self.layer_blocks) - i
            in_width, out_width = self.calculate_layer_widths(depth)
            if i != len(self.layer_blocks) - 1:
                # print(out_width, in_width, '!')
                self.attentionblocks.append(
                    MultiAttentionBlock(in_size=out_width, gate_size=in_width, inter_size=out_width,
                                        nonlocal_mode='concatenation', sub_sample_factor=(2, 2, 2),
                                        align_corners=align_corners)
                )

            if self.use_transposed_convolutions:
                self.pre_upsampling_blocks.append(nn.Sequential())
                # print(kernel_size, upsampling_scale)
                self.upsampling_blocks.append(nn.ConvTranspose3d(in_width, out_width, kernel_size=upsampling_scale,
                                                                 stride=upsampling_scale))
            else:
                self.pre_upsampling_blocks.append(resnet.conv1x1x1(in_width, out_width, stride=1))
                self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                                      mode=upsampling_mode, align_corners=align_corners))

            self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                     kernel_size=kernel_size))

    def calculate_layer_widths(self, depth):
        in_width, out_width = super().calculate_layer_widths(depth=depth)
        # if depth != len(self.layer_blocks) - 1:
        #     in_width *= 2
        print("Decoder {}:".format(depth), in_width, out_width)
        return in_width, out_width

    def forward(self, inputs):
        outputs = []
        x = inputs[0]
        # print(x.size())
        # x = self.gating(x)
        # print("x:", x.size())
        gating = self.gating(x)
        for i, (pre, up, lay, att) in enumerate(zip(self.pre_upsampling_blocks[:-1], self.upsampling_blocks[:-1],
                                                    self.layers[:-1], self.attentionblocks)):
            # print("lay(x):", x.size())
            x = pre(x)
            x = up(x)
            # print(inputs[i + 1].size(), x.size())
            g_conv, atti = att(inputs[i + 1], x)
            # print("up(x):", x.size())
            # print(x.size(), g_conv.size())
            x = torch.cat((x, g_conv), 1)
            x = lay(x)
            outputs.append(x)
            # print("cat(x):", x.size())
        x = self.pre_upsampling_blocks[-1](x)
        x = self.upsampling_blocks[-1](x)
        x = torch.cat((x, inputs[-1]), 1)
        x = self.layers[-1](x)
        outputs.append(x)
        return outputs


class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1, 1, 1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.BatchNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class MultiAttentionBlock(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor, align_corners=False):
        super(MultiAttentionBlock, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor, align_corners=align_corners)
        self.gate_block_2 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor, align_corners=align_corners)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size * 2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True)
                                           )

        # initialise the blocks
        # for m in self.children():
        #     if m.__class__.__name__.find('GridAttentionBlock3D') != -1: continue
        #     init_weights(m, init_type='kaiming')

    def forward(self, input, gating_signal):
        gate_1, attention_1 = self.gate_block_1(input, gating_signal)
        gate_2, attention_2 = self.gate_block_2(input, gating_signal)

        return self.combine_gates(torch.cat([gate_1, gate_2], 1)), torch.cat([attention_1, attention_2], 1)


class _GridAttentionBlockND(nn.Module):
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3, mode='concatenation',
                 sub_sample_factor=(2, 2, 2), align_corners=False):
        super(_GridAttentionBlockND, self).__init__()

        assert dimension in [2, 3]
        assert mode in ['concatenation', 'concatenation_debug', 'concatenation_residual']

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor
        self.align_corners = align_corners

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            bn = nn.BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # Output transform
        self.W = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0,
                             bias=False)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

        # Initialise weights
        # for m in self.children():
        #     init_weights(m, init_type='kaiming')

        # Define the operation
        if mode == 'concatenation':
            self.operation_function = self._concatenation
        elif mode == 'concatenation_debug':
            self.operation_function = self._concatenation_debug
        elif mode == 'concatenation_residual':
            self.operation_function = self._concatenation_residual
        else:
            raise NotImplementedError('Unknown operation function.')

    def forward(self, x, g):
        '''
        :param x: (b, c, t, h, w)
        :param g: (b, g_d)
        :return:
        '''

        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode,
                              align_corners=self.align_corners)

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)

        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode,
                                   align_corners=self.align_corners)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_debug(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.softplus(theta_x + phi_g)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

    def _concatenation_residual(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = F.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = F.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        f = self.psi(f).view(batch_size, 1, -1)
        sigm_psi_f = F.softmax(f, dim=2).view(batch_size, 1, *theta_x.size()[2:])

        # upsample the attentions and multiply
        sigm_psi_f = F.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f


class GridAttentionBlock3D(_GridAttentionBlockND):
    def __init__(self, in_channels, gating_channels, inter_channels=None, mode='concatenation',
                 sub_sample_factor=(2, 2, 2), align_corners=False):
        super(GridAttentionBlock3D, self).__init__(in_channels,
                                                   inter_channels=inter_channels,
                                                   gating_channels=gating_channels,
                                                   dimension=3, mode=mode,
                                                   sub_sample_factor=sub_sample_factor,
                                                   align_corners=align_corners
                                                   )


class My_Attention_Gated_UNet(AttentionGatedConvolutionalAutoEncoder):
    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetAttentionDecoder, n_outputs=1,
                 is_training, is_dsv, is_half, **kwargs):
        print("OK!")
        super().__init__(*args, encoder_class=encoder_class, decoder_class=decoder_class, n_outputs=n_outputs, **kwargs)
        # self.set_final_convolution(n_outputs=n_outputs)


class My_Attention_Cascade_Gated_UNet(AttentionCascadeConvolutionalAutoEncoder):
    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetAttentionDecoder, decoder_class2=UNetDecoder,
                 n_outputs=1, is_training, is_dsv, is_half, **kwargs):
        super().__init__(*args, encoder_class=encoder_class, decoder_class=decoder_class, decoder_class2=decoder_class2,
                         n_outputs=n_outputs, is_training=is_training, is_dsv=is_dsv, is_half=is_half, **kwargs)


class AutoImplantUNet(My_Attention_Gated_UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y = super(AutoImplantUNet, self).forward(x)
        return y - x

    def test(self, x):
        return super(AutoImplantUNet, self).forward(x)

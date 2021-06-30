import torch
from ..classification.myronenko import MyronenkoEncoder
from ..classification.decoder import MirroredDecoder
from ..autoencoder.variational import CascadeConvolutionalAutoEncoder


class UNetEncoder(MyronenkoEncoder):
    def forward(self, x, y=None):
        outputs = list()
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            x = layer(x)
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
            x = lay(x)
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


class Cascade_UNet(CascadeConvolutionalAutoEncoder):
    def __init__(self, *args, encoder_class=UNetEncoder, decoder_class=UNetDecoder1, n_outputs=1, is_training, is_dsv, is_half, **kwargs):
        super().__init__(*args, encoder_class=encoder_class, decoder_class=decoder_class, decoder_class2=decoder_class,
                         n_outputs=n_outputs, is_training=is_training, **kwargs)
        # self.set_final_convolution(n_outputs=n_outputs)


class AutoImplantUNet(Cascade_UNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        y = super(AutoImplantUNet, self).forward(x)
        return y - x

    def test(self, x):
        return super(AutoImplantUNet, self).forward(x)
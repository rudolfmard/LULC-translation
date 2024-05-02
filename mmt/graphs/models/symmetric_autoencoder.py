import torch.nn as nn
import torch
import torch.utils.checkpoint as checkpoint

from mmt.graphs.models.custom_layers import down_block
from mmt.graphs.models.custom_layers import up_block
from mmt.graphs.models.custom_layers import double_conv

Down = down_block.Down
Up = up_block.Up
DoubleConv = double_conv.DoubleConv


class AtrouMMU(nn.Module):
    def __init__(self, inf, scale_factor=10, bias=False):
        super(AtrouMMU, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                inf, inf, kernel_size=3, padding=1, stride=scale_factor, bias=bias
            ),
            nn.GroupNorm(1, inf),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                inf,
                inf,
                kernel_size=3,
                padding=6,
                dilation=6,
                stride=scale_factor,
                bias=bias,
            ),
            nn.GroupNorm(1, inf),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                inf,
                inf,
                kernel_size=3,
                padding=12,
                dilation=12,
                stride=scale_factor,
                bias=bias,
            ),
            nn.GroupNorm(1, inf),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                inf,
                inf,
                kernel_size=3,
                padding=18,
                dilation=18,
                stride=scale_factor,
                bias=bias,
            ),
            nn.GroupNorm(1, inf),
            nn.ReLU(),
        )
        self.avg = nn.AvgPool2d(scale_factor)
        self.fin = nn.Conv2d(inf * 5, inf, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.avg(x)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        return self.fin(x)

class Upsample(nn.Module):
    def __init__(self, scale_factor=2, mode="nearest"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x, scale_factor=self.scale_factor, mode=self.mode
        )

class SimpleDecoder(nn.Module):
    """
    This class represents the tail of ResNet. It performs a global pooling and maps the output to the
    correct class by using a fully connected layer.
    """

    def __init__(
        self,
        in_features,
        n_classes,
        depth=1,
        num_groups=4,
        nf=52,
        resize=None,
        atrou=True,
        bias=False,
    ):
        super().__init__()
        # self.pre=torch.nn.PixelShuffle(6)
        if num_groups is None:
            num_groups = nf

        self.decoder = torch.nn.Sequential()
        inc = in_features
        if resize is not None:
            if atrou:
                self.decoder.add_module(
                    "pool", AtrouMMU(inc, scale_factor=resize, bias=bias)
                )
            else:
                self.decoder.add_module("pool", nn.MaxPool2d(resize))
        for i in range(1, depth):
            self.decoder.add_module(
                "conv_{}".format(i),
                nn.Conv2d(inc, nf, kernel_size=3, padding=1, bias=bias),
            )
            self.decoder.add_module(
                "groupnorm_{}".format(i), nn.GroupNorm(num_groups, nf)
            )
            self.decoder.add_module("relu_{}".format(i), nn.ReLU(inplace=True))
            inc = nf
        self.decoder.add_module("conv_{}".format(depth), nn.Conv2d(inc, n_classes, 1))

    def forward(self, x):
        x = self.decoder(x)
        return x

class DUNet(nn.Module):
    def __init__(
        self,
        input_channels,
        number_feature_map=32,
        embedding_dim=32,
        mode="light",
        num_groups=None,
        up_mode="bilinear",
        memory_monger=False,
        resize=None,
        pooling_factors=[2, 2, 2, 2, 3],
        tlm_p=0,
        bias=False,
    ):
        down_mode = "maxpool"
        super().__init__()
        self.embedding_dim = embedding_dim

        if resize is not None:
            self.inc = nn.Sequential(
                Upsample(scale_factor=resize, mode="nearest"),
                DoubleConv(
                    input_channels, number_feature_map, num_groups=num_groups, bias=bias
                ),
            )
        else:
            self.inc = DoubleConv(
                input_channels, number_feature_map, num_groups=num_groups, bias=bias
            )

        self.down1 = Down(
            number_feature_map,
            number_feature_map,
            mode=down_mode,
            num_groups=num_groups,
            factor=pooling_factors[0],
            bias=bias,
        )
        self.down2 = Down(
            number_feature_map,
            number_feature_map,
            mode=down_mode,
            num_groups=num_groups,
            factor=pooling_factors[1],
            bias=bias,
        )
        self.down3 = Down(
            number_feature_map,
            number_feature_map,
            mode=down_mode,
            num_groups=num_groups,
            factor=pooling_factors[2],
            bias=bias,
        )
        self.down4 = Down(
            number_feature_map,
            number_feature_map,
            mode=down_mode,
            num_groups=num_groups,
            factor=pooling_factors[3],
            bias=bias,
        )
        self.up1 = Up(
            number_feature_map * 2,
            number_feature_map,
            up_mode,
            num_groups=num_groups,
            factor=pooling_factors[-1],
            bias=bias,
        )
        self.up2 = Up(
            number_feature_map * 2,
            number_feature_map,
            up_mode,
            num_groups=num_groups,
            factor=pooling_factors[-2],
            bias=bias,
        )
        self.up3 = Up(
            number_feature_map * 2,
            number_feature_map,
            up_mode,
            num_groups=num_groups,
            factor=pooling_factors[-3],
            bias=bias,
        )
        self.up4 = Up(
            number_feature_map * 2,
            number_feature_map,
            up_mode,
            num_groups=num_groups,
            factor=pooling_factors[-4],
            bias=bias,
        )
        self.outc = DoubleConv(number_feature_map, embedding_dim, num_groups=num_groups, bias=bias)
        #self.outc = nn.Conv2d(number_feature_map, embedding_dim, kernel_size=1)

    def encoder_part(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

    def decoder_part(self, x1, x2, x3, x4, x5):
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder_part(x)
        return self.decoder_part(x1, x2, x3, x4, x5)

class SymmetricAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        cat=False,
        mul=False,
        softpos=False,
        mode="light",
        number_feature_map=None,
        n_channels_hiddenlay=30,
        embedding_dim=None,
        n_channels_embedding=32,
        memory_monger=False,
        num_groups=None,
        up_mode="bilinear",
        decoder_depth=1,
        resize=None,
        pooling_factors=[3, 3, 3, 3, 3],
        encoder=DUNet,  # TODO: switch
        decoder=SimpleDecoder,  # TODO: switch
        decoder_atrou=True,
        tlm_p=0,
        bias=False,
        image_operator="mul",
        n_px_input = None, # Not used. Added for API compatibility
        n_px_embedding = None, # Not used. Added for API compatibility
    ):
        super().__init__()
        if number_feature_map is not None:
            n_channels_hiddenlay = number_feature_map
            print(f"<{self.__class__.__name__}>mmt-0.2 API changes: 'number_feature_map' is now renamed 'n_channels_hiddenlay'. Please use it for now on. Current value: n_channels_hiddenlay={n_channels_hiddenlay}")
        
        if embedding_dim is not None:
            n_channels_embedding = embedding_dim
            print(f"<{self.__class__.__name__}>mmt-0.2 API changes: 'embedding_dim' is now renamed 'n_channels_hiddenlay'. Please use it for now on. Current value: n_channels_embedding={n_channels_embedding}")
        
        if not isinstance(resize, list):
            enc_resize = resize
            dec_resize = resize
        else:
            enc_resize = resize[0]
            dec_resize = resize[1]
        
        self.encoder = encoder(
            in_channels,
            number_feature_map=n_channels_hiddenlay,
            embedding_dim=n_channels_embedding,
            mode=mode,
            num_groups=num_groups,
            up_mode=up_mode,
            memory_monger=memory_monger,
            resize=enc_resize,
            pooling_factors=pooling_factors,
            tlm_p=tlm_p,
            bias=bias,
        )

        in_dec = self.encoder.embedding_dim
        self.cat = cat
        self.mul = mul
        self.softpos = softpos
        if cat:
            in_dec = in_dec * 2
        
        self.decoder = decoder(
            in_dec,
            out_channels,
            depth=decoder_depth,
            num_groups=num_groups,
            nf=n_channels_hiddenlay,
            resize=dec_resize,
            atrou=decoder_atrou,
            bias=bias,
        )

        self.forward_method = self.classical_forward
        self.n_classes = out_channels
        self.image_mul = False
        if image_operator == "mul":
            self.image_mul = True

    def classical_forward(self, x, full=False, res=None, image=None):
        if full:
            x = self.encoder(x)
        if res is not None:
            if self.softpos:
                res = torch.softmax(res, 1)
            if self.cat:
                x = torch.cat((x, res), 1)
            elif self.mul:
                x *= res
            else:
                x += res
        if image is not None:
            if self.image_mul:
                x *= image
            else:
                x += image
        return x, self.decoder(x)

    def forward(self, x, full=False, res=None, image=None):
        return self.forward_method(x, full, res, image)

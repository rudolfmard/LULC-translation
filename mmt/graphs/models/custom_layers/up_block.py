import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch import tensor, cat

from mmt.graphs.models.custom_layers import double_conv

DoubleConv = double_conv.DoubleConv


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self,
        in_channels,
        out_channels,
        mode="transposeconv",
        factor=2,
        num_groups=None,
        size=None,
        bias=False,
    ):
        super().__init__()
        num_groups_conv_trans = num_groups

        if mode == "bilinear":
            if size is None:
                self.up = nn.Upsample(
                    scale_factor=factor, mode="bilinear", align_corners=True
                )
            else:
                self.up = nn.Upsample(size=size, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(
                in_channels, out_channels, num_groups=num_groups, bias=bias
            )
        elif mode == "transposeconv":
            if num_groups is None:
                num_groups_conv_trans = in_channels // 4
            # self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=factor, stride=factor)
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 4 * 2,
                    in_channels // 4,
                    kernel_size=factor,
                    stride=factor,
                    bias=bias,
                ),
                nn.GroupNorm(num_groups_conv_trans, in_channels // 4),
                nn.ReLU(inplace=True),
            )
            self.conv = DoubleConv(
                in_channels * 3 // 4, out_channels, num_groups=num_groups, bias=bias
            )
        elif mode == "transposeconv_leg":
            if num_groups is None:
                num_groups_conv_trans = in_channels // 2
            self.up = nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels // 2,
                    in_channels // 2,
                    kernel_size=factor,
                    stride=factor,
                    bias=bias,
                ),
                nn.GroupNorm(num_groups_conv_trans, in_channels // 2),
                nn.ReLU(inplace=True),
            )
            self.conv = DoubleConv(
                in_channels, out_channels, num_groups=num_groups, bias=bias
            )
        else:
            raise ValueError("Unknown upsizing mode : " + str(mode))

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        #print("---------- up block ----------")
        #print(f"Input x1 shape: {x1.shape}")
        #print(f"Input x2 shape: {x2.shape}")

        diffY = tensor([x2.size()[2] - x1.size()[2]])
        diffX = tensor([x2.size()[3] - x1.size()[3]])

        #print(f"diffY (x2.shape[2]-x1.shape[2]): {diffY}")
        #print(f"diffY (x2.shape[3]-x1.shape[3]): {diffX}")

        x1 = pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        #print(f"pad x1 with [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2] = {[diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2]}")
        #print("------------------------------")

        x = cat([x2, x1], dim=1)
        return self.conv(x)

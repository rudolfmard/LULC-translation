import torch
import torch.nn as nn

from mmt.graphs.models import attention_autoencoder, universal_embedding, position_encoding
from mmt.graphs.models.custom_layers import double_conv, down_block, up_block

Down = down_block.Down
Up = up_block.Up
DoubleConv = double_conv.DoubleConv

class DownscaleWithSinusoidalEmbedding(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()

        self.downscaling_block = nn.Sequential(
            # TODO: CHECK in_channels!!!
            nn.Conv2d(in_channels=13, out_channels=64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=3),
            nn.ReLU()
        )

        self.coordinate_encoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, coordinates):
        x = self.downscaling_block(x)
        x = x + self.coordinate_encoder(coordinates).unsqueeze(-1).unsqueeze(-1) # Broadcast the coordinate encoding vector along spatial dimensions
        return x

class EsawcToEsgp(nn.Module):
    # Accept the same arguments as EncDec modules
    def __init__(self):
        super(EsawcToEsgp, self).__init__()
 
        # Input layer: Resize esawc patch to the resolution of ecosgp and embed coordinate encoding
        self.inc = DownscaleWithSinusoidalEmbedding()

        # U-net body of the network
        self.down1 = Down(
            128,
            256,
            mode="maxpool",
            bias=True,
        )
        self.down2 = Down(
            256,
            512,
            mode="maxpool",
            bias=True,
        )
        self.up1 = Up(
            512+256,
            256,
            mode="bilinear",
            bias=True,
        )
        self.up2 = Up(
            256+128,
            128,
            mode="bilinear",
            bias=True,
        )

        # Output layer: change the number of channels to match number of labels in ecosgp
        # TODO: CHECK out_channels!!!
        self.outc = nn.Sequential(
            DoubleConv(128, 128),
            DoubleConv(128, 64),
            nn.Conv2d(64, 35, kernel_size=1),
        )

    def forward(self, x, coordinates):
        x1 = self.inc(x, coordinates)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        return self.outc(x)
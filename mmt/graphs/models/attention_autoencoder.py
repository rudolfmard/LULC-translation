#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module with elements to build attention-augmented Unet-like auto-encoders
"""

import numpy as np
import torch
import math
from torch import nn
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
from mmt.graphs.models.custom_layers import down_block

Down = down_block.Down

def prime_factorization(n):
    """Return the prime factorization of `n`.

    Parameters
    ----------
    n : int
        The number for which the prime factorization should be computed.

    Returns
    -------
    dict[int, int]
        List of tuples containing the prime factors and multiplicities of `n`.
    """
    prime_factors = {}

    i = 2
    while i**2 <= n:
        if n % i:
            i += 1
        else:
            n = int(n / i)
            try:
                prime_factors[i] += 1
            except KeyError:
                prime_factors[i] = 1

    if n > 1:
        try:
            prime_factors[n] += 1
        except KeyError:
            prime_factors[n] = 1

    return prime_factors


# BLOCKS
# ========


class UpConv(nn.Module):
    """ConvTranspose2d + BatchNorm2d + ReLU

    Will inflate the size of the image by the factor `resize`

    Example
    -------
    >>> up = UpConv(in_channels = 12, out_channels=17, resize = 3)
    >>> x = torch.rand(16, 12, 60, 60)
    >>> up(x).shape
    (16, 17, 180, 180)
    """

    def __init__(self, in_channels, out_channels, resize=1):
        super().__init__()
        if resize is None:
            resize = 1

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=resize, stride=resize, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class DownConv(nn.Module):
    """Conv2d + BatchNorm2d + ReLU

    Will reduce the size of the image by the factor `resize`

    Example
    -------
    >>> down = DownConv(in_channels = 12, out_channels=17, resize = 3)
    >>> x = torch.rand(16, 12, 60, 60)
    >>> down(x).shape
    (16, 17, 20, 20)
    """

    def __init__(self, in_channels, out_channels, resize=1):
        super().__init__()
        if resize is None:
            resize = 1

        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=resize, stride=resize, bias=False
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

class UpSC(nn.Module):
    """Upscaling then DoubleConv with skip connection
    
    
    Parameters
    ----------
    in_channels: int
        Number of channels in the input data. Must be the sum of the channels in `x1` and `x2`
        
    out_channels: int
        Number of channels in the output data
    
    out_size: int
        Number of pixels in the input data
    
    
    
    Examples
    --------
    >>> up = Up3(in_channels = 24, out_channels = 20, out_size = 100)
    >>> x1 = torch.rand(16, 12, 60, 60)
    >>> x2 = torch.rand(16, 12, 100, 100)
    >>> up(x1, x2).shape
    torch.Size([16, 20, 100, 100])
    """
    
    def __init__(self, in_channels, out_channels, out_size, interp_mode = "bilinear"):
        super().__init__()
        self.up = nn.Upsample(size = out_size, mode = interp_mode)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ConvInterp(nn.Module):
    """Conv + interpolation.
    
    
    Examples
    --------
    >>> ci = ConvInterp(in_channels = 12, out_channels = 20, out_size = 100)
    >>> x = torch.rand(16, 12, 60, 60)
    >>> ci(x).shape
    torch.Size([16, 20, 100, 100])
    """
    def __init__(self, in_channels, out_channels, out_size, interp_mode="bilinear"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias = False)
        self.interp = nn.Upsample(size = out_size, mode = interp_mode)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.interp(x)
        return x

class CrossResolutionAttention(nn.Module):
    """Will reduce the size of the image by the factor `resize`
    If resize < 1, the image size is inflated by the factor int(1.0/resize)

    Example
    -------
    >>> x = torch.rand(16, 12, 60, 60)
    >>> cra = CrossResolutionAttention(in_channels = 12, out_channels=17, resize = 3)
    >>> cra(x).shape
    (16, 17, 20, 20)

    >>> cra = CrossResolutionAttention(in_channels = 12, out_channels=17, resize = 1/3)
    >>> cra(x).shape
    (16, 17, 180, 180)
    """

    def __init__(self, in_channels, out_channels, qk_channels=10, resize=1):
        super().__init__()
        if resize is None:
            resize = 1

        if resize < 1:
            resize = int(1.0 / resize)
            self.query_conv = nn.ConvTranspose2d(
                in_channels, qk_channels, kernel_size=resize, stride=resize
            )
            self.key_conv = nn.ConvTranspose2d(
                in_channels, qk_channels, kernel_size=resize, stride=resize
            )
            self.value_conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=resize, stride=resize
            )
        else:
            resize = int(resize)
            self.query_conv = nn.Conv2d(
                in_channels, qk_channels, kernel_size=resize, stride=resize
            )
            self.key_conv = nn.Conv2d(
                in_channels, qk_channels, kernel_size=resize, stride=resize
            )
            self.value_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=resize, stride=resize
            )

    def scaled_dot_product_attention(self, Q, K, V):
        """Equivalent to the implementation of `torch.nn.functional.scaled_dot_product_attention`
        See: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html

        Solve ONNX export issue "Exporting the operator 'aten::scaled_dot_product_attention' to ONNX opset version 14 is not supported"
        See: https://github.com/pytorch/pytorch/issues/97262

        Last checked: 14 Sept. 2023
        """
        attn_weight = torch.softmax(
            (Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1
        )
        return attn_weight @ V

    def forward(self, x):
        q = self.query_conv(x).permute(0, 3, 2, 1).contiguous()
        k = self.key_conv(x).permute(0, 3, 2, 1).contiguous()
        v = self.value_conv(x).permute(0, 3, 2, 1).contiguous()
        # attention = nn.functional.scaled_dot_product_attention(q, k, v)
        attention = self.scaled_dot_product_attention(q, k, v)
        return attention.permute(0, 3, 2, 1)

    def check_shapes(self, x=None):
        """Display shapes of some tensors"""
        if x is None:
            x = torch.rand(10, 3, 600, 300)
            print(f"Random input: x = {x.shape}")
        else:
            print(f"Given input: x = {x.shape}")

        q = self.query_conv(x).permute(0, 3, 2, 1)
        print(f"q = {q.shape}")
        k = self.key_conv(x).permute(0, 3, 2, 1)
        print(f"k = {k.shape}")
        v = self.value_conv(x).permute(0, 3, 2, 1)
        print(f"v = {v.shape}")
        attention = nn.functional.scaled_dot_product_attention(q, k, v).permute(
            0, 3, 2, 1
        )
        print(f"attention = {attention.shape}")


class SelfAttention(nn.Module):
    """Self attention layer for images.


    Parameters
    ----------
    in_channels: int
        Number of channels in the input data

    size: int
        Number of pixels in the input data (only square images)

    n_heads: int
        Number of head in the multi-head attention layer


    Examples
    --------
    >>> sa = SelfAttention2(12, 60, 3)
    >>> x = torch.rand(16, 12, 60, 60)
    >>> sa(x).shape
    torch.Size([16, 12, 60, 60])
    """

    def __init__(self, in_channels, size, n_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.size = size
        self.n_heads = n_heads

        self.mha = nn.MultiheadAttention(in_channels, n_heads, batch_first=True)
        self.ln = nn.LayerNorm([size * size, in_channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([size * size, in_channels]),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),
        )

    def forward(self, x):
        batch_size, in_channels, size, size = x.shape
        x = x.view(batch_size, -1, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(batch_size, -1, size, size)


# ENCODER/DECODER
# =================


class AttentionUNet(nn.Module):
    """UNet with attention layer modifying the shape of the bottleneck (thus of the output)"""

    def __init__(self, in_channels, out_channels, h_channels=[32] * 3, resizes=[1] * 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h_channels = h_channels
        self.resizes = resizes

        self.inc = nn.Conv2d(in_channels, h_channels[0], kernel_size=1)
        self.down = torch.nn.Sequential()
        for i in range(1, len(h_channels)):
            self.down.add_module(
                f"down{i}",
                DownConv(h_channels[i - 1], h_channels[i], resize=resizes[i - 1]),
            )
        self.attn = CrossResolutionAttention(
            h_channels[-1], h_channels[-1], resize=resizes[-1]
        )
        self.up = torch.nn.Sequential()
        for i in range(1, len(h_channels)):
            self.up.add_module(
                f"up{i}",
                UpConv(h_channels[-i], h_channels[-i - 1], resize=resizes[-i - 1]),
            )
        self.outc = nn.Conv2d(h_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.inc(x)
        x = self.down(x)
        x = self.attn(x)
        x = self.up(x)
        logits = self.outc(x)
        return logits


class AttentionUNetSC(nn.Module):
    """UNet with attention layer and skip connections"""

    def __init__(self, in_channels, out_channels, h_channels=[32] * 3, resizes=[1] * 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h_channels = h_channels
        self.resizes = resizes

        self.inc = nn.Conv2d(in_channels, h_channels[0], kernel_size=1)
        self.down1 = DownConv(h_channels[0], h_channels[1], resize=resizes[0])
        self.down2 = DownConv(h_channels[1], h_channels[2], resize=resizes[1])
        self.attn = CrossResolutionAttention(
            h_channels[-1], h_channels[-1], resize=resizes[-1]
        )
        self.up1 = UpConv(2 * h_channels[2], h_channels[1], resize=resizes[1])
        self.up2 = UpConv(2 * h_channels[1], h_channels[0], resize=resizes[0])
        self.outc = nn.Conv2d(h_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.inc(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x = self.attn(x2)
        rsz = Resize(
            x.shape[2:], antialias=True, interpolation=InterpolationMode.NEAREST
        )
        x = self.up1(torch.cat([x, rsz(x2)], dim=1))
        rsz = Resize(
            x.shape[2:], antialias=True, interpolation=InterpolationMode.NEAREST
        )
        x = self.up2(torch.cat([x, rsz(x1)], dim=1))
        logits = self.outc(x)
        return logits


class ShortUNet(nn.Module):
    """Short UNet with resizing at the bottleneck.

    Will reduce the size of the image by the factor `resize`
    If resize < 1, the image size is inflated by the factor int(1.0/resize)
    """

    def __init__(self, in_channels, out_channels, h_channels=[32] * 3, resizes=[1] * 3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h_channels = h_channels
        self.resizes = resizes

        self.inc = nn.Conv2d(in_channels, h_channels[0], kernel_size=1)
        self.down1 = DownConv(h_channels[0], h_channels[1], resize=resizes[0])
        self.down2 = DownConv(h_channels[1], h_channels[2], resize=resizes[1])

        if resizes[-1] < 1:
            resize = int(1.0 / resizes[-1])
            self.resizeconv = UpConv(h_channels[-1], h_channels[-1], resize=resize)
        else:
            resize = int(resizes[-1])
            self.resizeconv = DownConv(h_channels[-1], h_channels[-1], resize=resize)

        self.up1 = UpConv(2 * h_channels[2], h_channels[1], resize=resizes[1])
        self.up2 = UpConv(2 * h_channels[1], h_channels[0], resize=resizes[0])
        self.outc = nn.Conv2d(h_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        x = self.inc(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x = self.resizeconv(x2)
        rsz = Resize(
            x.shape[2:], antialias=True, interpolation=InterpolationMode.NEAREST
        )
        x = self.up1(torch.cat([x, rsz(x2)], dim=1))
        rsz = Resize(
            x.shape[2:], antialias=True, interpolation=InterpolationMode.NEAREST
        )
        x = self.up2(torch.cat([x, rsz(x1)], dim=1))
        logits = self.outc(x)
        return logits


class DownAttnAttnUp(nn.Module):
    """DownConv + SelfAttention2 + CrossResolutionAttention + UpConv

    Parameters
    ----------
    in_channels: int
        Number of channels in the input data

    out_channels: int
        Number of channels in the output data

    in_size: int
        Number of pixels in the input data (only square images)

    out_size: int
        Number of pixels in the input data (only square images)

    h_channels: int
        Number of channels in the hidden layers


    Examples
    --------
    >>> encoder = DownAttnAttnUp(in_channels = 12, out_channels = 20, in_size = 60, out_size = 120)
    >>> x = torch.rand(16, 12, 60, 60)
    >>> encoder(x).shape
    torch.Size([16, 20, 120, 120])
    """

    def __init__(self, in_channels, out_channels, in_size, out_size, h_channels=32):
        super().__init__()
        print(
            f"Init {self.__class__.__name__} with in_channels={in_channels}, out_channels={out_channels}"
        )

        self.down = DownConv(
            in_channels=in_channels, out_channels=h_channels, resize=10
        )
        self.attn1 = SelfAttention(in_channels=h_channels, size=in_size // 10)
        self.attn2 = CrossResolutionAttention(
            in_channels=h_channels, out_channels=h_channels, resize=in_size / out_size
        )
        self.up = UpConv(in_channels=h_channels, out_channels=out_channels, resize=10)

    def forward(self, x):
        x = self.down(x)
        x = self.attn1(x)
        x = self.attn2(x)
        x = self.up(x)
        return x


class DownSelfattnInterpUp(nn.Module):
    """DownConv + SelfAttention + Upsample + UpConv


    Parameters
    ----------
    in_channels: int
        Number of channels in the input data

    out_channels: int
        Number of channels in the output data

    in_size: int
        Number of pixels in the input data (only square images)

    out_size: int
        Number of pixels in the input data (only square images)

    h_channels: int
        Number of channels in the hidden layers

    divide_size_by: int
        Factor to divise the spatial dimensions before applying attention layers

    n_heads: int
        Number of head in the multi-head attention layer


    Examples
    --------
    >>> encoder = DownSelfattnInterpUp(in_channels = 12, out_channels = 20, in_size = 60, out_size = 120)
    >>> x = torch.rand(16, 12, 60, 60)
    >>> encoder(x).shape
    torch.Size([16, 20, 120, 120])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        in_size,
        out_size,
        h_channels=32,
        interp_mode="bilinear",
        divide_size_by=10,
        n_heads=4,
    ):
        super().__init__()
        print(
            f"Init {self.__class__.__name__} with in_channels={in_channels}, out_channels={out_channels}"
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size
        self.interp_mode = interp_mode
        self.divide_size_by = divide_size_by

        self.down = DownConv(
            in_channels=in_channels, out_channels=h_channels, resize=divide_size_by
        )
        self.attn = SelfAttention(
            in_channels=h_channels, size=in_size // divide_size_by
        )
        self.interp = nn.Upsample(
            size=(out_size // divide_size_by, out_size // divide_size_by),
            mode=interp_mode,
        )
        self.up = UpConv(
            in_channels=h_channels, out_channels=out_channels, resize=divide_size_by
        )

    def forward(self, x):
        x = self.down(x)
        x = self.attn(x)
        x = self.interp(x)
        x = self.up(x)
        return x


class SelfattentionUnet(nn.Module):
    """DownConv + SelfAttention2 + Upsample + UpConv
    
    
    Parameters
    ----------
    in_channels: int
        Number of channels in the input data
        
    out_channels: int
        Number of channels in the output data
    
    in_size: int
        Number of pixels in the input data (only square images)
    
    out_size: int
        Number of pixels in the input data (only square images)
    
    h_channels: int
        Number of channels in the hidden layers
    
    interp_mode: str
        Type of interpolation used in `nn.Upsample` (mode argument)
        
    divide_size_by: int
        Factor to divise the spatial dimensions before applying attention layers
        
    n_heads: int
        Number of head in the multi-head attention layer. Must satisfy
        
    
    
    Examples
    --------
    >>> encoder = SelfattentionUnet(in_channels = 12, out_channels = 20, in_size = 60, out_size = 120)
    >>> x = torch.rand(16, 12, 60, 60)
    >>> encoder(x).shape
    torch.Size([16, 20, 120, 120])
    """
    def __init__(self, in_channels, out_channels, in_size, out_size, h_channels = 32, interp_mode = "bilinear", divide_size_by = 10, n_heads = 4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size
        self.interp_mode = interp_mode
        self.divide_size_by = divide_size_by
        
        self.interp = ConvInterp(in_channels = in_channels, out_channels = h_channels, out_size = out_size, interp_mode = interp_mode)
        self.down = Down(in_channels=h_channels, out_channels=h_channels, factor = divide_size_by)
        self.attn = SelfAttention(in_channels=h_channels, size = out_size//divide_size_by)
        self.up = UpSC(in_channels=2 * h_channels, out_channels=out_channels, out_size = out_size)
        
    def forward(self, x):
        x0 = self.interp(x)
        x1 = self.down(x0)
        x2 = self.attn(x1)
        x3 = self.up(x2, x0)
        return x3


# FULL NETWORKS
# ===============


class AttentionAutoEncoder(nn.Module):
    """Auto-encoder using AttentionUNet for both encoder and decoder"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n_channels_embedding=50,
        n_channels_hiddenlay=32,
        emb_size_ratio=60,
        resize=1,
    ):
        super().__init__()
        if resize is None:
            resize = 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels_embedding = n_channels_embedding
        self.emb_size_ratio = emb_size_ratio
        self.n_channels_hiddenlay = n_channels_hiddenlay
        self.resize = resize

        bottleneck_resize = int(emb_size_ratio / resize)
        pfactors = prime_factorization(bottleneck_resize)
        ae_resizes = []
        for p in pfactors.keys():
            for _ in range(pfactors[p]):
                ae_resizes.append(p)

        ae_resizes.sort()
        ae_resizes.reverse()
        self.encoder = AttentionUNet(
            in_channels=in_channels,
            out_channels=n_channels_embedding,
            h_channels=[n_channels_hiddenlay] * (len(ae_resizes) + 1),
            resizes=ae_resizes + [1 / resize],
        )
        self.decoder = AttentionUNet(
            in_channels=n_channels_embedding,
            out_channels=out_channels,
            h_channels=[n_channels_hiddenlay] * (len(ae_resizes) + 1),
            resizes=ae_resizes + [resize],
        )

    def forward(self, x, full=True, res=None):
        """Args `full` and `res` not used. Added for API compatibility"""
        emb = self.encoder(x)
        return emb, self.decoder(emb)


class AttentionAutoEncoderSC(nn.Module):
    """Auto-encoder using AttentionUNetSC for both encoder and decoder"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n_channels_embedding=50,
        n_channels_hiddenlay=32,
        emb_size_ratio=60,
        resize=1,
    ):
        super().__init__()
        if resize is None:
            resize = 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels_embedding = n_channels_embedding
        self.emb_size_ratio = emb_size_ratio
        self.n_channels_hiddenlay = n_channels_hiddenlay
        self.resize = resize

        bottleneck_resize = int(emb_size_ratio / resize)
        pfactors = prime_factorization(bottleneck_resize)
        ae_resizes = []
        for p in pfactors.keys():
            for _ in range(pfactors[p]):
                ae_resizes.append(p)

        ae_resizes.sort()
        ae_resizes.reverse()
        ae_resizes = ae_resizes + [1] * (3 - len(ae_resizes))
        self.encoder = AttentionUNetSC(
            in_channels=in_channels,
            out_channels=n_channels_embedding,
            h_channels=[n_channels_hiddenlay] * 3,
            resizes=[ae_resizes[0], ae_resizes[1], 1 / resize],
        )
        self.decoder = AttentionUNetSC(
            in_channels=n_channels_embedding,
            out_channels=out_channels,
            h_channels=[n_channels_hiddenlay] * 3,
            resizes=[ae_resizes[0], ae_resizes[1], resize],
        )

    def forward(self, x, full=True, res=None):
        """Args `full` and `res` not used. Added for API compatibility"""
        emb = self.encoder(x)
        return emb, self.decoder(emb)


class ShortUNetAutoEncoder(nn.Module):
    """Auto-encoder using ShortUNet for both encoder and decoder"""

    def __init__(
        self,
        in_channels,
        out_channels,
        n_channels_embedding=50,
        n_channels_hiddenlay=32,
        emb_size_ratio=60,
        resize=1,
    ):
        super().__init__()
        if resize is None:
            resize = 1

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels_embedding = n_channels_embedding
        self.emb_size_ratio = emb_size_ratio
        self.n_channels_hiddenlay = n_channels_hiddenlay
        self.resize = resize

        bottleneck_resize = int(emb_size_ratio / resize)
        pfactors = prime_factorization(bottleneck_resize)
        ae_resizes = []
        for p in pfactors.keys():
            for _ in range(pfactors[p]):
                ae_resizes.append(p)

        ae_resizes.sort()
        ae_resizes.reverse()
        ae_resizes = ae_resizes + [1] * (3 - len(ae_resizes))
        self.encoder = ShortUNet(
            in_channels=in_channels,
            out_channels=n_channels_embedding,
            h_channels=[n_channels_hiddenlay] * 3,
            resizes=[ae_resizes[0], ae_resizes[1], 1 / resize],
        )
        self.decoder = ShortUNet(
            in_channels=n_channels_embedding,
            out_channels=out_channels,
            h_channels=[n_channels_hiddenlay] * 3,
            resizes=[ae_resizes[0], ae_resizes[1], resize],
        )

    def forward(self, x, full=True, res=None):
        """Args `full` and `res` not used. Added for API compatibility"""
        emb = self.encoder(x)
        return emb, self.decoder(emb)


class SelfattentionAutoencoder(nn.Module):
    """Auto-encoder using DownSelfattnInterpUp for both encoder and decoder


    Parameters
    ----------
    in_channels: int
        Number of channels in the input data

    n_px_input: int
        Number of pixels in the input data (only square images)

    out_channels: int
        Number of channels in the output data (default is equal to the input data)

    n_channels_embedding: int
        Number of channels in the latent space

    h_channels: int
        Number of channels in the hidden layers

    divide_size_by: int
        Factor to divise the spatial dimensions before applying attention layers

    n_heads: int
        Number of head in the multi-head attention layer

    resize: None, UNUSED
        Unused, only added for API compatibility

    Examples
    --------
    >>> sa = SelfattentionAutoencoder(in_channels = 12, out_channels = 27, emb_channels = 20, in_size = 60, emb_size = 120)
    >>> x = torch.rand(16, 12, 60, 60)
    >>> emb, y = sa(x)
    >>> emb.shape
    torch.Size([16, 20, 120, 120])
    >>> y.shape
    torch.Size([16, 27, 60, 60])
    """
    
    def __init__(
        self,
        in_channels,
        n_px_input,
        n_px_embedding,
        out_channels=None,
        n_channels_embedding=50,
        n_channels_hiddenlay=32,
        interp_mode="bilinear",
        divide_size_by=10,
        n_heads=4,
        resize=None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = n_channels_embedding
        self.h_channels = n_channels_hiddenlay
        self.in_size = n_px_input
        self.emb_size = n_px_embedding

        self.encoder = DownSelfattnInterpUp(
            in_channels=in_channels,
            out_channels=n_channels_embedding,
            in_size=n_px_input,
            out_size=n_px_embedding,
            h_channels=n_channels_hiddenlay,
            interp_mode=interp_mode,
            divide_size_by=divide_size_by,
            n_heads=n_heads,
        )
        self.decoder = DownSelfattnInterpUp(
            in_channels=n_channels_embedding,
            out_channels=out_channels,
            in_size=n_px_embedding,
            out_size=n_px_input,
            h_channels=n_channels_hiddenlay,
            interp_mode=interp_mode,
            divide_size_by=divide_size_by,
            n_heads=n_heads,
        )

    def forward(self, x, full=True, res=None):
        """Args `full` and `res` not used. Added for API compatibility"""
        emb = self.encoder(x)
        return emb, self.decoder(emb)


class SelfattentionUnetAutoencoder(nn.Module):
    """Auto-encoder using DownSelfattnInterpUp for both encoder and decoder


    Parameters
    ----------
    in_channels: int
        Number of channels in the input data

    n_px_input: int
        Number of pixels in the input data (only square images)

    out_channels: int
        Number of channels in the output data (default is equal to the input data)

    n_channels_embedding: int
        Number of channels in the latent space

    h_channels: int
        Number of channels in the hidden layers

    divide_size_by: int
        Factor to divise the spatial dimensions before applying attention layers

    n_heads: int
        Number of head in the multi-head attention layer

    resize: None, UNUSED
        Unused, only added for API compatibility

    Examples
    --------
    >>> sa = SelfattentionAutoencoder(in_channels = 12, out_channels = 27, emb_channels = 20, in_size = 60, emb_size = 120)
    >>> x = torch.rand(16, 12, 60, 60)
    >>> emb, y = sa(x)
    >>> emb.shape
    torch.Size([16, 20, 120, 120])
    >>> y.shape
    torch.Size([16, 27, 60, 60])
    """
    
    def __init__(
        self,
        in_channels,
        n_px_input,
        n_px_embedding,
        out_channels=None,
        n_channels_embedding=50,
        n_channels_hiddenlay=32,
        interp_mode="bilinear",
        divide_size_by=10,
        n_heads=4,
        resize=None,
    ):
        super().__init__()

        if out_channels is None:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = n_channels_embedding
        self.h_channels = n_channels_hiddenlay
        self.in_size = n_px_input
        self.emb_size = n_px_embedding

        self.encoder = SelfattentionUnet(
            in_channels=in_channels,
            out_channels=n_channels_embedding,
            in_size=n_px_input,
            out_size=n_px_embedding,
            h_channels=n_channels_hiddenlay,
            interp_mode=interp_mode,
            divide_size_by=divide_size_by,
            n_heads=n_heads,
        )
        self.decoder = SelfattentionUnet(
            in_channels=n_channels_embedding,
            out_channels=out_channels,
            in_size=n_px_embedding,
            out_size=n_px_input,
            h_channels=n_channels_hiddenlay,
            interp_mode=interp_mode,
            divide_size_by=divide_size_by,
            n_heads=n_heads,
        )

    def forward(self, x, full=True, res=None):
        """Args `full` and `res` not used. Added for API compatibility"""
        emb = self.encoder(x)
        return emb, self.decoder(emb)



# EOF

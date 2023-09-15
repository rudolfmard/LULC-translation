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
            n = int(n/i)
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
    def __init__(self, in_channels, out_channels, resize = 1):
        super().__init__()
        if resize is None:
            resize = 1
        
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size = resize, stride = resize, bias = False)
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
    def __init__(self, in_channels, out_channels, resize = 1):
        super().__init__()
        if resize is None:
            resize = 1
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = resize, stride = resize, bias = False)
        self.norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
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
    def __init__(self, in_channels, out_channels, qk_channels=10, resize = 1):
        super().__init__()
        if resize is None:
            resize = 1
        
        if resize < 1:
            resize = int(1.0/resize)
            self.query_conv = nn.ConvTranspose2d(in_channels, qk_channels, kernel_size=resize, stride = resize)
            self.key_conv = nn.ConvTranspose2d(in_channels, qk_channels, kernel_size=resize, stride = resize)
            self.value_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=resize, stride = resize)
        else:
            resize = int(resize)
            self.query_conv = nn.Conv2d(in_channels, qk_channels, kernel_size=resize, stride = resize)
            self.key_conv = nn.Conv2d(in_channels, qk_channels, kernel_size=resize, stride = resize)
            self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=resize, stride = resize)
    
    def scaled_dot_product_attention(self, Q, K, V):
        """Equivalent to the implementation of `torch.nn.functional.scaled_dot_product_attention`
        See: https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        
        Solve ONNX export issue "Exporting the operator 'aten::scaled_dot_product_attention' to ONNX opset version 14 is not supported"
        See: https://github.com/pytorch/pytorch/issues/97262
        
        Last checked: 14 Sept. 2023
        """
        attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))), dim=-1)
        return attn_weight @ V
        
    def forward(self, x):
        q = self.query_conv(x).permute(0,3,2,1).contiguous()
        k = self.key_conv(x).permute(0,3,2,1).contiguous()
        v = self.value_conv(x).permute(0,3,2,1).contiguous()
        # attention = nn.functional.scaled_dot_product_attention(q, k, v)
        attention = self.scaled_dot_product_attention(q, k, v)
        return attention.permute(0,3,2,1)
    
    def check_shapes(self, x = None):
        """Display shapes of some tensors"""
        if x is None:
            x = torch.rand(10, 3, 600, 300)
            print(f"Random input: x = {x.shape}")
        else:
            print(f"Given input: x = {x.shape}")
        
        q = self.query_conv(x).permute(0,3,2,1)
        print(f"q = {q.shape}")
        k = self.key_conv(x).permute(0,3,2,1)
        print(f"k = {k.shape}")
        v = self.value_conv(x).permute(0,3,2,1)
        print(f"v = {v.shape}")
        attention = nn.functional.scaled_dot_product_attention(q, k, v).permute(0,3,2,1)
        print(f"attention = {attention.shape}")


class AttentionUNet(nn.Module):
    """UNet with attention layer modifying the shape of the bottleneck (thus of the output)
    """
    def __init__(self, in_channels, out_channels, h_channels = [32]*3, resizes = [1]*3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h_channels = h_channels
        self.resizes = resizes

        self.inc = nn.Conv2d(in_channels, h_channels[0], kernel_size=1)
        self.down = torch.nn.Sequential()
        for i in range(1, len(h_channels)):
            self.down.add_module(
                f"down{i}", DownConv(h_channels[i-1], h_channels[i], resize = resizes[i-1])
            )
        self.attn = CrossResolutionAttention(h_channels[-1], h_channels[-1], resize = resizes[-1])
        self.up = torch.nn.Sequential()
        for i in range(1, len(h_channels)):
            self.up.add_module(
                f"up{i}", UpConv(h_channels[-i], h_channels[-i-1], resize = resizes[-i-1])
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
    """UNet with attention layer and skip connections
    """
    def __init__(self, in_channels, out_channels, h_channels = [32]*3, resizes = [1]*3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.h_channels = h_channels
        self.resizes = resizes

        self.inc = nn.Conv2d(in_channels, h_channels[0], kernel_size=1)
        self.down1 = DownConv(h_channels[0], h_channels[1], resize = resizes[0])
        self.down2 = DownConv(h_channels[1], h_channels[2], resize = resizes[1])
        self.attn = CrossResolutionAttention(h_channels[-1], h_channels[-1], resize = resizes[-1])
        self.up1 = UpConv(2 * h_channels[2], h_channels[1], resize = resizes[1])
        self.up2 = UpConv(2 * h_channels[1], h_channels[0], resize = resizes[0])
        self.outc = nn.Conv2d(h_channels[0], out_channels, kernel_size=1)
    
    def forward(self, x):
        x = self.inc(x)
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x = self.attn(x2)
        rsz = Resize(
            x.shape[2:], antialias = True,interpolation=InterpolationMode.NEAREST
        )
        x = self.up1(torch.cat([x, rsz(x2)], dim=1))
        rsz = Resize(
            x.shape[2:], antialias = True,interpolation=InterpolationMode.NEAREST
        )
        x = self.up2(torch.cat([x, rsz(x1)], dim=1))
        logits = self.outc(x)
        return logits


class AttentionAutoEncoder(nn.Module):
    """Auto-encoder using AttentionUNet for both encoder and decoder
    """
    def __init__(self, in_channels, out_channels, emb_channels = 50, h_channels = 32, emb_size_ratio = 60, resize = 1):
        super().__init__()
        if resize is None:
            resize = 1
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.emb_size_ratio = emb_size_ratio
        self.h_channels = h_channels
        self.resize = resize
        
        bottleneck_resize = int(emb_size_ratio/resize)
        pfactors = prime_factorization(bottleneck_resize)
        ae_resizes = []
        for p in pfactors.keys():
            for _ in range(pfactors[p]):
                ae_resizes.append(p)
        
        ae_resizes.sort()
        ae_resizes.reverse()
        self.encoder = AttentionUNet(
            in_channels = in_channels,
            out_channels = emb_channels,
            h_channels = [h_channels]*(len(ae_resizes) + 1),
            resizes = ae_resizes + [1/resize],
        )
        self.decoder = AttentionUNet(
            in_channels = emb_channels,
            out_channels = out_channels,
            h_channels = [h_channels]*(len(ae_resizes) + 1),
            resizes = ae_resizes + [resize],
        )
        
    def forward(self, x, full=True, res=None):
        """Args `full` and `res` not used. Added for API compatibility"""
        emb = self.encoder(x)
        return emb, self.decoder(emb)

class AttentionAutoEncoderSC(nn.Module):
    """Auto-encoder using AttentionUNetSC for both encoder and decoder
    """
    def __init__(self, in_channels, out_channels, emb_channels = 50, h_channels = 32, emb_size_ratio = 60, resize = 1):
        super().__init__()
        if resize is None:
            resize = 1
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.emb_size_ratio = emb_size_ratio
        self.h_channels = h_channels
        self.resize = resize
        
        bottleneck_resize = int(emb_size_ratio/resize)
        pfactors = prime_factorization(bottleneck_resize)
        ae_resizes = []
        for p in pfactors.keys():
            for _ in range(pfactors[p]):
                ae_resizes.append(p)
        
        ae_resizes.sort()
        ae_resizes.reverse()
        ae_resizes = ae_resizes + [1]*(3-len(ae_resizes))
        self.encoder = AttentionUNetSC(
            in_channels = in_channels,
            out_channels = emb_channels,
            h_channels = [h_channels]*3,
            resizes = [ae_resizes[0], ae_resizes[1], 1/resize],
        )
        self.decoder = AttentionUNetSC(
            in_channels = emb_channels,
            out_channels = out_channels,
            h_channels = [h_channels]*3,
            resizes = [ae_resizes[0], ae_resizes[1], resize],
        )
        
    def forward(self, x, full=True, res=None):
        """Args `full` and `res` not used. Added for API compatibility"""
        emb = self.encoder(x)
        return emb, self.decoder(emb)


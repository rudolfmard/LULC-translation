#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module with embedding mixers
"""

from torch import nn

class LinearCombination(nn.Module):
    """Perform a linear combination between the channels of the embeddings"""
    def __init__(self, n_channels_embedding = 50):
        super().__init__()
        self.emb_mixer = nn.Conv2d(2*n_channels_embedding, n_channels_embedding, kernel_size = (1,1))
    
    def forward(self, x):
        return self.emb_mixer(x)

class MLP(nn.Module):
    """Perform a linear combination between the channels of the embeddings"""
    def __init__(self, n_channels_embedding = 50, h_channels = 64):
        super().__init__()
        self.emb_mixer = nn.Sequential(
            nn.Conv2d(2*n_channels_embedding, h_channels, kernel_size = (1,1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(h_channels, h_channels, kernel_size = (1,1)),
            nn.ReLU(inplace = True),
            nn.Conv2d(h_channels, n_channels_embedding, kernel_size = (1,1)),
        )
    
    def forward(self, x):
        return self.emb_mixer(x)

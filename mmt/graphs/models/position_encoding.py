#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module with position encoding models
"""

from torch import nn

class PositionEncoder(nn.Module):
    """Position encoder as described in Baudoux et al. (2022)"""
    def __init__(self, d = 128, h_channels = 300, n_channels_embedding = 50):
        
        super().__init__()
        self.d = d
        self.n_channels_embedding = n_channels_embedding
        self.pos_encoder = nn.Sequential(
            nn.Linear(d, h_channels),
            nn.ReLU(inplace=True),
            nn.Linear(h_channels, n_channels_embedding),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.pos_encoder(x)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Test transforms
"""

import numpy as np
import torch
from mmt.datasets import transforms

x = np.array(
    [
        [0,0,1,1,1],
        [0,0,2,2,2],
        [3,2,2,0,2],
        [3,3,3,2,2],
        [4,0,3,3,3],
    ]
)
X = torch.tensor(x)
print("BEFORE:", X)


fmws = transforms.FillMissingWithSea(sea_label = 10)
print("AFTER", fmws.__class__.__name__, fmws({"mask":X}))


X = torch.tensor(x)
fmwn = transforms.FillMissingWithNeighbors(neighboring_size = 1, key="image")
print("AFTER", fmwn.__class__.__name__, fmwn({"image":X}))


X = torch.tensor(x)
st = transforms.ScoreTransform(divide_by = 10)
print("AFTER", st.__class__.__name__, st(X))


x = np.array(
    [
        [0,0,80,80,80],
        [0,0,90,90,90],
        [100,90,90,0,90],
        [100,100,100,90,90],
        [95,0,100,100,100],
    ]
)
X = torch.tensor(x)
esat = transforms.EsawcTransform()
print("AFTER", esat.__class__.__name__, esat(X))

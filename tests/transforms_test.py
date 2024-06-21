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

X_expected = np.array(
    [
        [10,10,1,1,1],
        [10,10,2,2,2],
        [3,2,2,10,2],
        [3,3,3,2,2],
        [4,10,3,3,3],
    ]
)
assert np.abs(fmws(X).numpy() - X_expected).sum() == 0, f"Error with {fmws.__class__.__name__}"



X = torch.tensor(x)
fmwn = transforms.FillMissingWithNeighbors(neighboring_size = 1, key="image")
print("AFTER", fmwn.__class__.__name__, fmwn({"image":X}))

X_expected = np.array(
    [
        [0,1,1,1,1],
        [1,1,2,2,2],
        [3,2,2,2,2],
        [3,3,3,2,2],
        [4,3,3,3,3],
    ]
)
assert np.abs(fmwn(torch.tensor(x)).numpy() - X_expected).sum() == 0, f"Error with {fmwn.__class__.__name__}"



X = torch.tensor(x)
st = transforms.ScoreTransform(divide_by = 10)
print("AFTER", st.__class__.__name__, st(X))
X_expected = np.array(
    [
        [0.0000, 0.0000, 0.1000, 0.1000, 0.1000],
        [0.0000, 0.0000, 0.2000, 0.2000, 0.2000],
        [0.3000, 0.2000, 0.2000, 0.0000, 0.2000],
        [0.3000, 0.3000, 0.3000, 0.2000, 0.2000],
        [0.4000, 0.0000, 0.3000, 0.3000, 0.3000]
    ], dtype = np.float32
)
assert np.abs(st(X).numpy() - X_expected).sum() == 0, f"Error with {st.__class__.__name__}"



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
X_expected = np.array(
[   
        [ 8,  8,  8,  8,  8],
        [ 8,  8,  9,  9,  9],
        [11,  9,  9,  8,  9],
        [11, 11, 11,  9,  9],
        [10,  8, 11, 11, 11]
    ]
)
assert np.abs(esat(X).numpy() - X_expected).sum() == 0, f"Error with {esat.__class__.__name__}"

print("All transforms tested successfully")

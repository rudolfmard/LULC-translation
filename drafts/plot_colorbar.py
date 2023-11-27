#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Plot land cover colorbars
"""
import numpy as np
from mmt.datasets import transforms as mmt_transforms
from mmt.datasets import landcovers
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

esawc = landcovers.ESAWorldCover(transforms=mmt_transforms.EsawcTransform)
ecosg = landcovers.EcoclimapSG()
qflags = landcovers.QualityFlagsECOSGplus(transforms=mmt_transforms.FillMissingWithSea(0,6))

lc = landcovers.EcoclimapSG()
for lc in [esawc, ecosg, qflags]:
    cmap = ListedColormap(np.array(lc.cmap)/255)
    lbnames = lc.labels
    
    N = len(lbnames)
    m = np.zeros((N, N),dtype=np.uint8)
    for l in range(N):
        m[l,:] = l
    
    fig, ax = lc.plot({"mask":m})
    figpath = f"colorbar_{lc.__class__.__name__}.svg"
    fig.savefig(figpath)
    fig.show()
    print(f"Figure saved {figpath}")
    

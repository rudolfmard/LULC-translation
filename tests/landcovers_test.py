#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to test the access of data with landcover classes
"""

import os
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import domains
import matplotlib.pyplot as plt

qb = domains.dublin_city.to_tgbox()

for lc in [
    landcovers.EcoclimapSG(),
    landcovers.ESAWorldCover(transforms = mmt_transforms.EsawcTransform()),
    landcovers.EcoclimapSGplus(),
    landcovers.QualityFlagsECOSGplus(),
    landcovers.EcoclimapSGML(),
    landcovers.SpecialistLabelsECOSGplus(),
    landcovers.ScoreECOSGplus(transforms = mmt_transforms.ScoreTransform(divide_by=100)),
    landcovers.EcoclimapSGplusV2(),
]:
    x = lc[qb]
    shapes = {k:x[k].shape for k in ["mask", "image"] if k in x.keys()}
    print(f"Loaded from {lc.__class__.__name__}:\t {shapes}")
    lc.plot(x)
    plt.show(block=False)

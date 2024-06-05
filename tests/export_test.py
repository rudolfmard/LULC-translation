#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to test the exportation of data
"""

import os

from mmt.datasets import landcovers
from mmt.utils import domains

qb = domains.dublin_city.to_tgbox()
os.makedirs("tmp", exist_ok=True)

for lc in [
    landcovers.EcoclimapSGplus(),
    landcovers.ESAWorldCover(),
    landcovers.EcoclimapSG(),
]:
    x = lc[qb]
    print(f"Loaded {x['mask'].shape} from {lc.__class__.__name__}")

    for fmt in ["dir", "tif", "npy", "nc"]:
        newfile = lc.export(x, os.path.join("tmp", f"test.{fmt}"))
        print(f"   Written: {newfile}")

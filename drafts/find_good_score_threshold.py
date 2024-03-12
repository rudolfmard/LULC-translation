#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Qualitative evaluation of map translation. Pre-set patches.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse
from skimage.filters import threshold_otsu

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import transforms
from mmt.datasets import landcovers
from mmt.utils import domains


# Argument parsing
# ----------------
parser = argparse.ArgumentParser(prog=__name__)
parser.add_argument("--domainname", help="Geographical domain name", default="eurat")
parser.add_argument("--threshold", help="Threshold value for the ECOSG+ score", type=float, default=0)
parser.add_argument("--res", help="Resolution of the map (degree)", default=None)
args = parser.parse_args()

domainname = args.domainname
res = args.res

kwargs = {"score_min": args.threshold}
if res:
    res = float(res)
    kwargs.update(res = res)


esgpv2 = landcovers.EcoclimapSGplusV2(**kwargs)
print(f"Loaded: {esgpv2.__class__.__name__} with crs={esgpv2.crs}, res={esgpv2.res}")

qdomain = getattr(domains, domainname)
qb = qdomain.to_tgbox(esgpv2.crs)

x = esgpv2[qb]

print(f"Data loaded: shape=", x["image"].shape)
labels = x["mask"].numpy()
score = x["image"].numpy()

sea = labels == 1
zeros = score == 0

n_bins = 100
smin = threshold_otsu(score[~zeros].ravel(), nbins=n_bins)
print(f"Threshold estimated by Otsu method: {smin}")

plt.figure()
counts, _, _ = plt.hist(score[sea].ravel(), n_bins, alpha = 0.5, label = "Sea")
plt.hist(score[~sea].ravel(), n_bins, alpha = 0.5, label = "Land")
plt.plot([smin]*2, [0, max(counts)], "r--", label = "Otsu's threshold")
plt.xlabel("Score values")
plt.ylabel("# pixels")
plt.grid()
plt.legend()
plt.show(block=False)


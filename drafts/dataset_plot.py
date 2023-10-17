#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Have look at maps and export it
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from mmt.datasets import landcovers
from mmt.datasets import landcover_to_landcover

ds = landcover_to_landcover.EEEmapsDataset(path = "/data/trieutord/MLULC/new-large-domain-hdf5/", mode = "val")
idx = 97
lcname_to_lcclass = {
    "esawc":landcovers.ESAWorldCover(tgeo_init = False),
    "ecosg":landcovers.EcoclimapSG(tgeo_init = False),
    "esgp":landcovers.EcoclimapSGplus(tgeo_init = False),
    "coordinate":landcovers.OpenStreetMap(details=14),
}

x = ds[idx]

fig, axs = plt.subplots(2, 2, figsize=(12,12))
for k, lcname in enumerate(x.keys()):
    i = k // 2
    j = k % 2
    lc = lcname_to_lcclass[lcname]
    if lcname == "coordinate":
        fig, axs[i,j] = lc.plot(x, figax = (fig, axs[i,j]), rowcolidx = 224)
    else:
        lc.plot({"mask":x[lcname]}, figax = (fig, axs[i,j]))

lon, lat = x["coordinate"]
fig.suptitle(f"Item {idx}. Lat={lat} Lon={lon}")
fig.tight_layout()
fig.show()

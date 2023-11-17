#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Qualitative evaluation of map translation. Pre-set patches.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from mmt.datasets import transforms as mmt_transforms
from mmt.datasets import landcovers
from mmt.utils import domains


# Configs
#---------
print(f"Executing program {sys.argv[0]} in {os.getcwd()}")

storeImages = False
fmtImages = ".svg"
figureDir = ""

n_px_esawc = 1000
val_domains = ["snaefell_glacier", "nanterre", "iso_kihdinluoto", "bakar_bay_croatia", "portugese_crops", "elmenia_algeria"]


# Land cover loading
#--------------------
esawc = landcovers.ESAWorldCover(transforms=mmt_transforms.EsawcTransform)
ecosg = landcovers.EcoclimapSG()
esgp = landcovers.EcoclimapSGplus()
esgml = landcovers.EcoclimapSGML()
qflags = landcovers.QualityFlagsECOSGplus(transforms=mmt_transforms.FillMissingWithSea(0,6))
print(f"Landcovers loaded with native CRS and resolution")

# Inference
#----------------
fig, axs = plt.subplots(len(val_domains), 5, figsize = (12,16))
for i, domainname in enumerate(val_domains):
    dom = getattr(domains, domainname)
    if n_px_esawc is None:
        qb = dom.to_tgbox()
    else:
        qb = dom.centred_fixed_size(n_px_esawc, esawc.res).to_tgbox()
    
    x_esawc = esawc[qb]
    x_ecosg = ecosg[qb]
    x_esgp = esgp[qb]
    x_esgml = esgml[qb]
    x_qflags = qflags[qb]
    print(f"Domain {domainname}: {dom.central_point()}")
    # print("   " + " ".join([f"{x}.shape = {eval(x).get('mask').shape}," for x in ["x_esawc", "x_ecosg", "x_esgp", "x_esgml"]]))
    
    esawc.plot(x_esawc, figax = (fig, axs[i, 0]), show_titles=False, show_colorbar=False)
    ecosg.plot(x_ecosg, figax = (fig, axs[i, 1]), show_titles=False, show_colorbar=False)
    esgp.plot(x_esgp, figax = (fig, axs[i, 2]), show_titles=False, show_colorbar=False)
    esgml.plot(x_esgml, figax = (fig, axs[i, 3]), show_titles=False, show_colorbar=False)
    qflags.plot(x_qflags, figax = (fig, axs[i, 4]), show_titles=False, show_colorbar=False)
    
    
[ax.axis("off") for ax in axs.ravel()]
cols = ["ESAWC", "ECOSG", "ECOSG+", "ECOSG-ML", "QFLAGS"]
for ax, col in zip(axs[0], cols):
    ax.set_title(col)

v_esgml = esgml.get_version()
title = f"Qualitative check (ECOSG-ML version {v_esgml})"
figname = f"qualcheck_esgml{v_esgml}"
fig.suptitle(title)
fig.tight_layout()
if storeImages:
    figpath = os.path.join(figureDir, figname + fmtImages)
    fig.savefig(figpath)
    print("Figure saved:", figpath)

plt.show(block=False)

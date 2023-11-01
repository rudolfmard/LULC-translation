#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Qualitative evaluation of map translation. Pre-set patches.
"""

import os
import sys
import h5py
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pprint import pprint
import torchvision.transforms as tvt

from mmt.inference import io
from mmt.inference import translators
from mmt import _repopath_ as mmt_repopath
from mmt.datasets import transforms as mmt_transforms
from mmt.datasets import landcovers
from mmt.utils import config as utilconf
from mmt.utils import domains


# Configs
#---------
usegpu = True
device = "cuda" if usegpu else "cpu"


print(f"Executing program {sys.argv[0]} in {os.getcwd()}")

xp_name = "vanilla_eurat3"
val_domains = ["snaefell_glacier", "dublin_city", "iso_kihdinluoto", "elmenia_algeria"]


# Land cover loading
#--------------------
esgp = landcovers.EcoclimapSGplus()
ecosg = landcovers.EcoclimapSG()
esawc = landcovers.ESAWorldCover(transforms=mmt_transforms.EsawcTransform)
print(f"Landcovers loaded with native CRS and resolution")

# Loading models
#----------------
checkpoint_path = os.path.join(mmt_repopath, "saved_models", "vanilla_eurat3.ckpt")
print(f"Loading auto-encoders from {checkpoint_path}")
translator1 = translators.EsawcToEsgp(checkpoint_path = checkpoint_path)
translator2 = translators.EsawcEcosgToEsgpRFC(checkpoint_path = checkpoint_path)

# Inference
#----------------
fig, axs = plt.subplots(len(val_domains), 5, figsize = (12,16))
for i, domainname in enumerate(val_domains):
    dom = getattr(domains, domainname)
    
    qb = dom.centred_fixed_size(600, esawc.res).to_tgbox()
    x_esgp = esgp[qb]
    x_ecosg = ecosg[qb]
    x_esawc = esawc[qb]
    print(f"Domain {domainname}: x_ecosg.shape = {x_ecosg['mask'].shape}, x_esgp.shape = {x_esgp['mask'].shape}")
    
    esawc.plot(x_esawc, figax = (fig, axs[i, 0]), show_titles=False, show_colorbar=False)
    esgp.plot(x_esgp, figax = (fig, axs[i, 1]), show_titles=False, show_colorbar=False)
    ecosg.plot(x_ecosg, figax = (fig, axs[i, 2]), show_titles=False, show_colorbar=False)
    
    y1 = translator1(qb)
    y2 = translator2(qb)
    
    esgp.plot({"mask":y1}, figax = (fig, axs[i, -2]), show_titles=False, show_colorbar=False)
    esgp.plot({"mask":y2}, figax = (fig, axs[i, -1]), show_titles=False, show_colorbar=False)
    
    
[ax.axis("off") for ax in axs.ravel()]
cols = ["ESAWC", "ECOSG+", "ECOSG", translator1.__class__.__name__, translator2.__class__.__name__]
for ax, col in zip(axs[0], cols):
    ax.set_title(col)

title = f"From xp {xp_name}"
plt.suptitle(title)
plt.tight_layout()
plt.show(block=False)

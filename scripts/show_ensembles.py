#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Generation of land cover members on few patches
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import transforms as mmt_transforms
from mmt.datasets import landcovers
from mmt.inference import translators
from mmt.utils import domains


# Configs
#---------
print(f"Executing program {sys.argv[0]} in {os.getcwd()}")

storeImages = False
fmtImages = ".svg"
figureDir = ""

n_px_esawc = 900
val_domains = ["snaefell_glacier", "nanterre", "iso_kihdinluoto", "portugese_crops", "elmenia_algeria"]
checkpoint_path=os.path.join(mmt_repopath, "data", "saved_models", "mmt-weights-v1.0.ckpt")

# ### Random draw
# n_members = 4
# u_values = torch.rand(n_members)
### Fixed draw
u_values = torch.tensor([0.62, 0.29, 0.41, 0.78, 0.09])
n_members = len(u_values)


# Land cover loading
#--------------------
esgp = landcovers.EcoclimapSGplus(transforms=mmt_transforms.OneHot(35))
esgml = landcovers.EcoclimapSGML()
qflags = landcovers.QualityFlagsECOSGplus(transforms=mmt_transforms.FillMissingWithSea(0,6))
enslc = landcovers.InferenceResultsProba(path = "", tgeo_init = False)
esawc = landcovers.ESAWorldCover(transforms=mmt_transforms.EsawcTransform, res = esgp.res / 6)
esawc.res = esgp.res / 6
print(f"Landcovers loaded with native CRS and resolution")
    
translator = translators.EsawcToEsgpProba(checkpoint_path=checkpoint_path, always_predict = False)
merger = translators.MapMergerProba("", merge_criterion = "qflag2_nodata", output_dtype="float32", tgeo_init = False)

# Inference
#----------------
fig, axs = plt.subplots(len(val_domains), n_members + 1, figsize = (12,16))
for i, domainname in enumerate(val_domains):
    dom = getattr(domains, domainname)
    if n_px_esawc is None:
        qb = dom.to_tgbox()
    else:
        qb = dom.centred_fixed_size(n_px_esawc, esawc.res).to_tgbox()
    
    x_esawc = esawc[qb]
    x_esgp = esgp[qb]
    x_esgml = esgml[qb]
    x_qflags = qflags[qb]
    print(f"Domain {domainname}: {dom.central_point()}")
    print("   " + " ".join([f"{x}.shape = {eval(x).get('mask').shape}," for x in ["x_esawc", "x_esgp", "x_esgml"]]))
    
    esgml.plot(x_esgml, figax = (fig, axs[i, 0]), show_titles=False, show_colorbar=False)
    
    y = translator.predict_from_data(x_esawc["mask"])
    proba = merger.predict_from_data(torch.tensor(y), x_qflags["mask"].squeeze(), x_esgp["mask"].squeeze())
    
    for mb in range(n_members):
        x_ens = enslc.generate_member({"image":torch.tensor(proba)}, u = u_values[mb])
        esgml.plot({"mask":x_ens}, figax = (fig, axs[i, mb + 1]), show_titles=False, show_colorbar=False)
    
    
[ax.axis("off") for ax in axs.ravel()]
cols = ["ECOSG-ML"] + [f"Member {mb + 1} (u={np.round(u_values[mb].item(),2)})" for mb in range(n_members)]
for ax, col in zip(axs[0], cols):
    ax.set_title(col)

v_esgml = esgml.get_version()
title = f"Ensemble generation sampling inference probabilities"
figname = f"ensemble{n_members}_esgml{v_esgml}"
fig.suptitle(title)
fig.tight_layout()
if storeImages:
    figpath = os.path.join(figureDir, figname + fmtImages)
    fig.savefig(figpath)
    print("Figure saved:", figpath)

plt.show(block=False)

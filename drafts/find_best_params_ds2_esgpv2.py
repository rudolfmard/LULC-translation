#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

As per https://github.com/ThomasRieutord/MT-MLULC, the data is expected to be stored as follow:
```
data
 ├── outputs        -> where the inference output will be stored
 |
 ├── saved_models   -> where the model checkpoints are stored.
 |
 ├── tiff_data      -> where the original land cover maps are stored in TIF format
 |   ├── ECOCLIMAP-SG
 |   ├── ECOCLIMAP-SG-ML
 |   ├── ECOCLIMAP-SG-plus
 |   └── ESA-WorldCover-2021
 |
 └── hdf5_data      -> where the training data is stored
     ├── ecosg.hdf5
     ├── ecosg-train.hdf5
     ├── ecosg-test.hdf5
     ├── ecosg-val.hdf5
     ├── esawc.hdf5
     └── ...
```

This program will create the HDF5 files `{ecosg|esgp|esawc}-{train|test|val}.hdf5`.

It takes the data from the TIF files in `tiff_data` reproject it onto a common grid and create patches according to the given parameters.


Examples
--------
python prepare_hdf5_ds2.py --h5dir=test --npatches=100 --qscore=0.9
"""
import os
import sys
import argparse
from pprint import pprint

import h5py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import rasterio
import torch
import torchvision.transforms as tvt
import seaborn as sns
from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcover_to_landcover, landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import config as utilconf
from mmt.utils import domains, misc, plt_utils
from torchgeo import samplers
from torchgeo.datasets.utils import BoundingBox

def qscore_from_score(score, name = "patch_mean"):
    """Return the patch-averaged value of the score"""
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu().numpy()
    
    return eval(name)(score)

def no_missing(score):
    if (score==0).sum() > 0:
        return 0
    else:
        return 1

def patch_mean(score):
    return score.mean()

def above_threshold(score):
    return (score > 0.525).sum()/score.size

def show_esawc_score_hist(x_esawc, x_score):
    fig, axs = plt.subplots(1,3, figsize = (12,6))
    esawc.plot({"mask":x_esawc}, figax = (fig, axs[0]), show_titles=False, show_colorbar=False)
    score.plot({"image":x_score}, figax = (fig, axs[1]), show_titles=False, show_colorbar=False)
    axs[2].hist(x_score.numpy().ravel(), 50)
    fig.show()

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(prog="prepare_hdf5_datasets")
parser.add_argument("--npatches", help="Nb of patches to be created", default=200)
parser.add_argument("--qscore", help="Minimum quality score to reach for a patch to be selected", default=0.7)
parser.add_argument("--divscore", help="Minimum diversity score to reach for a patch to be selected", default=0.1)
parser.add_argument("--margin", help="Nb of extra pixels extracted in the lager patches", default=200)
parser.add_argument("--domainname", help="Domain on which the patches are randomly picked up", default="eurat")
parser.add_argument("--scorename", help="Domain on which the patches are randomly picked up", default="above_threshold")
args = parser.parse_args()

print(f"Executing program {sys.argv[0]} in {os.getcwd()} with {args}")
domainname = args.domainname
scorename = args.scorename
n_patches = int(args.npatches)
quality_threshold = float(args.qscore)
diversity_threshold = float(args.divscore)
margin = int(args.margin)

# Land cover loading
#--------------------
print(f"Loading landcovers with native CRS")
esawc = landcovers.ESAWorldCover(transforms=mmt_transforms.EsawcTransform())
esgp = landcovers.EcoclimapSGplusV2()
ecosg = landcovers.EcoclimapSG()
score = landcovers.ScoreECOSGplus(transforms=mmt_transforms.ScoreTransform(divide_by=100))

lc = esawc & esgp.maps

lcnames = ["esawc", "esgp", "ecosg"]
lcmaps = {"esawc":esawc, "esgp":esgp, "ecosg":ecosg}
n_pxs = {"esawc":600, "esgp":100, "ecosg":20}
n_px_esapatch = 600

# Sampler definition
#--------------------
subsets = ["train", "train2", "test", "val"]

n_px_max = 2 * n_px_esapatch + margin
sampler_length = 2 * n_patches
val_domain = getattr(domains, domainname).to_tgbox(esawc.crs)
sampler = samplers.RandomGeoSampler(lc, size=n_px_max, length = sampler_length, roi = val_domain)


# INIT MAIN LOOP
#================
i = 0   # Valid patches counter
j = 0   # All patches counter
selected_bboxes = []
missing = {k:0 for k in lcnames}
add_another_patch =True
sampler = iter(sampler)
qscores = []
divscores = []
patch_means = []
above_thresholds = []

print(f"Start sampling {n_patches} patches over {domainname}")

while add_another_patch:
    try:
        qb = next(sampler)
        j += 1
    except StopIteration:
        add_another_patch = False
        print(f"No more patch in sampler (size={sampler_length}). i={i}/{n_patches}, j={j}/{sampler_length}")
        break
    
    # Get the data from all maps
    #-----------------
    x_lc = lc[qb] # 'mask': (3, n_px_max, n_px_max) -> 3: esawc, splab, ecosg ; 'image': (1, n_px_max, n_px_max) -> score
    # x_labels = {
        # "esawc": x_lc["mask"][0],
        # "ecosg": x_lc["mask"][2],
        # "esgp": esgp.getitem_from_data(x_lc["image"], x_lc["mask"][1], x_lc["mask"][2])
    # }
    
    # Quality control
    #-----------------
    x_score = x_lc["image"].unsqueeze(0)
    x_esawc = x_lc["mask"][0].unsqueeze(0)
    
    tttv_score = misc.ccrop_and_split(x_score, n_px_esapatch)
    tttv_esawc = misc.ccrop_and_split(x_esawc, n_px_esapatch)
    
    qscore = [qscore_from_score(x, name = scorename) for x in tttv_score.values()]
    unmet_qscore = np.array(qscore) < quality_threshold
    divscore = [misc.divscore_from_esawc(x) for x in tttv_esawc.values()]
    unmet_divscore = np.array(divscore) < diversity_threshold 
    if any(unmet_qscore) or any(unmet_divscore):
        # print(f"[-] Unmet quality criterion: all-patch qscore={qscore_from_score(x_score)}. Move to next patch")
        # print(f"[-] Unmet criterion:\t unmet_qscore={unmet_qscore},\t unmet_divscore={unmet_divscore}")
        # print(f"    Unmet criterion:\t       qscore={qscores},\t       divscore={divscores}")
        # show_esawc_score_hist(x_esawc, x_score)
        continue
    
    qscores += qscore
    patch_means += [patch_mean(x.detach().cpu().numpy()) for x in tttv_score.values()]
    above_thresholds += [above_threshold(x.detach().cpu().numpy()) for x in tttv_score.values()]
    divscores += divscore
    
    selected_bboxes.append(domains.GeoRectangle(qb, fmt = "tgbox"))
    if i % int(n_patches/10) == 0:
        print(f"[i={i}/{n_patches}, j={j}/{sampler_length}] Missing patches count: " + ", ".join([f"{k}={missing[k]}" for k in lcnames]))
    
    i += 1
    if i >= n_patches:
        add_another_patch = False
        print(f"Number of patches requested has been reached. i={i}/{n_patches}, j={j}/{sampler_length}")


print(f"Acceptation rate: (# valid patches)/(# tested patches) = {i/j}")

b = np.linspace(0,1,21)
df = pd.DataFrame(
    {
        "divscore":divscores,
        "patchmean":patch_means,
        "abovethre":above_thresholds,
        "logdivscore":np.log10(divscores),
        "logpatchmean":np.log10(patch_means),
        # "logabovethre":np.log10(np.array(above_thresholds) + quality_threshold),
        "logabovethre":np.log10(above_thresholds),
    }
)
divscore_candidate = 0.1
qscore_candidate = 0.5

# sns.jointplot(data=df, x="patchmean", y="abovethre", kind = "hex", color="#4CB391")
sns.histplot(df, x="divscore", y="abovethre", color="blue", alpha=0.5, label = "above threshold")
ax = sns.histplot(df, x="divscore", y="patchmean", color="yellow", alpha=0.4, label = "patch mean")
rectangle = mpatches.Rectangle(
    xy=[divscore_candidate, qscore_candidate],
    width=1-divscore_candidate, height=1-qscore_candidate,
    facecolor='red',alpha=0.2
)
# sns.histplot(df, x="logdivscore", y="logabovethre", color="blue", alpha=0.5, label = "above threshold")
# ax = sns.histplot(df, x="logdivscore", y="logpatchmean", color="yellow", alpha=0.4, label = "patch mean")
# rectangle = mpatches.Rectangle(
    # xy=np.log10([divscore_candidate, qscore_candidate]),
    # width=-np.log10(divscore_candidate), height=-np.log10(qscore_candidate),
    # facecolor='red',alpha=0.2
# )

ax.add_patch(rectangle)
# sns.histplot(df, x="logdivscore", y="logabovethre", color="blue")
# sns.kdeplot(df, x="logdivscore", y="logabovethre", color="green")
plt.grid()
plt.title("Yellow=patch mean; Blue=above threshold")
plt.show(block=False)


plt.figure()
h = plt.hist2d(divscores, qscores, bins=20, cmap = "GnBu")
# h = plt.hist2d(divscores, patch_means, bins=np.linspace(0,1,21), cmap = "GnBu")
# h = plt.hist2d(divscores, above_thresholds, bins=np.linspace(0,1,21), cmap = "GnBu")
# plt.hist2d(np.log10(divscores), np.log10(qscores), bins=60, cmap = "GnBu")
plt.xlabel("Diversity score")
plt.ylabel("Quality score")
plt.show(block=False)



h1 = np.histogram2d(divscores, patch_means, bins=b)
h2 = np.histogram2d(divscores, above_thresholds, bins=b)
x, y = np.meshgrid((b[1:] + b[:-1])/2, (b[1:] + b[:-1])/2)

plt.figure()
plt.contourf(x, y, h1[0], cmap = "GnBu")
plt.contour(x, y, h2[0], colors="black")
plt.xlabel("Diversity score")
plt.ylabel("Quality score")
plt.show(block=False)


# Plot patches on a map
#-----------------
qdom = getattr(domains, domainname)
figname = "patchloc-" + "-".join([domainname, f"{n_patches}patches", f"{quality_threshold}qthresh", f"{diversity_threshold}divthresh"])
plt_utils.storeImages = False
plt_utils.figureDir = ""
plt_utils.patches_over_domain(qdom, selected_bboxes, background="osm", zoomout=0, details=4, figname = figname)

# EOF

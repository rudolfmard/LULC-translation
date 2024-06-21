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
python prepare_hdf5_ds2.py --h5dir=test --npatches=100 --qscore=0.2
"""
import os
import sys
import argparse
from pprint import pprint

import h5py
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
import torchvision.transforms as tvt
from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcover_to_landcover, landcovers
from mmt.datasets import  transforms as mmt_transforms
from mmt.utils import config as utilconf
from mmt.utils import domains, misc, plt_utils
from torchgeo import samplers
from torchgeo.datasets.utils import BoundingBox

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(prog="prepare_hdf5_datasets")
parser.add_argument("--h5dir", help="Path to the directory where the HDF5 will be stored.")
parser.add_argument("--npatches", help="Nb of patches to be created", default=5000)
parser.add_argument("--qscore", help="Minimum quality score to reach for a patch to be selected", default=0.7)
parser.add_argument("--divscore", help="Minimum diversity score to reach for a patch to be selected", default=0.1)
parser.add_argument("--npxemb", help="Nb of pixels of the patches in the latent space", default=600)
parser.add_argument("--margin", help="Nb of extra pixels extracted in the lager patches", default=200)
parser.add_argument("--domainname", help="Domain on which the patches are randomly picked up", default="eurat")
args = parser.parse_args()

print(f"Executing program {sys.argv[0]} in {os.getcwd()} with {args}")
domainname = args.domainname
n_patches = int(args.npatches)
quality_threshold = float(args.qscore)
diversity_threshold = float(args.divscore)
dump_dir = args.h5dir
margin = int(args.margin)
n_px_emb = int(args.npxemb)

# Land cover loading
#--------------------
print(f"Loading landcovers with native CRS")
esawc = landcovers.ESAWorldCover(transforms=mmt_transforms.EsawcTransform())
ecosg = landcovers.EcoclimapSG()
esgp = landcovers.EcoclimapSGplus()
qscore = landcovers.ScoreECOSGplus(transforms=mmt_transforms.ScoreTransform(divide_by=100))

lc = esawc & esgp & ecosg & qscore

print(f"Joint dataset has res={lc.res} and crs={lc.crs}")

lcnames = ["esawc", "ecosg", "esgp"]
lcmaps = {"esawc":esawc, "ecosg":ecosg, "esgp":esgp}
n_pxs = {"esawc":600, "ecosg":20, "esgp":100}

# Sampler definition
#--------------------
subsets = ["train", "train2", "test", "val"]

n_px_max = 2 * n_px_emb + margin
sampler_length = 14 * n_patches
val_domain = getattr(domains, domainname).to_tgbox(esawc.crs)
sampler = samplers.RandomGeoSampler(lc, size=n_px_max, length = sampler_length, roi = val_domain)


# Open HDF5 files
#--------------------
h5_path = dict()
h5f = dict()
for subset in subsets:
    h5_path[subset] = dict()
    h5f[subset] = dict()
    for lcname in lcnames:
        h5_path[subset][lcname] = os.path.join(
            dump_dir,
            f"{lcname}-{subset}.hdf5"
        )
        h5f[subset][lcname] = h5py.File(h5_path[subset][lcname], "w", libver='latest')
    

# INIT MAIN LOOP
#================
i = 0   # Valid patches counter
j = 0   # All patches counter
selected_bboxes = []
missing = {k:0 for k in lcnames}
add_another_patch =True
sampler = iter(sampler)
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
    
    x_labels = {
        "esawc": x_lc["mask"][0],
        "ecosg": x_lc["mask"][2],
        "esgp": x_lc["mask"][1],
    }
    
    # Quality control
    #-----------------
    x_score = x_lc["image"].unsqueeze(0)
    x_esawc = x_lc["mask"][0].unsqueeze(0)
    
    tttv_score = misc.ccrop_and_split(x_score, n_px_emb)
    tttv_esawc = misc.ccrop_and_split(x_esawc, n_px_emb)
    
    unmet_qscore = [misc.qscore_from_score(x) < quality_threshold for x in tttv_score.values()]
    unmet_divscore = [misc.divscore_from_esawc(x) < diversity_threshold for x in tttv_esawc.values()]
    if any(unmet_qscore) or any(unmet_divscore):
        continue
    
    # If passed, write in file
    #--------------------------
    for i_lc, lcname in enumerate(lcnames):
        x = x_labels[lcname].unsqueeze(0)
        
        tttv = misc.ccrop_and_split(x, n_px_emb)
        rsz = tvt.Resize(n_pxs[lcname], interpolation = tvt.functional.InterpolationMode.NEAREST)
        
        for subset in subsets:
            patch = rsz(tttv[subset].unsqueeze(0)).squeeze().detach().cpu().numpy()
            h5f[subset][lcname].create_dataset(str(i), data = patch)
            rd = h5f[subset][lcname].get(str(i))
            lon, lat = domains.GeoRectangle(qb, fmt = "tgbox").central_point()
            rd.attrs["x_coor"] = lon
            rd.attrs["y_coor"] = lat
    
    selected_bboxes.append(domains.GeoRectangle(qb, fmt = "tgbox"))
    if i % int(n_patches/10) == 0:
        print(f"[i={i}/{n_patches}, j={j}/{sampler_length}] Missing patches count: " + ", ".join([f"{k}={missing[k]}" for k in lcnames]))
    
    i += 1
    if i >= n_patches:
        add_another_patch = False
        print(f"Number of patches requested has been reached. i={i}/{n_patches}, j={j}/{sampler_length}")

print(f"Acceptation rate: (# valid patches)/(# tested patches) = {i/j}")

# Close HDF5 files
#-----------------
for subset in subsets:
    for lcname in lcnames:
        h5f[subset][lcname].attrs["name"] = lcname
        h5f[subset][lcname].attrs["year"] = 2024
        h5f[subset][lcname].attrs["type"] = "raster"
        h5f[subset][lcname].attrs["resolution"] = lcmaps[lcname].res
        h5f[subset][lcname].attrs["crs"] = lcmaps[lcname].crs.to_string()
        h5f[subset][lcname].attrs["patch_size"] = n_pxs[lcname]
        h5f[subset][lcname].attrs["n_channels"] = 1
        h5f[subset][lcname].attrs["numberclasses"] = lcmaps[lcname].n_labels
        h5f[subset][lcname].attrs["label_definition"] = lcmaps[lcname].labels
        h5f[subset][lcname].close()
        print(f"File {h5_path[subset][lcname]} written.")

# Write patches coords
#-----------------
coordspatches_filename = "coordspatches-" + "-".join([domainname, f"{n_patches}patches", f"{quality_threshold}qthresh"]) + ".txt"
with open(os.path.join(dump_dir, coordspatches_filename), "w") as f:
    f.write("\t".join(["idx", "minx", "miny", "maxx", "maxy"]) + "\n")
    for i, b in enumerate(selected_bboxes):
        f.write("\t".join([str(_) for _ in [i, b.min_longitude, b.min_latitude, b.max_longitude, b.max_latitude]]) + "\n")

# Plot patches on a map
#-----------------
qdom = getattr(domains, domainname)
figname = "patchloc-" + "-".join([domainname, f"{n_patches}patches", f"{quality_threshold}qthresh", f"{diversity_threshold}divthresh"])
plt_utils.DEFAULT_SAVEFIG = True
plt_utils.DEFAULT_FIGDIR = dump_dir
plt_utils.patches_over_domain(qdom, selected_bboxes, background="osm", zoomout=0, details=4, figname = figname)

# EOF

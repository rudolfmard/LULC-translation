#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to compute confusion matrix of MLCT-net prediction and compare them to ECOSG

Useful links
------------
Paper   ->  https://www.tandfonline.com/doi/pdf/10.1080/13658816.2022.2120996
Code    ->  https://github.com/LBaudoux/MLULC
Data    ->  https://zenodo.org/record/5843595
Last checked: 31 July 2023
"""
import os
import sys
import h5py
import torch
from torch import nn
from torchgeo.datasets.utils import BoundingBox
import torchgeo.datasets as tgd
import torchvision.transforms as tvt
from torchgeo import samplers
from torch.utils.data import DataLoader
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn import metrics
from wopt.ml import graphics

from mmt import _repopath_ as mmt_repopath
from mmt.graphs.models import universal_embedding
from mmt.datasets import landcover_to_landcover
from mmt.datasets import landcovers
from mmt.datasets import transforms
from mmt.utils import config as utilconf
from mmt.utils import domains
from mmt.utils import plt_utils
from mmt.inference import io

def qscore_from_qflags(qflags):
    """Return the proportion of quality flag with values 1 or 2"""
    if isinstance(qflags, torch.Tensor):
        qflags = qflags.detach().cpu().numpy()
    
    if (qflags==0).sum() > 0:
        return 0
    else:
        return (qflags < 3).sum()/qflags.size

def ccrop_and_split(x, n_px):
    """Center-crop and split into four patches
    
    Params
    ------
    x: torch.Tensor or dict
        Data to be cropped and split
    
    n_px: int
        Number of pixels in the split patches
    """
    ccrop = tvt.CenterCrop(2 * n_px)
    try:
        x = ccrop(x["mask"]).squeeze()
    except:
        x = ccrop(x).squeeze()
        
    x_train1 =  x[:n_px, :n_px]
    x_train2 =  x[n_px:, :n_px]
    x_test =    x[:n_px, n_px:]
    x_val =     x[n_px:, n_px:]
    
    return {"train1":x_train1, "train2":x_train2, "test":x_test, "val":x_val}

# PARAMETERS
#============
usegpu = True
domainname = "eurat"
n_patches = 4000
quality_threshold = 0.7

print(f"Executing program {sys.argv[0]} in {os.getcwd()}")
device = torch.device("cuda" if usegpu else "cpu")
config = utilconf.get_config(
    os.path.join(
        mmt_repopath,
        "configs",
        "new_config_template.yaml"
    )
)
dump_dir = os.path.join(config.paths.data_dir, "new-large-domain-hdf5")


# Land cover loading
#--------------------
print(f"Loading landcovers with native CRS")
qflag = landcovers.QualityFlagsECOSGplus()
esawc = landcovers.ESAWorldCover(transforms=transforms.EsawcTransform)
esgp = landcovers.EcoclimapSGplus()
ecosg = landcovers.EcoclimapSG()

lc = esawc & esgp & ecosg & qflag

lcnames = ["esawc", "esgp", "ecosg", "qflag"]
lcmaps = {"qflag":qflag, "esawc":esawc, "esgp":esgp, "ecosg":ecosg}

# Sampler definition
#--------------------
subsets = ["train1", "train2", "test", "val"]

margin = 200
n_px_max = 2 * config.dimensions.n_px_embedding + margin
n_px_emb = config.dimensions.n_px_embedding
sampler_length = 20 * n_patches
val_domain = getattr(domains, domainname).to_tgbox(esawc.crs)
sampler = samplers.RandomGeoSampler(lc, size=n_px_max, length = sampler_length, roi = val_domain)

n_pxs = {"qflag":100, "esawc":600, "esgp":100, "ecosg":20}

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
            # "-".join([lcname, domainname, subset, f"{n_patches}patches", f"{quality_threshold}qthresh"]) + ".hdf5"
            f"{lcname}-{subset}.hdf5"
        )
        h5f[subset][lcname] = h5py.File(h5_path[subset][lcname], "w", libver='latest')
    

# INIT MAIN LOOP
#================
i = 0   # Valid patches counter
j = 0   # All patches counter
selected_bboxes = []
missing = {k:0 for k in lcmaps.keys()}
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
    x_lc = lc[qb]
    
    # Quality control
    #-----------------
    x_qflag = x_lc["mask"][-1].unsqueeze(0)
    
    tttv_qflag = ccrop_and_split(x_qflag, n_px_emb)
    
    qscore = qscore_from_qflags(x_qflag)
    unmet_qscore = [qscore_from_qflags(x) < quality_threshold for x in tttv_qflag.values()]
    if any(unmet_qscore):
        # print(f"[-] Unmet quality criterion: all-patch qscore={qscore}. Move to next patch")
        continue
    
    # If passed, write in file
    #--------------------------
    for i_lc, lcname in enumerate(lcnames):
        x = x_lc["mask"][i_lc].unsqueeze(0)
        
        tttv = ccrop_and_split(x, n_px_emb)
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
        print(f"[i={i}/{n_patches}, j={j}/{sampler_length}] Missing patches count: " + ", ".join([f"{k}={missing[k]}" for k in lcmaps.keys()]))
    
    i += 1
    if i >= n_patches:
        add_another_patch = False
        print(f"Number of patches requested has been reached. i={i}/{n_patches}, j={j}/{sampler_length}")


# Close HDF5 files
#-----------------
for subset in subsets:
    for lcname in lcnames:
        h5f[subset][lcname].attrs["name"] = lcname
        h5f[subset][lcname].attrs["year"] = 2023
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
figname = "patchloc-" + "-".join([domainname, f"{n_patches}patches", f"{quality_threshold}qthresh"])
graphics.storeImages = True
graphics.figureDir = dump_dir
graphics.patches_over_domain(qdom, selected_bboxes, background="osm", zoomout=0, details=4, figname = figname)
# EOF

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
import torch
from torch import nn
from torchgeo.datasets.utils import BoundingBox
import torchgeo.datasets as tgd
from torchgeo import samplers
from torch.utils.data import DataLoader
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pprint import pprint
from sklearn import metrics
import pandas as pd
import seaborn as sns
import wopt.ml.utils
from wopt.ml import (
    landcovers,
    transforms,
    graphics,
    domains,
)

from mmt import _repopath_ as mmt_repopath
from mmt.graphs.models import universal_embedding
from mmt.datasets import landcover_to_landcover
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import config as utilconf
from mmt.inference import io


# Configs
#---------
usegpu = True
device = torch.device("cuda" if usegpu else "cpu")

woptconfig = wopt.ml.utils.load_config()
print(f"Executing program {sys.argv[0]} in {os.getcwd()}")

xp_name = "vanilla_with_esgp_v2"
domainname = "freastern_eurat"
n_val_patches = 60
lc_in="esawc"
lc_out="esgp"

mlulcconfig, _ = utilconf.get_config_from_json(
    os.path.join(
        mmt_repopath,
        "experiments",
        xp_name,
        "logs",
        "config.json",
    )
)

checkpoint_path = os.path.join(
    mmt_repopath,
    "experiments",
    xp_name,
    "checkpoints",
    "model_best.pth.tar",
)

graphics.storeImages = woptconfig["graphics"]["store_images"]
graphics.figureDir = woptconfig["graphics"]["figure_dir"]
graphics.fmtImages = "." + woptconfig["graphics"]["fmt_images"]
graphics.print_graphics_config()


# Land cover loading
#--------------------
dst_crs = rasterio.crs.CRS.from_epsg(3035)

print(f"Loading landcovers with CRS = {dst_crs}")
esawc = landcovers.ESAWorldCover(
    transforms=transforms.esawc_transform,
    crs = dst_crs,
)
esawc.crs = dst_crs
esawc.res = 10

esgp = landcovers.EcoclimapSGplus(crs=dst_crs, res=60)
esgp.crs = dst_crs
esgp.res = 60

ecosg = landcovers.TgEcoclimapSG(crs=dst_crs, res=60)
ecosg.crs = dst_crs
ecosg.res = 60

# Loading models
#----------------
print(f"Loading auto-encoders from {xp_name}")
model1 = io.load_pytorch_model(xp_name, "esawc", "esgp")
model1 = model1.to(device)
model2 = io.load_pytorch_model(xp_name, "esgp", "esgp")
model2 = model2.to(device)

toh1 = mmt_transforms.OneHot(esawc.n_labels + 1, device = device)
toh2 = mmt_transforms.OneHot(esgp.n_labels + 1, device = device)

# VALIDATION DOMAINS
#====================

# Wide random domain
#--------------------
n_px = int(6000/esawc.res)
sampler_length = 5 * n_val_patches
# val_domain = domains.freastern_eurat.to_tgbox(esawc.crs)
val_domain = getattr(domains, domainname).to_tgbox(esawc.crs)
sampler = samplers.RandomGeoSampler(esawc, size=n_px, length = sampler_length, roi = val_domain)

n_labels = len(esgp.labels)
cmx_esawc2esgp_esgp = np.zeros((n_labels, n_labels), dtype = np.int32)
cmx_esgp2esgp_esgp = np.zeros((n_labels, n_labels), dtype = np.int32)
cmx_ecosg_esgp = np.zeros((n_labels, n_labels), dtype = np.int32)
i = 0
missing_ecosg = 0
missing_esawc = 0
missing_esgp = 0
add_another_patch =True
sampler = iter(sampler)
print(f"Start inference on {n_val_patches} patches of {n_px*esawc.res} meters over {domainname}")
while add_another_patch:
    try:
        qb = next(sampler)
    except StopIteration:
        add_another_patch = False
        print(f"No more patch in sampler (size={sampler_length}). i={i}/{n_val_patches}")
        break
    
    try:    
        x_esgp = esgp[qb]
    except:
        missing_esgp += 1
        continue
        
    try:
        x_esawc = esawc[qb]
    except:
        missing_esawc += 1
    
    try:
        x_ecosg = ecosg[qb]
    except:
        missing_ecosg += 1
        continue
    
    if i % int(n_val_patches/10) == 0:
        print(f"[{i}/{n_val_patches}] Missing patches count: ecosg={missing_ecosg}, esgp={missing_esgp}, esawc={missing_esawc}")
    
    x1 = x_esawc["mask"]
    x2 = x_esgp["mask"]
    y_true = x_esgp["mask"]
    with torch.no_grad():
        x1 = toh1(x1)
        x2 = toh2(x2)
        y1 = model1(x1).argmax(1).cpu()
        y2 = model2(x2).argmax(1).cpu()
    
    cmx_esawc2esgp_esgp += metrics.confusion_matrix(y_true.numpy().ravel(), y1.numpy().ravel(), labels = np.arange(n_labels))
    cmx_esgp2esgp_esgp += metrics.confusion_matrix(y_true.numpy().ravel(), y2.numpy().ravel(), labels = np.arange(n_labels))
    cmx_ecosg_esgp += metrics.confusion_matrix(y_true.numpy().ravel(), x_ecosg["mask"].numpy().ravel(), labels = np.arange(n_labels))
    
    i += 1
    if i >= n_val_patches:
        add_another_patch = False
        print(f"Number of patches requested has been reached. i={i}/{n_val_patches}")


print(f"Bulk total overall accuracy (ESAWC -> ECOSG+): {np.diag(cmx_esawc2esgp_esgp).sum()/cmx_esawc2esgp_esgp.sum()}")
print(f"Bulk total overall accuracy (ECOSG+ -> ECOSG+): {np.diag(cmx_esawc2esgp_esgp).sum()/cmx_esawc2esgp_esgp.sum()}")
print(f"Bulk total overall accuracy (ECOSG): {np.diag(cmx_ecosg_esgp).sum()/cmx_ecosg_esgp.sum()}")

shortlbnames = np.array(
    """0.nodata
1.sea
2.lakes
3.rivers
4.bare land
5.bare rock
6.snow
7.bor bl dec
8.temp bl dec
9.trop bl dec
10.temp bl evg
11.trop bl evg
12.bor bl evg
13.temp nl evg
14.bor nl dec
15.shrubs
16.bor grass
17.temp grass
18.trop grass
19.winC3
20.sumC3
21.C4crops
22.fl trees
23.fl grass
24.LCZ1 c.hig
25.LCZ2 c.mid
26.LCZ3 c.low
27.LCZ4 o.hig
28.LCZ5 o.mid
29.LCZ6 o.low
30.LCZ7 lw.l
31.LCZ8 lar.l
32.LCZ9 spars
33.LCZ10indus""".split("\n")
)
shortlbnames = np.array(wopt.ml.utils.ecosg_label_names)
def t_(ls):
    return ["t"+l for l in ls]

def p_(ls):
    return ["p"+l for l in ls]

lh = landcovers.ecoclimapsg_label_hierarchy
n_priml = len(lh.keys())
oap = {}
oat = {}
print("Computing confusion matrices...")
for cmx, method, shortmet in zip(
        [cmx_esawc2esgp_esgp, cmx_esgp2esgp_esgp, cmx_ecosg_esgp],
        ["ESAWC -> ECOSG+", "ECOSG+ -> ECOSG+", "ECOSG"],
        ["esawc2esgp", "esgp2esgp", "ecosg"],
    ):
    print(f"Method = {method}")
    
    # Primary labels confusion matrix
    #---------------------
    pred_lnames = p_(shortlbnames)
    true_lnames = t_(shortlbnames)
    dfcmx = pd.DataFrame(data=cmx, index = true_lnames, columns = pred_lnames)
    famcmx = np.zeros((n_priml, n_priml), dtype = np.int32)
    for i, fam1 in enumerate(lh.keys()):
        for j, fam2 in enumerate(lh.keys()):
            famcmx[i,j] = dfcmx.loc[t_(lh[fam1]), p_(lh[fam2])].values.sum()
        
    dfamcmx = pd.DataFrame(data = famcmx, index = lh.keys(), columns = lh.keys())
    print(f"  Overall accuracy on primary labels ({method}): {np.diag(famcmx).sum()/famcmx.sum()}")
    oap[shortmet] = np.round(np.diag(famcmx).sum()/famcmx.sum(), 3)
    
    # graphics.plot_confusion_matrix(
        # dfamcmx,
        # figtitle = f"Primary label confusion matrix {method}",
        # figname = f"famcmx_{domainname}_{shortmet}",
        # accuracy_in_corner=True,
    # )
    keep_idxs = cmx.sum(axis=1) > 0
    keep_idxs[0] = False # Remove pixels where ref is "no data"
    kcmx = cmx[:, keep_idxs]
    kcmx = kcmx[keep_idxs, :]
    pred_lnames = p_(shortlbnames[keep_idxs])
    true_lnames = t_(shortlbnames[keep_idxs])
    lnames = shortlbnames[keep_idxs]
    nl = len(lnames)
    dfkcmx = pd.DataFrame(data=kcmx, index = true_lnames, columns = pred_lnames)
    
    # Confusion matrix
    #---------------------
    graphics.plot_confusion_matrix(
        dfkcmx,
        figtitle = f"Unnormed confusion matrix {method}",
        figname = f"cmx_{domainname}_{shortmet}",
        accuracy_in_corner = True
    )
    oat[shortmet] = np.round(np.diag(kcmx).sum()/kcmx.sum(), 3)
    print(f"  Overall accuracy on ECOSG labels ({method}): {np.diag(kcmx).sum()/kcmx.sum()}")
    
    # Normalization by actual amounts -> Recall matrix
    #--------------------------------
    reccmx = kcmx/np.repeat(kcmx.sum(axis=1), nl).reshape((nl,nl))
    graphics.plot_confusion_matrix(
        pd.DataFrame(data=reccmx, index = dfkcmx.index, columns = dfkcmx.columns),
        figtitle = f"Recall matrix (normed by reference) {method}",
        figname = f"reccmx_{domainname}_{shortmet}"
    )
    # # Normalization by predicted amounts -> Precision matrix
    # #--------------------------------
    # precmx = kcmx/np.repeat(kcmx.sum(axis=0), nl).reshape((nl,nl)).T
    # graphics.plot_confusion_matrix(
        # pd.DataFrame(data=precmx, index = dfkcmx.index, columns = dfkcmx.columns),
        # figtitle = f"Precision matrix (normed by predictions) {method}",
        # figname = f"precmx_{domainname}_{shortmet}"
    # )

print(f"""Recap of overall accuracies over {domainname}:
+------------+----------------+--------------+
| Method     | Primary labels | ECOSG labels |
+------------+----------------+--------------+
| ECOSG      | {oap['ecosg']}          | {oat['ecosg']}        |
| ESA trans  | {oap['esawc2esgp']}          | {oat['esawc2esgp']}        |
| ESG+ trans | {oap['esgp2esgp']}          | {oat['esgp2esgp']}        |
+------------+----------------+--------------+
""")
# EOF

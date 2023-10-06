#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to compute confusion matrix of MLCT-net prediction and compare them to ECOSG
"""
import os
import sys
import h5py
import torch
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
from sklearn import metrics
import pandas as pd
import seaborn as sns

from mmt import _repopath_ as mmt_repopath
from mmt.graphs.models import universal_embedding
from mmt.datasets import landcover_to_landcover
from mmt.datasets import landcovers
from mmt.datasets import transforms
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import config as utilconf
from mmt.utils import domains
from mmt.utils import plt_utils
from mmt.inference import io


shortlbnames = np.array(landcovers.ecoclimapsg_labels)
def t_(ls):
    return ["t"+l for l in ls]

def p_(ls):
    return ["p"+l for l in ls]

def accuracy(cmx):
    return np.diag(cmx).sum()/cmx.sum()

# Configs
#---------
usegpu = True
device = "cuda" if usegpu else "cpu"

print(f"Executing program {sys.argv[0]} in {os.getcwd()}")

xp_name = "vanilla"
domainname = "eurat"

config = utilconf.get_config(
    os.path.join(
        mmt_repopath,
        "experiments",
        xp_name,
        "logs",
        "config.yaml",
    )
)

# Loading models
#----------------
print(f"Loading auto-encoders from {xp_name}")
model1 = io.load_pytorch_model(xp_name, "esawc", "esgp")
model1 = model1.to(device)
model2 = io.load_pytorch_model(xp_name, "esgp", "esgp")
model2 = model2.to(device)

toh1 = mmt_transforms.OneHot(13, device = device)
toh2 = mmt_transforms.OneHot(35, device = device)
to_tensor = lambda x: torch.Tensor(x[:]).long()

# VALIDATION DOMAINS
#====================
# ldom_data_dir = os.path.join(config.paths.data_dir, "large-domain-200")
ldom_data_dir = os.path.join(config.paths.data_dir, "hdf5_data")
subset = "test"
lcnames = ["esawc", "ecosg", "esgp"]
h5_path = {}
h5f = {}
for lcname in lcnames:
    h5_path[lcname] = os.path.join(ldom_data_dir, f"{lcname}-{subset}.hdf5")
    assert os.path.isfile(h5_path[lcname]), f"File {h5_path[lcname]} does not exist"
    h5f[lcname] = h5py.File(h5_path[lcname], "r", libver='latest')

assert all([set(h5f["esawc"].keys()) == set(h5f[lcname].keys()) for lcname in ["ecosg", "esgp"]]), "HDF5 keys lists don't match"

n_labels = len(shortlbnames)
cmx_esawc2esgp_esgp = np.zeros((n_labels, n_labels), dtype = np.int32)
cmx_esgp2esgp_esgp = np.zeros((n_labels, n_labels), dtype = np.int32)
cmx_ecosg_esgp = np.zeros((n_labels, n_labels), dtype = np.int32)

print("Computing confusion matrices...")
for i in tqdm(h5f["esawc"].keys()):
    x1 = to_tensor(h5f["esawc"].get(i))
    x2 = to_tensor(h5f["esgp"].get(i))
    x3 = h5f["ecosg"].get(i)
    y_true = h5f["esgp"].get(i)
    with torch.no_grad():
        x1 = toh1(x1)
        x2 = toh2(x2)
        y1 = model1(x1.unsqueeze(0)).argmax(1).cpu()
        y2 = model2(x2.unsqueeze(0)).argmax(1).cpu()
    
    cmx_esawc2esgp_esgp += metrics.confusion_matrix(y_true[:].ravel(), y1.numpy().ravel(), labels = np.arange(n_labels))
    cmx_esgp2esgp_esgp += metrics.confusion_matrix(y_true[:].ravel(), y2.numpy().ravel(), labels = np.arange(n_labels))
    cmx_ecosg_esgp += metrics.confusion_matrix(y_true[:].ravel(), np.tile(x3[:], (5,5)).ravel(), labels = np.arange(n_labels))


print(f"Bulk total overall accuracy (ESAWC -> ECOSG+): {accuracy(cmx_esawc2esgp_esgp)}")
print(f"Bulk total overall accuracy (ECOSG+ -> ECOSG+): {accuracy(cmx_esgp2esgp_esgp)}")
print(f"Bulk total overall accuracy (ECOSG): {accuracy(cmx_ecosg_esgp)}")

lh = landcovers.ecoclimapsg_label_hierarchy
n_priml = len(lh.keys())
oap = {}
oat = {}
print("Drawing confusion matrices...")
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
    print(f"  Overall accuracy on primary labels ({method}): {accuracy(famcmx)}")
    oap[shortmet] = np.round(accuracy(famcmx), 3)
    
    # plt_utils.plot_confusion_matrix(
        # dfamcmx,
        # figtitle = f"Primary label confusion matrix {method}",
        # figname = f"famcmx_{domainname}_{shortmet}",
        # accuracy_in_corner=True,
    # )
    keep_idxs = cmx.sum(axis=1) > 0
    keep_idxs[0] = False # Remove pixels where ref is "no data"
    print(f"  {sum(keep_idxs)}/{len(keep_idxs)} labels kept. Removed = {shortlbnames[~keep_idxs]}")
    kcmx = cmx[:, keep_idxs]
    kcmx = kcmx[keep_idxs, :]
    pred_lnames = p_(shortlbnames[keep_idxs])
    true_lnames = t_(shortlbnames[keep_idxs])
    lnames = shortlbnames[keep_idxs]
    nl = len(lnames)
    dfkcmx = pd.DataFrame(data=kcmx, index = true_lnames, columns = pred_lnames)
    
    # Confusion matrix
    #---------------------
    plt_utils.plot_confusion_matrix(
        dfkcmx,
        figtitle = f"Unnormed confusion matrix {method}",
        figname = f"cmx_{domainname}_{shortmet}",
        accuracy_in_corner = True
    )
    oat[shortmet] = np.round(accuracy(kcmx), 3)
    print(f"  Overall accuracy on ECOSG labels ({method}): {accuracy(kcmx)}")
    
    # Normalization by actual amounts -> Recall matrix
    #--------------------------------
    reccmx = kcmx/np.repeat(kcmx.sum(axis=1), nl).reshape((nl,nl))
    plt_utils.plot_confusion_matrix(
        pd.DataFrame(data=reccmx, index = dfkcmx.index, columns = dfkcmx.columns),
        figtitle = f"Recall matrix (normed by reference) {method}",
        figname = f"reccmx_{domainname}_{shortmet}"
    )
    # # Normalization by predicted amounts -> Precision matrix
    # #--------------------------------
    # precmx = kcmx/np.repeat(kcmx.sum(axis=0), nl).reshape((nl,nl)).T
    # plt_utils.plot_confusion_matrix(
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

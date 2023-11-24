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
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import config as utilconf
from mmt.utils import domains
from mmt.utils import plt_utils
from mmt.utils import scores
from mmt.inference import io
from mmt.inference import translators

shortlbnames = np.array(landcovers.ecoclimapsg_labels)

# Configs
#---------
usegpu = True
device = "cuda" if usegpu else "cpu"

print(f"Executing program {sys.argv[0]} in {os.getcwd()}")

xp_name = "vanilla_eurat3"
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

# plt_utils.figureDir = os.path.join(
    # mmt_repopath,
    # "experiments",
    # xp_name,
    # "out"
# )
plt_utils.figureDir = ""
plt_utils.fmtImages = ".svg"
plt_utils.storeImages = False

# Loading models
#----------------
checkpoint_path = os.path.join(mmt_repopath, "saved_models", "vanilla_eurat3.ep169.ckpt")
print(f"Loading auto-encoders from {checkpoint_path}")
translator = translators.EsawcToEsgp(checkpoint_path = checkpoint_path)
epoch = io.get_epoch_of_best_model(checkpoint_path)
to_tensor = lambda x: torch.Tensor(x[:]).long().unsqueeze(0)

# VALIDATION DOMAINS
#====================
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
cmx_infres_esgp = np.zeros((n_labels, n_labels), dtype = np.int32)
cmx_ecosg_esgp = np.zeros((n_labels, n_labels), dtype = np.int32)

print("Computing confusion matrices...")
items = list(h5f["esawc"].keys())
for i in tqdm(items):
    x1 = to_tensor(h5f["esawc"].get(i))
    x3 = h5f["ecosg"].get(i)
    y_true = h5f["esgp"].get(i)
    
    y1 = translator.predict_from_data(x1)
    
    cmx_infres_esgp += metrics.confusion_matrix(y_true[:].ravel(), y1.ravel(), labels = np.arange(n_labels))
    cmx_ecosg_esgp += metrics.confusion_matrix(y_true[:].ravel(), np.tile(x3[:], (5,5)).ravel(), labels = np.arange(n_labels))


print(f"Bulk total overall accuracy (ECOSG): {scores.oaccuracy(cmx_ecosg_esgp)}")
print(f"Bulk total overall accuracy (INFRES): {scores.oaccuracy(cmx_infres_esgp)}")

lh = landcovers.ecoclimapsg_label_hierarchy
oap = {}
oat = {}
domainname = "eurat"
epoch = 174
pred_lnames = scores.p_(shortlbnames)
true_lnames = scores.t_(shortlbnames)

dfcmx_ecosg = pd.DataFrame(data=cmx_ecosg_esgp, index = true_lnames, columns = pred_lnames)
dfcmx_infres = pd.DataFrame(data=cmx_infres_esgp, index = true_lnames, columns = pred_lnames)

print("Drawing confusion matrices...")
for cmx, method, shortmet in zip(
        [cmx_infres_esgp, cmx_ecosg_esgp],
        ["ESAWC -> ECOSG+", "ECOSG"],
        ["infres", "ecosg"],
    ):
    print(f"Method = {method}")
    dfcmx = pd.DataFrame(data=cmx, index = true_lnames, columns = pred_lnames)
    dfkcmx = scores.remove_absent_labels(dfcmx)
    print(f"  Overall accuracy on secondary labels ({method}): {scores.oaccuracy(dfkcmx)}")
    oat[shortmet] = np.round(scores.oaccuracy(dfkcmx), 3)
    
    # Primary labels confusion matrix
    #---------------------
    dfamcmx = scores.sum_primary_labels(dfcmx, lh)
    print(f"  Overall accuracy on primary labels ({method}): {scores.oaccuracy(dfamcmx)}")
    oap[shortmet] = np.round(scores.oaccuracy(dfamcmx), 3)
    
    # Normalization by actual amounts -> Recall matrix
    #--------------------------------
    reccmx = scores.norm_matrix(dfkcmx, axis = 1)
    plt_utils.plot_confusion_matrix(
        pd.DataFrame(data=reccmx, index = dfkcmx.index, columns = dfkcmx.columns),
        figtitle = f"Recall matrix (normed by reference) {method}",
        figname = f"reccmx_{domainname}_{shortmet}_ep{epoch}"
    )

print(f"""Recap of overall accuracies over {domainname}:
+------------+----------------+--------------+
| Method     | Primary labels | ECOSG labels |
+------------+----------------+--------------+
| ECOSG      | {oap['ecosg']}          | {oat['ecosg']}        |
| ESA trans  | {oap['infres']}          | {oat['infres']}        |
+------------+----------------+--------------+
""")

scorename = "f1score"
methods = ["infres", "ecosg"]

print(f"\n   {scorename.upper()} FOR PRIMARY LABELS")
cmxs = [scores.sum_primary_labels(cmx, lh) for cmx in [dfcmx_infres, dfcmx_ecosg]]
pms = scores.permethod_scores(cmxs, methods, scorename)
scores.latexstyleprint(pms)

print(f"\n   {scorename.upper()} FOR SECONDARY LABELS")
cmxs = [dfcmx_infres, dfcmx_ecosg]
# cmxs = [scores.remove_absent_labels(cmx) for cmx in [dfcmx_infres, dfcmx_ecosg]]
pms = scores.permethod_scores(cmxs, methods, scorename)

scores.latexstyleprint(pms)
# EOF

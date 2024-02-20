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
import pickle
import argparse

from mmt import _repopath_ as mmt_repopath
from mmt.graphs.models import universal_embedding
from mmt.datasets import landcover_to_landcover
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import config as utilconf
from mmt.utils import domains
from mmt.utils import plt_utils
from mmt.utils import scores
from mmt.utils import misc
from mmt.inference import io
from mmt.inference import translators


# Argument parsing
# ----------------
parser = argparse.ArgumentParser(prog="scores_from_inference", description="Evaluate model on the dataset DS2 (EURAT-test), and provide confusion matrices")
parser.add_argument("--weights", help="Weight file, experience ID or path to the checkpoint to use for inference", default = "mmt-weights-v1.0.ckpt")
parser.add_argument("--device", help="Make inference on CPU or CUDA", default="cuda")
parser.add_argument("--figfmt", help="Format of the figure", default="svg")
parser.add_argument("--scorename", help="Score used to put in the tables (user_accuracy, prod_accuracy, f1score)", default="f1score")
parser.add_argument("--npatches", help="Number of patches on which the confusion matrices are computed", default=5000)
parser.add_argument("--figdir", help="Directory where figure will be saved", default=os.path.join(mmt_repopath, "figures"))
parser.add_argument("--savefig", help="Save the figures instead of plotting them", action = "store_true")
args = parser.parse_args()

device = args.device
n_patches = int(args.npatches)

if os.path.isfile(args.weights):
    checkpoint_path = args.weights
elif os.path.isfile(os.path.join(mmt_repopath, "data", "saved_models", args.weights)):
    checkpoint_path = os.path.join(mmt_repopath, "data", "saved_models", args.weights)
elif os.path.isfile(os.path.join(mmt_repopath, "experiments", args.weights, "checkpoints", "model_best.ckpt")):
    checkpoint_path = os.path.join(mmt_repopath, "experiments", args.weights, "checkpoints", "model_best.ckpt")
else:
    raise ValueError(f"Unable to find weights for {args.weights}")

plt_utils.figureDir = args.figdir
plt_utils.fmtImages = "." + args.figfmt
plt_utils.storeImages = args.savefig

# Loading models
#----------------
print(f"Loading auto-encoders from {checkpoint_path}")
translator = translators.EsawcToEsgp(checkpoint_path = checkpoint_path)
epoch = io.get_epoch_of_best_model(checkpoint_path)
to_tensor = lambda x: torch.Tensor(x[:]).long().unsqueeze(0)


# VALIDATION DOMAINS
#====================
ldom_data_dir = os.path.join(mmt_repopath, "data", "hdf5_data")
subset = "test"
lcnames = ["esawc", "ecosg", "esgp"]
h5_path = {}
h5f = {}
for lcname in lcnames:
    h5_path[lcname] = os.path.join(ldom_data_dir, f"{lcname}-{subset}.hdf5")
    assert os.path.isfile(h5_path[lcname]), f"File {h5_path[lcname]} does not exist"
    h5f[lcname] = h5py.File(h5_path[lcname], "r", libver='latest')

assert all([set(h5f["esawc"].keys()) == set(h5f[lcname].keys()) for lcname in ["ecosg", "esgp"]]), "HDF5 keys lists don't match"

shortlbnames = np.array(landcovers.ecoclimapsg_labels)
n_labels = len(shortlbnames)

cachedscore_header = {
    "weights": repr((io.get_epoch_of_best_model(checkpoint_path, return_iteration=True), translator)),
    "n_patches": repr(n_patches),
    "valdata": repr([(h5f[k].file, len(h5f[k]), hash(h5f[k])) for k in sorted(h5f.keys())])
}
cachedscore_file = os.path.join(
    mmt_repopath,
    "experiments",
    "cache",
    f"score-{misc.hashdict(cachedscore_header)}.pkl"
)
if os.path.isfile(cachedscore_file):
    print(f"Loading scores from {cachedscore_file}")
    with open(cachedscore_file, "rb") as f:
        cachedscore = pickle.load(f)
    
    cmx_infres_esgp = cachedscore["cmx_infres_esgp"]
    cmx_ecosg_esgp = cachedscore["cmx_ecosg_esgp"]
else:
    print(f"No cached scored found for {misc.hashdict(cachedscore_header)}. Computing confusion matrices...")
    cmx_infres_esgp = np.zeros((n_labels, n_labels), dtype = np.int32)
    cmx_ecosg_esgp = np.zeros((n_labels, n_labels), dtype = np.int32)
    
    items = list(h5f["esawc"].keys())[:n_patches]
    
    for i in tqdm(items):
        x1 = to_tensor(h5f["esawc"].get(i))
        x3 = h5f["ecosg"].get(i)
        y_true = h5f["esgp"].get(i)
        
        y1 = translator.predict_from_data(x1)
        
        cmx_infres_esgp += metrics.confusion_matrix(y_true[:].ravel(), y1.ravel(), labels = np.arange(n_labels))
        cmx_ecosg_esgp += metrics.confusion_matrix(y_true[:].ravel(), np.tile(x3[:], (5,5)).ravel(), labels = np.arange(n_labels))
    
    cachedscore = {**cachedscore_header, "cmx_infres_esgp":cmx_infres_esgp, "cmx_ecosg_esgp": cmx_ecosg_esgp}
    
    with open(cachedscore_file, "wb") as f:
        pickle.dump(cachedscore, f)

print(f"Bulk total overall accuracy (ECOSG): {scores.oaccuracy(cmx_ecosg_esgp)}")
print(f"Bulk total overall accuracy (INFRES): {scores.oaccuracy(cmx_infres_esgp)}")

lh = landcovers.ecoclimapsg_label_hierarchy
oap = {}
oat = {}
domainname = "eurat"
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

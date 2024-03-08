#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to compute confusion matrix of MLCT-net prediction and compare them to ECOSG
"""
import argparse
import os

import h5py

from mmt import _repopath_ as mmt_repopath
from mmt.inference import translators
from mmt.utils import plt_utils, scores

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(
    prog="scores_from_inference",
    description="Evaluate model on the dataset DS2 (EURAT-test), and provide confusion matrices",
)
parser.add_argument(
    "--weights",
    help="Weight file, experience ID or path to the checkpoint to use for inference",
    default="mmt-weights-v1.0.ckpt",
)
parser.add_argument(
    "--scorename",
    help="Score used to put in the tables (user_accuracy, prod_accuracy, f1score)",
    default="f1score",
)
parser.add_argument("--figfmt", help="Format of the figure", default="svg")
parser.add_argument(
    "--npatches",
    help="Number of patches on which the confusion matrices are computed",
    type=int,
    default=1e5,
)
parser.add_argument(
    "--figdir",
    help="Directory where figure will be saved",
    default=os.path.join(mmt_repopath, "figures"),
)
parser.add_argument(
    "--savefig", help="Save the figures instead of plotting them", action="store_true"
)
args = parser.parse_args()

scorename = args.scorename
weights_list = args.weights.split(",")

plt_utils.figureDir = args.figdir
plt_utils.fmtImages = "." + args.figfmt
plt_utils.storeImages = args.savefig


# Open HDF5 files with validation patches
# -----------------------------------------
ldom_data_dir = os.path.join(mmt_repopath, "data", "hdf5_data")
subset = "test"
lcnames = ["esawc", "ecosg", "esgp"]
h5_path = {}
h5f = {}
for lcname in lcnames:
    h5_path[lcname] = os.path.join(ldom_data_dir, f"{lcname}-{subset}.hdf5")
    assert os.path.isfile(h5_path[lcname]), f"File {h5_path[lcname]} does not exist"
    h5f[lcname] = h5py.File(h5_path[lcname], "r", libver="latest")

assert all(
    [
        set(h5f["esawc"].keys()) == set(h5f[lcname].keys())
        for lcname in ["ecosg", "esgp"]
    ]
), "HDF5 keys lists don't match"

n_patches = min(args.npatches, len(h5f["esawc"]))


# Instanciate all translators
# -----------------------------
translator_list = [
    translators.EsawcToEsgp(checkpoint_path=scores.weights_to_checkpoint(weights))
    for weights in weights_list
]


# Compute (or load) confusion matrices
# --------------------------------------
cmxs = {}
for translator in translator_list + ["ecosg"]:
    method = (
        scores.checkpoint_to_weight(translator.checkpoint_path)
        if hasattr(translator, "checkpoint_path")
        else translator
    )
    cmxs[method] = scores.look_in_cache_else_compute(translator, h5f, n_patches)
    print(f"Bulk total overall accuracy ({method}): {scores.oaccuracy(cmxs[method])}")


scores.pprint_oaccuracies(cmxs)

# Plot recall matrices
# ----------------------
for method, cmx in cmxs.items():
    reccmx = scores.norm_matrix(scores.remove_absent_labels(cmx), axis=1)
    plt_utils.plot_confusion_matrix(
        reccmx,
        figtitle=f"Recall matrix (normed by reference) {method}",
        figname=f"reccmx_{method}",
    )


# Per-label scores
# ----------------------
print(f"\n   {scorename.upper()} FOR PRIMARY LABELS")
pms = scores.permethod_scores(
    {k: scores.sum_primary_labels(v) for k, v in cmxs.items()}, scorename
)
scores.latexstyleprint(pms)

print(f"\n   {scorename.upper()} FOR SECONDARY LABELS")
pms = scores.permethod_scores(cmxs, scorename)
scores.latexstyleprint(pms)

# EOF

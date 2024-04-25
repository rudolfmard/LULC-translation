#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Generation of land cover members on few patches

python -i show_ensembles.py --weights v2outofbox2 --u 0.82,0.11,0.47,0.34,0.65
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.inference import translators
from mmt.utils import domains, misc


# Argument parsing
# ----------------
parser = argparse.ArgumentParser(
    prog="show_ensembles",
    description="Generation of land cover members on few patches",
    epilog="Example: python -i show_ensembles.py --weights mmt-weights-v1.0.ckpt --cpu",
)
parser.add_argument(
    "--locations",
    help="Domain names to look at",
    default="snaefell_glacier,nanterre,iso_kihdinluoto,portugese_crops,elmenia_algeria",
)
parser.add_argument(
    "--weights",
    help="Weight file, experience ID or path to the checkpoint to use for inference",
    default="mmt-weights-v1.0.ckpt",
)
parser.add_argument(
    "--u",
    help=f"Values for the random drawing of the ensemble",
    default="0.62,0.29,0.41,0.78,0.09",
)
parser.add_argument(
    "--npx", help="Size of patch (in number of ~10m pixels)", default=900, type=int
)
parser.add_argument("--figfmt", help="Format of the figure", default="png")
parser.add_argument(
    "--figdir",
    help="Directory where figure will be saved",
    default=os.path.join(mmt_repopath, "figures"),
)
parser.add_argument(
    "--savefig", help="Save the figures instead of plotting them", action="store_true"
)
parser.add_argument(
    "--cpu", help="Perform inference on CPU", action="store_true", default=False
)
args = parser.parse_args()

# Default resolution is the one of ESA World Cover (~10m)
res = 8.333e-5
# patch_size = float(args.patchsize)
# n_px = patch_size // lcs[0].res
n_px = args.npx
locations = args.locations.split(",")
device = "cpu" if args.cpu else "cuda"
u_values = [None] + [float(u) for u in args.u.split(",")]
n_members = len(u_values)

tr = translators.EsawcToEsgpShowEnsemble(
    checkpoint_path=misc.weights_to_checkpoint(args.weights), device=device
)
weights = misc.checkpoint_to_weight(tr.checkpoint_path)

# Inference
# ----------------
fig, axs = plt.subplots(len(locations), len(u_values), figsize=(12, 16))
for i, domainname in enumerate(locations):
    dom = getattr(domains, domainname)
    if n_px > 0:
        qb = dom.centred_fixed_size(n_px, res).to_tgbox()
    else:
        qb = dom.to_tgbox()
    
    print(f"Location {domainname} (lon, lat): {dom.central_point()}")
    for j, u_value in enumerate(u_values):
        tr.u = u_value
        x = tr[qb]
        tr.plot(x, figax=(fig, axs[i, j]), show_titles=False, show_colorbar=False)

[ax.axis("off") for ax in axs.ravel()]
cols = []
for j, u_value in enumerate(u_values):
    cols.append(f"Member {j} (u={u_value})")

for ax, col in zip(axs[0], cols):
    ax.set_title(col)

figname = f"_".join(
    [f"ensemble", weights, "-".join([str(u) for u in u_values]), "-".join([loc[:3] for loc in locations])]
)
title = f"Ensemble generation for {weights}"
fig.suptitle(title)
fig.tight_layout()

if args.savefig:
    figpath = os.path.join(args.figdir, f"{figname}.{args.figfmt}")
    fig.savefig(figpath)
    print("Figure saved:", figpath)

plt.show(block=False)

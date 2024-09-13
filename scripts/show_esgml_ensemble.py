#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Generation of land cover members on few patches. Same as drafts/show_ensembles.py but with the ECOSG-ML already exported.

python -i show_esgml_ensemble.py --locations portugese_crops,elmenia_algeria,iziaslav_ukraine,elhichria_tunisia,rural_andalousia --npx 1200
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmt import _repopath_ as mmt_repopath
from mmt.utils import aliases, domains, misc

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(
    prog="show_esgml_ensemble",
    description="Generation of land cover members on few patches",
    epilog="Example: python -i show_esgml_ensemble.py --locations portugese_crops,elmenia_algeria,iziaslav_ukraine,elhichria_tunisia,rural_andalousia --npx 1200",
)
parser.add_argument(
    "--locations",
    help="Domain names to look at",
    default="snaefell_glacier,nanterre,iso_kihdinluoto,portugese_crops,elmenia_algeria",
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
args = parser.parse_args()
print(f"Executing {sys.argv[0]} from {os.getcwd()} with args={args}")

# Default resolution is the one of ESA World Cover (~10m)
res = misc.DEFAULT_RESOLUTION_10M


n_px = args.npx
locations = args.locations.split(",")

# Load land cover maps
# ----------------
qscore = aliases.get_landcover_from_alias("qscore", cutoff=0.3)
esgml = aliases.get_landcover_from_alias("EcoclimapSGML")

u_values = np.zeros(esgml.n_members)
fig, axs = plt.subplots(len(locations), esgml.n_members + 1, figsize=(12, 16))
for i, domainname in enumerate(locations):
    dom = getattr(domains, domainname)
    if n_px > 0:
        qb = dom.centred_fixed_size(n_px, res).to_tgbox()
    else:
        qb = dom.to_tgbox()

    print(f"Location {domainname} (lon, lat): {dom.central_point()}")
    for mb in range(esgml.n_members):
        esgml.member = mb
        u_values[mb] = esgml.u
        x = esgml[qb]
        esgml.plot(x, figax=(fig, axs[i, mb]), show_titles=False, show_colorbar=False)

    qs = qscore[qb]
    qscore.plot(qs, figax=(fig, axs[i, -1]), show_titles=False, show_colorbar=False)

[ax.axis("off") for ax in axs.ravel()]
cols = []
for j, u_value in enumerate(u_values):
    cols.append(f"Member {j} (u={u_value})")

cols.append("QSCORE")

for ax, col in zip(axs[0], cols):
    ax.set_title(col)

figname = "esgml-ensemble_" + "-".join([loc[:3] for loc in locations])
title = f"Ensemble land cover ECOSG-ML"
fig.suptitle(title)
fig.tight_layout()

if args.savefig:
    figpath = os.path.join(args.figdir, f"{figname}.{args.figfmt}")
    fig.savefig(figpath)
    print("Figure saved:", figpath)

plt.show(block=False)

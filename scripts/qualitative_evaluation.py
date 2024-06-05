#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Qualitative evaluation of map translation. Pre-set patches.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.inference import translators
from mmt.utils import aliases, domains, misc

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(
    prog="qualitative_evalution",
    description="Compare a set of maps on a set of patches",
    epilog="Example: python -i qualitative_evaluation.py --lcnames esawc,ecosg,outofbox2,mmt-weights-v2.0.ckpt",
)
parser.add_argument(
    "--locations",
    help="Domain names to look at (see mmt/utils/domains.py). Default are: snaefell_glacier,nanterre,iso_kihdinluoto,portugese_crops,elmenia_algeria",
    default="snaefell_glacier,nanterre,iso_kihdinluoto,portugese_crops,elmenia_algeria",
)
parser.add_argument(
    "--lcnames",
    help="Land cover aliases (esawc, ecosg, esgp, esgml, qflags)",
    default="esawc,ecosg,esgp,esgml,qflags",
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
print(f"Executing {sys.argv[0]} from {os.getcwd()} with args={args}")

n_px = args.npx
locations = args.locations.split(",")
lcnames = args.lcnames.split(",")
device = "cpu" if args.cpu else "cuda"


# Default resolution is the one of ESA World Cover (~10m)
res = misc.DEFAULT_RESOLUTION_10M

# Land cover loading
# --------------------
lcs = []
for lcname in lcnames:
    lcs.append(aliases.get_landcover_from_alias(lcname))

print(f"Landcovers loaded with native CRS and resolution")

# Inference or access to data
# ----------------
fig, axs = plt.subplots(len(locations), len(lcnames), figsize=(12, 16))
for i, domainname in enumerate(locations):
    dom = getattr(domains, domainname)
    if n_px > 0:
        qb = dom.centred_fixed_size(n_px, res).to_tgbox()
    else:
        qb = dom.to_tgbox()

    print(f"Location {domainname} (lon, lat): {dom.central_point()}")
    for j, lc in enumerate(lcs):
        x = lc[qb]
        lc.plot(x, figax=(fig, axs[i, j]), show_titles=False, show_colorbar=False)


lcattrs = aliases.LANDCOVER_ALIASES
[ax.axis("off") for ax in axs.ravel()]
cols = []
for lcname in lcnames:
    if lcname in lcattrs.keys():
        cols.append(lcattrs[lcname]["colname"])
    else:
        cols.append(lcname)

for ax, col in zip(axs[0], cols):
    ax.set_title(col)

figname = f"_".join(
    ["qualcheck", "-".join(lcnames), "-".join([loc[:3] for loc in locations])]
)
fig.suptitle(figname)
fig.tight_layout()

if args.savefig:
    figpath = os.path.join(args.figdir, f"{figname}.{args.figfmt}")
    fig.savefig(figpath)
    print("Figure saved:", figpath)

plt.show(block=False)

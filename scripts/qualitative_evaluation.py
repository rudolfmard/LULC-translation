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
from mmt.datasets import landcovers, transforms
from mmt.inference import translators
from mmt.utils import domains, misc

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(
    prog="qualitative_evalution",
    description="Compare a set of maps on a set of patches",
    epilog="Example: python -i qualitative_evaluation.py --lcnames esawc,ecosg,outofbox2,mmt-weights-v1.0.ckpt --cpu",
)
parser.add_argument(
    "--locations",
    help="Domain names to look at",
    default="snaefell_glacier,nanterre,iso_kihdinluoto,portugese_crops,elmenia_algeria",
)
parser.add_argument(
    "--lcnames",
    help="Land cover maps short names (esawc, ecosg, esgp, esgml, qflags)",
    default="esawc,ecosg,esgp,esgml,qflags",
)
# parser.add_argument("--patchsize", help="Size of patch (in the first land cover map CRS). Put 0 to avoid cropping", default=0.08333)
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

# patch_size = float(args.patchsize)
# n_px = patch_size // lcs[0].res
n_px = args.npx
locations = args.locations.split(",")
lcnames = args.lcnames.split(",")
device = "cpu" if args.cpu else "cuda"

lcattrs = {
    "esawc": {
        "lcclass": "ESAWorldCover",
        "kwargs": {
            "transforms": transforms.EsawcTransform(),
        },
        "colname": "ESAWC",
    },
    "ecosg": {
        "lcclass": "EcoclimapSG",
        "kwargs": {},
        "colname": "ECOSG",
    },
    "esgp": {
        "lcclass": "EcoclimapSGplus",
        "kwargs": {},
        "colname": "ECOSG+",
    },
    "esgpv2": {
        "lcclass": "EcoclimapSGplusV2",
        "kwargs": {},
        "colname": "ECOSG+v2",
    },
    "bguess": {
        "lcclass": "SpecialistLabelsECOSGplus",
        "kwargs": {},
        "colname": "BGUESS",
    },
    "esgml": {
        "lcclass": "EcoclimapSGML",
        "kwargs": {},
        "colname": "ECOSG-ML",
    },
    "qflags": {
        "lcclass": "QualityFlagsECOSGplus",
        "kwargs": {
            "transforms": transforms.FillMissingWithSea(0, 6),
        },
        "colname": "QFLAGS",
    },
    "qscore": {
        "lcclass": "ScoreECOSGplus",
        "kwargs": {
            "transforms": transforms.ScoreTransform(divide_by=100),
        },
        "colname": "QSCORE",
    },
}
# Default resolution is the one of ESA World Cover (~10m)
res = 8.333e-5

# Land cover loading
# --------------------
lcs = []
for lcname in lcnames:
    if lcname in lcattrs.keys():
        lc_class = getattr(landcovers, lcattrs[lcname]["lcclass"])
        lcs.append(lc_class(**lcattrs[lcname]["kwargs"]))
    elif lcname == "ECOSG-ML":
        tr = translators.EsawcToEsgpShowEnsemble(
            checkpoint_path=misc.weights_to_checkpoint("v2outofbox2"), device=device
        )
        lcs.append(tr)
    else:
        tr = translators.EsawcToEsgpAsMap(
            checkpoint_path=misc.weights_to_checkpoint(lcname), device=device
        )
        lcs.append(tr)

print(f"Landcovers loaded with native CRS and resolution")

# Inference
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

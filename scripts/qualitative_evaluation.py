#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Qualitative evaluation of map translation. Pre-set patches.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import argparse

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import transforms
from mmt.datasets import landcovers
from mmt.utils import domains

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(prog="qualitative_evalution", description="Compare a set of maps on a set of patches")
parser.add_argument("--locations", help="Domain names to look at", default="snaefell_glacier,nanterre,iso_kihdinluoto,portugese_crops,elmenia_algeria")
parser.add_argument("--lcnames", help="Land cover maps short names (esawc, ecosg, esgp, esgml, qflags)", default="esawc,ecosg,esgp,esgml,qflags")
parser.add_argument("--patchsize", help="Size of patch (in the first land cover map CRS). Put 0 to avoid cropping", default=0.08333)
parser.add_argument("--figfmt", help="Format of the figure", default="png")
parser.add_argument("--figdir", help="Directory where figure will be saved", default=os.path.join(mmt_repopath, "figures"))
parser.add_argument("--savefig", help="Save the figures instead of plotting them", action = "store_true")
args = parser.parse_args()

patch_size = float(args.patchsize)
locations = args.locations.split(",")
lcnames = args.lcnames.split(",")

lcattrs = {
    "esawc":{
        "lcclass": "ESAWorldCover",
        "kwargs":{
            "transforms": transforms.EsawcTransform(),
        },
        "colname":"ESAWC",
    },
    "ecosg":{
        "lcclass": "EcoclimapSG",
        "kwargs":{},
        "colname":"ECOSG",
    },
    "esgp":{
        "lcclass": "EcoclimapSGplus",
        "kwargs":{},
        "colname":"ECOSG+",
    },
    "esgml":{
        "lcclass": "EcoclimapSGML",
        "kwargs":{},
        "colname":"ECOSG-ML",
    },
    "qflags":{
        "lcclass": "QualityFlagsECOSGplus",
        "kwargs":{
            "transforms":transforms.FillMissingWithSea(0,6),
        },
        "colname":"QFLAGS",
    },
}


# Land cover loading
#--------------------
lcs = []
for lcname in lcnames:
    lc_class = getattr(landcovers, lcattrs[lcname]["lcclass"])
    lcs.append(lc_class(**lcattrs[lcname]["kwargs"]))

print(f"Landcovers loaded with native CRS and resolution")
n_px = patch_size // lcs[0].res

# Inference
#----------------
fig, axs = plt.subplots(len(locations), len(lcnames), figsize = (12,16))
for i, domainname in enumerate(locations):
    dom = getattr(domains, domainname)
    if n_px > 0:
        qb = dom.centred_fixed_size(n_px, lcs[0].res).to_tgbox()
    else:
        qb = dom.to_tgbox()
    
    print(f"Location {domainname} (lon, lat): {dom.central_point()}")
    for j, lc in enumerate(lcs):
        x = lc[qb]
        lc.plot(x, figax = (fig, axs[i, j]), show_titles=False, show_colorbar=False)
    
    
[ax.axis("off") for ax in axs.ravel()]
cols = [lcattrs[lcname]["colname"] for lcname in lcnames]
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

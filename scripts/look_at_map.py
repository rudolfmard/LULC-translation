#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Have look at maps.


Examples
--------
python -i look_at_map.py --lcname=EcoclimapSGplus --domainname=eurat --res=0.1
    Plot ECOSG+ over the EURAT domain at 0.1 degree resolution and leave missing data unchanged

python -i look_at_map.py --lcname=ESAWorldCover --domainname=dublin_city
    Plot ESA World Cover over Dublin city at native resolution

python -i look_at_map.py --lcpath=/path/to/tif-files/ --domainname=ireland25 --res=0.01
    Plot whatever is in the specified dir (expecting TIF files with ECOSG labels) on the domain ireland25 at 0.01 resolution
"""

import argparse
import os
import sys
from pprint import pprint

import numpy as np
from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import aliases, domains

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(
    prog="look_at_map",
    description="Plot a map on a given domain at a given resolution",
    epilog="Example: python -i look_at_map.py --lcname=EcoclimapSGplus --domainname=eurat --res=0.1",
)
parser.add_argument(
    "--lcname",
    help="Land cover alias: class full name (see mmt/datasets/landcovers.py) or short name (see mmt/utils/aliases.py), directory with TIF files or checkpoint",
)
parser.add_argument(
    "--domainname", help="Geographical domain name (see mmt/utils/domains.py)"
)
parser.add_argument("--res", help="Resolution of the map (in degree)", default=None)
parser.add_argument(
    "--fillsea",
    help="Replace missing data by sea",
    action="store_true",
)
parser.add_argument(
    "--fillneighbors",
    help="Replace missing data by the most common label in the neighborhood",
    action="store_true",
)
parser.add_argument(
    "--other-kwargs",
    help="Additional arguments for the landcover init (ex: --lcname ScoreECOSGplus --other-kwargs cutoff=0.3,tgeo_init=False)",
    dest="okwargs",
)
parser.add_argument(
    "--npx", help="Size of patch (in number of ~10m pixels)", default=0, type=int
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

# Storing arguments in variables
lcname = args.lcname
domainname = args.domainname
res = args.res
fillsea = args.fillsea
fillneighbors = args.fillneighbors
n_px = args.npx

# Creating kwargs for the landcover class init
if args.okwargs is not None:
    kwargs = {e.split("=")[0]: eval(e.split("=")[1]) for e in args.okwargs.split(",")}
else:
    kwargs = dict()

if fillsea:
    kwargs.update(transforms=mmt_transforms.FillMissingWithSea(0, 1))

if fillneighbors:
    kwargs.update(transforms=mmt_transforms.FillMissingWithNeighbors(0, 1))

if res:
    res = float(res)
    kwargs.update(res=res)


# Loading map
# -----------
print(f"Loading {lcname} with")
lcmap = aliases.get_landcover_from_alias(lcname, print_kwargs=True, **kwargs)
print(f"Loaded: {lcmap.__class__.__name__} with crs={lcmap.crs}, res={lcmap.res}")

if domainname:
    qdomain = getattr(domains, domainname)
    if n_px > 0:
        qdomain = qdomain.centred_fixed_size(n_px, lcmap.res)

    qb = qdomain.to_tgbox(lcmap.crs)
else:
    qb = lcmap.bounds
    domainname = "all"


# Plotting map
# ------------
x_map = lcmap[qb]
fig, ax = lcmap.plot(x_map)
if args.savefig:
    figpath = os.path.join(
        args.figdir,
        f"{lcmap.__class__.__name__}_res{round(lcmap.res, 4)}_{domainname}.{args.figfmt}",
    )
    fig.savefig(figpath)
    print("Figure saved:", figpath)

fig.show()

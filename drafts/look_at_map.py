#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Have look at maps

Examples
--------
python -i look_at_map.py --lcname=EcoclimapSGplus --domainname=eurat --res=0.1 --no-fillsea
    Plot ECOSG+ over the EURAT domain at 0.1 degree resolution and leave missing data unchanged

python -i look_at_map.py --lcname=ESAWorldCover --domainname=dublin_city
    Plot ESA World Cover over Dublin city at native resolution

python -i look_at_map.py --lcpath=/data/trieutord/MLULC/outputs/ECOSGML-v0.4-onelabel.02Nov-13h13 --domainname=ireland25 --res=0.01
    Plot whatever is in the specified dir (expecting TIF files with ECOSG labels) on the domain ireland25 at 0.01 resolution
"""

import os
import numpy as np
from pprint import pprint
import argparse

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import domains

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(prog="look_at_map")
parser.add_argument("--lcname", help="Land cover class name")
parser.add_argument("--domainname", help="Geographical domain name")
parser.add_argument("--res", help="Resolution of the map (degree)", default=None)
parser.add_argument("--lcpath", help="Path to the directory with land cover TIF files. If provided, lcname will be forced to InferenceResults.", default="")
parser.add_argument('--no-fillsea', help="Do not replace missing data by sea", dest='fillsea', action='store_false')
args = parser.parse_args()

lcname = args.lcname
domainname = args.domainname
res = args.res
lcpath = args.lcpath
fillsea = args.fillsea

kwargs = dict()

if os.path.exists(lcpath):
    lcname = "InferenceResults"
    kwargs.update(path = lcpath)
else:
    assert lcname not in [None, "InferenceResults"], f"Conflicting arguments: lcpath={lcpath} does not exist and lcname={lcname}"

if lcname == "ESAWorldCover":
    kwargs.update(transforms=mmt_transforms.EsawcTransform)
    fillsea = False

if fillsea:
    kwargs.update(transforms=mmt_transforms.FillMissingWithSea(0,1))

if res:
    res = float(res)
    kwargs.update(res = res)

# Loading map
# -----------
print(f"Loading {lcname} with")
pprint(kwargs)
lc_class = getattr(landcovers, lcname)
lcmap = lc_class(**kwargs)
print(f"Loaded: {lcmap.__class__.__name__} with crs={lcmap.crs}, res={lcmap.res}")

if domainname:
    qdomain = getattr(domains, domainname)
    qb = qdomain.to_tgbox(lcmap.crs)
else:
    qb = lcmap.bounds
    domainname = "all"

# Plotting map
# ------------
x_map = lcmap[qb]
fig, ax = lcmap.plot(x_map)
figpath = os.path.join(mmt_repopath, "figures", f"{lcmap.__class__.__name__}__{lcmap.res}__{domainname}.png")
fig.savefig(figpath)
print("Figure saved:", figpath)
fig.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to export land cover to HDR/DIR format (SURFEX readable)


Examples
--------
python export_landcover_to_hdrdir.py --lcpath=/data/trieutord/MLULC/outputs/ECOSGML-v0.4-onelabel.02Nov-13h13 --output=test/hector.dir --domainname=irl750
    Export the domain 'irl750' of the land cover stored in lcpath (multiple TIF files) into the file output at native resolution

python export_landcover_to_hdrdir.py --lcpath=/data/trieutord/MLULC/outputs/ECOSGML-v0.4-onelabel.02Nov-13h13 --output=test/ireland25.dir --res=0.01
    Export the full domain of the land cover stored in lcpath (multiple TIF files) into the file output at 0.01 resolution

python export_landcover_to_hdrdir.py --lcpath=/data/trieutord/ECOCLIMAP-SG-ML/tif/ECOCLIMAP-SG-ML-eurat-v0.3 --output=/data/trieutord/ECOCLIMAP-SG-ML/dir-hdr/irl750-ecosg/COVER_ECOSGML_2023_v0.3.dir --domainname=irl750 --res=EcoclimapSG
    Export the domain 'irl750' of the land cover stored in lcpath (multiple TIF files) into the file output at the same resolution as EcoclimapSG
"""

import os
import numpy as np
from pprint import pprint
import argparse

from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import domains

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(prog="export_landcover_to_hdrdir")
parser.add_argument("--lcpath", help="Path to the directory with land cover TIF files.", type=str)
parser.add_argument("--output", help="Output file name (full path)", type=str)
parser.add_argument("--domainname", help="Geographical domain name")
parser.add_argument("--res", help="Resolution of the map (degree)", default=None)
parser.add_argument('--no-fillsea', help="Do not replace missing data by sea", dest='fillsea', action='store_false')
args = parser.parse_args()

lcpath = args.lcpath
output = args.output
domainname = args.domainname
res = args.res
fillsea = args.fillsea

# Argument checks
# ----------------
assert os.path.exists(lcpath), f"Path {lcpath} does not exist"

output_path = os.path.dirname(output)
if not os.path.exists(output_path):
    os.makedirs(output_path)

print(f"Exporting to {output}")

kwargs = dict(path = lcpath)

if fillsea:
    kwargs.update(transforms=mmt_transforms.FillMissingWithSea(0,1))

if res:
    changeres = True
    if hasattr(landcovers, res):
        copyresfrom = getattr(landcovers, res)()
        res = copyresfrom.res
    else:
        res = float(res)
        kwargs.update(res = res)
else:
    changeres = False

# Loading map
# -----------
lcmap = landcovers.InferenceResults(**kwargs)
if changeres:
    lcmap.res = res

print(f"Loaded: {lcmap.__class__.__name__} with crs={lcmap.crs}, res={lcmap.res}")

if domainname:
    qdomain = getattr(domains, domainname)
    qb = qdomain.to_tgbox(lcmap.crs)
else:
    qb = lcmap.bounds
    domainname = "all"

# Exporting map
# -------------
x_map = lcmap[qb]
ofn_dir, ofn_hdr = lcmap.export_to_dirhdr(x_map, ofn_dir = output)
print(f"Files created: {ofn_dir}, {ofn_hdr}")

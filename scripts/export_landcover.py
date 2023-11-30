#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to export land cover to HDR/DIR format (SURFEX readable)


Examples
--------
python export_landcover.py --lcname=EcoclimapSGML --output=test/hector.dir --domainname=irl750
    Export in raw binary the domain 'irl750' of the land cover stored in lcpath (multiple TIF files) into the file output at native resolution

python export_landcover.py --lcname=EcoclimapSGML --output=test/ireland25.nc --res=0.01
    Export in netCDF the full domain of the land cover stored in lcpath (multiple TIF files) into the file output at 0.01 resolution

python export_landcover.py --lcpath=/data/trieutord/ECOCLIMAP-SG-ML/tif/ECOCLIMAP-SG-ML-eurat-v0.3 --output=/data/trieutord/ECOCLIMAP-SG-ML/dir-hdr/irl750-ecosg/COVER_ECOSGML_2023_v0.3.dir --domainname=irl750 --res=EcoclimapSG
    Export the domain 'irl750' of the land cover stored in lcpath (multiple TIF files) into the file output at the same resolution as EcoclimapSG

python export_landcover.py --lcname=EcoclimapSGML --output=/data/trieutord/ECOCLIMAP-SG-ML/netcdf/COVER_ECOSGML_2023_v0.6.nc
    Export the domain 'irl750' of the land cover stored in lcpath (multiple TIF files) into the file output at the same resolution as EcoclimapSG
"""

import os
import psutil
import numpy as np
from pprint import pprint
import argparse

from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import domains

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(prog="export_landcover")
parser.add_argument("--lcname", help="Land cover class name")
parser.add_argument("--lcpath", help="Path to the directory with land cover TIF files.", default="", type=str)
parser.add_argument("--output", help="Output file name (full path)", type=str)
parser.add_argument("--domainname", help="Geographical domain name")
parser.add_argument("--res", help="Resolution of the map (degree)", default=None)
parser.add_argument('--no-fillsea', help="Do not replace missing data by sea", dest='fillsea', action='store_false')
args = parser.parse_args()

lcname = args.lcname
lcpath = args.lcpath
output = args.output
domainname = args.domainname
res = args.res
fillsea = args.fillsea


# Argument checks
# ----------------
assert output.endswith(".dir") or output.endswith(".nc"), f"Output format is not recognized. Please use .nc for netCDF4 or .dir for raw binary"
print(f"Exporting to {output}")

kwargs = dict()

if os.path.exists(lcpath):
    lcname = "InferenceResults"
    kwargs.update(path = lcpath)
else:
    assert lcname not in [None, "InferenceResults"], f"Conflicting arguments: lcpath={lcpath} does not exist and lcname={lcname}"

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
lc_class = getattr(landcovers, lcname)
assert hasattr(lc_class, "export_to_netcdf"), f"The class {lcname} has no export method"

lcmap = lc_class(**kwargs)
if changeres:
    lcmap.res = res

if domainname:
    qdomain = getattr(domains, domainname)
    qb = qdomain.to_tgbox(lcmap.crs)
else:
    qb = lcmap.bounds
    domainname = "all"

estimated_ram = lcmap.get_bytes_for_domain(qb)
available_ram = psutil.virtual_memory()
available_ram = available_ram.available

print(f"Loaded: {lcmap.__class__.__name__} with crs={lcmap.crs}, res={lcmap.res}. Expecting {estimated_ram/10**9} GB of data (available {available_ram/10**9} GB)")

if estimated_ram > 0.95 * available_ram:
    raise Exception("You are asking for too much memory. Please reduce the domain or resolution")
if estimated_ram > 0.6 * available_ram:
    answer = input(f"You are about to load {estimated_ram/10**9} GB of data, which more than 60% of your available RAM. Are you sure you want to continue?")
    if answer.lower() not in ["y", "yes", "oui"]:
        raise Exception("Stopped after warning about memory usage.")
    else:
        print("Extracting now...")

# Exporting map
# -------------
output_path = os.path.dirname(output)
if not os.path.exists(output_path) and len(output_path) > 0:
    os.makedirs(output_path)

x_map = lcmap[qb]
print(f"Data loaded (shape = {x_map['mask'].shape}). Now writing output")
if output.endswith(".nc"):
    ofn = lcmap.export_to_netcdf(x_map, output)
else:
    ofn = lcmap.export_to_dirhdr(x_map, output)

print(f"Files created: {ofn}")

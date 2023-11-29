#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to release a new version of ECOCLIMAP-SG-ML from existing inference results

The release consists in copying the outputs to a clean, well organized, format.
Two formats are stored: the set of ~200 TIF files and the ZIP archive of these same files.


Examples
--------
python release_ecosgml_version.py --lcpath=/data/trieutord/MLULC/outputs/ECOSGML-v0.5-wrt3pl.03Nov-22h00 --vtag=v0.5 --releasenotes="Short description + from experiment 'vanilla_eurat3' with mmt-0.2.0 and ECOCLIMAP-SG+ v0.3.e2"
    Copy the files in `lcpath` to the archive directory as the version with tag "v0.5".
"""

import os
import shutil
import time
import numpy as np
from pprint import pprint
import argparse

from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import domains

rootarchivedir = "/data/trieutord/ECOCLIMAP-SG-ML/"

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(prog="release_ecosgml_version")
parser.add_argument("--lcpath", help="Path to the directory with land cover TIF files.", type=str)
parser.add_argument("--vtag", help="Version tag (e.g. v0.5).", type=str)
parser.add_argument("--releasenotes", help="Short description of the changes in this version", type=str)
args = parser.parse_args()

lcpath = args.lcpath
vtag = args.vtag
releasenotes = args.releasenotes

# Argument checks
# ----------------
assert os.path.exists(lcpath), f"Path {lcpath} does not exist"

nametag = f"ECOCLIMAP-SG-ML-{vtag}"
output_tif_path = os.path.join(rootarchivedir, "tif", nametag)
output_zip_file = os.path.join(rootarchivedir, "zip", nametag)

exist = [os.path.exists(p) for p in [output_tif_path, output_zip_file + ".zip"]]
if any(exist):
    raise ValueError(f"This version ({vtag}) already exist ({exist}). Please the change the version tag or rename the existing directories.")

# Loading the map
# ----------------
lcmap = landcovers.InferenceResults(path = lcpath, tgeo_init = True)
print(f"Loaded: {lcmap.__class__.__name__} with crs={lcmap.crs}, res={lcmap.res}")

try:
    lcmap.get_version()
except AssertionError:
    print(f"Version file is missing. Created now with the release notes as description")
    version_file = os.path.join(lcpath, "version-infos.txt")
    with open(version_file, "w") as f:
        f.write(f'export_date="{time.ctime()}"\n')
        f.write(f'version="{vtag}"\n')
        f.write(f'description="{releasenotes}"\n')
        f.write(f'domain="EURAT"\n')
        f.write(f'crs="{lcmap.crs}"\n')

add_to_log = f"""
Version {vtag}
--------------
{releasenotes}

Additional infos:
  * release date = {time.ctime()}
  * source directory = {lcpath}
  * crs = {lcmap.crs}
  * domain = {lcmap.bounds}
  * resolution = {lcmap.res}
"""
print(add_to_log)

# Copying map
# -----------
print(f"Copying TIF files from {lcpath} to {output_tif_path}")
shutil.copytree(lcpath, output_tif_path)

# Zipping map
# -----------
print(f"Zipping TIF files from {lcpath} to {output_zip_file}")
shutil.make_archive(output_zip_file, "zip", lcpath)

# Updating version log
# -----------
versionlog = os.path.join(rootarchivedir, "versions-log.md")
with open(versionlog, "a") as vl:
    vl.write(add_to_log)

print("Updated: ", versionlog)

prologue = f"""
Version {vtag} of ECOSG-ML successfully released locally!

Next steps?
-----------
Export it to hdr/dir format (SURFEX readable):
    python export_landcover_to_hdrdir.py --lcpath={output_tif_path} --output=<path> --domainname=<within eurat> --res=<degree>

Push everything to ECMWF HPC:
    rsync -avz {rootarchivedir}/ hpc-login:/perm/<ec id>/data/ECOCLIMAP-SG-ML
"""
print(prologue)

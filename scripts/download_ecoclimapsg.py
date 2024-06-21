#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to download ECOCLIMAP-SG.

Download ECOCLIMAP-SG:
    wget --user ecoclimap --password ecoclimap ftp://ftp.umr-cnrm.fr/ECOCLIMAP-SG/V0/COVER/COVER_ECOSG_2010_V0.1.tgz
    tar xzf COVER_ECOSG_2010_V0.1.tgz

Estimate times
--------------
Download    2 minutes
Un-tar      1 minute
Extract     <1 minute
TOTAL:      3 minutes
"""
import argparse
import os
import time
import rasterio
import numpy as np
import pandas as pd
from pprint import pprint
from mmt.utils import domains, misc

parser = argparse.ArgumentParser(prog="download_ecoclimapsg", description="Download ECOCLIMAP-SG and extract data into a TIF file for the EURAT domain", epilog="Example: python download_ecoclimapsg.py --landingdir /data/trieutord/ECOCLIMAP-SG/tmp")
parser.add_argument("--landingdir", help="Path to the directory where the ECOCLIMAP-SG data will be stored.")
parser.add_argument("--tmpdir", help="Directory for temporary files.", default="")
args = parser.parse_args()

def create_tmp_files(orig_dir_file, tmpdir=args.tmpdir):
    """Create temporary files and links for the cropping of ECO-SG

    As the original files (a pair .hdr .dir) cannot be read by rasterio
    as is, this function tricks the reader with temporary, rasterio-readable, files.


    Parameters
    ----------
    tmpdir: str
        Path to the directory that will contain the temporary files.


    Returns
    -------
    tmp_fn_dir: str
        Temporary file name of the DIR file

    tmp_fn_hdr: str
        Temporary file name of the HDR file


    Notes
    -----
    Inspired from Geoffrey Bessardon's work: https://github.com/gbessardon/Cropecosg
    """
    fn_dir = os.path.basename(orig_dir_file)
    uid = misc.id_generator(forbidden="dir")
    fn_dir = fn_dir.replace(".dir", f".{uid}.dir")
    fn_hdr = fn_dir.replace(".dir", ".hdr")

    tmp_fn_dir = os.path.join(tmpdir, fn_dir)
    tmp_fn_hdr = os.path.join(tmpdir, fn_hdr)

    # Write the temporary DIR file
    if not os.path.lexists(tmp_fn_dir):
        os.symlink(orig_dir_file, tmp_fn_dir)

    # Write the temporary HDR file
    if not os.path.exists(tmp_fn_hdr):
        dictecosg = pd.read_csv(
            orig_dir_file.replace(".dir", ".hdr"),
            delimiter=":",
            names=["characteristics", "value"],
            index_col="characteristics",
            skiprows=1,
        ).to_dict()["value"]

        dictp = {}
        dictp["nodata"] = dictecosg["nodata"]
        dictp["nrows"] = dictecosg["rows"]
        dictp["ncols"] = dictecosg["cols"]
        dictp["ULXMAP"] = dictecosg["west"]
        dictp["ULYMAP"] = dictecosg["north"]
        dictp["XDIM"] = str(
            abs(
                (float(dictecosg["west"]) - float(dictecosg["east"]))
                / float(int(dictecosg["cols"]))
            )
        )
        dictp["YDIM"] = str(
            abs(
                (float(dictecosg["north"]) - float(dictecosg["south"]))
                / float(int(dictecosg["rows"]))
            )
        )
        with open(tmp_fn_hdr, "w+") as f:
            for k, v in dictp.items():
                f.write(k + " " + v + "\n")
            f.close()

    return tmp_fn_dir, tmp_fn_hdr

def remove_tmp_files(tmp_fn_dir, tmp_fn_hdr):
    """Remove the temporary files and links after the cropping of ECO-SG"""
    os.remove(tmp_fn_hdr)
    os.unlink(tmp_fn_dir)

def extract_domain_to_tif(domain, orig_dir_file, output_tif_file):
    """Extract a subdomain of the global data and store if into a TIF file


    Parameters
    ----------
    domain: tuple or `wopt.domains.GeoRectangle`
        Geographical domain of type `domain = lonmin, lonmax, latmin, latmax`

    output_tif_file: str
        Path and name of the TIF to be created


    Returns
    -------
    None. Create the TIF file at the specified location


    Notes
    -----
    Inspired from Geoffrey Bessardon's work: https://github.com/gbessardon/Cropecosg
    """

    if hasattr(domain, "to_llmm"):
        xmin, xmax, ymin, ymax = domain.to_llmm()
    else:
        xmin, xmax, ymin, ymax = domain

    tmp_fn_dir, tmp_fn_hdr = create_tmp_files(orig_dir_file)

    with rasterio.open(tmp_fn_dir) as src:
        w = src.window(xmin, ymin, xmax, ymax).round_offsets().round_lengths()
        cropdata = src.read(1, window=w)
        trans = rasterio.windows.transform(w, src.transform)

        trg = rasterio.open(
            output_tif_file,
            mode="w",
            Driver="gTiff",
            width=w.width,
            height=w.height,
            count=src.count,
            dtype=np.int16,
            crs={"init": "EPSG:4326"},
            transform=trans,
            nodata=src.nodata,
            compress="lzw",
        )
        trg.write(cropdata, 1)
        trg.close()

    remove_tmp_files(tmp_fn_dir, tmp_fn_hdr)

# landingdir_ecosg = os.path.join(mmt_repopath, "data", "tiff_data", "ECOCLIMAP-SG")
landingdir_ecosg = args.landingdir

if not os.path.isdir(landingdir_ecosg):
    os.makedirs(landingdir_ecosg)
    print(f"Creating directory {landingdir_ecosg}")

# Download
#----------
print(f" ==== Start download: {time.ctime()}")
url = "ftp://ftp.umr-cnrm.fr/ECOCLIMAP-SG/V0/COVER/COVER_ECOSG_2010_V0.1.tgz"
ecosg_tgz = os.path.join(landingdir_ecosg, os.path.basename(url))
if not os.path.isfile(ecosg_tgz):
    wget_cmd = f"wget --user ecoclimap --password ecoclimap {url} -O {ecosg_tgz}"
    print(wget_cmd)
    os.system(wget_cmd)
    print(f"ECOCLIMAP-SG downloaded in {ecosg_tgz}")
else:
    print(f"Already existing {ecosg_tgz}. Skip download")

# Un-tar
#--------
print(f" ==== Start un-tar: {time.ctime()}")
main_file_ecosg = ecosg_tgz.replace(".tgz", ".dir")
if not os.path.isfile(main_file_ecosg):
    tar_cmd = f"tar xvzf {ecosg_tgz} -C {landingdir_ecosg}"
    print(tar_cmd)
    os.system(tar_cmd)
    print(f"ECOCLIMAP-SG extracted in {landingdir_ecosg}:")
    pprint(os.listdir(landingdir_ecosg))
else:
    print(f"Already existing {main_file_ecosg}. Skip un-tar")

# Extract domain
#----------------
print(f" ==== Start extraction: {time.ctime()}")
domain = domains.eurat.enlarge(0.05)
tif_file_ecosg = "ECOCLIMAP-SG-Eurat.tif"
tif_file_ecosg_path = os.path.join(landingdir_ecosg, tif_file_ecosg)

if not os.path.isfile(tif_file_ecosg_path):
    extract_domain_to_tif(domain, main_file_ecosg, tif_file_ecosg_path)
    print(f"ECOCLIMAP-SG extracted over domain {domain} in file {tif_file_ecosg_path}")
else:
    print(f"Already existing {tif_file_ecosg_path}. Skip extraction")

print(f" ==== End of program: {time.ctime()}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Plot patches in HDF5 files


Example
-------
python plot_patches_hdf5_files.py --h5file ../data/hdf5_data/esgp.hdf5 --idxs 4,57,63,112
"""
import argparse
import os

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt

from mmt.utils import aliases


parser = argparse.ArgumentParser(
    prog="plot_patches_hdf5_files",
    description="Plot patches in HDF5 files",
    epilog="Example: python plot_patches_hdf5_files.py --h5file ../data/hdf5_data/esgp.hdf5 --idxs 4,57,63,112",
)
parser.add_argument(
    "--h5file",
    help="Path to the HDF5 file",
)
parser.add_argument(
    "--idxs",
    help="Indices of the patches to plot",
    default="4,57,63,112"
)
args = parser.parse_args()


# idxs = [int(i) for i in args.idxs.split(",")]
idxs = args.idxs.split(",")
assert len(idxs)==4, "Please provide exactly 4 indices"


h5base = os.path.basename(args.h5file)
lcname = h5base.split("-")[0] if "-" in h5base else h5base.split(".")[0]
lc = lcmap = aliases.get_landcover_from_alias(lcname)


h5f = h5py.File(args.h5file, "r", swmr=True, libver='latest')
fig, axs = plt.subplots(2, 2, figsize=(12, 16))

for i, ax in enumerate(axs.ravel()):
    idx = idxs[i]
    x = h5f.get(idx)
    x = torch.tensor(x[:])
    print(f"Accessing patch {idx}: {x.shape}")
    lc.plot({"mask":x}, figax=(fig, ax), show_titles=True, title = f"Patch #{idx}")

h5f.close()

fig.tight_layout()
fig.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Plot patches in HDF5 files


Example
-------
python -i plot_patches_hdf5_files2.py --h5dir ../data/hdf5_data --idx 57 --ds2-subset val
"""
import argparse
import os

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt

from mmt.utils import aliases


parser = argparse.ArgumentParser(
    prog="plot_patches_hdf5_files2",
    description="Plot patches in HDF5 files",
    epilog="Example: python -i plot_patches_hdf5_files2.py --h5dir ../data/hdf5_data --idx 57 --ds2-subset val",
)
parser.add_argument(
    "--h5dir",
    help="Directory of the HDF5 files",
)
parser.add_argument(
    "--idx",
    help="Indices of the patches to plot",
)
parser.add_argument(
    "--ds2-subset",
    help="Dataset DS2 subset (train, test, val). If None, takes DS1",
    dest="ds2subset",
)
args = parser.parse_args()


if args.idx is None:
    if args.ds2subset:
        h5file = os.path.join(args.h5dir, f"ecosg-{args.ds2subset}.hdf5")
    else:
        h5file = os.path.join(args.h5dir, f"ecosg.hdf5")
    
    with h5py.File(h5file, "r", swmr=True, libver='latest') as h5f:
        idx = np.random.choice(list(h5f.keys()))
else:
    idx = args.idx


fig, axs = plt.subplots(2, 2, figsize=(12, 16))

for i, ax, lcname in zip(range(4), axs.ravel(), ["ecosg", "esgp", "esawc", ""]):
    if len(lcname) == 0:
        continue
    
    if args.ds2subset:
        h5file = os.path.join(args.h5dir, f"{lcname}-{args.ds2subset}.hdf5")
    else:
        h5file = os.path.join(args.h5dir, f"{lcname}.hdf5")
    
    h5f = h5py.File(h5file, "r", swmr=True, libver='latest')
    x = h5f.get(idx)
    x = torch.tensor(x[:])
    print(f"Accessing patch {idx} from {lcname}: {x.shape}")
    
    lc = lcmap = aliases.get_landcover_from_alias(lcname)
    lc.plot({"mask":x}, figax=(fig, ax), show_titles=True, title = f"{lcname} #{idx}")

    h5f.close()

fig.tight_layout()
fig.show()


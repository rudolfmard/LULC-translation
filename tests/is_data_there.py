#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Check if the data is at the correct location and in the correct format.


Examples
--------
If you want to visualize land covers
    python is_data_there.py --tiff

If you want to make inference
    python is_data_there.py --tiff --weights

If you want to train the model
    python is_data_there.py --tiff --weights --hdf5


Organisation of the data folder
-----------
data
 ├── outputs        -> where the inference output will be stored
 |
 ├── saved_models   -> where the model checkpoints are stored.
 |
 ├── tiff_data      -> where the original land cover maps are stored in TIF format
 |   ├── ECOCLIMAP-SG
 |   ├── ECOCLIMAP-SG-ML
 |   ├── ECOCLIMAP-SG-plus
 |   └── ESA-WorldCover-2021
 |
 └── hdf5_data      -> where the training data is stored
     ├── ecosg.hdf5
     ├── ecosg-train.hdf5
     ├── ecosg-test.hdf5
     ├── ecosg-val.hdf5
     ├── esawc.hdf5
     └── ...
"""

import argparse
import os

from mmt import _repopath_ as mmt_repopath

parser = argparse.ArgumentParser(prog="is_data_there", description='Check if the data is at the correct location and in the correct format.')
parser.add_argument('--tiff', help="Check if land cover maps are correctly stored in TIF files. Needed to visualize land covers.", action='store_true')
parser.add_argument('--weights', help="Check if model checkpoints are correctly stored. Needed to make inference.", action='store_true')
parser.add_argument('--hdf5', help="Check if training data is correctly stored in HDF5 files. Needed to train the model.", action='store_true')
parser.add_argument('--all', help="Set --tiff --weights --hdf5 to True. Default behaviour when no arg is given", action='store_true')
args = parser.parse_args()

tiff = args.tiff
weights = args.weights
hdf5 = args.hdf5
checkall = args.all

if all([b is False for b in [tiff, weights, hdf5, checkall]]):
    checkall = True

if checkall:
    tiff = True
    weights = True
    hdf5 = True

if tiff:
    from mmt.datasets import landcovers
    
    print(f"Loading landcovers with native CRS and resolution")
    for lcname in ["ESAWorldCover", "EcoclimapSG", "EcoclimapSGplus", "EcoclimapSGML", "ScoreECOSGplus"]:
        lc_class = getattr(landcovers, lcname)
        lcmap = lc_class()
        print(f"  Loaded: {lcmap.__class__.__name__} with crs={lcmap.crs}, res={lcmap.res}")
    
if weights:
    from mmt.inference import io, translators
    
    translator = translators.EsawcToEsgp()
    epoch, iteration = io.get_epoch_of_best_model(translator.checkpoint_path, return_iteration=True)    
    print(f"Loaded {translator} at epoch={epoch}, iteration={iteration}")
    
if hdf5:
    from mmt.datasets import landcover_to_landcover
    from mmt.utils import config as utilconf
    config = utilconf.get_config(
        os.path.join(
            mmt_repopath,
            "configs",
            "new_config_template.yaml",
        )
    )
    datasets = ["esawc.hdf5", "ecosg.hdf5", "esgp.hdf5"]
    print("\nChecking dataset DS1 (France mainland)")
    dataloader = landcover_to_landcover.LandcoverToLandcoverDataLoader(
        config,
        datasets,
        num_workers=0,
        dataset_class = "LandcoverToLandcover",
        to_one_hot=True,
        pos_enc=False,
        ampli=False
    )
    for source, targetval in dataloader.train.items():
        print(f"  source={source}")
        for target, val in targetval.items():
            print(f"    target={target}: dataset={val} ({len(val)} items)")
        
    
    datasets = ["esawc-train.hdf5", "ecosg-train.hdf5", "esgp-train.hdf5"]
    print("\nChecking dataset DS2 (EURAT domain)")
    dataloader = landcover_to_landcover.LandcoverToLandcoverDataLoader(
        config,
        datasets,
        num_workers=0,
        dataset_class = "LandcoverToLandcoverNoJson",
        to_one_hot=True,
        pos_enc=False,
        ampli=False
    )
    for source, targetval in dataloader.train.items():
        print(f"  source={source}")
        for target, val in targetval.items():
            print(f"    target={target}: dataset={val} ({len(val)} items)")

print("\nAll tested data is there")

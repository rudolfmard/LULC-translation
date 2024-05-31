#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to prepare a HDF5 datasets compatible with MLULC map translation.

As per https://github.com/ThomasRieutord/MT-MLULC, the data is expected to be stored as follow:
```
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
```

This program will create the HDF5 files `ecosg.hdf5`, `esgp.hdf5` and `esawc.hdf5`.

It takes the data from the TIF files in `tiff_data` and copy the structure of a template HDF5 file.
Such template HDF5 file is taken from [Luc Baudoux's archive](https://zenodo.org/records/5843595).


Examples
--------
python prepare_hdf5_ds1.py --h5template=../data/hdf5_data/mos.hdf5 --lcnames=ecosg,esgp
"""
import os
import rasterio
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from torchgeo.datasets.utils import BoundingBox
from mmt.datasets import landcovers
from mmt.datasets import transforms
from mmt.utils import misc

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(prog="prepare_hdf5_ds1", description="Create the dataset DS1 (France mainland), which is used to reproduce the result fo Luc Baudoux with new land covers")
parser.add_argument("--h5template", help="Path to a template HDF5 file. Newly created HDF5 files will be stored in the same directory.")
parser.add_argument("--lcnames", help="Land cover maps to be created", default="esawc,ecosg,esgp")
args = parser.parse_args()

lcnames = args.lcnames.split(",")
print(f"Creating {len(lcnames)} HDF5 files based on {args.h5template}")

hdf5_dir = os.path.dirname(args.h5template)
patch_size = 6000 # meters

lcattrs = {
    "esawc":{
        "lcclass": "ESAWorldCover",
        "kwargs":{
            "transforms": transforms.EsawcTransform(),
            "res": 10,
            "crs": rasterio.crs.CRS.from_epsg(2154),
        },
        "producer": "ESA",
        "year": 2021,
    },
    "ecosg":{
        "lcclass": "EcoclimapSG",
        "kwargs":{
            "res": 300,
            "crs": rasterio.crs.CRS.from_epsg(2154),
        },
        "producer": "Meteo France",
        "year": 2018,
    },
    "esgp":{
        "lcclass": "EcoclimapSGplus",
        "kwargs":{
            "res": 60,
            "crs": rasterio.crs.CRS.from_epsg(2154),
        },
        "producer": "Met Eireann",
        "year": 2024,
    },
}

# Template land cover
#---------------------
th5 = h5py.File(args.h5template, "r", swmr=True, libver='latest')

# New land cover
#----------------
for lcname in lcnames:
    h5_lc_path = os.path.join(hdf5_dir, lcname + ".hdf5")
    eh5 = h5py.File(h5_lc_path, "w", libver='latest')
    
    lc_class = getattr(landcovers, lcattrs[lcname]["lcclass"])
    lc = lc_class(**lcattrs[lcname]["kwargs"])
    # lc.crs = lcattrs[lcname]["kwargs"]["crs"]
    # lc.res = lcattrs[lcname]["kwargs"]["res"]
    
    ccrop = transforms.tvt.CenterCrop(int(patch_size/lcattrs[lcname]["kwargs"]["res"]))
    
    print(f"\nWrite patches of the new map in {h5_lc_path}. New map is:")
    # print(lc)
    print(f"crs={lc.crs}, res={lc.res}")

    # Dataset creation
    #------------------
    for i,k in enumerate(th5.keys()):
        tdata = th5.get(k)
        
        xmin, ymin, xmax, ymax = misc.get_bbox_from_coord(tdata.attrs["x_coor"], tdata.attrs["y_coor"], patch_size, location = 'upper-left')
        qb = BoundingBox(xmin, xmax, ymin, ymax, 0, 1e12)
        
        rdata = ccrop(lc[qb]["mask"])
        
        if i % 1000 == 0:
            print(f"[{i}/{len(th5)}] \t rdata.shape={rdata.shape}\t tdata.shape={tdata.shape}, n_zeros={(rdata==0).sum()}")
        
        eh5.create_dataset(k, data=rdata.numpy())
        rd = eh5.get(k)
        rd.attrs["x_coor"] = tdata.attrs["x_coor"]
        rd.attrs["y_coor"] = tdata.attrs["y_coor"]
    
    # Default: copy attributes from template
    for a in th5.attrs.keys():
        eh5.attrs[a] = th5.attrs[a]
    
    # Change the ones we know about
    eh5.attrs["name"] = lcname
    eh5.attrs["year"] = lcattrs[lcname]["year"]
    eh5.attrs["type"] = "raster"
    eh5.attrs["producer"] = lcattrs[lcname]["producer"]
    eh5.attrs["crs"] = lc.crs.to_string()
    eh5.attrs["resolution"] = lc.res
    eh5.attrs["patch_size"] = rdata.shape[-1]
    eh5.attrs["n_channels"] = 1
    eh5.attrs["numberclasses"] = lc.n_labels
    eh5.attrs["label_definition"] = lc.labels
    
    eh5.close()
    print(f"File {h5_lc_path} written.")
    
    # Light check
    #-------------
    eh5 = h5py.File(h5_lc_path, "r", swmr=True, libver='latest')
    print(f"\n --- Checking {h5_lc_path} - len={len(eh5)} ---")
    for a in eh5.attrs.keys():
        print(a, "=", eh5.attrs[a])
    
th5.close()

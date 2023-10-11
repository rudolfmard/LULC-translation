#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to export land cover to HDR/DIR format (SURFEX readable)
"""

import os
import numpy as np
import rasterio
from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.utils import config as utilconf

xp_name = "vanilla"
config = utilconf.get_config(
    os.path.join(
        mmt_repopath,
        "experiments",
        xp_name,
        "logs",
        "config.yaml",
    )
)
mergedmap_dump_dir = os.path.join(config.paths.data_dir, "outputs", f"Merged_vanilla_ireland_esawc_esgp")

print(f"Loading landcover with orginal CRS and resolution")
merged = landcovers.MergedMap(path = mergedmap_dump_dir)
print(f"Loaded: {merged.__class__.__name__} with crs={merged.crs} and res={merged.res}")
data = merged[merged.bounds]
print(f"Data loaded. shape={data['mask'].shape}")

fig, ax = merged.plot(data)
fig.show()

def export_to_dirhdr(sample, ofn_dir):
    """Export a sample to the SURFEX-readable format DIR/HDR
    """
    # HDR file
    ofn_hdr = ofn_dir.replace(".dir", ".hdr")
    hdr_dict = {
        "nodata":0,
        "north":sample["bbox"].maxy,
        "south":sample["bbox"].miny,
        "west":sample["bbox"].maxx,
        "east":sample["bbox"].minx,
        "rows":data["mask"].shape[-2],
        "cols":data["mask"].shape[-1],
        "recordtype": "integer 8 bits",
    }
    with open(ofn_hdr, "w") as hdr:
        hdr.write(os.path.basename(ofn_dir).split(".")[0] + "\n")
        for k,v in hdr_dict.items():
            hdr.write(f"{k}: {v}\n")
        
    
    # DIR file
    kwargs = {
        "driver": "gTiff",
        "count": 1,
        "dtype": np.uint8,
        "crs": data["crs"],
        "width": data["mask"].shape[-1],
        "height": data["mask"].shape[-2],
    }
    
    with rasterio.open(ofn_dir, "w", **kwargs) as dst:
        dst.write(data["mask"].squeeze().numpy(), 1)
    
    return ofn_dir, ofn_hdr

ofn_dir = os.path.join(mergedmap_dump_dir, "COVER_ECOSGpp_2023_v0.4.dir")
export_to_dirhdr(data, ofn_dir)

# hdr_dict = {
    # "nodata":0,
    # "north":merged.bounds.maxy,
    # "south":merged.bounds.miny,
    # "west":merged.bounds.maxx,
    # "east":merged.bounds.minx,
    # "rows":data.shape[0],
    # "cols":data.shape[1],
    # "recordtype": "integer 8 bits",
# }
# ofn_hdr = ofn_dir.replace(".dir", ".hdr")
# with open(ofn_hdr, "w") as hdr:
    # hdr.write(os.path.basename(ofn_dir).split(".")[0] + "\n")
    # for k,v in hdr_dict.items():
        # hdr.write(f"{k}:{v}\n")
    
# kwargs = {
    # "driver": "gTiff",
    # "count": 1,
    # "dtype": np.uint8,
    # "crs": data["crs"],
    # "width": data["mask"].shape[-1],
    # "height": data["mask"].shape[-2],
# }

# with rasterio.open(ofn_dir, "w", **kwargs) as dst:
    # dst.write(data["mask"].squeeze().numpy(), 1)


# ls = os.listdir(mergedmap_dump_dir)
# ds = [rasterio.open(os.path.join(mergedmap_dump_dir, f), "r") for f in ls]
# rasterio.merge.merge(ds)

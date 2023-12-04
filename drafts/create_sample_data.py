#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Create test dataset
"""

import os
import sys
import rasterio
import numpy as np
from torchgeo import samplers

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import transforms as mmt_transforms
from mmt.datasets import landcovers
from mmt.utils import domains


# Config
#--------
domainname = "ireland"
dump_dir = os.path.join(mmt_repopath, "sample-data")


# Land cover loading
#--------------------
esawc = landcovers.ESAWorldCover()
ecosg = landcovers.EcoclimapSG()
esgp = landcovers.EcoclimapSGplus()
esgml = landcovers.EcoclimapSGML()
qflags = landcovers.QualityFlagsECOSGplus()
print(f"Landcovers loaded with native CRS and resolution")


# Extract and save data
#-----------------------
qdomain = getattr(domains, domainname)
qb = qdomain.to_tgbox()

for lc in [ecosg, esgp, esgml, qflags]:
    
    tiffiledir = lc.path.replace("/data/", "/sample-data/")
    if not os.path.exists(tiffiledir):
        os.makedirs(tiffiledir)
    
    tiffilename = os.path.join(
        tiffiledir,
        ".".join([lc.__class__.__name__, domainname, "tif"])
    )
    
    print(f"Extracting {lc.__class__.__name__} over {domainname} in {tiffilename}")
    x = lc[qb]["mask"].squeeze().numpy()
    
    xmin, ymin, xmax, ymax = rasterio.warp.transform_bounds(
        rasterio.crs.CRS.from_epsg(4326), lc.crs, *qdomain.to_lbrt()
    )
    width = x.shape[1]
    height = x.shape[0]
    transform = rasterio.transform.from_bounds(
        xmin, ymin, xmax, ymax, width, height
    )
    kwargs = {
        "driver": "gTiff",
        "dtype": "int8",
        "nodata": 0,
        "count": 1,
        "crs": lc.crs,
        "transform": transform,
        "width": width,
        "height": height,
    }
    with rasterio.open(tiffilename, "w", **kwargs) as dst:
        dst.write(x, 1)

# Extract ESA World Cover
#-------------------------
tiffiledir = esawc.path.replace("/data/", "/sample-data/")
if not os.path.exists(tiffiledir):
    os.makedirs(tiffiledir)

print(f"Extracting {esawc.__class__.__name__} over {domainname} in {tiffiledir}")

sampler = samplers.GridGeoSampler(
    esawc, size=9000, stride=8000, roi=qb
)
for i, iqb in enumerate(iter(sampler)):
    tiffilename = os.path.join(
        tiffiledir,
        ".".join([esawc.__class__.__name__, domainname, f"i{i}", "tif"])
    )
    if i % 10 == 0:
        print(f"  [{i}/{len(sampler)}] tiffilename={tiffilename}")
    
    x = esawc[iqb]["mask"].squeeze().numpy()
    
    xmin = iqb.minx
    ymin = iqb.miny
    xmax = iqb.maxx
    ymax = iqb.maxy
    width = x.shape[1]
    height = x.shape[0]
    transform = rasterio.transform.from_bounds(
        xmin, ymin, xmax, ymax, width, height
    )
    kwargs = {
        "driver": "gTiff",
        "dtype": "int16",
        "nodata": 0,
        "count": 1,
        "crs": esawc.crs,
        "transform": transform,
        "width": width,
        "height": height,
    }
    with rasterio.open(tiffilename, "w", **kwargs) as dst:
        dst.write(x, 1)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Export land cover on MERA domain.
"""

import os
import yaml
import shutil

import easydict
import rasterio
import torchvision.transforms as tvt
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import torchgeo.datasets as tgd

from mmt.datasets import landcovers
from mmt.utils import domains


merageomfile = (
    "/home/trieutord/Works/mera-explorer/mera_explorer/data/mera-grid-geometry.yaml"
)
mllamdataroot = "/home/trieutord/Works/neural-lam/data/mera_example"
xyfile = "/home/trieutord/Works/mera-explorer/mera_explorer/data/nwp_xy.npy"
orofile = "/home/trieutord/Works/mera-explorer/mera_explorer/data/orography.npy"

with open(merageomfile, "r") as f:
    geom = yaml.safe_load(f)

xy = np.load(xyfile)
ll = xy[0, 0, 0], xy[1, 0, 0]
lr = xy[0, 0, -1], xy[1, 0, -1]
ul = xy[0, -1, 0], xy[1, -1, 0]
ur = xy[0, -1, -1], xy[1, -1, -1]
rsz = tvt.Compose(
    [tvt.Resize(xy[0].shape, antialias=True), tvt.functional.vflip]
)

z_grib = np.load(orofile)
print(f"z_grib = {z_grib.shape}")

g = easydict.EasyDict(geom["geometry"])
meracrs = rasterio.crs.CRS.from_proj4(
    f"+proj=lcc +lat_0={g.projlat} +lon_0={g.projlon} +lat_1={g.projlat} +lat_2={g.projlat2} +x_0={g.polon} +y_0={g.polat} +datum=WGS84 +units=m +no_defs"
)
datacrs = ccrs.LambertConformal(
    central_longitude=g.projlon,
    central_latitude=g.projlat,
    standard_parallels=(g.projlat, g.projlat2),
)

# Accessing the data
#================

# A specific class
#------------------
class MeraOrography(tgd.RasterDataset):
    path = "/home/trieutord/Works/mera-explorer/mera_explorer/data"
    is_image = True
    element_size = 32  # Bytes per pixel
    separate_files = False
    # orig_crs = meracrs
    filename_glob = "mera_orography.tif"
    
    def __init__(self, **kwargs):
        super().__init__(self.path, **kwargs)

        # if crs is not None:
            # self.crs = crs
        # else:
            # self.crs = self.crs = self.orig_crs

        # if res is not None:
            # self.res = res

# A native RasterDataset
#-----------------------
oro = tgd.RasterDataset("/home/trieutord/Works/mera-explorer/mera_explorer/data")

# Directly from the TIF file
#---------------------------
with rasterio.open("/home/trieutord/Works/mera-explorer/mera_explorer/data/mera_surface_geopotential.tif", "r") as src:
    z_tiff = src.read(1)/9.81
    tif_trans = src.transform
    tif_bounds = src.bounds

print(f"z_tiff = {z_tiff.shape}")
print(f"||z_grib - z_tiff|| = {np.abs(z_grib - z_tiff[::-1,:]).max()}")


lc = landcovers.EcoclimapSGML()
both = oro & lc


# Bounding Boxes
#================

# External one
#--------------
# qb = domains.ireland25.to_tgbox(both.crs)

# From the lat/lon grid
#--------------
# qb = tgd.BoundingBox(minx=ll[0], maxx=ur[0], miny=ll[1], maxy=ur[1], mint=0, maxt=10e8)

# From the tif file
#--------------
# qb = tgd.BoundingBox(minx=tif_bounds.left, maxx=tif_bounds.right, miny=tif_bounds.bottom, maxy=tif_bounds.top, mint=0, maxt=10e8)
# qb = oro.bounds

# 1. Elevation
qb = oro.bounds
x = both[qb]
print(f"Loaded {x['mask'].shape}, {x['image'].shape} from {lc.__class__.__name__}")
z_tgeo = rsz(x["image"]).squeeze()
print(f"||z_grib - z_tgeo|| = {np.abs(z_grib - z_tgeo.numpy()/9.81).mean()}")

# 2. Land cover
qb = tgd.BoundingBox(minx=ll[0], maxx=ur[0], miny=ll[1], maxy=ur[1], mint=0, maxt=10e8)
x = both[qb]
print(f"Loaded {x['mask'].shape}, {x['image'].shape} from {lc.__class__.__name__}")
c_tgeo = rsz(x["mask"]).squeeze()

# plt.figure()
# plt.imshow(x["image"].squeeze().numpy(), cmap="terrain")
# plt.show(block=False)

# fig, ax = lc.plot(x)
# fig.show()

fig = plt.figure()
ax = plt.subplot(projection=ccrs.PlateCarree())
ax.set_extent([-22, 15, 40, 65])
ax.coastlines(resolution="50m", color="black", linewidth=0.5)
ax.pcolormesh(xy[0], xy[1], c_tgeo, transform = datacrs, alpha = 0.9, cmap="YlGn")
ax.pcolormesh(xy[0], xy[1], z_tgeo, transform = datacrs, alpha = 0.5, cmap="Reds")
ax.pcolormesh(xy[0], xy[1], z_grib, transform = datacrs, alpha = 0.5, cmap="Blues")
# ax.pcolormesh(xy[0], xy[1], rsz(x["image"]).squeeze() - z_grib, transform = datacrs)
# ax.pcolormesh(xy[0], xy[1], z_grib, transform = datacrs)
fig.show()

# lc.export({"mask":c_tgeo}, "ecosgml-v2.0-mb000-mera-2.5km.npy")


# Export all members
#--------------------
for mb in range(lc.n_members):
    lcnpy_file = f"/data/trieutord/Marwa/npy/ecosgml-v2.0-mb00{mb}-mera-2.5km.npy"
    lc = landcovers.EcoclimapSGML(member = mb)
    both = oro & lc
    qb = tgd.BoundingBox(minx=ll[0], maxx=ur[0], miny=ll[1], maxy=ur[1], mint=0, maxt=10e8)
    x = both[qb]
    c_tgeo = rsz(x["mask"]).squeeze()
    lc.export({"mask":c_tgeo}, lcnpy_file)
    print("Written: ", lcnpy_file)

shutil.copy(orofile, "/data/trieutord/Marwa/npy/mera.orography.npy")
print("Written: ", "/data/trieutord/Marwa/npy/mera.orography.npy")
shutil.copy(xyfile, "/data/trieutord/Marwa/npy/mera.geographic_coordinates.npy")
print("Written: ", "/data/trieutord/Marwa/npy/mera.geographic_coordinates.npy")

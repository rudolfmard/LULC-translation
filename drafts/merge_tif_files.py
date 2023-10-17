#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Have look at maps and export it
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy.cluster import hierarchy

from mmt.datasets import landcovers
# from mmt.datasets import transforms as mmt_transforms
from mmt.utils import domains

mergedmap_tif_dir = "/data/trieutord/MLULC/outputs/Inference_vanilla_eurat_eurat_esawc_esgp"
mergedmap_out_dir = "/data/trieutord/MLULC/outputs/Inference200_vanilla_eurat_eurat_esawc_esgp"
# mergedmap_out_dir = "/data/trieutord/ECOCLIMAP-SG-ML/tif/ECOCLIMAP-SG-ML-eurat-v0.3"
eurat = domains.eurat

ls = np.array([i for i in os.listdir(mergedmap_tif_dir) if i.endswith(".tif")])
lats = []
lons = []
for i in ls:
    n, e = i.split("_")
    lats.append(float(n[1:]))
    lons.append(float(e[1:-4]))
    
# ddeg = 4
# res = 0.0005389221556886219
# llats = np.arange(eurat.min_latitude, eurat.max_latitude, ddeg)
# llons = np.arange(eurat.min_longitude, eurat.max_longitude, ddeg)

# seafile = os.path.join(mergedmap_out_dir, "sea_file.tif")

# for llat in llats:
    # for llon in llons:
        # dst_path = os.path.join(mergedmap_out_dir, f"ECOSGML_v0.2_N{llat}_E{llon}.tif")
        # inside = np.logical_and(
            # np.logical_and(lats >= llat, lats < llat + ddeg),
            # np.logical_and(lons >= llon, lons < llon + ddeg)
        # )
        # print(f"[llat={llat}, llon={llon}] {inside.sum()} inside")
        # if inside.sum() > 0:
            # rasterio.merge.merge(
                # [os.path.join(mergedmap_tif_dir, i) for i in ls[inside]],
                # dst_path = os.path.join(mergedmap_out_dir, f"ECOSGML_v0.2_N{llat}_E{llon}.tif")
            # )
        # else:
            # n_px = int(ddeg / res) + 1
            # data = np.zeros((n_px, n_px))
            # trans = rasterio.transform.from_bounds(
                # llon, llat, llon + ddeg, llat + ddeg, n_px, n_px
            # )
            # kwargs = {
                # "driver": "gTiff",
                # "count": 1,
                # "nodata": 0,
                # "dtype": np.uint8,
                # "transform": trans,
                # "crs": rasterio.crs.CRS.from_epsg(4326),
                # "width": n_px,
                # "height": n_px,
            # }
            # with rasterio.open(dst_path, "w", **kwargs) as dst:
                # dst.write(data, 1)
        
X = np.array([lons, lats]).T
Z = hierarchy.linkage(X, method = "centroid")
print(f"Hierarchical clustering done.")
n_clusters = 200
idx = hierarchy.fcluster(Z, t = n_clusters, criterion="maxclust")

for k in range(1, n_clusters+1):
    dst_path = os.path.join(mergedmap_out_dir, f"ECOSGML_v0.3_K{k}.tif")
    print(f"[{k}/{n_clusters}] Merging {len(ls[idx == k])} files into {dst_path}")
    rasterio.merge.merge(
        [os.path.join(mergedmap_tif_dir, i) for i in ls[idx == k]],
        dst_path = dst_path
    )


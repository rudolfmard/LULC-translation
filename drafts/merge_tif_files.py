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

mergedmap_tif_dir = "/data/trieutord/MLULC/outputs/Merged_vanilla_fin_south_deode_esawc_esgp"
mergedmap_out_dir = "/data/trieutord/ECOCLIMAP-SG-ML/tif/ECOCLIMAP-SG-ML-finsouthdeode-v0.3"

ls = np.array([i for i in os.listdir(mergedmap_tif_dir) if i.endswith(".tif")])
lats = []
lons = []
for i in ls:
    n, e = i.split("_")
    lats.append(float(n[1:]))
    lons.append(float(e[1:-4]))

X = np.array([lons, lats]).T
Z = hierarchy.linkage(X, method = "centroid")
print(f"Hierarchical clustering done.")
n_clusters = 200
idx = hierarchy.fcluster(Z, t = n_clusters, criterion="maxclust")

n_files = 0
for k in range(1, n_clusters+1):
    dst_path = os.path.join(mergedmap_out_dir, f"ECOSGML_v0.3_K{k}.tif")
    if len(ls[idx == k]) > 0:
        print(f"[{k}/{n_clusters}] Merging {len(ls[idx == k])} files into {dst_path}")
        rasterio.merge.merge(
            [os.path.join(mergedmap_tif_dir, i) for i in ls[idx == k]],
            dst_path = dst_path
        )
        n_files += 1
    else:
        print(f"[{k}/{n_clusters}] Merging {len(ls[idx == k])} files into {dst_path}. File not created.")


print(f"Merging into {n_files} files complete. Files are in {mergedmap_out_dir}")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to test the access of data with landcover classes
"""
import os

import rasterio
import matplotlib.pyplot as plt

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import domains

dst_crs = rasterio.crs.CRS.from_epsg(3035)
dst_res = 1000
qb = domains.dublin_county.to_tgbox(dst_crs)

# dst_crs = None
# dst_res = None
# qb = domains.portugese_crops.centred_fixed_size(100, 0.0005).to_tgbox()

path_to_infres = os.path.join(
    mmt_repopath, "data", "outputs", "v2", "ecosgml-v2.0-v2outofbox2.eurat.u0.47.sm0.3"
)

for lc in [
    landcovers.EcoclimapSG(crs=dst_crs, res=dst_res),
    landcovers.ESAWorldCover(
        transforms=mmt_transforms.EsawcTransform(), crs=dst_crs, res=dst_res
    ),
    landcovers.EcoclimapSGML(member=2, crs=dst_crs, res=dst_res),
    landcovers.SpecialistLabelsECOSGplus(crs=dst_crs, res=dst_res),
    landcovers.ScoreECOSGplus(
        transforms=mmt_transforms.ScoreTransform(divide_by=100),
        cutoff=0.3,
        crs=dst_crs,
        res=dst_res,
    ),
    landcovers.EcoclimapSGplus(crs=dst_crs, res=dst_res),
    landcovers.InferenceResults(path_to_infres, crs=dst_crs, res=dst_res),
    landcovers.EcoclimapSGMLcomposite(
        path_to_infres, crs=dst_crs, res=dst_res, score_lim=0.6
    ),
]:
    x = lc[qb]
    shapes = {k: x[k].shape for k in ["mask", "image"] if k in x.keys()}
    print(f"Loaded from {lc.__class__.__name__}:\t {shapes}")
    lc.plot(x)

lc = landcovers.EcoclimapSGML(member=0)
qb = domains.portugese_crops.centred_fixed_size(100, 0.0005).to_tgbox(lc.crs)
lc.plot_all_members(qb)
plt.show(block=False)

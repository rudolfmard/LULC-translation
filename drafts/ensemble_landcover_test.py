#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Testing inference run module
"""

import os
import rasterio
import torch
import numpy as np
import matplotlib.pyplot as plt
import onnx
import onnxruntime as ort
from copy import deepcopy
from tqdm import tqdm
from torchgeo import samplers

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import config as utilconf
from mmt.inference import io
from mmt.inference import translators
from mmt.utils import domains

from wopt.ml import graphics

# Configs
#------------
xp_name = "vanilla"
n_members = 5

device = torch.device("cuda")
domainname = "montpellier_agglo"

config = utilconf.get_config(
    os.path.join(
        mmt_repopath,
        "experiments",
        xp_name,
        "logs",
        "config.yaml",
    )
)

# Landcovers
#------------
esawc = landcovers.ESAWorldCover(transforms=mmt_transforms.EsawcTransform)
ecosg = landcovers.EcoclimapSG()
esgp = landcovers.EcoclimapSGplus()
qflags = landcovers.QualityFlagsECOSGplus(transforms=mmt_transforms.FillMissingWithSea(0,6))
print(f"Landcovers loaded with native CRS and resolution")

# Loading query
#----------------
qdomain = getattr(domains, domainname)
qb = qdomain.to_tgbox(esawc.crs)

# Loading models
#----------------

# Pytorch
esawc_encoder = io.load_pytorch_model(xp_name, lc_in = "esawc", lc_out = "encoder")
ecosg_encoder = io.load_pytorch_model(xp_name, lc_in = "ecosg", lc_out = "encoder")
esgp_encoder = io.load_pytorch_model(xp_name, lc_in = "esgp", lc_out = "encoder")
position_encoder = io.load_pytorch_posenc(xp_name)
esgp_decoder = io.load_pytorch_model(xp_name, lc_in = "esgp", lc_out = "decoder")

esawc_encoder.to(device)
esgp_decoder.to(device)

esawc_transform = mmt_transforms.OneHot(esawc.n_labels + 1, device = device)
pos_transform = mmt_transforms.GeolocEncoder()


# Tiling query domain
#----------------
margin = 120
n_px_max = 2400
sampler = samplers.GridGeoSampler(esawc, size=n_px_max, stride = n_px_max - margin, roi = qb)

patches = []
for iqb in sampler:
    patches.append(domains.GeoRectangle(iqb, fmt = "tgbox"))

inference_template_dir = os.path.join(config.paths.data_dir, "ensembles", f"Inference_{xp_name}_{domainname}_esawc_esgp_[member]")
mergedmap_template_dir = os.path.join(config.paths.data_dir, "ensembles", f"Merged_{xp_name}_{domainname}_esawc_esgp_[member]")

for mb in range(n_members):
    inference_dump_dir = inference_template_dir.replace("[member]", str(mb).zfill(2))
    mergedmap_dump_dir = mergedmap_template_dir.replace("[member]", str(mb).zfill(2))
    if not os.path.exists(inference_dump_dir):
        os.makedirs(inference_dump_dir)
    if not os.path.exists(mergedmap_dump_dir):
        os.makedirs(mergedmap_dump_dir)
    
    u = torch.rand(1)

    # Perform inference on each tile
    #----------------
    for iqb in tqdm(sampler, desc = f"[member {mb}] Inference over {len(sampler)} patches"):
        
        tifpatchname = f"N{iqb.minx}_E{iqb.maxy}.tif"
        if os.path.exists(os.path.join(inference_dump_dir, tifpatchname)):
            continue
        
        # Inference
        # - - - - -
        x_esawc = esawc[iqb]
        x_esawc = esawc_transform(x_esawc["mask"])
        
        geoloc = {"coordinate": (iqb.minx, iqb.maxy)}
        pos_enc = pos_transform(geoloc)
        
        with torch.no_grad():
            emb = esawc_encoder(x_esawc.float())
            # emb += position_encoder(torch.Tensor(pos_enc["coordenc"])).unsqueeze(0).unsqueeze(2).unsqueeze(3)
            logits = esgp_decoder(emb)
        
        proba = logits.softmax(1).squeeze().cpu()
        cdf = proba.cumsum(0)
        
        y_esgp = (cdf < u).sum(0)
        width, height = y_esgp.shape
        transform = rasterio.transform.from_bounds(
            iqb.minx, iqb.miny, iqb.maxx, iqb.maxy, width, height
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
        with rasterio.open(os.path.join(inference_dump_dir, tifpatchname), "w", **kwargs) as f:
            f.write(y_esgp, 1)
        
    print("Inference complete.")
    
    infres = landcovers.InferenceResults(path = inference_dump_dir, res = esgp.res)
    infres.res = esgp.res
    
    for iqb in tqdm(sampler, desc = f"[member {mb}] Merging maps over {len(sampler)} patches"):
        
        # Merging
        # - - - - -
        x_infres = infres[iqb]
        x_qflags = qflags[iqb]
        x_esgp = esgp[iqb]
        
        x_merge = deepcopy(x_esgp["mask"])
        w_infres = torch.logical_and(x_qflags["mask"] > 2, x_esgp["mask"] != 1)
        x_merge[w_infres] = x_infres["mask"][w_infres]
        x_merge = x_merge.squeeze().numpy()
        
        tifpatchname = f"N{iqb.minx}_E{iqb.maxy}.tif"
        width, height = x_merge.shape
        transform = rasterio.transform.from_bounds(
            iqb.minx, iqb.miny, iqb.maxx, iqb.maxy, width, height
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
        with rasterio.open(os.path.join(mergedmap_dump_dir, tifpatchname), "w", **kwargs) as f:
            f.write(x_merge, 1)
        
    print("Merging complete.")
    
    
    # View results
    #----------------
    
    infres = landcovers.InferenceResults(path = inference_dump_dir, res = esgp.res)
    infres.res = esgp.res
    
    x_infres = infres[qb]
    fig, ax = infres.plot(x_infres)
    fig.savefig(os.path.join(inference_dump_dir, f"{domainname}_infres.png"))
    fig.show()
    infres.export_to_dirhdr(x_infres, ofn_dir = os.path.join(inference_dump_dir, f"COVER_INFRES_mb{str(mb).zfill(2)}.dir"))
    
    merged = landcovers.MergedMap(path = mergedmap_dump_dir, res = esgp.res)
    merged.res = esgp.res
    
    x_merged = merged[qb]
    fig, ax = merged.plot(x_merged)
    fig.savefig(os.path.join(mergedmap_dump_dir, f"{domainname}_merged_mb{mb}.png"))
    fig.show()
    infres.export_to_dirhdr(x_infres, ofn_dir = os.path.join(mergedmap_dump_dir, f"COVER_MERGED_mb{str(mb).zfill(2)}.dir"))

# x_esgp = esgp[qb]
# fig, ax = esgp.plot(x_esgp)
# # fig.savefig(os.path.join(mergedmap_dump_dir, f"{domainname}_esgp.png"))
# fig.show()



# # Merge where qflags are bad
# #----------------

# x_qflags = qflags[qb]
# x_esgp = esgp[qb]

# fig, ax = qflags.plot(x_qflags)
# fig.savefig(os.path.join(inference_dump_dir, f"{domainname}_qflags.png"))
# # fig.show()

# w_infres = qflags[qb]["mask"] > 2
# print(f"{w_infres.sum().item()} ({100*w_infres.sum().item()/w_infres.nelement()} %) pixels with qflag > 2")

# x_merge = deepcopy(x_esgp["mask"])
# x_merge[w_infres] = x_infres["mask"][w_infres]

# fig, ax = infres.plot({"mask":x_merge}, suptitle = "Merge of inference and ECOSG+")
# fig.savefig(os.path.join(inference_dump_dir, f"{domainname}_merge.png"))
# # fig.show()

# fig, ax = esgp.plot(x_esgp)
# fig.savefig(os.path.join(inference_dump_dir, f"{domainname}_esgp.png"))
# # fig.show()

# x_diff = x_esgp["mask"] != x_merge
# print(f"{x_diff.sum().item()} ({100*x_diff.sum().item()/x_diff.nelement()} %) modified pixels in merge")
# # x_diff = x_esgp["mask"] - x_infres["mask"]

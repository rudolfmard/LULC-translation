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
inference_dump_dir = os.path.join(config.paths.data_dir, "outputs", f"Inference_c54asd_{xp_name}_{domainname}_esawc_esgp")
mergedmap_dump_dir = os.path.join(config.paths.data_dir, "outputs", f"Merged_c54asd_{xp_name}_{domainname}_esawc_esgp")
stitchmap_dump_dir = os.path.join(config.paths.data_dir, "outputs", f"Stitched_c54asd_{xp_name}_{domainname}_esawc_esgp")

if not os.path.exists(inference_dump_dir):
    os.makedirs(inference_dump_dir)
if not os.path.exists(mergedmap_dump_dir):
    os.makedirs(mergedmap_dump_dir)

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
# qdomain = domains.GeoRectangle([4.71, 4.93, 43.87, 44.05])
qb = qdomain.to_tgbox(esawc.crs)

# x_esawc = esawc[qb]
# x_ecosg = ecosg[qb]
# x_esgp = esgp[qb]

# print(f"Original shapes: x_esawc={x_esawc['mask'].shape}, x_ecosg={x_ecosg['mask'].shape}, x_esgp={x_esgp['mask'].shape}")

# Loading models
#----------------

# # ONNX
# onnxfilepath = os.path.join(mmt_repopath, "experiments", xp_name, "checkpoints")
# esawc_encoder = translators.OnnxModel(os.path.join(onnxfilepath, "esawc_encoder.onnx"))
# ecosg_encoder = translators.OnnxModel(os.path.join(onnxfilepath, "ecosg_encoder.onnx"))
# esgp_encoder = translators.OnnxModel(os.path.join(onnxfilepath, "esgp_encoder.onnx"))
# position_encoder = translators.OnnxModel(os.path.join(onnxfilepath, "position_encoder.onnx"))
# esgp_decoder = translators.OnnxModel(os.path.join(onnxfilepath, "esgp_decoder.onnx"))

# Pytorch
esawc_encoder = io.load_pytorch_model(xp_name, lc_in = "esawc", lc_out = "encoder")
ecosg_encoder = io.load_pytorch_model(xp_name, lc_in = "ecosg", lc_out = "encoder")
esgp_encoder = io.load_pytorch_model(xp_name, lc_in = "esgp", lc_out = "encoder")
position_encoder = io.load_pytorch_posenc(xp_name)
emb_mixer = io.load_pytorch_embmix(xp_name)
esgp_decoder = io.load_pytorch_model(xp_name, lc_in = "esgp", lc_out = "decoder")

esawc_encoder.to(device)
ecosg_encoder.to(device)
esgp_decoder.to(device)
emb_mixer.to(device)
position_encoder.to(device)

esawc_transform = mmt_transforms.OneHot(esawc.n_labels + 1, device = device)
ecosg_transform = mmt_transforms.OneHot(ecosg.n_labels + 1, device = device)
pos_transform = mmt_transforms.GeolocEncoder()

# Tiling query domain
#----------------
n_px_max = 600
margin = n_px_max // 6
sampler = samplers.GridGeoSampler(esawc, size=n_px_max, stride = n_px_max - margin, roi = qb)

patches = []
for iqb in sampler:
    patches.append(domains.GeoRectangle(iqb, fmt = "tgbox"))

# exit("Intentional stop")
# graphics.patches_over_domain(qdomain, patches, background="osm", zoomout=0.2, details=2)

# Perform inference on each tile
#----------------
for iqb in tqdm(sampler, desc = f"Inference over {len(sampler)} patches"):
    
    tifpatchname = f"N{iqb.minx}_E{iqb.maxy}.tif"
    if os.path.exists(os.path.join(inference_dump_dir, tifpatchname)):
        continue
    
    # Inference
    # - - - - -
    x_esawc = esawc[iqb]
    x_esawc = esawc_transform(x_esawc["mask"])
    x_ecosg = ecosg[iqb]
    x_ecosg = ecosg_transform(x_ecosg["mask"])
    
    geoloc = {"coordinate": (iqb.minx, iqb.maxy)}
    pos_enc = pos_transform(geoloc)
    
    with torch.no_grad():
        emba = esawc_encoder(x_esawc.float())
        embo = ecosg_encoder(x_ecosg.float())
        # emb = emba
        emb = emb_mixer(torch.cat([emba, torch.nn.functional.interpolate(embo, emba.shape[-2:], mode="bicubic")], dim=1))
        # emb += position_encoder(torch.Tensor(pos_enc["coordenc"]).to(device)).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        y_esgp = esgp_decoder(emb)
    
    y_esgp = y_esgp.argmax(1).squeeze().cpu().numpy()
    
    io.dump_labels_in_tif(
        y_esgp, iqb, esawc.crs, os.path.join(inference_dump_dir, tifpatchname)
    )
    
print("Inference complete.")

infres = landcovers.InferenceResults(path = inference_dump_dir, res = esgp.res)
infres.res = esgp.res

for iqb in tqdm(sampler, desc = f"Merging maps over {len(sampler)} patches"):
    
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
    io.dump_labels_in_tif(
        x_merge, iqb, esawc.crs, os.path.join(mergedmap_dump_dir, tifpatchname)
    )
    
print("Merging complete.")

io.stitch_tif_files(mergedmap_dump_dir, stitchmap_dump_dir, n_max_files = 16)

print("inference_dump_dir=", inference_dump_dir)
print("mergedmap_dump_dir=", mergedmap_dump_dir)

# View results
#----------------

infres = landcovers.InferenceResults(path = inference_dump_dir, res = esgp.res)
infres.res = esgp.res

x_infres = infres[qb]
fig, ax = infres.plot(x_infres)
fig.savefig(os.path.join(inference_dump_dir, f"{domainname}_infres.png"))
fig.show()

merged = landcovers.MergedMap(path = stitchmap_dump_dir, res = esgp.res)
merged.res = esgp.res

x_merged = merged[qb]
fig, ax = merged.plot(x_merged)
fig.savefig(os.path.join(stitchmap_dump_dir, f"{domainname}_merged.png"))
fig.show()

# x_esgp = esgp[qb]
# fig, ax = esgp.plot(x_esgp)
# # fig.savefig(os.path.join(mergedmap_dump_dir, f"{domainname}_esgp.png"))
# fig.show()

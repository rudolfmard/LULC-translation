#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Weather-oriented physiography toolbox (WOPT)

Program to make prediction with MLCT-net.

Useful links
------------
Paper   ->  https://www.tandfonline.com/doi/pdf/10.1080/13658816.2022.2120996
Code    ->  https://github.com/LBaudoux/MLULC
Data    ->  https://zenodo.org/record/5843595
Last checked: 31 July 2023
"""
import os
import sys
import torch
from torch import nn
from torchinfo import summary
from torchgeo.datasets.utils import BoundingBox
import numpy as np
import rasterio.crs
import matplotlib.pyplot as plt
from pprint import pprint
import wopt.ml.utils
from wopt.ml import (
    landcovers,
    transforms,
    domains,
)

from mmt import _repopath_ as mmt_repopath
from mmt.graphs.models import universal_embedding
from mmt.datasets import landcover_to_landcover
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import config as utilconf


# Must run in the MLULC code directory
woptconfig = wopt.ml.utils.load_config()
print(f"Executing program {sys.argv[0]} from {os.getcwd()}")

# Configs
#---------
xp_name = "test_if_it_runs"
mlulcconfig, _ = utilconf.get_config_from_json(
    os.path.join(
        mmt_repopath,
        "configs",
        "universal_embedding.json",
    )
)

checkpoint_path = os.path.join(
    mmt_repopath,
    "experiments",
    xp_name,
    "checkpoints",
    "model_best.pth.tar",
)

# Landcovers
#------------
dst_crs = rasterio.crs.CRS.from_epsg(3035)

print(f"Loading landcovers with CRS = {dst_crs}")
in_lc = landcovers.ESAWorldCover(
    transforms=transforms.fdiv_10_minus_1,
    crs = dst_crs,
)
in_lc.crs = dst_crs
in_lc.res = 10

out_lc = landcovers.EcoclimapSGplus(crs=dst_crs, res=60)
out_lc.crs=dst_crs
out_lc.res=60


# Loading models
#----------------
checkpoint = torch.load(checkpoint_path)

esawc_autoencoder_state = checkpoint["encoder_state_dict_esawc.hdf5"]
esgp_autoencoder_state = checkpoint["encoder_state_dict_esgp.hdf5"]
coord_model_state = checkpoint["image_state_dict_esgp.hdf5"]

data_loader = landcover_to_landcover.LandcoverToLandcoverDataLoader(
    config=mlulcconfig, device="cuda", pos_enc=True
)

print(f"Loading auto-encoders from {checkpoint_path}")
resizes = mlulcconfig.embedding_dim[1]//np.array(data_loader.real_patch_sizes)
resizes=np.where(resizes==1,None,resizes)

EncDec = universal_embedding.UnivEmb
esawc_autoencoder = EncDec(
    in_channels = in_lc.n_labels + 1,
    n_classes = in_lc.n_labels + 1,
    softpos = mlulcconfig.softpos,
    number_feature_map = mlulcconfig.number_of_feature_map,
    embedding_dim = mlulcconfig.embedding_dim[0],
    memory_monger = mlulcconfig.memory_monger,
    up_mode = mlulcconfig.up_mode,
    num_groups = mlulcconfig.group_norm,
    decoder_depth = mlulcconfig.decoder_depth,
    mode = mlulcconfig.mode,
    resize = None,
    cat=False,
    pooling_factors = mlulcconfig.pooling_factors,
    decoder_atrou = mlulcconfig.decoder_atrou,
)
esgp_autoencoder = EncDec(
    in_channels = out_lc.n_labels + 1,
    n_classes = out_lc.n_labels + 1,
    softpos = mlulcconfig.softpos,
    number_feature_map = mlulcconfig.number_of_feature_map,
    embedding_dim = mlulcconfig.embedding_dim[0],
    memory_monger = mlulcconfig.memory_monger,
    up_mode = mlulcconfig.up_mode,
    num_groups = mlulcconfig.group_norm,
    decoder_depth = mlulcconfig.decoder_depth,
    mode = mlulcconfig.mode,
    resize = 6,
    cat=False,
    pooling_factors = mlulcconfig.pooling_factors,
    decoder_atrou = mlulcconfig.decoder_atrou,
)
coord_model = nn.Sequential(
    nn.Linear(128, 300),
    nn.ReLU(inplace=True),
    nn.Linear(300, 50),
    nn.ReLU(inplace=True),
)

esawc_autoencoder.load_state_dict(esawc_autoencoder_state)
esgp_autoencoder.load_state_dict(esgp_autoencoder_state)
coord_model.load_state_dict(coord_model_state)

ce = mmt_transforms.CoordEnc(None)

# Loading query
#----------------
qdomain = domains.dublin_city
qb = qdomain.to_tgbox(dst_crs)

x = in_lc[qb]
y_true = out_lc[qb]
in_lc.plot(x)
plt.show(block=False)
out_lc.plot(y_true)
plt.show(block=False)

x["coordinate"] = (x["bbox"].minx, x["bbox"].maxy)
pos_enc = torch.Tensor(ce(x).get("coordenc"))
res = coord_model(pos_enc.float()).unsqueeze(0).unsqueeze(2).unsqueeze(3)

x = torch.nn.functional.one_hot(x["mask"].squeeze(),num_classes = in_lc.n_labels + 1).permute(2,0,1).unsqueeze(0)
k = out_lc.res/in_lc.res
ccrop = transforms.t.CenterCrop(size=[int(k*(d // k)) for d in x.shape[-2:]])
x = ccrop(x)
z = torch.nn.functional.one_hot(y_true["mask"].squeeze(),num_classes = out_lc.n_labels + 1).permute(2,0,1).unsqueeze(0)

# Inference
#-----------
emb0, x2 = esawc_autoencoder(x.float(), full = True, res = None)
emb, x2 = esawc_autoencoder(x.float(), full = True, res = res)
emb2, z2 = esgp_autoencoder(z.float(), full = True, res = res)
logits = esgp_autoencoder.decoder(emb)
proba = logits.detach().softmax(1)
y = logits.detach().argmax(1)

# Show results
#-----------
inner_shape = [min(s1,s2) for (s1,s2) in zip(y.shape[1:], y_true["mask"].shape[1:])]
ccrop = transforms.t.CenterCrop(size=inner_shape)
acc = (ccrop(y) == ccrop(y_true["mask"])).sum()/ccrop(y).numel()
print(f"Overall accuracy over this patch: {acc}")
out_lc.plot({"mask":y})
plt.show(block=False)

# Export
#-----------
model = nn.Sequential(
    esawc_autoencoder.encoder,
    esgp_autoencoder.decoder
)
summary(model, x.shape)
# torch.onnx.export(model, "esawc2esgp_test.onnx", verbose=True)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Testing inference run module
"""

import os
import pickle
import rasterio
import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import onnx
import onnxruntime as ort
from copy import deepcopy
from tqdm import tqdm
from torchgeo import samplers
import torchvision.transforms as tvt
from sklearn.ensemble import RandomForestClassifier

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
device = torch.device("cuda")
domainname = "dublin_city"

config = utilconf.get_config(
    os.path.join(
        mmt_repopath,
        "configs",
        "new_config_template.yaml",
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

lc = esawc & ecosg & esgp

x = lc[qb]

# Loading models
#----------------
xp_name = "vanilla"

# Pytorch
esawc_encoder = io.load_pytorch_model(xp_name, lc_in = "esawc", lc_out = "encoder")
ecosg_encoder = io.load_pytorch_model(xp_name, lc_in = "ecosg", lc_out = "encoder")
esgp_encoder = io.load_pytorch_model(xp_name, lc_in = "esgp", lc_out = "encoder")
esgp_decoder = io.load_pytorch_model(xp_name, lc_in = "esgp", lc_out = "decoder")

esawc_encoder.to(device)
ecosg_encoder.to(device)
esgp_decoder.to(device)

esawc_transform = mmt_transforms.OneHot(esawc.n_labels + 1, device = device)
ecosg_transform = mmt_transforms.OneHot(ecosg.n_labels + 1, device = device)

flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
avg = torch.nn.AvgPool2d((6, 6))
avg30 = torch.nn.AvgPool2d((30, 30))
ccrop = tvt.CenterCrop([30*(s // 30) for s in x["mask"].shape[-2:]])


# Save/load model
#------------
pklfile = os.path.join(mmt_repopath, "saved_models", "rfc_1000trees.pkl")

with open(pklfile, "rb") as f:
    rfc = pickle.load(f)
    rfc.verbose = 0
    print(f"Model loaded from {f.name}")
    
# Inference
#----------------
xc = ccrop(x["mask"])
x_esawc = xc[0].unsqueeze(0)
x_ecosg = xc[1].unsqueeze(0)
x_esawc = esawc_transform(x_esawc)
x_ecosg = avg30(ecosg_transform(x_ecosg))

with torch.no_grad():
    emba = esawc_encoder(x_esawc.float())
    embo = ecosg_encoder(x_ecosg.float())

femba = flatten(avg(emba.squeeze())).cpu().numpy()
fembo = flatten(avg(embo.squeeze())).cpu().numpy()
X = np.concatenate([femba.T, fembo.T], axis = 1)
trg_shape = avg(xc[2].unsqueeze(0)).shape[-2:]
y = flatten(avg(xc[2].unsqueeze(0))).cpu().numpy()

y_pred = rfc.predict(X)
print(f"Accuracy: {(y_pred == y).sum()/y.size}")

# View results
#----------------
fig, ax = esgp.plot({"mask":y.reshape(trg_shape)})
fig.show()

y_esgp = y_pred.reshape(trg_shape)
fig, ax = esgp.plot({"mask":y_esgp})
fig.show()


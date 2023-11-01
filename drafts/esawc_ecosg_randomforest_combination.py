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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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
domainname = "nanterre"

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
# qdomain = domains.GeoRectangle([4.71, 4.93, 43.87, 44.05])
qb = qdomain.to_tgbox(esawc.crs)

n_pxs = {"esawc":600, "esgp":100, "ecosg":20}
h5_path = dict()
h5f = dict()
for lcname in n_pxs.keys():
    h5_path[lcname] = os.path.join(config.paths.data_dir, "hdf5_data", f"{lcname}-train.hdf5")
    h5f[lcname] = h5py.File(h5_path[lcname], "r", libver='latest')


# Loading models
#----------------
xp_name = "vanilla"
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
esgp_decoder = io.load_pytorch_model(xp_name, lc_in = "esgp", lc_out = "decoder")

esawc_encoder.to(device)
ecosg_encoder.to(device)
esgp_decoder.to(device)
position_encoder.to(device)

esawc_transform = tvt.Compose([
    mmt_transforms.OneHot(esawc.n_labels + 1, device = device),
    # tvt.CenterCrop((600,600))
])
ecosg_transform = tvt.Compose([
    mmt_transforms.OneHot(ecosg.n_labels + 1, device = device),
    # tvt.CenterCrop((20,20))
])
pos_transform = mmt_transforms.GeolocEncoder()

# Find balanced dataset
# - - - - -
# lbcount = {k:0 for k in range(ecosg.n_labels + 1)}
# first_patch_to_have_it = {k:None for k in range(ecosg.n_labels + 1)}

# patches_to_use = []
# while len(patches_to_use) < 20:
    # for it in np.random.choice(list(h5f["esawc"].keys()), 2000):
        # x_esawc = h5f["esawc"][it]
        # x_ecosg = h5f["ecosg"][it]
        # x_esgp = h5f["esgp"][it]
        # geoloc = {"coordinate": (
            # h5f["esgp"][it].attrs["x_coor"],
            # h5f["esgp"][it].attrs["y_coor"]
        # )}
        # for k in range(ecosg.n_labels + 1):
            # if (x_esgp[:] == k).any() and lbcount[k] == 0:
                # lbcount[k] += 1
                # # print(f"First patch to have {ecosg.labels[k]}: {it}")
                # first_patch_to_have_it[k] = it
            
    # balanced_patches = np.unique([i for i in first_patch_to_have_it.values() if i is not None])
    # patches_to_use = np.unique(np.concatenate([patches_to_use, balanced_patches]))
patches_to_use = ['0', '10', '100', '1000', '1001', '1003', '1004', '1032', '1033',
       '1046', '107', '1112', '1210'] + ['1045', '1338', '144', '1459', '1506', '18', '2152', '243', '2696',
       '316', '3549', '3901', '570'] + ['1859', '2096', '2411', '2569', '2755', '2814', '2912', '3444',
       '3511', '3777', '3921', '418', '645', '972']


# Prepare dataset
# - - - - -
flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
avg = torch.nn.AvgPool2d((6,6))
X = []
y = []

for it in patches_to_use:
    x_esawc = torch.Tensor(h5f["esawc"][it][:]).long()
    x_ecosg = torch.Tensor(h5f["ecosg"][it][:]).long()
    # x_esgp = torch.Tensor(h5f["esgp"][it][:])
    geoloc = {"coordinate": (
        h5f["esgp"][it].attrs["x_coor"],
        h5f["esgp"][it].attrs["y_coor"]
    )}
    
    pos_enc = pos_transform(geoloc)
    x_esawc = esawc_transform(x_esawc)
    x_ecosg = ecosg_transform(x_ecosg)
    
    with torch.no_grad():
        emba = esawc_encoder(x_esawc.float().unsqueeze(0))
        embo = ecosg_encoder(x_ecosg.float().unsqueeze(0))
    
    femba = flatten(avg(emba.squeeze())).cpu().numpy()
    fembo = flatten(avg(embo.squeeze())).cpu().numpy()
    fembp = np.tile(pos_enc["coordenc"], (100*100, 1))
    # X_loc = np.concatenate([femba.T, fembo.T, fembp], axis = 1)
    X_loc = np.concatenate([femba.T, fembo.T], axis = 1)
    y_loc = h5f["esgp"][it][:].ravel()
    if len(X) == 0:
        X = deepcopy(X_loc)
        y = deepcopy(y_loc)
    else:
        X = np.concatenate([X,X_loc])
        y = np.concatenate([y, y_loc])
    
# x_esawc = esawc[qb]
# x_esawc = esawc_transform(x_esawc["mask"])

# x_ecosg = ecosg[qb]
# x_ecosg = ecosg_transform(x_ecosg["mask"])
# ccrop = tvt.CenterCrop((100,100))
# x_esgp = ccrop(esgp[qb]["mask"])

# geoloc = {"coordinate": (qb.minx, qb.maxy)}
# pos_enc = pos_transform(geoloc)
# pos_enc = torch.Tensor(pos_enc["coordenc"]).to(device)

# with torch.no_grad():
    # emba = esawc_encoder(x_esawc.float())
    # embo = ecosg_encoder(x_ecosg.float())
    # embp = position_encoder(pos_enc)
    # emb = (emba + embo)/2 + embp.unsqueeze(0).unsqueeze(2).unsqueeze(3)
    # y_esgp = esgp_decoder(emb)

# # Combine embedding
# # - - - - -

# flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
# avg = torch.nn.AvgPool2d((6,6))
# femba = flatten(avg(emba.squeeze())).cpu().numpy()
# fembo = flatten(avg(embo.squeeze())).cpu().numpy()
# fembp = np.tile(pos_enc.cpu().numpy(), (100*100, 1))
# X = np.concatenate([femba.T, fembo.T, fembp], axis = 1)
# y = x_esgp.squeeze().cpu().numpy().ravel()

print(f"Shapes: X={X.shape}, y={y.shape}")
rfc = RandomForestClassifier(n_estimators = 1000, verbose=2, n_jobs = 4, max_depth=20)
# rfc = AdaBoostClassifier(
    # estimator=DecisionTreeClassifier(max_depth=20), n_estimators=200
# )
rfc.fit(X, y)
# y_pred = rfc.predict(X)

# Save/load model
#------------
pklfile = os.path.join(mmt_repopath, "saved_models", "rfc_1000trees.pkl")
with open(pklfile, "wb") as f:
    pickle.dump(rfc, f)
    print(f"Model saved at {f.name}")

with open(pklfile, "rb") as f:
    rfc = pickle.load(f)
    print(f"Model loaded from {f.name}")
    
# Testing on few patches
#----------------
patches_to_test = ["546", "2187", "3"]
X_test = []
y_test = []

for it in patches_to_test:
    x_esawc = torch.Tensor(h5f["esawc"][it][:]).long()
    x_ecosg = torch.Tensor(h5f["ecosg"][it][:]).long()
    # x_esgp = torch.Tensor(h5f["esgp"][it][:])
    geoloc = {"coordinate": (
        h5f["esgp"][it].attrs["x_coor"],
        h5f["esgp"][it].attrs["y_coor"]
    )}
    
    pos_enc = pos_transform(geoloc)
    x_esawc = esawc_transform(x_esawc)
    x_ecosg = ecosg_transform(x_ecosg)
    
    with torch.no_grad():
        emba = esawc_encoder(x_esawc.float().unsqueeze(0))
        embo = ecosg_encoder(x_ecosg.float().unsqueeze(0))
    
    femba = flatten(avg(emba.squeeze())).cpu().numpy()
    fembo = flatten(avg(embo.squeeze())).cpu().numpy()
    fembp = np.tile(pos_enc["coordenc"], (100*100, 1))
    # X_loc = np.concatenate([femba.T, fembo.T, fembp], axis = 1)
    X_loc = np.concatenate([femba.T, fembo.T], axis = 1)
    y_loc = h5f["esgp"][it][:].ravel()
    if len(X_test) == 0:
        X_test = deepcopy(X_loc)
        y_test = deepcopy(y_loc)
    else:
        X_test = np.concatenate([X_test,X_loc])
        y_test = np.concatenate([y_test, y_loc])

y_pred = rfc.predict(X_test)
print(f"Accuracy: {(y_pred == y_test).sum()/y_test.size}")

# View results
#----------------
it = "100"
x_esawc = torch.Tensor(h5f["esawc"][it][:]).long()
x_ecosg = torch.Tensor(h5f["ecosg"][it][:]).long()
# x_esgp = torch.Tensor(h5f["esgp"][it][:])
geoloc = {"coordinate": (
    h5f["esgp"][it].attrs["x_coor"],
    h5f["esgp"][it].attrs["y_coor"]
)}

pos_enc = pos_transform(geoloc)
x_esawc = esawc_transform(x_esawc)
x_ecosg = ecosg_transform(x_ecosg)

with torch.no_grad():
    emba = esawc_encoder(x_esawc.float().unsqueeze(0))
    embo = ecosg_encoder(x_ecosg.float().unsqueeze(0))

femba = flatten(avg(emba.squeeze())).cpu().numpy()
fembo = flatten(avg(embo.squeeze())).cpu().numpy()
fembp = np.tile(pos_enc["coordenc"], (100*100, 1))
# X_loc = np.concatenate([femba.T, fembo.T, fembp], axis = 1)
X_loc = np.concatenate([femba.T, fembo.T], axis = 1)
y_loc = rfc.predict(X_loc)

fig, ax = esgp.plot({"mask":h5f["esgp"][it][:]})
fig.show()

y_esgp = y_loc.reshape((100, 100))
fig, ax = esgp.plot({"mask":y_esgp})
fig.show()


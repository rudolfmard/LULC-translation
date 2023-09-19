#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to make prediction with MLCT-net.
"""
import os
import sys
import torch
import matplotlib.pyplot as plt
import rasterio.crs
from mmt.inference import io
from mmt.utils import domains
from mmt.utils import config as utilconf
from mmt.datasets import landcovers
from mmt.datasets import transforms
from torchvision import transforms as tvt
# import wopt.ml.utils
# from wopt.ml import (
    # landcovers,
    # transforms,
    # domains,
# )

# Configs
#---------
xp_name = "vanilla_no0"
domainname = "dublin_city"
lc_in="esawc"
lc_out="esgp"
usegpu = True
device = torch.device("cuda" if usegpu else "cpu")

# woptconfig = wopt.ml.utils.load_config()
print(f"Executing program {sys.argv[0]} from {os.getcwd()}")


# Landcovers
#------------
dst_crs = rasterio.crs.CRS.from_epsg(3035)

print(f"Loading landcovers with CRS = {dst_crs}")
in_lc = landcovers.ESAWorldCover(
    transforms = transforms.FloorDivMinus(10,1),
    crs = dst_crs,
)
in_lc.crs = dst_crs
in_lc.res = 10

out_lc = landcovers.EcoclimapSGplus(crs=dst_crs, res=60)
out_lc.crs=dst_crs
out_lc.res=60


# Loading query
#----------------
qdomain = getattr(domains, domainname)
qb = qdomain.to_tgbox(dst_crs)

t1h = transforms.OneHot(in_lc.n_labels + 1, device = device)
x = t1h(in_lc[qb]["mask"])
k = out_lc.res/in_lc.res
ccrop = tvt.CenterCrop(size=[int(k*(d // k)) for d in x.shape[-2:]])
x = ccrop(x)

# Loading model
#----------------
print(f"Loading auto-encoders from {xp_name}")
model = io.load_pytorch_model(xp_name, lc_in, lc_out)
model = model.to(device)

# Apply model
#----------------
with torch.no_grad():
    logits = model(x)

y = logits.detach().argmax(1).cpu()

print(f"Show inference on {domainname}")
in_lc.plot(in_lc[qb])
out_lc.plot(out_lc[qb])
out_lc.plot({"mask":y})
plt.show(block=False)

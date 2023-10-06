#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Testing inference I/O module
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from mmt import _repopath_ as mmt_repopath
from mmt.graphs.models import (
    universal_embedding,
    transformer_embedding,
    position_encoding,
    attention_autoencoder
)
from mmt.datasets import landcovers
from mmt.datasets import landcover_to_landcover
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import config as utilconf
from mmt.inference import io

patch_size_metres = landcover_to_landcover.patch_size_metres

xp_name = "vanilla"
lc_in="esawc"
lc_out="esgp"

config = utilconf.get_config(
    os.path.join(
        mmt_repopath,
        "experiments",
        xp_name,
        "logs",
        "config.yaml",
    )
)

checkpoint_path = os.path.join(
    mmt_repopath,
    "experiments",
    xp_name,
    "checkpoints",
    "model_best.pth.tar",
)
assert os.path.isfile(checkpoint_path), f"No checkpoint found at {checkpoint_path}"

res_in = landcover_to_landcover.resolution_dict[lc_in + ".hdf5"]
res_out = landcover_to_landcover.resolution_dict[lc_out + ".hdf5"]
n_channels_in = len(landcover_to_landcover.label_dict[lc_in + ".hdf5"]) + 1
n_channels_out = len(landcover_to_landcover.label_dict[lc_out + ".hdf5"]) + 1

if config.model.type == "transformer_embedding":
    EncDec = getattr(transformer_embedding, config.model.name)
elif config.model.type == "universal_embedding":
    EncDec = getattr(universal_embedding, config.model.name)
elif config.model.type == "attention_autoencoder":
    EncDec = getattr(attention_autoencoder, config.model.name)
else:
    raise ValueError(f"Unknown model.type = {config.model.type}. Please change config to one among ['transformer_embedding', 'universal_embedding', 'attention_autoencoder']")

autoenc_in = EncDec(
    n_channels_in,
    n_channels_in,
    resize = io.get_resize_from_mapname(lc_in, config),
    n_channels_hiddenlay = config.dimensions.n_channels_hiddenlay,
    n_channels_embedding = config.dimensions.n_channels_embedding,
    **config.model.params
)
autoenc_out = EncDec(
    n_channels_out,
    n_channels_out,
    resize = io.get_resize_from_mapname(lc_out, config),
    n_channels_hiddenlay = config.dimensions.n_channels_hiddenlay,
    n_channels_embedding = config.dimensions.n_channels_embedding,
    **config.model.params
)

checkpoint = torch.load(checkpoint_path)

autoenc_in.load_state_dict(checkpoint[f"encoder_state_dict_{lc_in}.hdf5"])
autoenc_out.load_state_dict(checkpoint[f"encoder_state_dict_{lc_out}.hdf5"])
print(f"Loaded model at epoch {checkpoint['epoch']}, iteration {checkpoint['iteration']}")


posenc_in = position_encoding.PositionEncoder(
    n_channels_embedding = config.dimensions.n_channels_embedding
)
posenc_out = position_encoding.PositionEncoder(
    n_channels_embedding = config.dimensions.n_channels_embedding
)
sd_posenc_in = checkpoint[f"image_state_dict_{lc_in}.hdf5"]
posenc_in.load_state_dict({"pos_encoder." + k: v for k,v in sd_posenc_in.items()})

# Fake data
#-----------
def generate_fake_data(n_labels, size_patch):
    """Generate a patch of size `size_patch` with all labels between 0 and `n_labels`"""
    fake_data = torch.zeros(size_patch, size_patch)
    idxs = np.arange(0, size_patch, size_patch / n_labels)
    idxs = idxs.astype(int)
    for i in range(n_labels):
        idx = idxs[i]
        fake_data[idx:,idx:] = i
    
    return fake_data
    
esawc = landcovers.ESAWorldCover(tgeo_init = False)
esgp = landcovers.EcoclimapSGplus(tgeo_init = False)

xy = (0.17, 45.14)
esawc_template = generate_fake_data(12, 600)
esawc.plot({"mask":esawc_template})
plt.show(block=False)
# esgp_template = generate_fake_data(34, 100)
# esgp.plot({"mask":esgp_template})
# plt.show()



gle = mmt_transforms.GeolocEncoder()
toh = mmt_transforms.OneHot(nclasses = 13)

pos_enc = gle({"coordinate": xy})
data = toh(esawc_template.unsqueeze(0).long())

pos_enc_in = posenc_in(torch.Tensor(pos_enc["coordenc"]))
emb = autoenc_in.encoder(data) + pos_enc_in.unsqueeze(0).unsqueeze(2).unsqueeze(3)
rec_data = autoenc_in.decoder(emb).argmax(1)

esawc.plot({"mask":rec_data})
plt.show(block=False)

io.export_autoencoder_to_onnx(xp_name = "vanilla_euratposenc", lc_in = "esawc", lc_out = "encoder", onnxfilename = "test_[default].onnx")
io.export_autoencoder_to_onnx(xp_name = "vanilla_euratposenc", lc_in = "ecosg", lc_out = "encoder", onnxfilename = "test_[default].onnx")
io.export_autoencoder_to_onnx(xp_name = "vanilla_euratposenc", lc_in = "esgp", lc_out = "encoder", onnxfilename = "test_[default].onnx")
io.export_autoencoder_to_onnx(xp_name = "vanilla_euratposenc", lc_in = "esgp", lc_out = "decoder", onnxfilename = "test_[default].onnx")
io.export_autoencoder_to_onnx(xp_name = "vanilla_euratposenc", lc_in = "esawc", lc_out = "esgp", onnxfilename = "test_[default].onnx")
io.export_position_encoder_to_onnx(xp_name = "vanilla_euratposenc", onnxfilename = "[default]_test.onnx")




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
checkpoint_path=os.path.join(mmt_repopath, "saved_models", "vanilla_eurat3.ep169.ckpt")
large_domain = "montpellier_agglo"
small_domain = "dublin_city"

# Load translators
#------------
translator1 = translators.EsawcToEsgpProba(checkpoint_path=checkpoint_path, always_predict = False)

# Loading landcovers
#----------------
esgp = landcovers.EcoclimapSGplus()
esawc = landcovers.ESAWorldCover(transforms=mmt_transforms.EsawcTransform)



# SMALL DOMAIN TESTS
# ==================
qdomain = getattr(domains, small_domain)
qb = qdomain.centred_fixed_size(600, esawc.res).to_tgbox()
print(f"Testing translator {translator1.shortname} on {small_domain}")

# Predict from data
#-------------------
x = esawc[qb]
y = translator1.predict_from_data(x["mask"])
print(f"  Predict from data: {y.shape}")

# Predict from domain
#---------------------
y1 = translator1.predict_from_domain(qb)
print(f"  Predict from domain: {y1.shape}")



# LARGE DOMAIN TESTS
# ==================
qdomain = getattr(domains, large_domain)

print(f"Testing translator1 {translator1.shortname} on {large_domain}")
inference_dump_dir = translator1.predict_from_large_domain(qdomain, output_dir=f"{large_domain}.[id]", tmp_dir=f"{large_domain}.[id]", n_px_max=600, n_max_files=12)

# View results
#----------------
infres = landcovers.InferenceResultsProba(path = inference_dump_dir, res = esgp.res)
infres.res = esgp.res

qb = qdomain.to_tgbox(esgp.crs)
x_infres = infres[qb]
fig, ax = infres.plot(x_infres, title = f"Inference from {translator1.shortname}")
fig.savefig(os.path.join(inference_dump_dir, f"{large_domain}_infres.png"))
fig.show()

# Compare to results
#----------------
x_esgp = esgp[qb]
fig, ax = esgp.plot(x_esgp)
fig.show()



# MERGING TESTS
# =============
print("Merging the inference with ECOSG+")
merger = translators.MapMergerProba(inference_dump_dir, merge_criterion = "qflag2_nodata", output_dtype="float64")
merging_dump_dir = merger.predict_from_large_domain(qdomain, output_dir=f"merger.{large_domain}.[id]", tmp_dir=f"merger.{large_domain}.[id]", n_px_max=100, n_max_files=12)

infres = landcovers.InferenceResultsProba(path = merging_dump_dir)
x_infres = infres[qb]
fig, ax = infres.plot(x_infres, title = f"Merged map from {translator1.shortname}")
fig.show()

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
checkpoint_path=os.path.join(mmt_repopath, "saved_models", "vanilla.ckpt")
classifier_path=os.path.join(mmt_repopath, "saved_models", "rfc_1000trees.pkl")
domainname = "montpellier_agglo"

# Load translators
#------------
translator1 = translators.EsawcToEsgp(checkpoint_path=checkpoint_path)
translator2 = translators.EsawcEcosgToEsgpRFC(checkpoint_path=checkpoint_path, classifier_path=classifier_path)

# Loading query
#----------------
qdomain = getattr(domains, domainname)

# RUN TRANSLATION
#=================
esgp = landcovers.EcoclimapSGplus()

for translator in [translator1, translator2]:
    print(f"Testing translator {translator.shortname} on {domainname}")
    inference_dump_dir = translator.predict_from_large_domain(qdomain, output_dir=f"{domainname}.[id]", tmp_dir=f"{domainname}.[id]", n_max_files=12)
    
    # View results
    #----------------
    infres = landcovers.InferenceResults(path = inference_dump_dir, res = esgp.res)
    infres.res = esgp.res
    
    qb = qdomain.to_tgbox(esgp.crs)
    x_infres = infres[qb]
    fig, ax = infres.plot(x_infres)
    fig.savefig(os.path.join(inference_dump_dir, f"{domainname}_infres.png"))
    fig.show()

x_esgp = esgp[qb]
fig, ax = esgp.plot(x_esgp)
fig.show()

print("Merging the inference with ECOSG+")
merger = translators.MapMerger(inference_dump_dir)
merging_dump_dir = merger.predict_from_large_domain(qdomain, output_dir=f"merger.{domainname}.[id]", tmp_dir=f"merger.{domainname}.[id]", n_px_max=100, n_max_files=12)

infres = landcovers.InferenceResults(path = merging_dump_dir)
x_infres = infres[qb]
fig, ax = infres.plot(x_infres)
fig.show()

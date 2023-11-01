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
checkpoint_path=os.path.join(mmt_repopath, "saved_models", "vanilla.ckpt")
classifier_path=os.path.join(mmt_repopath, "saved_models", "rfc_1000trees.pkl")
domainname = "montpellier_agglo"
# esawc2esgp = translators.EsawcToEsgp(checkpoint_path=checkpoint_path)
esawc2esgp = translators.EsawcEcosgToEsgpRFC(checkpoint_path=checkpoint_path, classifier_path=classifier_path)

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

# RUN TRANSLATION
#=================
inference_dump_dir = esawc2esgp.predict_from_large_domain(qdomain, output_dir=f"{domainname}.[id]", tmp_dir=f"{domainname}.[id]")

# View results
#----------------
infres = landcovers.InferenceResults(path = inference_dump_dir, res = esgp.res)
infres.res = esgp.res

x_infres = infres[qb]
fig, ax = infres.plot(x_infres)
fig.savefig(os.path.join(inference_dump_dir, f"{domainname}_infres.png"))
fig.show()

x_esgp = esgp[qb]
fig, ax = esgp.plot(x_esgp)
fig.show()


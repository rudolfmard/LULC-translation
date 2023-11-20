#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Testing inference run module
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.inference import translators
from mmt.utils import domains

# Config
#--------
checkpoint_path=os.path.join(mmt_repopath, "saved_models", "vanilla_eurat3.ep169.ckpt")
large_domain = "bakar_bay_croatia"
n_members = 3

# Loading land covers and translator
#------------------------------------
esgp = landcovers.EcoclimapSGplus()
esgp0 = landcovers.EcoclimapSGplus(transforms=mmt_transforms.OneHot(35))
esawc = landcovers.ESAWorldCover(transforms=mmt_transforms.EsawcTransform)
translator1 = translators.EsawcToEsgpProba(checkpoint_path=checkpoint_path, always_predict = True)

qdomain = getattr(domains, large_domain)
qdomain = qdomain.centred_fixed_size(2000, res = esawc.res)

# Perform translation
#---------------------
print(f"Using translator {translator1.shortname} on {large_domain}")
inference_dump_dir = translator1.predict_from_large_domain(qdomain, output_dir=f"{large_domain}.[id]", tmp_dir=f"{large_domain}.[id]", n_px_max=600, n_max_files=0)

# Loading results
#----------------
ensinf = landcovers.InferenceResultsProba(path = inference_dump_dir, res = esgp.res)
ensinf.res = esgp.res

qb = qdomain.to_tgbox(esgp.crs)
x_ensinf = ensinf[qb]

# Viewing results
#----------------

### 1. Coloured probabilies
fig, ax = ensinf.plot(x_ensinf, title = f"Inference from {translator1.shortname}")
fig.savefig(os.path.join(inference_dump_dir, f"{large_domain}_ensinf.png"))
fig.show()

### 2. Uncertainty quantification
fig, ax = ensinf.plot_uncertainty(x_ensinf, title = f"Uncertainty on {translator1.shortname}", logscale=False)
fig.savefig(os.path.join(inference_dump_dir, f"uq_{large_domain}_ensinf.png"))
fig.show()

### 3. Reference: ECOSG+
x_esgp = esgp[qb]
fig, ax = esgp.plot(x_esgp)
fig.show()

### 4. Sanity check of coloured probabilies: should be the same as ECOSG+
x_esgp0 = esgp0[qb]
fig, ax = ensinf.plot({"image":x_esgp0["mask"].squeeze()}, title = f"Check proba plot: this should be the same as ECOSG+")
fig.show()

# Generate ensembles
#----------------
for mb in range(n_members):
    print(f"Generating member {mb}")
    x_mb = ensinf.generate_member(x_ensinf, print_u = True)
    fig, ax = esgp.plot({"mask":x_mb}, title = f"Member {mb}")
    fig.show()

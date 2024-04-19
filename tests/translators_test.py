#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to test the inference with the map translator classes
"""

import os
# from mmt.datasets import landcovers
# from mmt.datasets import transforms as mmt_transforms
from mmt.inference import translators
from mmt.utils import domains, misc
import matplotlib.pyplot as plt

# Default resolution is the one of ESA World Cover (~10m)
res = 8.333e-5
n_px = 600
qb = domains.dublin_city.centred_fixed_size(n_px, res).to_tgbox()
checkpoint_path=misc.weights_to_checkpoint("v2outofbox2")

for tr in [
    translators.EsawcToEsgp(checkpoint_path=checkpoint_path),
    translators.EsawcToEsgpProba(checkpoint_path=checkpoint_path),
]:
    x = tr[qb]
    shapes = {k:x[k].shape for k in ["mask", "image"] if k in x.keys()}
    print(f"Loaded from {tr.__class__.__name__}:\t {shapes}")
    tr.plot(x)
    plt.show(block=False)

qdomain = domains.montpellier_agglo
os.makedirs("tmp", exist_ok=True)

for tr in [
    translators.EsawcToEsgp(checkpoint_path=checkpoint_path),
    translators.EsawcToEsgpProba(checkpoint_path=checkpoint_path),
]:
    odir = tr.predict_from_large_domain(
        qdomain,
        output_dir="tmp/[id]",
        tmp_dir="tmp/[id]",
        n_px_max = n_px,
        n_max_files = 0,
    )
    print(f"TIF files written in {odir}")
    
    lcclass = tr.__class__.__bases__[1]
    lc = lcclass(path=odir)
    lc.plot(lc[lc.bounds])
    plt.show(block=False)

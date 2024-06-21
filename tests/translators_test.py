#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Program to test the inference with the map translator classes
"""

import os

import matplotlib.pyplot as plt
from mmt.inference import translators
from mmt.utils import domains, misc

# Default resolution is the one of ESA World Cover (~10m)
res = misc.DEFAULT_RESOLUTION_10M
n_px = 600
qb = domains.dublin_city.centred_fixed_size(n_px, res).to_tgbox()
checkpoint_path=misc.weights_to_checkpoint("mmt-weights-v2.0.ckpt")

for tr in [
    translators.EsawcToEsgpAsMap(checkpoint_path=checkpoint_path),
    translators.EsawcToEsgpShowEnsemble(checkpoint_path=checkpoint_path, u = 0.34),
]:
    x = tr[qb]
    shapes = {k:x[k].shape for k in ["mask", "image"] if k in x.keys()}
    print(f"Loaded from {tr.__class__.__name__}:\t {shapes}")
    tr.plot(x)
    plt.show(block=False)



print("Translators tested successfully")

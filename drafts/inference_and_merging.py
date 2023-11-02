#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Run inference and merge with ECOSG+ on large domains
"""

import os

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.utils import domains
from mmt.inference import translators


# Configs
#------------
checkpoint_path=os.path.join(mmt_repopath, "saved_models", "vanilla_eurat3.ep169.ckpt")
# classifier_path=os.path.join(mmt_repopath, "saved_models", "rfc_1000trees.pkl")
domainname = "eurat"
n_px_max = 5400 # Maximum that could fit on the GPU
inference_dump_dir = os.path.join(mmt_repopath, "data", "outputs")

# Load translators
#------------
translator = translators.EsawcToEsgp(checkpoint_path=checkpoint_path, remove_tmpdirs = False)

qdomain = getattr(domains, domainname)

# Run inference
#------------
inference_tif_dir = translator.predict_from_large_domain(
    qdomain,
    output_dir=os.path.join(inference_dump_dir, f"{domainname}.[id]"),
    tmp_dir=os.path.join(inference_dump_dir, "eurat.tmp.2e7cat.01Nov-17h29"),#f"{domainname}.[id]"),
    n_px_max=5400,
    n_max_files = 0,
)

# Merge with ECOSG+
#------------
print("Merging the inference with ECOSG+")
merger = translators.MapMerger(inference_tif_dir)
print(f"Merging criterion: {merger.merge_criterion.__doc__}")
merging_dump_dir = merger.predict_from_large_domain(
    qdomain,
    output_dir=os.path.join(inference_dump_dir, "ECOSGML-v0.4-[id]"),
    tmp_dir=os.path.join(inference_dump_dir, f"merger.{domainname}.[id]"),
    n_px_max = 828, # Make sure that the patches are smaller than the size of a file
    # Thumb rule: < n_px_max*esawc.res/esgp.res
    n_max_files = 200,
)
print(f"Merged map created at {merging_dump_dir}")

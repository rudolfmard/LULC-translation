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
version = "v0.5"
n_px_max = 5400 # Maximum that could fit on the GPU
inference_dump_dir = os.path.join(mmt_repopath, "data", "outputs")

# Load translators
#------------
translator = translators.EsawcToEsgp(checkpoint_path=checkpoint_path, remove_tmpdirs = False, always_predict = False)

qdomain = getattr(domains, domainname)

# Run inference
#------------
inference_tif_dir = translator.predict_from_large_domain(
    qdomain,
    output_dir=os.path.join(inference_dump_dir, f"infres-{version}.{domainname}.[id]"),
    tmp_dir=os.path.join(inference_dump_dir, f"infres-{version}.{domainname}.[id]"),
    n_px_max=5400,
    n_max_files = 1000,
)
print(f"Inference complete. inference_tif_dir = {inference_tif_dir}")
# inference_tif_dir = os.path.join(inference_dump_dir, "eurat.1000clusters.2e7cat.01Nov-17h29")

# Merge with ECOSG+
#------------
print("Merging the inference with ECOSG+")
merger = translators.MapMerger(inference_tif_dir, merge_criterion = "qflag2_nodata")
print(f"Merging criterion: {merger.merge_criterion.__doc__}")
merging_dump_dir = merger.predict_from_large_domain(
    qdomain,
    output_dir=os.path.join(inference_dump_dir, f"ECOSGML-{version}-[id]"),
    tmp_dir=os.path.join(inference_dump_dir, f"merger-{version}.{domainname}.[id]"),
    n_px_max = 828, # Make sure that the patches are smaller than the size of a file
    # Thumb rule: < n_px_max*esawc.res/esgp.res
    n_max_files = 200,
)
print(f"Merged map created at {merging_dump_dir}")

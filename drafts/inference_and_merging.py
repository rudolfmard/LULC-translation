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
checkpoint_path=os.path.join(mmt_repopath, "data", "saved_models", "mmt-weights-v1.0.ckpt")
domainname = "montpellier_agglo"
version = "v0.6"
n_px_max = 600 # Maximum that could fit on the GPU?
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
    n_px_max = n_px_max,
    n_max_files = 0,
)
print(f"Inference complete. inference_tif_dir = {inference_tif_dir}")

# Merge with ECOSG+
#------------
print("Merging the inference with ECOSG+")
merger = translators.MapMerger(inference_tif_dir, merge_criterion = "qflag2_nodata")
print(f"Merging criterion: {merger.merge_criterion.__doc__}")
merging_dump_dir = merger.predict_from_large_domain(
    qdomain,
    output_dir=os.path.join(inference_dump_dir, f"ECOSGML-{version}-[id]"),
    tmp_dir=os.path.join(inference_dump_dir, f"merger-{version}.{domainname}.[id]"),
    n_px_max = 80, # Make sure that the patches are smaller than the size of a file
    # Thumb rule: < n_px_max*esawc.res/esgp.res
    n_max_files = 12, # Set to 0 to avoid clustering (only copy from tmp_dir to output_dir)
)
print(f"Merged map created at {merging_dump_dir}")

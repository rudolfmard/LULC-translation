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
inference_dump_dir = os.path.join(mmt_repopath, "data", "output")

# Load translators
#------------
translator = translators.EsawcToEsgp(checkpoint_path=checkpoint_path, remove_tmpdirs = False)

qdomain = getattr(domains, domainname)

# Run inference
#------------
inference_tif_dir = translator.predict_from_large_domain(
    qdomain,
    output_dir=os.path.join(inference_dump_dir, f"{domainname}.[id]"),
    tmp_dir=f"{domainname}.[id]",
    n_px_max=5400,
    n_max_files = 0,
)

# Merge with ECOSG+
#------------
print("Merging the inference with ECOSG+")
merger = translators.MapMerger(inference_tif_dir)
merging_dump_dir = merger.predict_from_large_domain(
    qdomain,
    output_dir=os.path.join(inference_dump_dir, "ECOSGML-v0.4-[id]"),
    tmp_dir=f"merger.{domainname}.[id]",
    n_px_max = 900,
    n_max_files = 200,
)
print(f"Merged map created at {merging_dump_dir}")

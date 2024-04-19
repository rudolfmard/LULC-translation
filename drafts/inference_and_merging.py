#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Run inference and merge with ECOSG+ on large domains
"""

import argparse
import os

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.utils import domains, misc
from mmt.inference import translators


# Configs
#------------
default_output = os.path.join(mmt_repopath, "data", "outputs")

parser = argparse.ArgumentParser(
    prog="inference_and_merging",
    description="Produce the ECOSG-ML map with the given weights on the given domain",
    epilog="Example: python -i inference_and_merging.py --weights outofbox2 --npatches 200 --cpu",
)
parser.add_argument(
    "--weights",
    help="Weight file, experience ID or path to the checkpoint to use for inference",
    default="mmt-weights-v1.0.ckpt",
)
parser.add_argument(
    "--output",
    help=f"Output directory is {default_output}",
    default=default_output,
)
parser.add_argument("--domainname", help="Geographical domain name", default="montpellier_agglo")
parser.add_argument("--patchsize", help="Size (#px of 10m) of the patches in the sampler", default=600, type=int) # Maximum that could fit on the GPU?
parser.add_argument(
    "--cpu", help="Perform inference on CPU", action="store_true", default=False
)
args = parser.parse_args()


device = "cpu" if args.cpu else "cuda"
checkpoint_path = misc.weights_to_checkpoint(args.weights)
weights = misc.checkpoint_to_weight(checkpoint_path)
domainname = args.domainname
n_px_max = args.patchsize
inference_dump_dir = args.output

# Load translators
#------------
translator = translators.EsawcToEsgp(checkpoint_path=checkpoint_path, remove_tmpdirs = False, always_predict = False)

qdomain = getattr(domains, domainname)

# Run inference
#------------
inference_tif_dir = translator.predict_from_large_domain(
    qdomain,
    output_dir=os.path.join(inference_dump_dir, f"infres-{weights}.{domainname}.[id]"),
    tmp_dir=os.path.join(inference_dump_dir, f"infres-{weights}.{domainname}.[id]"),
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
    output_dir=os.path.join(inference_dump_dir, f"ECOSGML-{weights}-[id]"),
    tmp_dir=os.path.join(inference_dump_dir, f"merger-{weights}.{domainname}.[id]"),
    n_px_max = 80, # Make sure that the patches are smaller than the size of a file
    # Thumb rule: < n_px_max*esawc.res/esgp.res
    n_max_files = 12, # Set to 0 to avoid clustering (only copy from tmp_dir to output_dir)
)
print(f"Merged map created at {merging_dump_dir}")

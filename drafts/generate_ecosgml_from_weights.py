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
    prog="generate_ecosgml_from_weights",
    description="Produce the ECOSG-ML map with the given weights on the given domain",
    epilog="Example: python generate_ecosgml_from_weights.py --weights v2outofbox2 --domainname montpellier_agglo --n_cluster_files 10 --u 0.82,0.47,0.11,0.34,0.65",
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
parser.add_argument(
    "--n_cluster_files",
    help=f"Merge the inference data into n_cluster_files TIF files (0 means no merging)",
    default=0,
    type=int,
)
parser.add_argument(
    "--n_members",
    help=f"Size of the ensemble (does not include the control member which is always added)",
    default=5,
    type=int,
)
parser.add_argument(
    "--u",
    help=f"Values for the random drawing of the ensemble (overrides --n_members)",
    default=None,
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
n_cluster_files = args.n_cluster_files
inference_dump_dir = args.output

if args.u is None:
    u_values = None
    n_members = args.n_members
else:
    u_values = [float(u) for u in args.u.split(",")]
    n_members = len(u_values)

# Load translators
#------------
translator = translators.EsawcToEsgpThenMergeMembers(checkpoint_path=checkpoint_path, remove_tmpdirs = True, always_predict = False)
qdomain = getattr(domains, domainname)


# Run inference
#------------
inference_tif_dir = translator.predict_members_from_large_domain(
    qdomain,
    output_dir=os.path.join(inference_dump_dir, f"ecosgml-{weights}.{domainname}.[id]"),
    tmp_dir=os.path.join(inference_dump_dir, f"ecosgml-{weights}.{domainname}.[id]"),
    n_px_max = n_px_max,
    n_max_files = n_cluster_files,
    n_members = n_members,
    u_values = u_values,
)
print(f"Inference complete. inference_tif_dir = {inference_tif_dir}")

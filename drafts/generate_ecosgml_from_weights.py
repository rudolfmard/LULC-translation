#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Run inference and merge with ECOSG+ on large domains

Test on a small domain
----------------------
python generate_ecosgml_from_weights.py --weights v2outofbox2 --domainname montpellier_agglo --n_cluster_files 10
python generate_ecosgml_from_weights.py --weights v2outofbox2 --domainname montpellier_agglo --n_cluster_files 10 --u 0.82
python generate_ecosgml_from_weights.py --weights v2outofbox2 --domainname montpellier_agglo --n_cluster_files 10 --u 0.47
python generate_ecosgml_from_weights.py --weights v2outofbox2 --domainname montpellier_agglo --n_cluster_files 10 --u 0.11
python generate_ecosgml_from_weights.py --weights v2outofbox2 --domainname montpellier_agglo --n_cluster_files 10 --u 0.34
python generate_ecosgml_from_weights.py --weights v2outofbox2 --domainname montpellier_agglo --n_cluster_files 10 --u 0.65


Run on a large domain
----------------------
python generate_ecosgml_from_weights.py --weights v2outofbox2 --domainname eurat --n_cluster_files 200
python generate_ecosgml_from_weights.py --weights v2outofbox2 --domainname eurat --n_cluster_files 200 --u 0.82
python generate_ecosgml_from_weights.py --weights v2outofbox2 --domainname eurat --n_cluster_files 200 --u 0.47
python generate_ecosgml_from_weights.py --weights v2outofbox2 --domainname eurat --n_cluster_files 200 --u 0.11
python generate_ecosgml_from_weights.py --weights v2outofbox2 --domainname eurat --n_cluster_files 200 --u 0.34
python generate_ecosgml_from_weights.py --weights v2outofbox2 --domainname eurat --n_cluster_files 200 --u 0.65
"""

import argparse
import os

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.utils import domains, misc
from mmt.inference import translators

from multiprocessing import Pool

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
    "--u",
    help=f"Value for the random drawing of the ensemble",
    default=None,
    type=float,
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
u_value = args.u


# Load translators
#------------
tr = translators.EsawcToEsgpThenMergeMembers(checkpoint_path=checkpoint_path, remove_tmpdirs = True, always_predict = True, u=u_value)
qdomain = getattr(domains, domainname)


# Run inference
#------------
inference_tif_dir = tr.predict_from_large_domain(
    qdomain,
    output_dir=os.path.join(inference_dump_dir, f"ecosgml-{weights}.{domainname}.u{u_value}"),
    tmp_dir=os.path.join(inference_dump_dir, f"ecosgml-{weights}.{domainname}.u{u_value}.TMP"),
    n_px_max = n_px_max,
    n_max_files = n_cluster_files,
)
print(f"Inference complete. inference_tif_dir = {inference_tif_dir}")

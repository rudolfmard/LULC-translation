#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Run inference and merge with ECOSG+ on large domains


Run on a large domain
----------------------
python inference_and_merging.py --weights v2outofbox2 --domainname eurat
python inference_and_merging.py --weights v2outofbox2 --domainname eurat --u 0.82
python inference_and_merging.py --weights v2outofbox2 --domainname eurat --u 0.11
python inference_and_merging.py --weights v2outofbox2 --domainname eurat --u 0.47
python inference_and_merging.py --weights v2outofbox2 --domainname eurat --u 0.34
python inference_and_merging.py --weights v2outofbox2 --domainname eurat --u 0.65
"""

import argparse
import os

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.utils import domains, misc
from mmt.inference import translators


# Configs
#------------
default_output = os.path.join(mmt_repopath, "data", "outputs", "v2")

parser = argparse.ArgumentParser(
    prog="inference_and_merging",
    description="Produce the ECOSG-ML map with the given weights on the given domain",
    epilog="Example: python inference_and_merging.py --weights v2outofbox2 --domainname montpellier_agglo --patchsize 600 --output test --cpu",
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
parser.add_argument("--scoremin", help="Score threshold for the transition", default=0.525, type=float)
parser.add_argument(
    "--cpu", help="Perform inference on CPU", action="store_true", default=False
)
parser.add_argument(
    "--u",
    help=f"Value for the random drawing of the ensemble",
    default=None,
    type=float,
)
parser.add_argument(
    "--skip-inference",
    help="Do not run the inference",
    dest="skip_inference",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--skip-merging",
    help="Do not run the merging",
    dest="skip_merging",
    action="store_true",
    default=False,
)
args = parser.parse_args()


device = "cpu" if args.cpu else "cuda"
checkpoint_path = misc.weights_to_checkpoint(args.weights)
weights = misc.checkpoint_to_weight(checkpoint_path)
domainname = args.domainname
inference_dump_dir = args.output
u_value = args.u

# Load translators
#------------

qdomain = getattr(domains, domainname)

if qdomain.to_tgbox("EPSG:3035").area > 1e10:
    # If the domain is greater than 10,000 km2, cluster the TIF files
    n_px_max1 = 600
    n_cluster_files1 = 1000
    n_px_max2 = 600
    n_cluster_files2 = 200
else:
    n_px_max1 = 600
    n_cluster_files1 = 0
    n_px_max2 = 80
    n_cluster_files2 = 0
    


# Run inference
#------------
inference_tif_dir = os.path.join(inference_dump_dir, f"infres-v2.0-{weights}.{domainname}.u{u_value}")

if not args.skip_inference:
    translator = translators.EsawcToEsgpMembers(checkpoint_path=checkpoint_path, remove_tmpdirs = True, always_predict = True, u = u_value)
    translator.predict_from_large_domain_parallel(
        qdomain,
        output_dir=inference_tif_dir,
        tmp_dir=inference_tif_dir + ".TMP",
        n_px_max = n_px_max1,
        n_cluster_files = n_cluster_files1,
    )

print(f"Inference complete. inference_tif_dir = {inference_tif_dir}")

# Merge with ECOSG+
#------------
print("Merging the inference with ECOSG+")
merging_dump_dir = os.path.join(inference_dump_dir, f"ecosgml-v2.0-{weights}.{domainname}.u{u_value}.sm{args.scoremin}")
if not args.skip_merging:
    merger = translators.MapMergerV2(inference_tif_dir, score_min = args.scoremin)
    merging_dump_dir = merger.predict_from_large_domain(
        qdomain,
        output_dir=merging_dump_dir,
        tmp_dir=merging_dump_dir + ".TMP",
        n_px_max = n_px_max2, # Make sure that the patches are smaller than the size of a file
        # Thumb rule: < n_px_max*esawc.res/esgp.res
        n_cluster_files = n_cluster_files2, # Set to 0 to avoid clustering (only copy from tmp_dir to output_dir)
    )

print(f"Plot it: python -i ../scripts/look_at_map.py --lcpath {merging_dump_dir}")

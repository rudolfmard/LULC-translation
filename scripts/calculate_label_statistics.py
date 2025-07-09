# Imports:

import os
import torch
from torchgeo.datasets.utils import BoundingBox
from tqdm import tqdm
import time

import mmt
from mmt.utils import domains, misc
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.inference.translators import sample_domain_with_patches

def calculate_label_statistics(landcover, landcover_transform=None, domainname="eurat"):
    start_t = time.time()
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

    tmp_dir ="/project/project_465000527/LULC_translation/LULC-translation/scripts"
    patches_definition_file  = sample_domain_with_patches(qdomain, landcover, n_px_max1, tmp_dir)
    
    with open(patches_definition_file, "r") as f:
        patches = f.readlines()

    first_patch = True
    for tifpatchname in tqdm(patches, desc=f"Loop over {len(patches)} patches"):
        tifpatchname = tifpatchname.strip()
        if os.path.exists(os.path.join(tmp_dir, tifpatchname)):
            continue
        qb = BoundingBox(*[float(s[4:]) for s in tifpatchname[:-4].split("_")], 0, 1e12)
        patch = landcover[qb]
        x = patch["mask"]
        one_hot = landcover_transform(x)

        if first_patch == True:
            one_hot_sum = torch.sum(one_hot, dim=(-1,-2), dtype=torch.int)
            first_patch = False
        else:
            one_hot_sum += torch.sum(one_hot, dim=(-1,-2), dtype=torch.int)
    print(f"Label frequencies: f{one_hot_sum}")
    print(f"Total number of pixels: f{one_hot_sum.sum()}")
    os.remove(os.path.join(tmp_dir, "patches_definition_file.txt"))
    print(f"Calculating stats took {time.time()-start_t} seconds.")

def main():
    print("Calculate label statistics for ECOSGML:")
    ecosgml = landcovers.EcoclimapSGML()
    print(f"\tLandcover file: {ecosgml.path}")
    ecosgml_transform = mmt_transforms.OneHotTorchgeo(ecosgml.n_labels + 1, device="cpu")
    calculate_label_statistics(ecosgml, ecosgml_transform)

    print("Calculate label statistics for ECOSGP:")
    ecosgp = landcovers.EcoclimapSGplus()
    print(f"\tLandcover file: {ecosgp.path}")
    ecosgp_transform = mmt_transforms.OneHotTorchgeo(ecosgp.n_labels + 1, device="cpu")
    calculate_label_statistics(ecosgp, ecosgp_transform)

if __name__ == "__main__":
    main()
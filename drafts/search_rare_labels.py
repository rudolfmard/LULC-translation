#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Find approriate patches for qualitative evaluation

Criteria:
  * Balance among all present labels in Europe
  * Few patches
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pprint import pprint
from torchgeo import samplers

from mmt.datasets import transforms as mmt_transforms
from mmt.datasets import landcovers
from mmt.utils import domains


n_px = 150
esgp = landcovers.EcoclimapSGplus()

# # Non-present labels
# #--------------------
# # From `python -i stats_on_labels.py --lcname=EcoclimapSGplus --no-fillsea`
# countfile = "../data/esgp_fullres_labelcount.csv"
# df = pd.read_csv(countfile, sep = ";", skiprows=1)
# fig, ax = plt.subplots(figsize = (10,10))
# ax.set_axisbelow(True)
# ax.grid(color='gray', linestyle='dashed')
# plt.bar(df.labels, df.counts/df.counts.sum(), color = np.array(esgp.cmap)/255)
# plt.xticks(rotation=90)
# fig.show()

not_in_eurat = [
    "0. no data",
    "11. tropical broadleaf evergreen",
    "30: LCZ7: lightweight low-rise",
]

rare_labels = [
    "9. tropical broadleaf deciduous",
    "14. boreal needleleaf deciduous",
    "22. flooded trees",
    "30: LCZ7: lightweight low-rise",
]

# Count labels on pre-set patches
#---------------------------------
lcount = {k: 0 for k in esgp.labels}
selected_patches = []
sampler = samplers.GridGeoSampler(esgp, size=n_px, stride = n_px - 10)

for iqb in tqdm(sampler, desc=f"Search rare labels over {len(sampler)} patches"):
    select_this_patch = False
    x_esgp = esgp[iqb]["mask"]
    
    for lb in rare_labels:
        i_lb = esgp.labels.index(lb)
        count = (x_esgp == i_lb).sum().item()
        if count > 0:
            if lcount[lb] == 0:
                select_this_patch = True
                print(f"Found label {lb} at {iqb}")
            
            lcount[lb] += count
        
    if select_this_patch:
        selected_patches.append(iqb)

print(f"\n{len(selected_patches)} patches selected")

for iqb in selected_patches:
    x_esgp = esgp[iqb]
    fig, ax = esgp.plot(x_esgp)
    fig.show()


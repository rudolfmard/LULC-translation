#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Have look at maps

Examples
--------
python -i stats_on_labels.py --lcname=EcoclimapSGplus --domainname=eurat --res=0.1 --no-fillsea
    Plot distribution of ECOSG+ labels over the EURAT domain at 0.1 degree resolution and leave missing data unchanged

python -i stats_on_labels.py --lcname=EcoclimapSG --rmzeros
    Plot distribution of ECOSG labels and remove the labels that are not presents

python -i stats_on_labels.py --lcname=QualityFlagsECOSGplus --domainname=eurat --res=0.1 --no-fillsea --charttype=pie
    Plot proportion of ECOSG+ quality flags over the EURAT domain at 0.1 in a pie chart

python -i stats_on_labels.py --lcpath=/data/trieutord/MLULC/outputs/ECOSGML-v0.4-onelabel.02Nov-13h13 --domainname=ireland25 --res=0.01
    Plot whatever is in the specified dir (expecting TIF files with ECOSG labels) on the domain ireland25 at 0.01 resolution
"""

import os
import psutil
import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import argparse

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import domains

# Argument parsing
# ----------------
parser = argparse.ArgumentParser(prog="stats_on_labels")
parser.add_argument("--lcname", help="Land cover class name")
parser.add_argument("--domainname", help="Geographical domain name")
parser.add_argument("--res", help="Resolution of the map (degree)", default=None)
parser.add_argument("--lcpath", help="Path to the directory with land cover TIF files. If provided, lcname will be forced to InferenceResults.", default="")
parser.add_argument('--no-fillsea', help="Do not replace missing data by sea", dest='fillsea', action='store_false')
parser.add_argument("--charttype", help="Type of chart to be drawn: pie or bars", default="bars")
parser.add_argument('--rmzeros', help="Do not include labels that are absent", action='store_true')
args = parser.parse_args()

lcname = args.lcname
domainname = args.domainname
res = args.res
lcpath = args.lcpath
fillsea = args.fillsea
charttype = args.charttype
rmzeros = args.rmzeros

assert charttype in ["bars", "pie"], "Invalid chart type. Please choose `bars` or `pie`"

if charttype == "pie":
    rmzeros = True
    print(f"With {charttype} chart, non-present labels are always removed, thus rmzeros={rmzeros}")

kwargs = dict()

if os.path.exists(lcpath):
    lcname = "InferenceResults"
    kwargs.update(path = lcpath)
else:
    assert lcname not in [None, "InferenceResults"], f"Conflicting arguments: lcpath={lcpath} does not exist and lcname={lcname}"

if lcname == "ESAWorldCover":
    kwargs.update(transforms=mmt_transforms.EsawcTransform)
    fillsea = False

if fillsea:
    kwargs.update(transforms=mmt_transforms.FillMissingWithSea(0,1))

if res:
    res = float(res)
    kwargs.update(res = res)

# Loading map
# -----------
print(f"Loading {lcname} with")
pprint(kwargs)
lc_class = getattr(landcovers, lcname)
lcmap = lc_class(**kwargs)

if domainname:
    qdomain = getattr(domains, domainname)
    qb = qdomain.to_tgbox(lcmap.crs)
else:
    qb = lcmap.bounds
    domainname = "all"


# Avoid server crashing
# ---------------------
estimated_ram = lcmap.get_bytes_for_domain(qb)
available_ram = psutil.virtual_memory()
available_ram = available_ram.available

print(f"Loaded: {lcmap.__class__.__name__} with crs={lcmap.crs}, res={lcmap.res}. Expecting {estimated_ram/10**9} GB of data (available {available_ram/10**9} GB)")

if estimated_ram > 0.95 * available_ram:
    raise Exception("You are asking for too much memory. Please reduce the domain or resolution")
if estimated_ram > 0.6 * available_ram:
    answer = input(f"You are about to load {estimated_ram/10**9} GB of data, which more than 60% of your available RAM. Are you sure you want to continue?")
    if answer.lower() not in ["y", "yes", "oui"]:
        raise Exception("Stopped after warning about memory usage.")
    else:
        print("Extracting now...")

x_map = lcmap[qb]


# Counting labels
# ---------------
print("Land cover loaded. Counting labels now...")
ulabels, ucounts = np.unique(x_map["mask"].numpy(), return_counts=True)
if rmzeros:
    counts = ucounts
    labels = [lcmap.labels[l] for l in ulabels]
    cmap = [np.array(lcmap.cmap[l])/255. for l in ulabels]
else:
    counts = np.zeros(lcmap.n_labels)
    counts[ulabels] = ucounts
    zlabels = counts == 0
    labels = np.array(lcmap.labels)
    cmap = np.array(lcmap.cmap)/255.
    print(f"{zlabels.sum()} non-present labels: {labels[zlabels]}")

fig, ax = plt.subplots(figsize = (10,10))
if charttype == "bars":
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')
    plt.bar(labels, counts/counts.sum(), color = cmap)
    if not rmzeros:
        plt.plot(np.arange(lcmap.n_labels)[zlabels], 0.01*np.ones(zlabels.sum()), "o", color = "gray", alpha = 0.2, label = "Not present")
        plt.legend()
    plt.xticks(rotation=90)
else:
    plt.pie(counts, labels=labels, colors = cmap, autopct='%.0f%%')

plt.title(f"Distribution of {lcname} labels over {domainname}")
figpath = os.path.join(mmt_repopath, "figures", f"{lcmap.__class__.__name__}__{lcmap.res}__{domainname}_propoflabels_{charttype}.svg")
fig.savefig(figpath)
print(f"Figure saved at {figpath}")
fig.show()

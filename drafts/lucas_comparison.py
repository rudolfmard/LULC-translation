#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Compare landcovers to LUCAS
"""
import argparse
import os
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import domains, misc, scores


basepath_to_tif = "/data/trieutord/MLULC/outputs/v2/infres-v2.0-v2outofbox2.eurat"
path_to_lucas = "/data/trieutord/LUCAS/EU_LUCAS_2022.csv"

u_values = [None, 0.82, 0.11, 0.47, 0.34, 0.65]
# u_values = [0.82, 0.11, 0.47, 0.34, 0.65]
score_mins = [0.001, 0.1, 0.3, 0.5, 0.7]

# Read LUCAS
# ----------
lucas_label_hierarchy = {
    "water": [
        "G1", "G2", "G3", "G4"
    ],
    "bareland": [
        "F1","F2", "F4"
    ],
    "snow": [
        "G5",
    ],
    "trees": [
        "C"
    ],
    "shrubs": [
        "D"
    ],
    "grassland": [
        "E"
    ],
    "crops": [
        "B"
    ],
    "flooded_veg": [
        "H", "F3"
    ],
    "urban": [
        "A"
    ],
}

primary_labels = list(lucas_label_hierarchy.keys())
tlabels, plabels = scores.prefix_labels(primary_labels)

def convert_lucas_to_primary(lucaslabel):
    for primary in lucas_label_hierarchy.keys():
        if any([lucaslabel.startswith(s) for s in lucas_label_hierarchy[primary]]):
            return primary

def convert_ilabel_to_primary(ilabel):
    for primary in landcovers.ecoclimapsg_label_hierarchy.keys():
        if landcovers.ecoclimapsg_labels[ilabel] in landcovers.ecoclimapsg_label_hierarchy[primary]:
            return primary

def get_lonlat_value(lon, lat, landcover):
    qb = domains.GeoRectangle(
        misc.get_bbox_from_coord(lon, lat, landcover.res, location = "center"),
        fmt = "lbrt"
    ).to_tgbox(landcover.crs)
    
    if landcover.is_image:
        key = "image"
    else:
        key = "mask"
        
    return landcover[qb][key].item()

@misc.memoize
def compute_lucas_confusion_matrix(llons, llats, llabl, landcover, desc = "Comp. to LUCAS"):
    cmx = pd.DataFrame(index = tlabels, columns = plabels, data = np.zeros((len(primary_labels), len(primary_labels))))
    missing_lucas = 0
    missing_pred = 0
    
    for lon, lat, lab in tqdm(zip(llons, llats, llabl), total = len(llabl), desc = desc):
        if lab is None:
            missing_lucas += 1
            continue
        
        plab = convert_ilabel_to_primary(get_lonlat_value(lon, lat, landcover))
        
        if plab is None:
            missing_pred += 1
            continue
            
        cmx.loc["t" + lab, "p" + plab] += 1
    
    return cmx

lucas = pd.read_csv(path_to_lucas, dtype=str, usecols = ["POINT_LONG", "POINT_LAT", "SURVEY_LC1"]).dropna()
print(f"LUCAS data loaded: {lucas.shape}")
llons = lucas["POINT_LONG"].values.astype(float)
llats = lucas["POINT_LAT"].values.astype(float)
llabl = np.array([convert_lucas_to_primary(l) for l in lucas["SURVEY_LC1"]])
# nmax = 2000
# idx = np.random.randint(0, len(llabl), nmax)
# llons = llons[idx]
# llats = llats[idx]
# llabl = llabl[idx]

print(f"\n        ======= ECOSG =======\t  {time.ctime()}")
ecosg = landcovers.EcoclimapSG()
cmxsg = compute_lucas_confusion_matrix(llons=llons, llats=llats, llabl=llabl, landcover=ecosg, desc = f"Comp. ECOSG to LUCAS")
oaccsg = scores.oaccuracy(cmxsg)
print(f"ECOSG overall accuracy: {oaccsg}\n")

print(f"\n        ======= ECOSG+ =======\t  {time.ctime()}")
oaccsgp = []
for ism, score_min in enumerate(score_mins):
    print(f"    ----- score_min={score_min} -----\t [{ism}/{len(score_mins)}]")
    esgpv2 = landcovers.EcoclimapSGplusV2p1(score_min = score_min)
    cmx = compute_lucas_confusion_matrix(llons=llons, llats=llats, llabl=llabl, landcover=esgpv2, desc = f"Comp. SG+ to LUCAS (smin = {score_min})")
    oacc = scores.oaccuracy(cmx)
    oaccsgp.append(oacc)
    print(f"[smin = {score_min}] Overall accuracy: {oacc}\n")


qscore = landcovers.ScoreECOSGplus(transforms=mmt_transforms.ScoreTransform(divide_by=100))
oaccs = {}
for u in u_values:
    path_to_tif = basepath_to_tif + ".u" + str(u)
    print(path_to_tif, os.path.isdir(path_to_tif))
    oaccs["u" + str(u)] = []

for iu, u in enumerate(u_values):
    ustr = "u" + str(u)
    print(f"\n        ======= {ustr} =======\t [{iu}/{len(u_values)}] {time.ctime()}")
    path_to_tif = basepath_to_tif + ".u" + str(u)
    infres = landcovers.InferenceResults(path = path_to_tif)
    
    for ism, score_min in enumerate(score_mins):
        print(f"    ----- score_min={score_min} -----\t [{ism}/{len(score_mins)}]")
        esgpv2 = landcovers.EcoclimapSGplusV2p1(score_min = score_min)
        cmx = pd.DataFrame(index = tlabels, columns = plabels, data = np.zeros((len(primary_labels), len(primary_labels))))
        missing_lucas = 0
        missing_pred = 0
        for lon, lat, lab in tqdm(zip(llons, llats, llabl), total = len(llabl), desc = f"Comp. to LUCAS ({ustr}, smin = {score_min})"):
            if lab is None:
                missing_lucas += 1
                continue
            scoreval = get_lonlat_value(lon, lat, qscore)
            
            if scoreval > score_min:
                # plab = sgplab
                plab = convert_ilabel_to_primary(get_lonlat_value(lon, lat, esgpv2))
            else:
                # plab = ifrlab
                plab = convert_ilabel_to_primary(get_lonlat_value(lon, lat, infres))
                
            if plab is None:
                missing_pred += 1
                continue
            
            cmx.loc["t" + lab, "p" + plab] += 1
        
        oacc = scores.oaccuracy(cmx)
        oaccs[ustr].append(oacc)
        print(f"[{ustr}, smin = {score_min}] Overall accuracy: {scores.oaccuracy(cmx)} Missing: {missing_lucas/len(llabl)} (LUCAS), {missing_pred/len(llabl)} (SG-ML)\n")


plt.figure()
markers = iter(["*", "o", "s", "v", "^", "d", "+"])
for ustr in oaccs.keys():
    plt.plot(score_mins, oaccs[ustr], "--", marker = next(markers), label = "u="+ustr[1:])

plt.plot(score_mins, oaccsgp, "k-", marker=".", label = "ECOSG+")
plt.plot(score_mins, [oaccsg]*len(score_mins), "k:", label = "ECOSG")
plt.grid()
plt.title("ECOSG-ML evaluation against LUCAS for multiple score thresholds")
plt.xlabel("Quality score threshold")
plt.ylabel("Overall accuracy")
plt.legend()
figname = "ecosgml_esgp_ecosg_vs_lucas"
figdir = os.path.join(mmt_repopath, "figures")
figpath = os.path.join(figdir, figname + ".svg")
plt.savefig(figpath)
print("Figure saved:", figpath)
plt.show(block=False)

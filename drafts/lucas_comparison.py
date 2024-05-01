#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Compare landcovers to LUCAS
"""
import argparse
import os

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
        "D", "E1"
    ],
    "grassland": [
        "E2", "E3"
    ],
    "crops": [
        "B",
    ],
    "flooded_veg": [
        "H", "F3"
    ],
    "urban": [
        "A1", "A2"
    ],
}
# primary_labels = [s[0].upper() + s[1:] for s in lucas_label_hierarchy.keys()]
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

lucas = pd.read_csv(path_to_lucas, dtype=str, usecols = ["POINT_LONG", "POINT_LAT", "SURVEY_LC1"]).dropna()
print(f"LUCAS data loaded: {lucas.shape}")
llons = lucas["POINT_LONG"].values.astype(float)
llats = lucas["POINT_LAT"].values.astype(float)
llabl = np.array([convert_lucas_to_primary(l) for l in lucas["SURVEY_LC1"]])
nmax = 1000
idx = np.random.randint(0, len(llabl), nmax)
llons = llons[idx]
llats = llats[idx]
llabl = llabl[idx]

qscore = landcovers.ScoreECOSGplus(transforms=mmt_transforms.ScoreTransform(divide_by=100))

oaccs = {}
# u_values = [None, 0.82, 0.11, 0.47, 0.34]
u_values = [0.82, 0.11]
score_mins = [0.1, 0.3, 0.5, 0.7]
for u in u_values:
    path_to_tif = basepath_to_tif + ".u" + str(u)
    print(path_to_tif, os.path.isdir(path_to_tif))
    oaccs["u" + str(u)] = []

for iu, u in enumerate(u_values):
    ustr = "u" + str(u)
    print(f"\n        ======= {ustr} =======\t [{iu}/{len(u_values)}]")
    path_to_tif = basepath_to_tif + ".u" + str(u)
    infres = landcovers.InferenceResults(path = path_to_tif, transforms=mmt_transforms.FillMissingWithSea(0, 1))
    
    for ism, score_min in enumerate(score_mins):
        print(f"    ----- score_min={score_min} -----\t [{ism}/{len(score_mins)}]")
        esgpv2 = landcovers.EcoclimapSGplusV2p1(score_min = score_min)
        cmx = pd.DataFrame(index = tlabels, columns = plabels, data = np.zeros((len(primary_labels), len(primary_labels))))
        for lon, lat, lab in tqdm(zip(llons, llats, llabl), total = len(llabl), desc = f"Comp. to LUCAS ({ustr}, smin = {score_min})"):
            if lab is None:
                continue
            
            # qb0 = domains.GeoRectangle(
                # misc.get_bbox_from_coord(lon, lat, infres.res, location = "center"),
                # fmt = "lbrt"
            # ).to_tgbox(infres.crs)
            # ifrlab = convert_ilabel_to_primary(infres[qb0]["mask"].item())
            
            qb1 = domains.GeoRectangle(
                misc.get_bbox_from_coord(lon, lat, qscore.res, location = "center"),
                fmt = "lbrt"
            ).to_tgbox(qscore.crs)
            scoreval = qscore[qb1]["image"].item()
            
            # qb2 = domains.GeoRectangle(
                # misc.get_bbox_from_coord(lon, lat, esgpv2.res, location = "center"),
                # fmt = "lbrt"
            # ).to_tgbox(esgpv2.crs)
            # altlab = convert_ilabel_to_primary(esgpv2[qb2]["mask"].item())
            
            if scoreval > score_min:
                # plab = altlab
                qb2 = domains.GeoRectangle(
                    misc.get_bbox_from_coord(lon, lat, esgpv2.res, location = "center"),
                    fmt = "lbrt"
                ).to_tgbox(esgpv2.crs)
                plab = convert_ilabel_to_primary(esgpv2[qb2]["mask"].item())
            else:
                # plab = ifrlab
                qb0 = domains.GeoRectangle(
                    misc.get_bbox_from_coord(lon, lat, infres.res, location = "center"),
                    fmt = "lbrt"
                ).to_tgbox(infres.crs)
                plab = convert_ilabel_to_primary(infres[qb0]["mask"].item())
            
            if plab is None:
                continue
            
            cmx.loc["t" + lab, "p" + plab] += 1
        
        oaccs[ustr].append(scores.oaccuracy(cmx))
        print(f"[{ustr}, smin = {score_min}] Overall accuracy: {scores.oaccuracy(cmx)}\n")

plt.figure()
markers = iter(["*", "o", "s", "v", "^", "d", "+"])
for ustr in oaccs.keys():
    plt.plot(score_mins, oaccs[ustr], "--", marker = next(markers), label = "u="+ustr[1:])

plt.grid()
plt.title("ECOSG-ML evaluation against LUCAS for multiple score thresholds")
plt.xlabel("Quality score threshold")
plt.ylabel("Overall accuracy")
plt.legend()
plt.show(block=False)

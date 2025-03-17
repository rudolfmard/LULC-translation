"""
This script iterates through the training dataset calculates the minimum and maximum of coordinate values.
Also, the map patch pairs are checked to have the same coordinates.
"""

import os
import itertools

import torch
import h5py
import numpy as np

from mmt.datasets import landcover_to_landcover
from mmt.utils import config as utilconf
from mmt.utils.misc import rmsuffix

# Get the configs:
config = utilconf.process_config("replicatep1_dummy.yaml")

# Create a dataloader:
data_loader = landcover_to_landcover.LandcoverToLandcoverDataLoader(config, config.dataloader.params.datasets)
train_data_loader = {
    source: {target: iter(val) for target, val in targetval.items()}
    for source, targetval in data_loader.train_loader.items()
}

max_x = -10000000
min_x = 10000000
max_y = -10000000
min_y = 10000000
end = False
mismatches=0
while not end:
    for source, targetval in train_data_loader.items():   # Iterate over source maps
        for target, dl in targetval.items():        # Iterate over target maps of current source
            ### Load data
            try:
                data = next(dl)                     # Fetch one batch corresponding to the current source and target map
            except:
                end = True
                break
            coord_t = data.get("coordinate_target")
            x_t = coord_t[0].item()
            y_t = coord_t[1].item()
            
            coord = data.get("coordinate")
            x = coord[0].item()
            y = coord[1].item()

            if round(x_t) != round(x) or round(y_t) != round(y):
                print(f"Coordinate mismatch!\ncoord: {coord}, coord_target: {coord_t}")
                print(f"Source: {source}, Target: {target}")
                print("-----------------------------")
                mismatches+=1
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y
        if end:
            break
print("finished")
print(f"Number of coordinate mismathces: {mismatches}")
print(f"X max: {max_x}, X min: {min_x}")
print(f"Y max: {max_y}, Y min: {min_y}")
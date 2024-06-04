#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module with commonly used functions
"""
import json
import logging
import os
import pickle
import random
import string
import time
from hashlib import blake2b

import numpy as np
import torch
import torchvision.transforms as tvt

from mmt import _repopath_ as mmt_repopath

CACHE_DIRECTORY = os.path.join(mmt_repopath, "experiments", "cache")


# DECORATORS
# ==========


def timeit(f):
    """Decorator to time Any Function"""

    def timed(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        seconds = end_time - start_time
        logging.getLogger("Timer").info(
            "   [-] %s : %2.5f sec, which is %2.5f min, which is %2.5f hour"
            % (f.__name__, seconds, seconds / 60, seconds / 3600)
        )
        return result

    return timed


def memoize(func):
    """Decorator for caching the results of a function using Pickle"""

    def wrapper(**kwargs):
        headerhash = hashdict(kwargs)
        cachedfile = os.path.join(CACHE_DIRECTORY, f"{func.__name__}-{headerhash}.pkl")
        if os.path.isfile(cachedfile):
            with open(cachedfile, "rb") as f:
                result = pickle.load(f)
        else:
            result = func(**kwargs)
            with open(cachedfile, "wb") as f:
                pickle.dump(result, f)

        return result

    return wrapper


# FUNCTIONS
# =========


def id_generator(size=6, chars=string.ascii_lowercase + string.digits, forbidden="_"):
    """Generate random strings of characters and digits than can be used as
    unique identifier


    Parameters
    ----------
    size: int
        Length of the returned string of characters

    chars: list
        Admissible characters. Default are lower-case alphanumeric ASCII characters


    Returns
    -------
    idstr: str
        String of `size` characters randomly taken among `chars`
    """
    idstr = "".join([random.choice(chars) for _ in range(size)])
    while forbidden in idstr:
        idstr = "".join([random.choice(chars) for _ in range(size)])

    return idstr


def hashdict(d):
    """Return the hash digest of a dictionary with str keys

    Source (2024/02/20): https://stackoverflow.com/questions/5884066/hashing-a-dictionary/22003440#22003440
    """
    h = blake2b(digest_size=8)

    try:
        s = json.dumps(d, sort_keys=True)
    except TypeError:
        s = json.dumps({k: repr(v) for k, v in d.items()}, sort_keys=True)

    h.update(s.encode("utf8"))

    return h.hexdigest()


def weights_to_checkpoint(weights):
    """Return the absolute path to the checkpoint from a weights name"""
    if os.path.isfile(weights):
        checkpoint_path = weights
    elif os.path.isfile(os.path.join(mmt_repopath, "data", "saved_models", weights)):
        checkpoint_path = os.path.join(mmt_repopath, "data", "saved_models", weights)
    elif os.path.isfile(
        os.path.join(
            mmt_repopath, "experiments", weights, "checkpoints", "model_best.ckpt"
        )
    ):
        checkpoint_path = os.path.join(
            mmt_repopath, "experiments", weights, "checkpoints", "model_best.ckpt"
        )
    else:
        raise ValueError(f"Unable to find weights for {weights}")

    return checkpoint_path


def checkpoint_to_weight(checkpoint_path):
    """Return the weights short name from the checkpoint absolute path"""
    weights = os.path.basename(checkpoint_path).split(".")[0]

    if weights == "model_best":
        weights = os.path.basename(os.path.dirname(os.path.dirname(checkpoint_path)))

    return weights


def create_directories(*indirs):
    """Give default name when the keyword '[id]' is present and create the directories

    Example
    -------
    >>> create_directories("tmp_[id]", "tmp", "out_[id]")
    ['tmp_cg242z.25Apr-09h35', 'tmp', 'out_cg242z.25Apr-09h35']
    """
    dir_id = id_generator()
    outdirs = []
    for d in indirs:
        if "[id]" in d:
            d = d.replace("[id]", dir_id + time.strftime(".%d%b-%Hh%M"))
        os.makedirs(d, exist_ok=True)
        outdirs.append(d)

    return outdirs

def rmsuffix(s, startchar="-", stopchar="."):
    """Remove suffix between `startchar` and `stopchar`
    
    
    Example
    -------
    >>> rmsuffix("esawc-trefs41.hdf5")
    'esawc.hdf5'
    """
    if startchar in s:
        return s.split(startchar)[0] + stopchar + s.split(stopchar)[1]
    else:
        return s

def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")
    import sys
    from subprocess import call

    import torch

    logger.info("__Python VERSION:  {}".format(sys.version))
    logger.info("__pyTorch VERSION:  {}".format(torch.__version__))
    logger.info("__CUDA VERSION")
    try:
        call(["nvcc", "--version"])
    except:
        logger.info("nvcc not found")
    logger.info("__CUDNN VERSION:  {}".format(torch.backends.cudnn.version()))
    logger.info("__Number CUDA Devices:  {}".format(torch.cuda.device_count()))
    logger.info("__Devices")
    call(
        [
            "nvidia-smi",
            "--format=csv",
            "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free",
        ]
    )
    logger.info("Active CUDA Device: GPU {}".format(torch.cuda.current_device()))
    logger.info("Available devices  {}".format(torch.cuda.device_count()))
    logger.info("Current cuda device  {}".format(torch.cuda.current_device()))


def get_bbox_from_coord(x_coor, y_coor, patch_size, location="upper-left"):
    """Return a bounding box (xmin, ymin, xmax, ymax) from a single point coordinate.


    Parameters
    ----------
    x_coor: float
        Easting coordinate of the single point

    y_coor: float
        Northing coordinate of the single point

    patch_size: float
        Size of the bounding box, expressed as a difference of coordinate (xmax -xmin)

    location: {'lower-left', 'center', 'upper-right', 'upper-left'}
        Indicator to locate the single point coordinate w.r.t the bounding box.
        For example, if location='upper-left' (default), the single point is assumed
        to be located in the upper left corner of the bounding box that will be returned.


    Returns
    --------
    xmin, ymin, xmax, ymax: 4-tuple of float
        Bounding box (left, bottom, right, top)
    """
    if location == "lower-left":
        xmin, ymin, xmax, ymax = (
            x_coor,
            y_coor,
            x_coor + patch_size,
            y_coor + patch_size,
        )
    elif location == "center":
        xmin, ymin, xmax, ymax = (
            x_coor - patch_size / 2,
            y_coor - patch_size / 2,
            x_coor + patch_size / 2,
            y_coor + patch_size / 2,
        )
    elif location == "upper-right":
        xmin, ymin, xmax, ymax = (
            x_coor - patch_size,
            y_coor - patch_size,
            x_coor,
            y_coor,
        )
    elif location == "upper-left":
        xmin, ymin, xmax, ymax = (
            x_coor,
            y_coor - patch_size,
            x_coor + patch_size,
            y_coor,
        )
    else:
        raise ValueError(
            f"Unsupported location key: {location}. Supported keys are 'lower-left', 'center', 'upper-right', 'upper-left'."
        )

    return xmin, ymin, xmax, ymax


def get_bytes_for_domain(qdomain, res, element_size=8):
    """Return the size in bytes that would be necessary to load the query domain (does not load anything)"""

    nlon = (qdomain.max_longitude - qdomain.min_longitude) / res
    nlat = (qdomain.max_latitude - qdomain.min_latitude) / res

    return nlon * nlat * element_size


def haversine_formula(lon1, lon2, lat1, lat2, degrees=True, r=6378100):
    """Give an estimate of the great-circle distance between two points
    in lon-lat coordinates.

    Source (2023/11/09): https://en.wikipedia.org/wiki/Haversine_formula
    """
    if degrees:
        lon1, lon2, lat1, lat2 = [a * np.pi / 180 for a in [lon1, lon2, lat1, lat2]]

    return (
        2
        * r
        * np.arcsin(
            np.sqrt(
                np.sin((lat2 - lat1) / 2) ** 2
                + np.cos(lat2) * np.cos(lat1) * np.sin((lon2 - lon1) / 2) ** 2
            )
        )
    )


def qscore_from_qflags(qflags):
    """Return the proportion of quality flag with values 1 or 2"""
    raise DeprecationWarning(
        f"{__name__}.{sys._getframe().f_code.co_name}: This function is deprecated"
    )
    if isinstance(qflags, torch.Tensor):
        qflags = qflags.detach().cpu().numpy()

    if (qflags == 0).sum() > 0:
        return 0
    else:
        return (qflags < 3).sum() / qflags.size


def qscore_from_score(score):
    """Return the patch-averaged value of the score"""
    if isinstance(score, torch.Tensor):
        score = score.detach().cpu().numpy()

    return (score > 0.525).sum() / score.size
    # return score.mean()


def divscore_from_esawc(esawc):
    """Return the proportion of pixels with the dominant label"""
    if isinstance(esawc, torch.Tensor):
        esawc = esawc.detach().cpu().numpy()

    _, c = np.unique(esawc, return_counts=True)
    return 1 - c.max() / c.sum()


def ccrop_and_split(x, n_px):
    """Center-crop and split into four patches

    Params
    ------
    x: torch.Tensor or dict
        Data to be cropped and split

    n_px: int
        Number of pixels in the split patches
    """
    ccrop = tvt.CenterCrop(2 * n_px)
    try:
        x = ccrop(x["mask"]).squeeze()
    except:
        x = ccrop(x).squeeze()

    x_train1 = x[:n_px, :n_px]
    x_train2 = x[n_px:, :n_px]
    x_test = x[:n_px, n_px:]
    x_val = x[n_px:, n_px:]

    return {"train": x_train1, "train2": x_train2, "test": x_test, "val": x_val}

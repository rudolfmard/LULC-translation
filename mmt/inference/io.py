#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module to load and export pretrained models for inference
"""

import os
import sys

import numpy as np
import rasterio
import torch
from scipy.cluster import hierarchy
from sklearn.cluster import KMeans
from torchgeo.datasets.utils import BoundingBox

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcover_to_landcover
from mmt.graphs.models import attention_autoencoder, universal_embedding
from mmt.utils import config as utilconf
from mmt.utils import misc

PATCH_SIZE_METRES = landcover_to_landcover.PATCH_SIZE_METRES


def cluster_tif_files(
    input_dir,
    output_dir,
    n_max_files=200,
    clustering="kmeans",
    prefix="stitched",
    verbose=True,
) -> int:
    """Merge a large number of TIF files into a smaller number of TIF files.

    The files are merged according to the lon-lat coordinates found in their names
    with the clustering method specified (default is K-means with 1 init and 10 iteration max).
    The expected pattern for the names of the TIF files is
        "N<lat>_E<lon>.tif"
        Ex: "N4.09_E43.82.tif"

    Used in mmt.inference.translators._MapTranslator.predict_from_large_domain


    Parameters
    ----------
    input_dir: str
        Path to the directory where are stored the input TIF files.

    output_dir: str
        Path to the directory where are stored the output TIF files.
        If it doesn't exists, it is created.

    n_max_files: int
        Maximum number of TIF files in the output directory.

    clustering: {"kmeans", "hierarchical"}
        Clustering method to be used (K-means or hierarchical with centroid linkage).
        Default if K-means with 1 init and 10 iteration max.

    prefix: str
        Prefix for the output file names.

    verbose: bool
        If False, remove all prints.


    Returns
    -------
    n_files: int
        The final number of files created after clustering (= n_max_files - #empty files)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ls = np.array([i for i in os.listdir(input_dir) if i.endswith(".tif")])
    lats = []
    lons = []
    for i in ls:
        w, _, s, _ = i.split("_")
        lats.append(float(s[4:]))
        lons.append(float(w[4:]))

    if verbose:
        print(
            f"Stitching {ls.size} TIF files from {input_dir} to <= {n_max_files} TIF files at {output_dir}."
        )

    X = np.array([lons, lats]).T
    del lats, lons

    if verbose:
        print(
            f"Starting stitching {len(ls)} TIF files into {n_max_files} files with {clustering}"
        )

    if clustering == "hierarchical":
        Z = hierarchy.linkage(X, method="centroid")
        if verbose:
            print(f"Hierarchical clustering done.")

        idx = hierarchy.fcluster(Z, t=n_max_files, criterion="maxclust")
        del X, Z
    elif clustering == "kmeans":
        km = KMeans(n_clusters=n_max_files, max_iter=10, n_init=1, init="k-means++")
        idx = km.fit_predict(X) + 1
        if verbose:
            print(f"K-means clustering done.")

        del X, km
    else:
        raise ValueError(f"Unsupported clustering method: {clustering}")

    n_files = 0
    for k in range(1, n_max_files + 1):
        dst_path = os.path.join(output_dir, f"{prefix}_K{k}.tif")
        if len(ls[idx == k]) > 0:
            if verbose and k % max(n_max_files // 10, 1) == 0:
                print(
                    f"[{k}/{n_max_files}] Merging {len(ls[idx == k])} files into {dst_path}"
                )

            try:
                rasterio.merge.merge(
                    [os.path.join(input_dir, i) for i in ls[idx == k]],
                    dst_path=dst_path,
                )
                n_files += 1
            except:
                print(f"Problem with these {len(ls[idx == k])} files:", ls[idx == k])
                raise Exception("Stop merging TIF files")
        else:
            if verbose:
                print(
                    f"[{k}/{n_max_files}] Merging {len(ls[idx == k])} files into {dst_path}. File not created."
                )

    if verbose:
        print(
            f"Stitching TIFs into {n_files} files complete. Files are in {output_dir}"
        )

    return n_files


def dump_labels_in_tif(labels, domain, crs, tifpath, dtype="int16") -> None:
    """Write land cover labels in a GeoTIFF file.


    Parameters
    ----------
    labels: ndarray
        Matrix of land cover labels

    domain: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
        Domain covered by the labels

    crs: `rasterio.crs.CRS`
        Coordinate reference system

    tifpath: str
        Path of the file to be written
    """

    if not isinstance(domain, BoundingBox):
        domain = domain.to_tgbox(crs)

    if labels.ndim == 2:
        height, width = labels.shape
        n_channels = 1
    elif labels.ndim == 3:
        n_channels, height, width = labels.shape
    else:
        raise ValueError(
            f"Input tensor has unexpected number of dimensions: {labels.ndim}"
        )

    transform = rasterio.transform.from_bounds(
        domain.minx, domain.miny, domain.maxx, domain.maxy, width, height
    )
    kwargs = {
        "driver": "gTiff",
        "dtype": dtype,
        "nodata": 0,
        "count": n_channels,
        "crs": crs,
        "transform": transform,
        "width": width,
        "height": height,
    }
    with rasterio.open(tifpath, "w", **kwargs) as f:
        if n_channels > 1:
            for c in range(n_channels):
                f.write(labels[c, :, :], c + 1)

        else:
            f.write(labels, 1)


def get_epoch_of_best_model(xp_name, return_iteration=False) -> int:
    """Read the value of epoch recorded in the best model checkpoint.


    Parameters
    ----------
    xp_name: str
        Alias to the weights (experiment name, saved weights or absolute path)
        See `mmt.utils.misc.weights_to_checkpoint`

    return_iteration: bool
        If True, returns a tuple (epoch, iteration).


    Returns
    -------
    epoch [, iteration]: int
        Epoch (and iteration, if `return_iteration=True`) of the model
    """
    checkpoint_path = misc.weights_to_checkpoint(xp_name)
    checkpoint = torch.load(checkpoint_path)

    if return_iteration:
        return checkpoint["epoch"], checkpoint["iteration"]
    else:
        return checkpoint["epoch"]


def get_patchsize_from_mapname(mapname) -> int:
    """Return the patch size (number of pixels) of a given land cover map"""
    if not mapname.endswith(".hdf5"):
        mapname += ".hdf5"

    resolution = landcover_to_landcover.RESOLUTION_CATALOG[mapname]
    return PATCH_SIZE_METRES // resolution


def get_resize_from_mapname(mapname, config) -> int:
    """Return the resizing factor (resolution map/resolution embedding)
    of a given land cover map.


    Parameters
    ----------
    mapname: str
        Short name of the map (esawc, ecosg, esgp)

    config: easydict
        The configuration of the experiment


    Returns
    -------
    resize: int or None
        The resizing factor (resolution map/resolution embedding)
    """
    if not mapname.endswith(".hdf5"):
        mapname += ".hdf5"

    n_px_emb = config.dimensions.n_px_embedding
    model_type = config.model.type

    resolution = landcover_to_landcover.RESOLUTION_CATALOG[mapname]
    resize = int(n_px_emb * resolution / PATCH_SIZE_METRES)

    if model_type == "universal_embedding" and resize == 1:
        resize = None

    return resize


def load_pytorch_model(
    xp_name, lc_in="esawc", lc_out="esgp", train_mode=False, device="cpu"
) -> torch.nn.Module:
    """Load a pre-trained Pytorch model to make map translation


    Parameters
    ----------
    xp_name: str
        Alias to the weights (experiment name, saved weights or absolute path)
        See `mmt.utils.misc.weights_to_checkpoint`

    lc_in: str
        Input map's short name (esawc, ecosg, esgp)

    lc_out: str
        Input map's short name (esawc, ecosg, esgp, encoder, decoder)
        If lc_out="encoder", the loaded model translates `lc_in` to the latent space
        if lc_out="decoder", the loaded model translates the latent space to `lc_in`

    train_mode: bool, optional
        If True, the training mode is set on

    device: {"cuda", "cpu"}
        The device on which the model is loaded


    Returns
    -------
    model: torch.nn.Module
        Map translation model loaded with weights of `xp_name` translating
        from `lc_in` to `lc_out`
    """

    checkpoint_path = misc.weights_to_checkpoint(xp_name)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = utilconf.get_config(checkpoint_path.replace("ckpt", "config.yaml"))

    if config.model.type == "universal_embedding":
        EncDec = getattr(universal_embedding, config.model.name)
    elif config.model.type == "attention_autoencoder":
        EncDec = getattr(attention_autoencoder, config.model.name)
    else:
        raise ValueError(
            f"Unknown model.type = {config.model.type}. Please change config to one among ['transformer_embedding', 'universal_embedding', 'attention_autoencoder']"
        )

    res_in = landcover_to_landcover.RESOLUTION_CATALOG[lc_in + ".hdf5"]
    n_channels_in = len(landcover_to_landcover.LABELS_CATALOG[lc_in + ".hdf5"]) + 1

    autoenc_in = EncDec(
        in_channels=n_channels_in,
        out_channels=n_channels_in,
        n_px_input=get_patchsize_from_mapname(lc_in),
        resize=get_resize_from_mapname(lc_in, config),
        n_px_embedding=config.dimensions.n_px_embedding,
        n_channels_hiddenlay=config.dimensions.n_channels_hiddenlay,
        n_channels_embedding=config.dimensions.n_channels_embedding,
        **config.model.params,
    )

    autoenc_in.load_state_dict(checkpoint[f"encoder_state_dict_{lc_in}.hdf5"])

    if lc_out not in ["encoder", "decoder"]:
        res_out = landcover_to_landcover.RESOLUTION_CATALOG[lc_out + ".hdf5"]
        n_channels_out = (
            len(landcover_to_landcover.LABELS_CATALOG[lc_out + ".hdf5"]) + 1
        )

        autoenc_out = EncDec(
            in_channels=n_channels_out,
            out_channels=n_channels_out,
            n_px_input=get_patchsize_from_mapname(lc_out),
            resize=get_resize_from_mapname(lc_out, config),
            n_px_embedding=config.dimensions.n_px_embedding,
            n_channels_hiddenlay=config.dimensions.n_channels_hiddenlay,
            n_channels_embedding=config.dimensions.n_channels_embedding,
            **config.model.params,
        )

        autoenc_out.load_state_dict(checkpoint[f"encoder_state_dict_{lc_out}.hdf5"])

    print(
        f"Loaded model at epoch {checkpoint['epoch']}, iteration {checkpoint['iteration']}"
    )

    if lc_out == "encoder":
        model = autoenc_in.encoder
    elif lc_out == "decoder":
        model = autoenc_in.decoder
    else:
        model = torch.nn.Sequential(autoenc_in.encoder, autoenc_out.decoder)

    model.train(mode=train_mode)
    return model

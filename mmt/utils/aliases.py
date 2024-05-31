#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module with aliases utilities.

Conversely to misc, which uses only third-party packages, aliases includes mostly first-party modules.
"""
import os
from pprint import pprint

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.inference import translators
from mmt.utils import misc

LANDCOVER_ALIASES = {
    "esawc": {
        "lcclass": "ESAWorldCover",
        "kwargs": {
            "transforms": mmt_transforms.EsawcTransform(),
        },
        "colname": "ESAWC",
    },
    "ecosg": {
        "lcclass": "EcoclimapSG",
        "kwargs": {},
        "colname": "ECOSG",
    },
    "esgp": {
        "lcclass": "EcoclimapSGplus",
        "kwargs": {},
        "colname": "ECOSG+",
    },
    "esgpv2": {
        "lcclass": "EcoclimapSGplusV2",
        "kwargs": {},
        "colname": "ECOSG+v2",
    },
    "bguess": {
        "lcclass": "SpecialistLabelsECOSGplus",
        "kwargs": {},
        "colname": "BGUESS",
    },
    "esgml": {
        "lcclass": "EcoclimapSGML",
        "kwargs": {},
        "colname": "ECOSG-ML",
    },
    "qflags": {
        "lcclass": "QualityFlagsECOSGplus",
        "kwargs": {
            "transforms": mmt_transforms.FillMissingWithSea(0, 6),
        },
        "colname": "QFLAGS",
    },
    "qscore": {
        "lcclass": "ScoreECOSGplus",
        "kwargs": {
            "transforms": mmt_transforms.ScoreTransform(divide_by=100),
        },
        "colname": "QSCORE",
    },
}


# FUNCTIONS
# =========
def merge_kwargs(kwargs1, kwargs2):
    """Merge arguments with care about multiple transforms


    Example
    -------
    >>> kwargs1 = {
        'cutoff': 0.3,
        'transforms': <mmt.datasets.transforms.FillMissingWithSea object at 0x14ea2ae32d90>
    }
    >>> kwargs2 = {
        'transforms': <mmt.datasets.transforms.ScoreTransform object at 0x14ea2f995e90>
    }
    >>> merge_kwargs(kwargs1, kwargs2)
    {
        'cutoff': 0.3,
        'transforms': Compose(
            <mmt.datasets.transforms.ScoreTransform object at 0x14ea2f995e90>
            <mmt.datasets.transforms.FillMissingWithSea object at 0x14ea2ae32d90>
        )
    }
    """
    if "transforms" in kwargs1.keys() and "transforms" in kwargs2.keys():
        trans1 = kwargs2.pop("transforms")
        trans2 = kwargs1.pop("transforms")

        # Make sure the key ("mask" or "image") is consistent
        if hasattr(trans1, "key") and hasattr(trans2, "key"):
            trans2.key = trans1.key

        transf = mmt_transforms.tvt.Compose([trans1, trans2])
        kwargs1.update({"transforms": transf})

    kwargs1.update(kwargs2)
    return kwargs1


def get_landcover_from_alias(lcname, print_kwargs=False, **kwargs):
    """Return the landcover object corresponding to the alias


    Parameters
    ----------
    lcname: str
        The landcover alias (short name, full class name or path)

    kwargs:
        Any extra argument to pass on to the constructor.


    Returns
    -------
    landcover: tgd.RasterDataset
        Landcover object corresponding to the alias.


    Examples
    --------
    >>> get_landcover_from_alias("esawc")
    <mmt.datasets.landcovers.ESAWorldCover object at 0x1544df313dd0>
    >>> get_landcover_from_alias("EcoclimapSG")
    <mmt.datasets.landcovers.EcoclimapSG object at 0x15442934f7d0>
    >>> get_landcover_from_alias("data/outputs/v2/infres-v2.0-v2outofbox2.eurat.u0.34")
    <mmt.datasets.landcovers.InferenceResults object at 0x15442a1de090>
    """
    try:
        checkpoint_path = misc.weights_to_checkpoint(lcname)
        is_checkpoint = True
    except ValueError:
        is_checkpoint = False

    if hasattr(landcovers, lcname):
        lc_class = getattr(landcovers, lcname)
    elif lcname in LANDCOVER_ALIASES.keys():
        kwargs = merge_kwargs(kwargs, LANDCOVER_ALIASES[lcname]["kwargs"])
        lc_class = getattr(landcovers, LANDCOVER_ALIASES[lcname]["lcclass"])
    elif os.path.isdir(lcname):
        kwargs.update({"path": lcname})
        lc_class = landcovers.InferenceResults
    elif is_checkpoint:
        kwargs.update({"checkpoint_path": checkpoint_path})
        lc_class = translators.EsawcToEsgpAsMap
    else:
        raise ValueError(
            f"Unable to find the landcover corresponding to the alias {lcname}"
        )

    if print_kwargs:
        pprint(kwargs)

    return lc_class(**kwargs)

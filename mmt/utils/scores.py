#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module with confusion matrix utilities
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn import metrics
from torch import tensor
from tqdm import tqdm

from mmt import _repopath_ as mmt_repopath
from mmt.datasets.landcovers import (ecoclimapsg_label_hierarchy,
                                     ecoclimapsg_labels)
from mmt.inference import io
from mmt.utils import misc

CACHE_DIRECTORY = os.path.join(mmt_repopath, "experiments", "cache")

if not os.path.isdir(CACHE_DIRECTORY):
    os.makedir(CACHE_DIRECTORY)

to_tensor = lambda x: tensor(x[:]).long().unsqueeze(0)


def _t(ls):
    return ["t" + l for l in ls]


def _p(ls):
    return ["p" + l for l in ls]


def prefix_labels(labels):
    """Add prefixes ('t' for 'true', 'p' for 'predicted') to a list of labels"""
    return _t(labels), _p(labels)


def oaccuracy(cmx):
    """Calculate the overall accuracy for the given confusion matrix"""
    if isinstance(cmx, pd.DataFrame):
        cmx = cmx.values

    return np.diag(cmx).sum() / cmx.sum()


def pprint_oaccuracies(cmxs):
    """Pretty print of the scores on a set of confusion matrices"""

    msg = """\n   OVERALL ACCURACIES:
+---------------------+----------------+----------------+
| Method              | Primary labels | ECOSG labels   |
+---------------------+----------------+----------------+
"""
    for method, cmx in cmxs.items():
        oa1 = str(
            np.round(oaccuracy(sum_primary_labels(cmx, ecoclimapsg_label_hierarchy)), 3)
        )
        oa2 = str(
            np.round(
                oaccuracy(
                    remove_absent_labels(cmx),
                ),
                3,
            )
        )
        msg += (
            "| " + "| ".join([method.ljust(20), oa1.ljust(15), oa2.ljust(15)]) + "|\n"
        )

    msg += "+---------------------+----------------+----------------+"
    print(msg)


def sum_primary_labels(cmx, label_hierarchy=ecoclimapsg_label_hierarchy):
    """Return the confusion matrix for primary labels


    Parameters
    ----------
    cmx: `pandas.DataFrame`
        Input confusion matrix

    label_hierarchy: dict
        Label names hierarchicaly organized in a dict


    Returns
    -------
    famcmx: `pandas.DataFrame`
        Output confusion matrix with primary labels
    """
    n_priml = len(label_hierarchy.keys())
    famcmx = np.zeros((n_priml, n_priml), dtype=np.int32)
    for i, fam1 in enumerate(label_hierarchy.keys()):
        for j, fam2 in enumerate(label_hierarchy.keys()):
            famcmx[i, j] = cmx.loc[
                _t(label_hierarchy[fam1]), _p(label_hierarchy[fam2])
            ].values.sum()

    return pd.DataFrame(
        data=famcmx,
        index=_t(label_hierarchy.keys()),
        columns=_p(label_hierarchy.keys()),
    )


def remove_absent_labels(cmx, axis=1):
    """Remove labels that have zero occurence in the given axis of the confusion matrix


    Parameters
    ----------
    cmx: `pandas.DataFrame`
        Input confusion matrix

    axis: {0, 1}
        Axis on which the absence are counted.
        If axis = 0, remove the labels absent in the prediction
        If axis = 1, remove the labels absent in the reference (default)


    Returns
    -------
    kcmx: `pandas.DataFrame`
        Output confusion matrix without zero-support labels
    """
    drop_idxs = cmx.values.sum(axis=axis) == 0
    drop_idxs[0] = True  # Remove pixels where ref is "no data"
    kcmx = cmx.drop(cmx.index[drop_idxs], axis=0, inplace=False)
    return kcmx.drop(kcmx.columns[drop_idxs], axis=1, inplace=False)


def norm_matrix(cmx, axis=1):
    """Normalize the confusion matrix along the given axis.


    Parameters
    ----------
    cmx: `pandas.DataFrame`
        Input confusion matrix

    axis: {0, 1}
        Axis on which the normalization is done
        If axis = 0, divide by the predicted amounts (precision matrix)
        If axis = 1, divide by the reference amounts (recall matrix, default)


    Returns
    -------
    kcmx: `pandas.DataFrame`
        Recall (axis = 1) or precision (axis = 0) matrix
    """
    kcmx = cmx.values
    nl = cmx.shape[0]
    denom = np.repeat(kcmx.sum(axis=axis), nl).reshape((nl, nl))
    if axis == 0:
        denom = denom.T

    return pd.DataFrame(data=kcmx / denom, index=cmx.index, columns=cmx.columns)


def perlabel_scores(cmx, scores=["user_accuracy", "prod_accuracy", "f1score"]):
    """Compute scores for each label


    Parameters
    ----------
    cmx: `pandas.DataFrame`
        Input confusion matrix

    scores: list of str
        List of scores to compute. Among "user_accuracy", "prod_accuracy", "f1score"


    Returns
    -------
    plscores: `pandas.DataFrame`
        Data frame with labels in index and the list of scores in columns


    Notes
    -----
    Some disambiguation on the scores:

    User accuracy = Precision = Positive predictive value:
        How often the class on the map will actually be present on the ground?

              #Correctly classified sites
        UA = -----------------------------
              #Total of classified sites

    Producer accuracy = Recall = Sensitivity = True positive rate:
        How often are real features on the ground correctly shown on the classified map?

              #Correctly classified sites
        PA = -----------------------------
               #Total of reference sites

    F1-score:
        Harmonic mean of precision and recall

              2 * UA * PA
        F1 = -------------
                PA + UA

    Sources:
        https://gsp.humboldt.edu/olm_2019/courses/GSP_216_Online/lesson6-2/metrics.html
        https://en.wikipedia.org/wiki/Confusion_matrix
    """
    labels = [s[1:] for s in cmx.index]
    plscores = pd.DataFrame(index=labels, columns=scores, dtype=float)
    for k, l in enumerate(labels):
        with warnings.catch_warnings(category=RuntimeWarning):
            warnings.simplefilter("ignore")
            if "user_accuracy" in scores:
                plscores.loc[l, "user_accuracy"] = cmx.iloc[k, k] / cmx.iloc[:, k].sum()
            if "prod_accuracy" in scores:
                plscores.loc[l, "prod_accuracy"] = cmx.iloc[k, k] / cmx.iloc[k, :].sum()
            if "f1score" in scores:
                precision = cmx.iloc[k, k] / cmx.iloc[k, :].sum()
                recall = cmx.iloc[k, k] / cmx.iloc[k, :].sum()
                plscores.loc[l, "f1score"] = (
                    2 * precision * recall / (precision + recall)
                )

    return plscores


def permethod_scores(cmxs, scorename="f1score"):
    """Compute scores for each label and each method.


    Parameters
    ----------
    cmxs: dict of `pandas.DataFrame`
        Confusion matrices obtained for each method

    scorename: {"user_accuracy", "prod_accuracy", "f1score"}
        Name of the score to use in the comparison


    Returns
    -------
    pmscores: `pandas.DataFrame`
        Score values for each label (rows) and each method (columns).
        The last column contain the name of the method with the highest score.
    """
    labels = [s[1:] for s in list(cmxs.values())[0].index]
    pmscores = pd.DataFrame(
        index=labels, columns=list(cmxs.keys()) + ["best_" + scorename], dtype=float
    )
    for method, cmx in cmxs.items():
        pls = perlabel_scores(cmx)
        pmscores.loc[:, method] = pls[scorename]

    pmscores["best_" + scorename] = pmscores.idxmax(axis=1)

    return pmscores


def latexstyleprint(pms):
    """Print a per-method score dataframe in a style close to a Latex table"""
    pms = pms.round(4)
    latexstyle = (
        " & ".join([" LABELS".rjust(36)] + [j.upper().rjust(15) for j in pms.columns])
        + "\n"
    )
    for i in pms.index:
        latexstyle += (
            " & ".join(
                [i.rjust(36)] + [str(pms.loc[i, j]).rjust(15) for j in pms.columns]
            )
            + "\n"
        )

    print(latexstyle)


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


def _compute_confusion_matrix_translator(translator, h5f, n_patches):
    """Compute the confusion matrix for a list of a single translator"""
    assert hasattr(translator, "checkpoint_path") and os.path.isfile(
        translator.checkpoint_path
    ), f"Invalid translator provided: {translator}"
    n_labels = h5f["esgp"].attrs["numberclasses"]
    cmx = np.zeros((n_labels, n_labels), dtype=np.int32)
    items = list(h5f["esawc"].keys())[:n_patches]

    for i in tqdm(items):
        x = to_tensor(h5f["esawc"].get(i))
        y_true = h5f["esgp"].get(i)

        y = translator.predict_from_data(x)

        cmx += metrics.confusion_matrix(
            y_true[:].ravel(), y.ravel(), labels=np.arange(n_labels)
        )

    return cmx


def _compute_confusion_matrix_translators(translator_list, h5f, n_patches):
    """Compute the confusion matrix for a list of translators"""
    n_labels = h5f["esgp"].attrs["numberclasses"]
    cmxs = {
        checkpoint_to_weight(tr.checkpoint_path): np.zeros(
            (n_labels, n_labels), dtype=np.int32
        )
        for tr in translator_list
    }
    items = list(h5f["esawc"].keys())[:n_patches]

    for i in tqdm(items):
        x = to_tensor(h5f["esawc"].get(i))
        y_true = h5f["esgp"].get(i)

        for tr in translator_list:
            y = tr.predict_from_data(x)

            cmxs[checkpoint_to_weight(tr.checkpoint_path)] += metrics.confusion_matrix(
                y_true[:].ravel(), y.ravel(), labels=np.arange(n_labels)
            )

    return cmxs


def _compute_confusion_matrix_ecosg(h5f, n_patches):
    """Compute the confusion matrix between ECOSG+ and ECOSG.
    The ECOSG labels are duplicated (x5) to match the ECOSG+ resolution."""
    n_labels = h5f["esgp"].attrs["numberclasses"]
    cmx = np.zeros((n_labels, n_labels), dtype=np.int32)
    items = list(h5f["esawc"].keys())[:n_patches]

    for i in tqdm(items):
        x = h5f["ecosg"].get(i)
        y_true = h5f["esgp"].get(i)
        cmx += metrics.confusion_matrix(
            y_true[:].ravel(), np.tile(x[:], (5, 5)).ravel(), labels=np.arange(n_labels)
        )

    return cmx


def compute_confusion_matrix(translator, h5f, n_patches):
    """Compute the confusion matrix between ECOSG+ and the inference
    performed by the translator on the validation set of pathches.


    Parameters
    ----------
    translator: `mmt.inference.translators.EsawcToEsgp` or list or "ecosg"
        Map translator performing the inference from ESAWC patches.
        If translator="ecosg", the ECOSG labels are duplicated to match
        the ECOSG+ resolution.

    h5f: dict
        Handles of the HDF5 open files containing the patches of the validation set

    n_patches: int
        Number of patches used in the computation


    Returns
    -------
    cmx: ndarray of shape (n_labels, n_labels) or dict
        Confusion matrix. If `translator` is a list, it returns a dict with
        the weights names as keys and the confusion matrices as values.
    """
    if isinstance(translator, list):
        return _compute_confusion_matrix_translators(translator, h5f, n_patches)
    if translator == "ecosg":
        return _compute_confusion_matrix_ecosg(h5f, n_patches)
    else:
        return _compute_confusion_matrix_translator(translator, h5f, n_patches)

    return cmx


def _pandafy(cmx):
    """Convert ndarray confusion matrix to Pandas data frame"""
    true_lnames, pred_lnames = prefix_labels(ecoclimapsg_labels)
    if isinstance(cmx, dict):
        return {
            k: pd.DataFrame(data=v, index=true_lnames, columns=pred_lnames)
            for k, v in cmx
        }
    else:
        return pd.DataFrame(data=cmx, index=true_lnames, columns=pred_lnames)


def look_in_cache_else_compute(translator, h5f, n_patches, pandas=True):
    """Look in the cache if these scores have been previously computed and
    compute them if they have not.



    Parameters
    ----------
    translator: `mmt.inference.translators.EsawcToEsgp` or list or "ecosg"
        Map translator performing the inference from ESAWC patches.
        If translator="ecosg", the ECOSG labels are duplicated to match
        the ECOSG+ resolution.

    h5f: dict
        Handles of the HDF5 open files containing the patches of the validation set

    n_patches: int
        Number of patches used in the computation

    pandas: bool
        If True, confusion matrices are returned as `pandas.DataFrame` instead
        of ndarray (default)


    Returns
    -------
    cmx: `pandas.DataFrame` of shape (n_labels, n_labels) or dict
        Confusion matrix. If `translator` is a list, it returns a dict with
        the weights names as keys and the confusion matrices as values.
    """
    cachedcmx_header = {
        "n_patches": repr(n_patches),
        "valdata": repr(
            [(h5f[k].file, len(h5f[k]), hash(h5f[k])) for k in sorted(h5f.keys())]
        ),
    }
    if hasattr(translator, "checkpoint_path"):
        cachedcmx_header["weights"] = repr(
            (
                io.get_epoch_of_best_model(
                    translator.checkpoint_path, return_iteration=True
                ),
                translator,
            )
        )

    cachedcmx_file = os.path.join(
        CACHE_DIRECTORY, f"cmx-{misc.hashdict(cachedcmx_header)}.pkl"
    )
    if os.path.isfile(cachedcmx_file):
        print(f"Loading scores from {cachedcmx_file}")
        with open(cachedcmx_file, "rb") as f:
            cachedcmx = pickle.load(f)

        cmx = cachedcmx["confusion_matrix"]
    else:
        print(
            f"No cached scored found for {misc.hashdict(cachedcmx_header)}. Computing confusion matrices..."
        )
        cmx = compute_confusion_matrix(translator, h5f, n_patches)
        cachedcmx = {**cachedcmx_header, "confusion_matrix": cmx}

        with open(cachedcmx_file, "wb") as f:
            pickle.dump(cachedcmx, f)

    if pandas:
        return _pandafy(cmx)
    else:
        return cmx


# EOF

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module with confusion matrix utilities
"""

import numpy as np
from sklearn import metrics
import pandas as pd

def t_(ls):
    return ["t"+l for l in ls]

def p_(ls):
    return ["p"+l for l in ls]

def oaccuracy(cmx):
    """Calculate the overall accuracy for the given confusion matrix"""
    if isinstance(cmx, pd.DataFrame):
        cmx = cmx.values
    
    return np.diag(cmx).sum()/cmx.sum()

def sum_primary_labels(cmx, label_hierarchy):
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
    famcmx = np.zeros((n_priml, n_priml), dtype = np.int32)
    for i, fam1 in enumerate(label_hierarchy.keys()):
        for j, fam2 in enumerate(label_hierarchy.keys()):
            famcmx[i,j] = cmx.loc[t_(label_hierarchy[fam1]), p_(label_hierarchy[fam2])].values.sum()
        
    return pd.DataFrame(data = famcmx, index = t_(label_hierarchy.keys()), columns = p_(label_hierarchy.keys()))

def remove_absent_labels(cmx, axis = 1):
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
    drop_idxs[0] = True # Remove pixels where ref is "no data"
    kcmx = cmx.drop(cmx.index[drop_idxs], axis = 0, inplace=False)
    return kcmx.drop(kcmx.columns[drop_idxs], axis = 1, inplace=False)

def norm_matrix(cmx, axis = 1):
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
    denom = np.repeat(kcmx.sum(axis=axis), nl).reshape((nl,nl))
    if axis == 0:
        denom = denom.T
    
    return pd.DataFrame(data=kcmx/denom, index = cmx.index, columns = cmx.columns)

def perlabel_scores(cmx, scores = ["user_accuracy", "prod_accuracy", "f1score"]):
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
    plscores = pd.DataFrame(index = labels, columns = scores, dtype = float)
    for k,l in enumerate(labels):
        if "user_accuracy" in scores:
            plscores.loc[l, "user_accuracy"] = cmx.iloc[k,k]/cmx.iloc[:,k].sum()
        if "prod_accuracy" in scores:
            plscores.loc[l, "prod_accuracy"] = cmx.iloc[k,k]/cmx.iloc[k,:].sum()
        if "f1score" in scores:
            precision = cmx.iloc[k,k]/cmx.iloc[k,:].sum()
            recall = cmx.iloc[k,k]/cmx.iloc[k,:].sum()
            plscores.loc[l, "f1score"] = 2 * precision * recall / (precision + recall)
    
    return plscores

def permethod_scores(cmxs, methods, scorename = "f1score"):
    """Compute scores for each label and each method.
    
    
    Parameters
    ----------
    cmxs: list of `pandas.DataFrame`
        Confusion matrices obtained for each method
    
    methods: list of str
        Name of each method
    
    scorename: {"user_accuracy", "prod_accuracy", "f1score"}
        Name of the score to use in the comparison
    
    
    Returns
    -------
    pmscores: `pandas.DataFrame`
        Score values for each label (rows) and each method (columns).
        The last column contain the name of the method with the highest score.
    """
    labels = [s[1:] for s in cmxs[0].index]
    pmscores = pd.DataFrame(index = labels, columns = methods + ["best_" + scorename], dtype = float)
    for cmx, method in zip(cmxs, methods):
        pls = perlabel_scores(cmx)
        pmscores.loc[:, method] = pls[scorename]
    
    pmscores["best_" + scorename] = pmscores.idxmax(axis = 1)
    
    return pmscores

def latexstyleprint(pms):
    """Print a per-method score dataframe in a style close to a Latex table"""
    latexstyle = " \t& ".join([" LABELS".rjust(36)] + [j.upper() for j in pms.columns]) + "\n"
    for i in pms.index:
        latexstyle += " \t& ".join([i.rjust(36)] + [str(pms.loc[i,j])[:5] for j in pms.columns]) + "\n"
    
    print(latexstyle)

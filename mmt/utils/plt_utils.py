#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module for graphics utilities
"""
import os

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATASET_COLORS = {
    "esawc.hdf5": cm.Set1.colors[0],
    "esawc-train.hdf5": cm.Set1.colors[0],
    "esgp.hdf5": cm.Set1.colors[1],
    "esgp-train.hdf5": cm.Set1.colors[1],
    "ecosg.hdf5": cm.Set1.colors[2],
    "ecosg-train.hdf5": cm.Set1.colors[2],
    "clc.hdf5": cm.Set1.colors[3],
    "oso.hdf5": cm.Set1.colors[4],
    "mos.hdf5": cm.Set1.colors[5],
    "cgls.hdf5": cm.Set1.colors[6],
}

DEFAULT_FIGFMT = (
    ".png"  # Images will be saved under this format (suffix of plt.savefig)
)

DEFAULT_FIGDIR = (
    "../figures"  # Images will be saved in this directory (prefix of plt.savefig)
)

DEFAULT_SAVEFIG = False  # If True, figures are saved in files but not shown. Else, figures are not saved in files but are shown


def plot_loss(
    train_loss, valid_loss, figsize=(10, 10), savefig=None, display=False
) -> None:
    """Plot the learning curve


    Parameters
    ----------
    train_loss: dict
        Loss values on the training set for each dataset

    valid_loss: dict
        Loss values on the validation set each dataset

    figsize: tuple of int
        Figure size

    savefig: str
        If provided, save the figure at the given path

    display: bool
        If True, displays the figure
    """
    fig = plt.figure(figsize=figsize)

    for k, v in train_loss.items():
        try:
            v = np.array(v)
            plt.plot(
                v[:, 0], v[:, 1], "--", color=DATASET_COLORS[k], label="training " + k
            )

        except:
            print(f"Error plotting loss for {k}. Values are {v}")

    for k, v in valid_loss.items():
        v = np.array(v)
        plt.plot(v[:, 0], v[:, 1], color=DATASET_COLORS[k], label="validation " + k)

    plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left", borderaxespad=0.5)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.grid()
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    if savefig:
        fig.savefig(savefig, bbox_inches="tight")
    if display:
        plt.show()

    plt.close(fig)


class PltPerClassMetrics(object):
    """Estimate scores at the end of the training. Not used in the paper."""

    def __call__(
        self, conf_matrix, labels=None, figsize=(20, 20), savefig=None, display=False
    ):
        for source, target_data in conf_matrix.items():
            for target, cm in target_data.items():
                source = os.path.basename(source)
                target = os.path.basename(target)
                TP = np.diag(cm)
                FP = np.sum(cm, axis=0) - TP
                FN = np.sum(cm, axis=1) - TP
                num_classes = cm.shape[0]

                tp_fp = TP + FP
                tp_fn = TP + FN
                precision = np.divide(
                    TP, tp_fp, out=np.zeros_like(TP), where=tp_fp != 0
                )
                recall = np.divide(TP, tp_fn, out=np.zeros_like(TP), where=tp_fn != 0)
                prec_rec = precision + recall
                fscore = np.divide(
                    2 * precision * recall,
                    prec_rec,
                    out=np.zeros_like(TP),
                    where=prec_rec != 0,
                )

                indices = np.arange(num_classes)

                if labels is None:
                    labels = np.array([str(i) for i in indices])
                suf = tp_fn > 0
                indices = indices[suf]  # [1:]
                labels = labels[suf]  # [1:]
                precision = precision[suf]  # [1:]
                recall = recall[suf]  # [1:]
                fscoret = fscore[suf]  # [1:]

                f, ax = plt.subplots(1, 2, figsize=figsize)
                plt.title("Source : {} Target: {}".format(source, target))
                ax[0].barh(indices, precision, 0.2, label="precision", color="navy")
                ax[0].barh(indices + 0.3, recall, 0.2, label="recall", color="c")
                ax[0].barh(
                    indices + 0.6, fscoret, 0.2, label="f-score", color="darkorange"
                )
                ax[0].set_yticks(())
                ax[0].legend(loc="best")

                for i, c in zip(indices, labels):
                    ax[0].text(-0.3, i, c)

                x = np.log10(tp_fn, out=np.zeros_like(tp_fn), where=tp_fn != 0)
                ax[1].scatter(
                    x,
                    fscore,
                )
                for i, c in zip(indices, labels):
                    ax[1].annotate(c, (x[i], fscore[i]))
                ax[1].set_xlabel("log support")
                ax[1].set_ylabel("fscore")

                if savefig:
                    f.savefig(
                        savefig + "_source_{}_target_{}.png".format(source, target),
                        bbox_inches="tight",
                    )
                if display:
                    plt.show()
                plt.close(f)
                f = None
                labels = None


def plot_confusion_matrix(
    dfcmx, accuracy_in_corner=False, annot=False, figname=None, figtitle=None
) -> None:
    """Heatmap of the confusion matrix coefficients


    Parameters
    ----------
    dfcmx: `pandas.DataFrame`
        Confusion matrix in a data frame

    accuracy_in_corner: bool
        If True, the overall accuracy is displayed in the upper-right corner

    annot: bool
        If True, the cells are annotated with the values of the matrix

    figname: str
        The figure name

    figtitle: str
        The figure title
    """

    if figtitle is None:
        figtitle = "Confusion matrix"
    if figname is None:
        figname = "onepatchplot"

    fig = plt.figure(figsize=(12, 10))
    ax = sns.heatmap(
        dfcmx, annot=annot, cmap=sns.cubehelix_palette(as_cmap=True), vmin=0, vmax=1
    )
    ax.set_title(figtitle)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Reference")
    if accuracy_in_corner:
        nx, ny = dfcmx.shape
        oa = np.round(np.diag(dfcmx.values).sum() / dfcmx.values.sum(), 3)
        ax.text(0.8 * nx, 0.1 * ny, f"OA={oa}", fontsize=18)
    fig.add_axes(ax)
    if DEFAULT_SAVEFIG:
        figpath = os.path.join(DEFAULT_FIGDIR, figname + DEFAULT_FIGFMT)
        fig.savefig(figpath)
        plt.close()
        print("Figure saved:", figpath)
    else:
        fig.show()


def patches_over_domain(
    qdom, bboxs, zoomout=3, background="osm", details=8, figname=None, figtitle=None
) -> None:
    """Locate the patch on a map.


    Parameters
    ----------
    qdom: `wopt.domains.GeoRectangle` or (2,2)-tuple
        Query domain bounding box (upper-left, lower-right) to be located.
        Will be displayed in red.

    bboxs: list of `wopt.domains.GeoRectangle`
        List of bounding boxes covering the domain. Will be displayed in blue

    zoomout: float
        Zoom-out level (coefficient applied to the size of `bbox` on both
        dimensions). The higher the larger will be the backgournd extent

    background: {"osm", "terrain"}
        The map to be drawn in background, Select "osm" for Open Street Map
        features. Select "terrain" for terrain elevation.

    details: int, default=8
        Level of details to be displayed in the background. The higher
        the more detailed is the backgound but the heavier is the figure

    figname: str
        Name of the figure to be saved

    figtitle: str
        Title of the figure
    """
    if figtitle is None:
        figtitle = f"Patches location over {background} background"
    if figname is None:
        figname = f"patches_over_{background}"

    if hasattr(qdom, "to_tlbr"):
        (ulx, uly), (lrx, lry) = qdom.to_tlbr()
    else:
        (ulx, uly), (lrx, lry) = qdom

    dlat = abs(uly - lry)
    dlon = abs(ulx - lrx)
    locextent = [
        ulx - zoomout * dlon,
        lrx + zoomout * dlon,
        lry - zoomout * dlat,
        uly + zoomout * dlat,
    ]
    xticks = np.linspace(locextent[0], locextent[1], 5)
    yticks = np.linspace(locextent[2], locextent[3], 5)

    if background in ["osm", "OSM"]:
        background_image = cimgt.OSM()
    elif background in ["terrain", "relief", "stamen"]:
        background_image = cimgt.Stamen("terrain-background")
    else:
        raise ValueError(f"Unknown background: {background}")

    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(1, 1, 1, projection=background_image.crs)
    rectangle = mpatches.Rectangle(
        xy=[ulx, lry],
        width=dlon,
        height=dlat,
        facecolor="red",
        alpha=0.2,
        transform=ccrs.PlateCarree(),
    )
    ax.set_extent(locextent)
    ax.add_image(background_image, details)
    # ax.add_patch(rectangle)
    for bbox in bboxs:
        if hasattr(bbox, "to_tlbr"):
            (ulx, uly), (lrx, lry) = bbox.to_tlbr()
        else:
            (ulx, uly), (lrx, lry) = bbox
        dlat = abs(uly - lry)
        dlon = abs(ulx - lrx)
        rectangle = mpatches.Rectangle(
            xy=[ulx, lry],
            width=dlon,
            height=dlat,
            facecolor="blue",
            alpha=0.5,
            transform=ccrs.PlateCarree(),
        )
        ax.add_patch(rectangle)
    ax.set_title(figtitle)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels(np.round(xticks, 3))
    ax.set_yticklabels(np.round(yticks, 3))
    fig.tight_layout()

    if DEFAULT_SAVEFIG:
        figpath = os.path.join(DEFAULT_FIGDIR, figname + DEFAULT_FIGFMT)
        fig.savefig(figpath)
        plt.close()
        print("Figure saved:", figpath)
    else:
        fig.show()


# EOF

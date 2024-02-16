import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
from torch.nn import Softmax2d
import torch
import os
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt

coloring = {
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


fmtImages = ".png"
# Images will be saved under this format (suffix of plt.savefig)

figureDir = "../figures"
# Images will be saved in this directory (prefix of plt.savefig)

storeImages = False
# If True, figures are saved in files but not shown
# If False, figures are not saved in files but always shown


class plt_loss(object):
    def __call__(
        self, train_loss, valid_loss, figsize=(10, 10), savefig=None, display=False
    ):
        train_loss = np.array(train_loss)
        valid_loss = np.array(valid_loss)
        f = plt.figure(figsize=figsize)
        plt.plot(train_loss[:, 0], train_loss[:, 1], c="blue", label="training loss")
        plt.plot(valid_loss[:, 0], valid_loss[:, 1], c="green", label="validation loss")

        plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left", borderaxespad=0.5)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        if savefig:
            f.savefig(savefig, bbox_inches="tight")
        if display:
            plt.show()
        plt.close(f)
        f = None


class plt_loss2(object):
    def __call__(
        self, train_loss, valid_loss, figsize=(10, 10), savefig=None, display=False
    ):
        # train_loss=np.array(train_loss)
        # valid_loss=np.array(valid_loss)
        f = plt.figure(figsize=figsize)
        for k, v in train_loss.items():
            try:
                v = np.array(v)
                plt.plot(
                    v[:, 0], v[:, 1], "--", color=coloring[k], label="training " + k
                )

                # xnew = np.linspace(int(np.min(v[:,0])), int(np.max(v[:,0])), int(np.max(v[:,0])-np.min(v[:,0])*10) )
                # spl = make_interp_spline(v[:,0], v[:,1], k=3)  # type: BSpline
                # power_smooth = np.convolve(v[:,1],[1/30]*30, 'same',)
                # plt.plot(v[30:-30,0], power_smooth[30:-30], label="Interpolated training " + k)
                # plt.plot(v[:, 0], v[:, 1], label="training_ " + k)
            except:
                print(f"Error plotting loss for {k}. Values are {v}")
        for k, v in valid_loss.items():
            v = np.array(v)
            plt.plot(v[:, 0], v[:, 1], color=coloring[k], label="validation " + k)

        plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left", borderaxespad=0.5)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.grid()
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        if savefig:
            f.savefig(savefig, bbox_inches="tight")
        if display:
            plt.show()
        plt.close(f)
        f = None


class plt_loss3(object):
    def __init__(self, loss={}):
        self.loss = loss

    def _setloss(self, l):
        for k, v in l.items():
            if self.loss.get(k) is not None:
                self.loss[k].append(v)
            else:
                self.loss[k] = [v]

    def __call__(self, loss, figsize=(10, 10), savefig=None, display=False):
        f = plt.figure(figsize=figsize)
        self._setloss(loss)
        for k, v in self.loss.items():
            v = np.array(v)
            plt.plot(v[:, 0], v[:, 1], label=k)

        plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left", borderaxespad=0.5)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        if savefig:
            f.savefig(savefig, bbox_inches="tight")
        if display:
            plt.show()
        plt.close(f)


class PltPerClassMetrics(object):
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
                # TN = []
                # for i in range(num_classes):
                #     temp = np.delete(cm, i, 0)  # delete ith row
                #     temp = np.delete(temp, i, 1)  # delete ith column
                #     TN.append(sum(sum(temp)))

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
                # ax[0].subplots_adjust(left=.25)
                # ax[0].subplots_adjust(top=.95)
                # ax[0].subplots_adjust(bottom=.05)

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
                # ax[1].set_xscale('log')

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


class plt_kappa(object):
    def __call__(
        self, train_kappa, valid_kappa, figsize=(10, 10), savefig=None, display=False
    ):
        train_kappa = np.array(train_kappa)
        valid_kappa = np.array(valid_kappa)
        f = plt.figure(figsize=figsize)
        plt.plot(train_kappa[:, 0], train_kappa[:, 1], c="blue", label="training kappa")
        plt.plot(
            valid_kappa[:, 0], valid_kappa[:, 1], c="green", label="validation kappa"
        )

        plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left", borderaxespad=0.5)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.xlabel("epoch")
        plt.ylabel("loss")
        if savefig:
            f.savefig(savefig, bbox_inches="tight")
        if display:
            plt.show()
        plt.close(f)
        f = None


class plt_scatter(object):
    def __call__(
        self,
        X,
        Y,
        C,
        label,
        xlabel,
        ylabel,
        figsize=(10, 10),
        savefig=None,
        display=False,
    ):
        f = plt.figure(figsize=figsize)
        for x, y, c, lab in zip(X, Y, C, label):
            plt.scatter(x, y, c=np.array([c]), label=lab, edgecolors="black", s=100)

        plt.legend(bbox_to_anchor=(1.0, 0.5), loc="center left", borderaxespad=0.5)
        plt.tight_layout(rect=[0, 0, 1, 1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if savefig:
            f.savefig(savefig, bbox_inches="tight")
        if display:
            plt.show()
        plt.close(f)
        f = None


class plt_image(object):
    def __init__(self):
        pass

    def masks_to_img(self, masks):
        return np.argmax(masks, axis=0) + 1

    def probability_map(self, output):
        m = Softmax2d()
        prob, _ = torch.max(m(output), dim=1)
        return prob

    def colorbar(self, fig, ax, cmap, labels_name):
        n_labels = len(labels_name)
        mappable = cm.ScalarMappable(cmap=cmap)
        mappable.set_array([])
        mappable.set_clim(0.5, n_labels + 0.5)
        colorbar = fig.colorbar(mappable, ax=ax)
        colorbar.set_ticks(np.linspace(1, n_labels + 1, n_labels + 1))
        colorbar.set_ticklabels(labels_name)
        if len(labels_name) > 30:
            colorbar.ax.tick_params(labelsize=5)
        else:
            colorbar.ax.tick_params(labelsize=15)

    def normalize_value_for_diplay(self, inputs, src_type):
        for i in range(len(src_type.id_labels)):
            inputs = np.where(inputs == src_type.id_labels[i], i + 1, inputs)
        return inputs

    def show_res(
        self, inputs, src_type, labels, tgt_type, outputs, save_path, display=False
    ):
        fig, axs = plt.subplots(3, 3, figsize=(30, 20))
        for i in range(3):
            show_labels = self.masks_to_img(labels.cpu().numpy()[i]).astype("uint8")
            if outputs.shape[1] > 1:
                show_outputs = self.masks_to_img(
                    outputs.cpu().detach().numpy()[i]
                ).astype("uint8")
            else:
                show_outputs = show_outputs.cpu().detach().numpy()[i][0]
            input = inputs.cpu().detach().numpy()[i][0]
            input = self.normalize_value_for_diplay(input, src_type)

            axs[i][0].imshow(
                input,
                cmap=src_type.matplotlib_cmap,
                vmin=1,
                vmax=len(src_type.labels_name),
                interpolation="nearest",
            )
            axs[i][0].axis("off")
            self.colorbar(
                fig, axs[i][0], src_type.matplotlib_cmap, src_type.labels_name
            )

            axs[i][1].imshow(
                show_labels,
                cmap=tgt_type.matplotlib_cmap,
                vmin=1,
                vmax=len(tgt_type.labels_name),
                interpolation="nearest",
            )
            axs[i][1].axis("off")
            self.colorbar(
                fig, axs[i][1], tgt_type.matplotlib_cmap, tgt_type.labels_name
            )

            axs[i][2].imshow(
                show_outputs,
                cmap=tgt_type.matplotlib_cmap,
                vmin=1,
                vmax=len(tgt_type.labels_name),
                interpolation="nearest",
            )
            axs[i][2].axis("off")
            self.colorbar(
                fig, axs[i][2], tgt_type.matplotlib_cmap, tgt_type.labels_name
            )

        # plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        if display:
            plt.show()
        plt.close(fig)
        fig = None

    def show_one_res(
        self, inputs, src_type, labels, tgt_type, outputs, save_path, display=False
    ):
        fig, axs = plt.subplots(1, 3, figsize=(30, 20))
        show_labels = self.masks_to_img(labels.cpu().numpy()[0]).astype("uint8")
        if outputs.shape[1] > 1:
            show_outputs = self.masks_to_img(outputs.cpu().detach().numpy()[0]).astype(
                "uint8"
            )
        else:
            show_outputs = outputs.cpu().detach().numpy()[0][0]
        input = inputs.cpu().detach().numpy()[0][0]
        input = self.normalize_value_for_diplay(input, src_type)

        axs[0].imshow(
            input,
            cmap=src_type.matplotlib_cmap,
            vmin=1,
            vmax=len(src_type.labels_name),
            interpolation="nearest",
        )
        axs[0].axis("off")
        self.colorbar(fig, axs[0], src_type.matplotlib_cmap, src_type.labels_name)

        axs[1].imshow(
            show_labels,
            cmap=tgt_type.matplotlib_cmap,
            vmin=1,
            vmax=len(tgt_type.labels_name),
            interpolation="nearest",
        )
        axs[1].axis("off")
        self.colorbar(fig, axs[1], tgt_type.matplotlib_cmap, tgt_type.labels_name)

        axs[2].imshow(
            show_outputs,
            cmap=tgt_type.matplotlib_cmap,
            vmin=1,
            vmax=len(tgt_type.labels_name),
            interpolation="nearest",
        )
        axs[2].axis("off")
        self.colorbar(fig, axs[2], tgt_type.matplotlib_cmap, tgt_type.labels_name)

        # plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        if display:
            plt.show()
        plt.close(fig)
        fig = None

    def show_probability_map(
        self, inputs, src_type, labels, tgt_type, outputs, save_path, display=False
    ):
        fig, axs = plt.subplots(3, 3, figsize=(30, 20))
        for i in range(3):
            show_labels = self.masks_to_img(labels.cpu().numpy()[i]).astype("uint8")
            show_outputs = self.masks_to_img(outputs.cpu().detach().numpy()[i]).astype(
                "uint8"
            )
            prob = self.probability_map(outputs).cpu().detach().numpy()[i]

            p = axs[i][2].imshow(prob, cmap="magma", vmin=0, vmax=1)
            axs[i][2].axis("off")
            fig.colorbar(p, ax=axs[i][2])

            axs[i][0].imshow(
                show_labels,
                cmap=tgt_type.matplotlib_cmap,
                vmin=1,
                vmax=len(tgt_type.labels_name),
                interpolation="nearest",
            )
            axs[i][0].axis("off")
            self.colorbar(
                fig, axs[i][0], tgt_type.matplotlib_cmap, tgt_type.labels_name
            )

            axs[i][1].imshow(
                show_outputs,
                cmap=tgt_type.matplotlib_cmap,
                vmin=1,
                vmax=len(tgt_type.labels_name),
                interpolation="nearest",
            )
            axs[i][1].axis("off")
            self.colorbar(
                fig, axs[i][1], tgt_type.matplotlib_cmap, tgt_type.labels_name
            )

        # plt.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        if display:
            plt.show()
        plt.close(fig)
        fig = None

def plot_confusion_matrix(dfcmx, accuracy_in_corner = False, annot=False, figname=None, figtitle=None):
    """Heatmap of the confusion matrix coefficients"""
    
    if figtitle is None:
        figtitle = "Confusion matrix"
    if figname is None:
        figname = "onepatchplot"
    
    fig = plt.figure(figsize=(12,10))
    ax = sns.heatmap(dfcmx, annot=annot, cmap=sns.cubehelix_palette(as_cmap=True))
    ax.set_title(figtitle)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Reference")
    if accuracy_in_corner:
        nx, ny = dfcmx.shape
        oa = np.round(np.diag(dfcmx.values).sum()/dfcmx.values.sum(), 3)
        ax.text(0.8*nx, 0.1*ny, f"OA={oa}", fontsize=18)
    fig.add_axes(ax)
    if storeImages:
        figpath = os.path.join(figureDir, figname + fmtImages)
        plt.savefig(figpath)
        plt.close()
        print("Figure saved:", figpath)
    else:
        plt.show(block=False)


def patches_over_domain(qdom, bboxs, zoomout=3, background="osm", details=8, figname=None, figtitle=None):
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
    
    dlat = abs(uly-lry)
    dlon = abs(ulx-lrx)
    locextent = [ulx - zoomout*dlon, lrx + zoomout*dlon, lry - zoomout*dlat, uly + zoomout*dlat]
    xticks = np.linspace(locextent[0],locextent[1],5)
    yticks = np.linspace(locextent[2],locextent[3],5)
    
    if background in ["osm", "OSM"]:
        background_image = cimgt.OSM()
    elif background in ["terrain", "relief", "stamen"]:
        background_image = cimgt.Stamen('terrain-background')
    else:
        raise ValueError(f"Unknown background: {background}")
    
    fig = plt.figure(figsize=(20,15))
    ax = fig.add_subplot(1, 1, 1, projection=background_image.crs)
    rectangle = mpatches.Rectangle(
        xy=[ulx, lry],
        width=dlon, height=dlat,
        facecolor='red',alpha=0.2, transform=ccrs.PlateCarree()
    )
    ax.set_extent(locextent)
    ax.add_image(background_image, details)
    # ax.add_patch(rectangle)
    for bbox in bboxs:
        if hasattr(bbox, "to_tlbr"):
            (ulx, uly), (lrx, lry) = bbox.to_tlbr()
        else:
            (ulx, uly), (lrx, lry) = bbox
        dlat = abs(uly-lry)
        dlon = abs(ulx-lrx)
        rectangle = mpatches.Rectangle(
            xy=[ulx, lry],
            width=dlon, height=dlat,
            facecolor='blue',alpha=0.5, transform=ccrs.PlateCarree()
        )
        ax.add_patch(rectangle)
    ax.set_title(figtitle)
    ax.set_xticks(xticks, crs = ccrs.PlateCarree())
    ax.set_yticks(yticks, crs = ccrs.PlateCarree())
    ax.set_xticklabels(np.round(xticks,3))
    ax.set_yticklabels(np.round(yticks,3))
    fig.tight_layout()
        
    if storeImages:
        figpath = os.path.join(figureDir, figname + fmtImages)
        plt.savefig(figpath)
        plt.close()
        print("Figure saved:", figpath)
    else:
        plt.show(block=False)


# EOF

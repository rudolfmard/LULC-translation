#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module with inference map translator
"""

import os
import shutil
import time
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import onnx
import pickle
import onnxruntime as ort
import torch
from torchgeo.datasets.utils import BoundingBox as TgeoBoundingBox
from torchgeo import samplers

from mmt import _repopath_ as mmt_repopath
from mmt.inference import io
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import misc


# BASE CLASSES
# ============
class OnnxModel:
    """Wrapper for inference from ONNX model"""

    def __init__(self, onnxfilename, inputname=None, outputname=None):
        self.onnxfilename = onnxfilename
        self.ort_session = ort.InferenceSession(
            onnxfilename, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        if outputname is None:
            self.outputname = (
                os.path.basename(onnxfilename).split(".")[0].split("_")[-1]
            )
        if inputname is None:
            self.inputname = os.path.basename(onnxfilename).split(".")[0].split("_")[-2]

        # Check the model
        onnx_model = onnx.load(onnxfilename)
        onnx.checker.check_model(onnx_model)

    def __call__(self, x):
        outputs = self.ort_session.run(
            [self.outputname],
            {self.inputname: x},
        )
        return outputs[0]


class MapTranslator:
    """Abstract class to make map translation models in inference mode.
    
    Define a set of nested methods that help scaling up the inference area
    and give a common API for all models (segmentation vs pixelwise).
    By default, the method `predict_from_domain` is called when a translator
    is used as a transform.
    
    
    Parameters
    ----------
    checkpoint_path: str
        Path to the Pytorch checkpoint from which are loaded the auto-encoders.
        For the child class `MapMerger`, this attribute is not used and set to "merger"
    
    device: {"cuda", "cpu"}
        Device on which the inference is done.
    
    remove_tmpdirs: bool, default = True
        If True, temporary directory are erased at the of the function using them.
    
    
    Main attributes
    ---------------
    checkpoint_path: str
        See parameters
        
    landcover: `mmt.datasets.landcovers.TorchgeoLandcover`
        Land cover dataset that will be used to define the sampler and the CRS
        of the query domain.
    
    
    Main methods
    ------------
    predict_from_data:
        Apply translation to matrices of land cover labels.
    
    predict_from_domain:
        Apply translation to a geographical domain.
        Calls `predict_from_data`
        
    predict_from_large_domain:
        Apply translation to a large domain in made in tiling the large domain into small patches.
        Calls `predict_from_domain`
    """
    def __init__(
        self,
        checkpoint_path=None,
        device="cuda",
        remove_tmpdirs=True,
    ):
        self.checkpoint_path = checkpoint_path
        self.remove_tmpdirs = remove_tmpdirs
        self.device = torch.device(device)
        self.shortname = self.__class__.__name__ + "." + os.path.basename(checkpoint_path)
        self.landcover = None
        
    def __call__(self, qb):
        return self.predict_from_domain(qb)
    
    def __repr__(self):
        return str(self.__class__) + " checkpoint_path=" + self.checkpoint_path
        
    def predict_from_data(self, x):
        """Apply translation to matrices of land cover labels.
        
        
        Parameters
        ----------
        x: ndarray or `torch.Tensor` of shape (1, H, W)
            Matrix of land cover labels in a Pytorch tensor with H=height, W=width
        
        
        Returns
        -------
        y: ndarray of shape (H', W')
            Matrix of predicted ECOSG+ land cover labels
        """
        raise NotImplementedError
    
    def predict_from_domain(self, qb):
        """Apply translation to a geographical domain.
        
        
        Parameters
        ----------
        qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
            Geographical domain
        
        
        Returns
        -------
        y: ndarray of shape (H', W')
            Matrix of predicted ECOSG+ land cover labels
        """
        raise NotImplementedError
    
    def predict_from_large_domain(
        self, qb, output_dir="[id]", tmp_dir="[id]", n_max_files=200, n_px_max=600
    ):
        """Apply translation to a large domain in made in tiling the large domain into small patches.
        
        As big domains cannot fit in memory, the inference is done on small patches
        (of size `n_px_max` pixels). The result of this inference is sotred in TIF files
        that can be read by Torchgeo. However, as the number of TIF files can be very
        large, there are stitched together into a smaller number (no more than `n_max_files`).
        
        
        Parameters
        ----------
        qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
            Geographical domain
        
        output_dir: str
            Path to the output directory where the TIF files are stored.
            If the keyword "[id]" is found, it will be replaced by a 6 random
            digits and the date (ex: "aa2xzh.01Nov-14h38")
        
        tmp_dir: str
            Path to the temporary directory. Used for the direct inference output (many TIF files)
            If the keyword "[id]" is found, it will be replaced by a 6 random
            digits and the date (ex: "aa2xzh.01Nov-14h38").
            Removed at the end of this function if `remove_tmpdirs=True`
        
        n_max_files: int, default=200
            Maximum number of TIF files created in `output_dir`.
            If `n_max_files=0`, no stitching is made, `output_dir` is then
            a copy of `tmp_dir`
        
        n_px_max: int, default=600
            Size (in pixels) of a single patch for inference.
        
        
        Returns
        -------
        output_dir: str
            Path to the output directory where the TIF files are stored (with completion)
        """
        if not isinstance(qb, TgeoBoundingBox):
            qb = qb.to_tgbox(self.landcover.crs)
        
        dir_id = misc.id_generator()
        if "[id]" in output_dir:
            output_dir = output_dir.replace(
                "[id]", dir_id + time.strftime(".%d%b-%Hh%M")
            )
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if "[id]" in tmp_dir:
            tmp_dir = tmp_dir.replace(
                "[id]", "tmp." + dir_id + time.strftime(".%d%b-%Hh%M")
            )
        if not os.path.exists(tmp_dir):
            os.makedirs(tmp_dir)

        margin = n_px_max // 6
        sampler = samplers.GridGeoSampler(
            self.landcover, size=n_px_max, stride=n_px_max - margin, roi=qb
        )
        if len(sampler) == 0:
            raise ValueError(f"Empty sampler. size={n_px_max}, stride={n_px_max - margin}, roi={qb}, landcover bounds={self.landcover.bounds}")

        for iqb in tqdm(sampler, desc=f"Inference over {len(sampler)} patches"):
            tifpatchname = f"N{iqb.minx}_E{iqb.maxy}.tif"
            if os.path.exists(os.path.join(tmp_dir, tifpatchname)):
                continue

            y = self.predict_from_domain(iqb)

            io.dump_labels_in_tif(
                y, iqb, self.landcover.crs, os.path.join(tmp_dir, tifpatchname)
            )
        
        if n_max_files > 0:
            io.stitch_tif_files(tmp_dir, output_dir, n_max_files=n_max_files, prefix = self.__class__.__name__, verbose = True)
        else:
            shutil.copytree(tmp_dir, output_dir, dirs_exist_ok=True)

        if self.remove_tmpdirs:
            shutil.rmtree(tmp_dir)
        else:
            print(f"Kept: tmp_dir={tmp_dir}")

        return output_dir


# CHILD CLASSES
# =============

class EsawcToEsgp(MapTranslator):
    """Translate from ESA World Cover to ECOCLIMAP-SG+ with map translation auto-encoders"""
    def __init__(
        self,
        checkpoint_path=os.path.join(mmt_repopath, "saved_models", "vanilla.ckpt"),
        device="cuda",
        remove_tmpdirs=True,
        always_predict=True,
    ):
        super().__init__(checkpoint_path, device, remove_tmpdirs)
        self.always_predict = always_predict
        
        self.esawc = landcovers.ESAWorldCover(transforms=mmt_transforms.EsawcTransform)
        self.esawc_transform = mmt_transforms.OneHot(
            self.esawc.n_labels + 1, device=self.device
        )
        self.encoder_decoder = io.load_pytorch_model(
            checkpoint_path, lc_in="esawc", lc_out="esgp"
        )
        self.encoder_decoder.to(self.device)
        self.landcover = self.esawc

    def predict_from_data(self, x):
        """Run the translation from matrices of land cover labels
        :x: `torch.Tensor` of shape (N, 1, H, W)
        """
        if not self.always_predict:
            x.to(self.device)
            _, c = torch.unique(x, return_counts = True)
            if any(c/c.sum() > 0.9):
                return np.zeros([s//6 for s in x.shape[-2:]])
                
        x = self.esawc_transform(x)
        with torch.no_grad():
            y = self.encoder_decoder(x.float())
        
        return y.argmax(1).squeeze().cpu().numpy()

    def predict_from_domain(self, qb):
        """Run the translation from geographical domain
        :qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
        """
        if not isinstance(qb, TgeoBoundingBox):
            qb = qb.to_tgbox(self.esawc.crs)

        x = self.esawc[qb]
        return self.predict_from_data(x["mask"])


class EsawcEcosgToEsgpRFC(MapTranslator):
    """Translate from ESA World Cover and ECOCLIMAP-SG to ECOCLIMAP-SG+ with a random forest classifer"""
    def __init__(
        self,
        checkpoint_path=os.path.join(mmt_repopath, "saved_models", "vanilla.ckpt"),
        classifier_path=os.path.join(mmt_repopath, "saved_models", "rfc_200trees.pkl"),
        device="cuda",
        remove_tmpdirs=True,
    ):
        super().__init__(checkpoint_path, device, remove_tmpdirs)
        
        # Landcovers
        self.esawc = landcovers.ESAWorldCover(transforms=mmt_transforms.EsawcTransform)
        self.esawc_transform = mmt_transforms.OneHot(
            self.esawc.n_labels + 1, device=self.device
        )
        self.ecosg = landcovers.EcoclimapSG()
        self.ecosg_transform = mmt_transforms.OneHot(
            self.ecosg.n_labels + 1, device=self.device
        )
        self.landcover = self.esawc & self.ecosg
        
        # Models
        self.esawc_encoder = io.load_pytorch_model(checkpoint_path, lc_in = "esawc", lc_out = "encoder")
        self.ecosg_encoder = io.load_pytorch_model(checkpoint_path, lc_in = "ecosg", lc_out = "encoder")
        self.esawc_encoder.to(self.device)
        self.ecosg_encoder.to(self.device)
        with open(classifier_path, "rb") as f:
            self.rfc = pickle.load(f)
            self.rfc.verbose = 0
            print(f"Model loaded from {f.name}")
        
        self.flatten = torch.nn.Flatten(start_dim=1, end_dim=-1)
        self.avg = torch.nn.AvgPool2d((6, 6))
        self.avg30 = torch.nn.AvgPool2d((30, 30))

    def predict_from_data(self, x_esawc, x_ecosg):
        """Run the translation from matrices of land cover labels
        :x: `torch.Tensor` of shape (N, 1, H, W)
        """
        assert all([s % 30 == 0 for s in x_esawc.shape[-2:]]), f"Invalid shape {x_esawc.shape[-2:]}. Must be a multiple of 30"
        tgt_shape = [s // 6 for s in x_esawc.shape[-2:]]
        
        x_esawc = self.esawc_transform(x_esawc)
        x_ecosg = self.ecosg_transform(x_ecosg)
        
        if sum(np.array(x_ecosg.shape[-2:])- np.array(x_esawc.shape[-2:])) == 0:
            x_ecosg = self.avg30(x_ecosg)
        elif sum(np.array(x_ecosg.shape[-2:])- np.array(x_esawc.shape[-2:])//30) == 0:
            pass
        else:
            raise ValueError(f"Shapes of ESAWC data {x_esawc.shape[-2:]} and ECOSG {x_ecosg.shape[-2:]} mismatch")
        
        with torch.no_grad():
            emba = self.esawc_encoder(x_esawc.float())
            embo = self.ecosg_encoder(x_ecosg.float())
        
        femba = self.flatten(self.avg(emba.squeeze())).cpu().numpy()
        fembo = self.flatten(self.avg(embo.squeeze())).cpu().numpy()
        X = np.concatenate([femba.T, fembo.T], axis = 1)
        
        y = self.rfc.predict(X)
        
        return y.reshape(tgt_shape)

    def predict_from_domain(self, qb):
        """Run the translation from geographical domain
        :qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
        """
        if not isinstance(qb, TgeoBoundingBox):
            qb = qb.to_tgbox(self.esawc.crs)
        
        x = self.landcover[qb]
        
        orig_shape = x["mask"].shape[-2:]
        if not all([s % 30 == 0 for s in orig_shape]):
            ccrop = tvt.CenterCrop([30*(s // 30) for s in orig_shape])
            x = ccrop(x["mask"])
            print(f"Original shape {orig_shape} adjusted to {x.shape[-2:]}")
        else:
            x = x["mask"]
            
        return self.predict_from_data(x[0].unsqueeze(0), x[1].unsqueeze(0))


class MapMerger(MapTranslator):
    """Merge map with ECOCLIMAP-SG+ according to a quality flag criterion"""
    
    def __init__(
        self,
        source_map_path,
        device="cpu",
        remove_tmpdirs=True,
        merge_criterion = "qflag2_nosea",
    ):
        super().__init__("merger", device, remove_tmpdirs)
        
        self.esgp = landcovers.EcoclimapSGplus()
        self.qflags = landcovers.QualityFlagsECOSGplus(transforms=mmt_transforms.FillMissingWithSea(0,6))
        self.landcover = landcovers.InferenceResults(path = source_map_path, res = self.esgp.res)
        self.landcover.res = self.esgp.res
        if callable(merge_criterion):
            self.merge_criterion = merge_criterion
        else:
            self.merge_criterion = eval(merge_criterion)
        
    def predict_from_domain(self, qb):
        """Run the translation from geographical domain
        :qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
        """
        if not isinstance(qb, TgeoBoundingBox):
            qb = qb.to_tgbox(self.landcover.crs)
        
        x_infres = self.landcover[qb]
        x_qflags = self.qflags[qb]
        x_esgp = self.esgp[qb]
        
        x_merge = deepcopy(x_esgp["mask"])
        
        if self.merge_criterion.__name__ == "qflag2_nodata":
            where_infres = self.merge_criterion(x_qflags["mask"], x_infres["mask"])
        else:
            where_infres = self.merge_criterion(x_qflags["mask"], x_esgp["mask"])
        
        x_merge[where_infres] = x_infres["mask"][where_infres]
        
        return x_merge.squeeze().numpy()


# MERGING CRITERIA
# ================

def qflag2_nodata(x_qflags, x_infres):
    """Use inference result when quality flag beyond 2 except for no data"""
    return torch.logical_and(x_qflags > 2, x_infres > 0)

def qflag2_nosea(x_qflags, x_esgp):
    """Use inference result when quality flag beyond 2 except for sea pixels"""
    return torch.logical_and(x_qflags > 2, x_esgp != 1)

def qflag2_onelabel(x_qflags, x_esgp):
    """Use inference result when quality flag beyond 2 except for patches with only one label"""
    return torch.logical_and(x_qflags > 2, torch.Tensor([np.unique(x_esgp).size > 1]))

def qflag2_nodominant(x_qflags, x_esgp):
    """Use inference result when quality flag beyond 2 except for patches with strongly dominant label"""
    _, c = np.unique(x_esgp, return_counts = True)
    return torch.logical_and(x_qflags > 2, torch.Tensor([any(c/c.sum() < 0.9)]))

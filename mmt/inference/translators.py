#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module with inference map translator


Class diagram
-------------
OnnxModel
MapTranslator
 ├── EsawcToEsgp
 |    └── EsawcToEsgpProba
 ├── EsawcEcosgToEsgpRFC
 ├── MapMerger
 └── MapMergerProba
"""

import os
import shutil
import time
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import pickle
import torch
from torchgeo.datasets.utils import BoundingBox as TgeoBoundingBox
from torchgeo import samplers
# from memory_profiler import profile

from mmt import _repopath_ as mmt_repopath
from mmt.inference import io
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import misc


# from multiprocessing import Pool
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

# BASE CLASSES
# ============
class OnnxModel:
    """Wrapper for inference from ONNX model
    
    Not used: remove?
    """

    def __init__(self, onnxfilename, inputname=None, outputname=None):
        import onnx
        import onnxruntime as ort
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
        output_dtype="int16",
    ):
        self.checkpoint_path = checkpoint_path
        self.remove_tmpdirs = remove_tmpdirs
        self.output_dtype = output_dtype
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
    
    # @profile
    def predict_from_large_domain(
        self, qb, output_dir="[id]", tmp_dir="[id]", n_cluster_files=200, n_px_max=600
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
        tmp_dir, output_dir = misc.create_directories(tmp_dir, output_dir)
        
        patches_definition_file = os.path.join(tmp_dir, "patches_definition_file.txt")
        if not os.path.isfile(patches_definition_file):
            sample_domain_with_patches(qb, self.landcover, n_px_max, tmp_dir)
        
        with open(patches_definition_file, "r") as f:
            patches = f.readlines()
        
        for tifpatchname in tqdm(patches, desc=f"Inference over {len(patches)} patches"):
            tifpatchname = tifpatchname.strip()
            if os.path.exists(os.path.join(tmp_dir, tifpatchname)):
                continue
            
            iqb = TgeoBoundingBox(*[float(s[4:]) for s in tifpatchname[:-4].split("_")], 0, 1e12)
            
            l_pred = self.predict_from_domain(iqb)
            
            io.dump_labels_in_tif(
                l_pred, iqb, self.landcover.crs, os.path.join(tmp_dir, tifpatchname), self.output_dtype
            )
        
        
        # with PatchIterator(tmp_dir) as pi:
            # for tifpatchname, iqb in tqdm(pi, desc=f"Inference over {len(pi)} patches"):
                # print("Next file to be written:", os.path.join(tmp_dir, tifpatchname))
                # l_pred = self.predict_from_domain(iqb)
                
                # io.dump_labels_in_tif(
                    # l_pred, iqb, self.landcover.crs, os.path.join(tmp_dir, tifpatchname), self.output_dtype
                # )
        
        if n_cluster_files > 0:
            io.stitch_tif_files(tmp_dir, output_dir, n_max_files=n_cluster_files, prefix = self.__class__.__name__, verbose = True)
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
        checkpoint_path=os.path.join(mmt_repopath, "data", "saved_models", "mmt-weights-v1.0.ckpt"),
        device="cuda",
        remove_tmpdirs=True,
        output_dtype="int16",
        always_predict=True,
    ):
        super().__init__(checkpoint_path, device, remove_tmpdirs, output_dtype)
        self.always_predict = always_predict
        
        self.esawc = landcovers.ESAWorldCover(transforms=mmt_transforms.EsawcTransform())
        self.esawc_transform = mmt_transforms.OneHot(
            self.esawc.n_labels + 1, device=self.device
        )
        self.encoder_decoder = io.load_pytorch_model(
            checkpoint_path, lc_in="esawc", lc_out="esgp", device = device
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
                return np.zeros(self.get_output_shape(x))
                
        x = self.esawc_transform(x)
        with torch.no_grad():
            y = self.encoder_decoder(x.float())
        
        return self.logits_transform(y)
        
    def get_output_shape(self, x):
        """Return the expected shape of the output with `x` in input"""
        return [s//6 for s in x.shape[-2:]]
        
    def logits_transform(self, logits):
        """Transform applied to the logits"""
        return logits.argmax(1).squeeze().cpu().numpy()

    def predict_from_domain(self, qb):
        """Run the translation from geographical domain
        :qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
        """
        if not isinstance(qb, TgeoBoundingBox):
            qb = qb.to_tgbox(self.esawc.crs)
        
        x = self.esawc[qb]
        return self.predict_from_data(x["mask"])


class EsawcToEsgpAsMap(EsawcToEsgp, landcovers.InferenceResults):
    """Map translator with plotting methods inherited from a land cover class
    
    Used in: scripts/qualitative_evalutation.py
    """
    def __getitem__(self, qb):
        return {"mask": self.predict_from_domain(qb)}

class EsawcToEsgpMembers(EsawcToEsgp):
    
    def __init__(
        self,
        checkpoint_path=os.path.join(mmt_repopath, "data", "saved_models", "mmt-weights-v1.0.ckpt"),
        device="cuda",
        remove_tmpdirs=True,
        output_dtype="int16",
        always_predict=True,
        u=None,
    ):
        self.u = u
        super().__init__(checkpoint_path, device, remove_tmpdirs, output_dtype, always_predict)
    
    def _logits_to_member0(self, logits):
        return logits.argmax(1).squeeze().cpu()
    
    def _logits_to_member(self, logits):
        assert self.u is not None, "Please provide a u value (in ]0,1[) to generate member"
        proba = logits.softmax(1).squeeze().cpu()
        cdf = proba.cumsum(0) / proba.sum(0)
        labels = (cdf < self.u).sum(0)
        return labels
    
    def logits_transform(self, logits):
        """Transform applied to the logits"""
        if self.u is None:
            return self._logits_to_member0(logits)
        else:
            return self._logits_to_member(logits)
    

class EsawcToEsgpShowEnsemble(EsawcToEsgpAsMap):
    
    def __init__(
        self,
        checkpoint_path=os.path.join(mmt_repopath, "data", "saved_models", "mmt-weights-v1.0.ckpt"),
        device="cuda",
        remove_tmpdirs=True,
        output_dtype="int16",
        always_predict=True,
        u = None,
    ):
        super().__init__(checkpoint_path, device, remove_tmpdirs, output_dtype, always_predict)
        
        self.u = u
        
        self.auxmap = landcovers.ScoreECOSGplus(
            transforms=mmt_transforms.ScoreTransform(divide_by=100),
            crs = self.esawc.crs,
            res = self.esawc.res * 6,
        )
        self.auxmap.crs = self.esawc.crs
        self.auxmap.res = self.esawc.res * 6
        
        self.score_min = self.auxmap.cutoff
        
        self.bottommap = landcovers.EcoclimapSGplusV2p1(
            score_min=self.score_min,
            crs = self.esawc.crs,
            res = self.esawc.res * 6,
        )
        self.bottommap.crs = self.esawc.crs
        self.bottommap.res = self.esawc.res * 6
    
    def predict_from_domain(self, qb):
        """Run the translation from geographical domain
        :qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
        """
        if not isinstance(qb, TgeoBoundingBox):
            qb = qb.to_tgbox(self.esawc.crs)
        
        x = self.esawc[qb]
        top = self.predict_from_data(x["mask"])
        aux = self.auxmap[qb]["image"]
        bottom = self.bottommap[qb]["mask"]
        
        return torch.where(self.criterion(top, aux), top, bottom).squeeze()
    
    def criterion(self, top, aux):
        return torch.logical_and(top != 0, aux < self.score_min)
    
    def _logits_to_member0(self, logits):
        return logits.argmax(1).squeeze().cpu()
    
    def _logits_to_member(self, logits):
        assert self.u is not None, "Please provide a u value (in ]0,1[) to generate member"
        proba = logits.softmax(1).squeeze().cpu()
        cdf = proba.cumsum(0) / proba.sum(0)
        labels = (cdf < self.u).sum(0)
        return labels
    
    def logits_transform(self, logits):
        """Transform applied to the logits"""
        if self.u is None:
            return self._logits_to_member0(logits)
        else:
            return self._logits_to_member(logits)
    

class EsawcToEsgpProba(EsawcToEsgp):
    """Return probabilities of classes instead of classes"""
    def __init__(
        self,
        checkpoint_path=os.path.join(mmt_repopath, "data", "saved_models", "mmt-weights-v1.0.ckpt"),
        device="cuda",
        remove_tmpdirs=True,
        output_dtype="float32",
        always_predict=True,
    ):
        super().__init__(checkpoint_path, device, remove_tmpdirs, output_dtype, always_predict)
    
    def get_output_shape(self, x):
        """Return the expected shape of the output with `x` in input"""
        return [35] + [s//6 for s in x.shape[-2:]]
        
    def logits_transform(self, logits):
        """Transform applied to the logits"""
        return logits.softmax(1).squeeze().cpu().numpy()


class MapMerger(MapTranslator):
    """Merge map with ECOCLIMAP-SG+ according to a quality flag criterion"""
    
    def __init__(
        self,
        source_map_path,
        device="cpu",
        remove_tmpdirs=True,
        output_dtype="int16",
        merge_criterion = "qflag2_nosea",
    ):
        super().__init__("merger", device, remove_tmpdirs, output_dtype)
        
        self.esgp = landcovers.EcoclimapSGplus()
        self.qflags = landcovers.QualityFlagsECOSGplus(transforms=mmt_transforms.FillMissingWithSea(0,6))
        self.landcover = landcovers.InferenceResults(path = source_map_path, res = self.esgp.res)
        self.landcover.res = self.esgp.res
        if callable(merge_criterion):
            self.merge_criterion = merge_criterion
        else:
            self.merge_criterion = eval(merge_criterion)
        
    def predict_from_domain(self, qb):
        """Run the translation from geographical domain"""
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

class MapMergerV2(MapTranslator):
    """Merge map with ECOCLIMAP-SG+ according to a quality flag criterion"""
    
    def __init__(
        self,
        source_map_path,
        device="cpu",
        remove_tmpdirs=True,
        output_dtype="int16",
        score_min = None,
    ):
        super().__init__("merger", device, remove_tmpdirs, output_dtype)
        
        self.auxmap = landcovers.ScoreECOSGplus(
            transforms=mmt_transforms.ScoreTransform(divide_by=100),
        )
        if score_min is None:
            self.score_min = self.auxmap.cutoff
        else:
            self.score_min = score_min
        
        self.bottommap = landcovers.EcoclimapSGplusV2p1(
            score_min=self.score_min,
        )
        
        self.topmap = landcovers.InferenceResults(path = source_map_path, res = self.bottommap.res)
        self.topmap.res = self.bottommap.res
        
        self.landcover = self.bottommap
        
    def predict_from_domain(self, qb):
        """Run the translation from geographical domain
        :qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
        """
        if not isinstance(qb, TgeoBoundingBox):
            qb = qb.to_tgbox(self.bottommap.crs)
        
        top = self.topmap[qb]["mask"]
        aux = self.auxmap[qb]["image"]
        bottom = self.bottommap[qb]["mask"]
        
        return torch.where(self.criterion(top, aux), top, bottom).squeeze()
    
    def criterion(self, top, aux):
        return torch.logical_and(top != 0, aux < self.score_min)


class MapMergerV3(MapTranslator):
    """Merge map with ECOCLIMAP-SG+ according to a quality flag criterion"""
    
    def __init__(
        self,
        source_map_path,
        device="cpu",
        remove_tmpdirs=True,
        output_dtype="int16",
        score_min = None,
    ):
        super().__init__("merger", device, remove_tmpdirs, output_dtype)
        
        auxmap = landcovers.ScoreECOSGplus(
            transforms=mmt_transforms.ScoreTransform(divide_by=100),
        )
        if score_min is None:
            self.score_min = self.auxmap.cutoff
        else:
            self.score_min = score_min
        
        bottommap = landcovers.EcoclimapSGplusV2p1(
            score_min=self.score_min,
        )
        
        topmap = landcovers.InferenceResults(path = source_map_path)
        
        self.landcover = auxmap & topmap & bottommap
        
    def predict_from_domain(self, qb):
        """Run the translation from geographical domain
        :qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
        """
        if not isinstance(qb, TgeoBoundingBox):
            qb = qb.to_tgbox(self.bottommap.crs)
        
        top = self.landcover[qb]["mask"][0]
        aux = self.landcover[qb]["image"]
        bottom = self.landcover[qb]["mask"][1]
        
        return torch.where(self.criterion(top, aux), top, bottom).squeeze()
    
    def criterion(self, top, aux):
        return torch.logical_and(top != 0, aux < self.score_min)


class MapMergerProba(MapTranslator):
    """Merge map with ECOCLIMAP-SG+ according to a quality flag criterion"""
    
    def __init__(
        self,
        source_map_path,
        device="cpu",
        remove_tmpdirs=True,
        output_dtype="int16",
        merge_criterion = "qflag2_nosea",
        tgeo_init = True,
    ):
        super().__init__("mergerproba", device, remove_tmpdirs, output_dtype)
        
        self.tgeo_init = tgeo_init
        
        self.esgp = landcovers.EcoclimapSGplus(transforms=mmt_transforms.OneHot(35), tgeo_init = tgeo_init)
        self.qflags = landcovers.QualityFlagsECOSGplus(transforms=mmt_transforms.FillMissingWithSea(0,6), tgeo_init = tgeo_init)
        if tgeo_init:
            self.landcover = landcovers.InferenceResultsProba(path = source_map_path, res = self.esgp.res, tgeo_init = tgeo_init)
            self.landcover.res = self.esgp.res
        else:
            self.landcover = landcovers.InferenceResultsProba(path = source_map_path, tgeo_init = tgeo_init)
        
        if callable(merge_criterion):
            self.merge_criterion = merge_criterion
        else:
            self.merge_criterion = eval(merge_criterion)
        
    def predict_from_data(self, x_infres, x_qflags, x_esgp):
        
        x_merge = deepcopy(x_esgp)
        
        if self.merge_criterion.__name__ == "qflag2_nodata":
            where_infres = self.merge_criterion(x_qflags, x_infres.sum(0))
        else:
            where_infres = self.merge_criterion(x_qflags, x_esgp)
        
        where_infres = where_infres.squeeze()
        x_merge[:, where_infres] = x_infres[:, where_infres]
        
        return x_merge.squeeze().numpy()
        
    def predict_from_domain(self, qb):
        """Run the translation from geographical domain"""
        if not isinstance(qb, TgeoBoundingBox):
            qb = qb.to_tgbox(self.landcover.crs)
        
        x_infres = self.landcover[qb]
        x_qflags = self.qflags[qb]
        x_esgp = self.esgp[qb]
        
        return self.predict_from_data(x_infres["image"].squeeze(), x_qflags["mask"].squeeze(), x_esgp["mask"].squeeze())



# MERGING CRITERIA
# ================

def sample_domain_with_patches(domain, landcover, patch_size, tmp_dir):
    """Sample create a list of TIF file names that cover the whole domain
    
    Example
    -------
    >>> from mmt.utils import domains
    >>> from mmt.datasets import landcovers
    >>> from mmt.inference import translators
    >>> landcover = landcovers.EcoclimapSG()
    >>> domain = domains.eurat
    >>> patch_size = 20
    >>> tmp_dir = "tmp"
    >>> translators.sample_domain_with_patches(domain, landcover, patch_size, tmp_dir)
    """
    if not isinstance(domain, TgeoBoundingBox):
        qb = domain.to_tgbox(landcover.crs)
    
    margin = patch_size // 6
    sampler = samplers.GridGeoSampler(
        landcover, size=patch_size, stride=patch_size - margin, roi=qb
    )
    if len(sampler) == 0:
        raise ValueError(f"Empty sampler. size={patch_size}, stride={patch_size - margin}, roi={qb}, landcover bounds={landcover.bounds}")
    
    patches_definition_file = os.path.join(tmp_dir, "patches_definition_file.txt")
    with open(patches_definition_file, "w") as f:
        for iqb in sampler:
            f.write(
                "_".join([f"{k}{getattr(iqb,k)}" for k in ["minx", "maxx", "miny", "maxy"]]) + ".tif\n"
            )
    
    return patches_definition_file


class PatchIterator:
    def __init__(self, directory):
        self.directory = directory
        self.patches_definition_file = os.path.join(directory, "patches_definition_file.txt")
        self.i = 0
    
    def __enter__(self):
        self.openfile = open(self.patches_definition_file, "r")
        self.i = 0
        l = next(iter(self.openfile))
        while os.path.exists(os.path.join(self.directory, l.strip())):
            l = next(iter(self.openfile))
            self.i += 1
        
        print(f"Opening {self.patches_definition_file} at line #{self.i}: {l.strip()}")
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.openfile.close()
        print(f"Closing {self.patches_definition_file} at line #{self.i}")
        
    def __iter__(self):
        return self
    
    def __next__(self):
        l = next(iter(self.openfile))
        self.i += 1
        tiff = l.strip()
        bbox = TgeoBoundingBox(*[float(s[4:]) for s in tiff[:-4].split("_")], 0, 1e12)
        return tiff, bbox
    
    def __len__(self):
        if not hasattr(self, "_len"):
            with open(self.patches_definition_file, "r") as f:
               count = sum(1 for _ in f)
            
            self._len = count
        
        return self._len - self.i


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

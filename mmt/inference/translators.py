#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module with inference map translator
"""

import os
import shutil
import time
import numpy as np
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
# ==============
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
    """Apply map translation models in inference mode"""

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

    def predict_from_data(self, x):
        """Run the translation from matrices of land cover labels
        :x: `torch.Tensor` of shape (N, 1, H, W)
        """
        raise NotImplementedError
    
    def predict_from_domain(self, qb):
        """Run the translation from geographical domain
        :qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
        """
        raise NotImplementedError
    
    def predict_from_large_domain(
        self, qb, output_dir="[id]", tmp_dir="[id]", n_max_files=200, n_px_max=600
    ):
        """Inference over large domain in made in tiling the large domain into small patches"""
        raise NotImplementedError
    
    def __call__(self, qb):
        return self.predict_from_domain(qb)
    
    def __repr__(self):
        return str(self.__class__) + " checkpoint_path=" + self.checkpoint_path


# PRACTICAL CLASSES
# =================

class EsawcToEsgp(MapTranslator):
    """Translate from ESA World Cover to ECOCLIMAP-SG+"""
    def __init__(
        self,
        checkpoint_path=os.path.join(mmt_repopath, "saved_models", "vanilla.ckpt"),
        device="cuda",
        remove_tmpdirs=True,
    ):
        super().__init__(checkpoint_path, device, remove_tmpdirs)

        self.esawc = landcovers.ESAWorldCover(transforms=mmt_transforms.EsawcTransform)
        self.esawc_transform = mmt_transforms.OneHot(
            self.esawc.n_labels + 1, device=self.device
        )
        self.encoder_decoder = io.load_pytorch_model(
            checkpoint_path, lc_in="esawc", lc_out="esgp"
        )
        self.encoder_decoder.to(self.device)

    def predict_from_data(self, x):
        """Run the translation from matrices of land cover labels
        :x: `torch.Tensor` of shape (N, 1, H, W)
        """
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

    def predict_from_large_domain(
        self, qb, output_dir="[id]", tmp_dir="[id]", n_max_files=200, n_px_max=600
    ):
        """Inference over large domain in made in tiling the large domain into small patches"""
        if not isinstance(qb, TgeoBoundingBox):
            qb = qb.to_tgbox(self.esawc.crs)
        
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
            self.esawc, size=n_px_max, stride=n_px_max - margin, roi=qb
        )

        for iqb in tqdm(sampler, desc=f"Inference over {len(sampler)} patches"):
            tifpatchname = f"N{iqb.minx}_E{iqb.maxy}.tif"
            if os.path.exists(os.path.join(tmp_dir, tifpatchname)):
                continue

            y = self.predict_from_domain(iqb)

            io.dump_labels_in_tif(
                y, iqb, self.esawc.crs, os.path.join(tmp_dir, tifpatchname)
            )

        io.stitch_tif_files(tmp_dir, output_dir, n_max_files=16)

        if self.remove_tmpdirs:
            shutil.rmtree(tmp_dir)

        return output_dir


class EsawcEcosgToEsgpRFC(MapTranslator):
    """Translate from ESA World Cover and ECOCLIMAP-SG to ECOCLIMAP-SG+
    with a random forest classifer"""
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

    def predict_from_large_domain(
        self, qb, output_dir="[id]", tmp_dir="[id]", n_max_files=200, n_px_max=600
    ):
        """Inference over large domain in made in tiling the large domain into small patches"""
        if not isinstance(qb, TgeoBoundingBox):
            qb = qb.to_tgbox(self.esawc.crs)
        
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

        for iqb in tqdm(sampler, desc=f"Inference over {len(sampler)} patches"):
            tifpatchname = f"N{iqb.minx}_E{iqb.maxy}.tif"
            if os.path.exists(os.path.join(tmp_dir, tifpatchname)):
                continue

            y = self.predict_from_domain(iqb)

            io.dump_labels_in_tif(
                y, iqb, self.landcover.crs, os.path.join(tmp_dir, tifpatchname)
            )

        io.stitch_tif_files(tmp_dir, output_dir, n_max_files=16)

        if self.remove_tmpdirs:
            shutil.rmtree(tmp_dir)

        return output_dir
    
    
    

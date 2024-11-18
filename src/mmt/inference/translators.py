#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module with inference map translator


Class diagram
-------------
_MapTranslator
 ├── MapMerger
 └── EsawcToEsgp
     ├── EsawcToEsgpMembers
     ├── EsawcToEsgpProba
     └── EsawcToEsgpAsMap -- landcovers.InferenceResults
            └── EsawcToEsgpShowEnsemble
"""

import os
import shutil

import torch
from torchgeo import samplers
from torchgeo.datasets.utils import BoundingBox
from tqdm import tqdm

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import landcovers
from mmt.datasets import transforms as mmt_transforms
from mmt.inference import io
from mmt.utils import misc

# BASE CLASSES
# ============


class _MapTranslator:
    """Abstract class to make map translation models in inference mode.

    Define a set of nested methods that help scaling up the inference area
    and give a common API for all models or task (inference and merging).
    By default, the method `predict_from_domain` is called when a translator
    is used as a transform.


    Main attributes
    ---------------
    checkpoint_path: str
        See parameters

    landcover: `torchgeo.datasets.RasterDataset`
        Land cover dataset that will be used to define the sampler and the CRS
        of the query domain.


    Main methods
    ------------
    predict_from_data(x) -> torch.Tensor:
        Apply translation to tensor of land cover labels.

    predict_from_domain(qb) -> torch.Tensor:
        Apply translation to a geographical domain.
        May call `predict_from_data`

    predict_from_large_domain(qb, **kwargs) -> str:
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
        """Constructor.


        Parameters
        ----------
        checkpoint_path: str
            Path to the Pytorch checkpoint from which are loaded the auto-encoders.
            For the child class `MapMerger`, this attribute is not used and set to "merger"

        device: {"cuda", "cpu"}
            Device on which the inference is done.

        remove_tmpdirs: bool, default = True
            If True, temporary directory are erased at the of the function using them.

        output_dtype: str
            Data type of the output
        """
        self.checkpoint_path = checkpoint_path
        self.remove_tmpdirs = remove_tmpdirs
        self.output_dtype = output_dtype
        self.device = torch.device(device)
        self.shortname = (
            self.__class__.__name__ + "." + os.path.basename(checkpoint_path)
        )
        self.landcover = None

    def __call__(self, qb):
        return self.predict_from_domain(qb)

    def __repr__(self):
        return str(self.__class__) + " checkpoint_path=" + self.checkpoint_path

    def predict_from_data(self, x) -> torch.Tensor:
        """Apply translation to tensor of land cover labels.


        Parameters
        ----------
        x: `torch.Tensor` of shape (1, H, W)
            Tensor of land cover labels, with H=height, W=width


        Returns
        -------
        y: `torch.Tensor` of shape (H', W')
            Tensor of predicted land cover labels (usually ECOSG+)
        """
        raise NotImplementedError

    def predict_from_domain(self, qb) -> torch.Tensor:
        """Apply translation to a geographical domain.


        Parameters
        ----------
        qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
            Geographical domain


        Returns
        -------
        y: `torch.Tensor` of shape (H', W')
            Tensor of predicted land cover labels (usually ECOSG+)
        """
        raise NotImplementedError

    def predict_from_large_domain(
        self, qb, output_dir="[id]", tmp_dir="[id]", n_cluster_files=200, n_px_max=600
    ) -> str:
        """Apply translation to a large domain in made in tiling the large domain into small patches.

        As big domains cannot fit in memory, the inference is done on small patches
        (of size `n_px_max` pixels). The result of this inference is sotred in TIF files
        that can be read by Torchgeo. However, as the number of TIF files can be very
        large, there are clustered together into a smaller number (no more than `n_cluster_files`).


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

        n_cluster_files: int, default=200
            Maximum number of TIF files created in `output_dir`.
            If `n_max_files=0`, no clustering is made, `output_dir` is then
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

        # Inference on small patches
        for tifpatchname in tqdm(
            patches, desc=f"Inference over {len(patches)} patches"
        ):
            tifpatchname = tifpatchname.strip()
            if os.path.exists(os.path.join(tmp_dir, tifpatchname)):
                continue

            iqb = BoundingBox(
                *[float(s[4:]) for s in tifpatchname[:-4].split("_")], 0, 1e12
            )

            l_pred = self.predict_from_domain(iqb)

            io.dump_labels_in_tif(
                l_pred.numpy(),
                iqb,
                self.landcover.crs,
                os.path.join(tmp_dir, tifpatchname),
                self.output_dtype,
            )

        # If needed, clustering of the output TIF files
        if n_cluster_files > 0:
            io.cluster_tif_files(
                tmp_dir,
                output_dir,
                n_max_files=n_cluster_files,
                prefix=os.path.basename(output_dir),
                verbose=True,
            )
        else:
            shutil.copytree(tmp_dir, output_dir, dirs_exist_ok=True)

        if self.remove_tmpdirs:
            shutil.rmtree(tmp_dir)
        else:
            print(f"Kept: tmp_dir={tmp_dir}")

        return output_dir


# CHILD CLASSES
# =============


class EsawcToEsgp(_MapTranslator):
    """Translate from ESA World Cover to ECOCLIMAP-SG+ with map translation auto-encoders


    Main attributes
    ---------------
    esawc: `mmt.datasets.landcovers.ESAWorldCover`
        The input map of the map translation

    encoder_decoder: `torch.nn.Module`
        The pre-trained map translation model


    Main methods
    ------------
    Same as _MapTranslator

    logits_transform(logits) -> torch.Tensor
        Transform applied to the logits
        Hook for alternative transforms in child classes
    """

    def __init__(
        self,
        checkpoint_path=os.path.join(
            mmt_repopath, "data", "saved_models", "mmt-weights-v2.0.ckpt"
        ),
        device="cuda",
        remove_tmpdirs=True,
        output_dtype="int16",
        always_predict=True,
    ):
        """Constructor. Instanciate the input land cover and load the pre-trained model


        Parameters
        ----------
        Same as _MapTranslator.__init__, except for

        always_predict: bool, default = True
            If True, does not apply the diversity criterion and predict all the time
        """
        super().__init__(checkpoint_path, device, remove_tmpdirs, output_dtype)
        self.always_predict = always_predict

        self.esawc = landcovers.ESAWorldCover(
            transforms=mmt_transforms.EsawcTransform()
        )
        self.esawc_transform = mmt_transforms.OneHotTorchgeo(
            self.esawc.n_labels + 1, device=self.device
        )
        self.encoder_decoder = io.load_pytorch_model(
            checkpoint_path, lc_in="esawc", lc_out="esgp", device=device
        )
        self.encoder_decoder.to(self.device)
        self.landcover = self.esawc

    def predict_from_data(self, x) -> torch.Tensor:
        """Apply translation to tensor of land cover labels.


        Parameters
        ----------
        x: `torch.Tensor` of shape (1, H, W)
            Tensor of land cover labels, with H=height, W=width


        Returns
        -------
        y: `torch.Tensor` of shape (H', W')
            Tensor of predicted ECOSG+ land cover labels
        """
        if not self.always_predict:
            x.to(self.device)
            _, c = torch.unique(x, return_counts=True)
            if any(c / c.sum() > 0.9):
                return torch.zeros(self.get_output_shape(x))

        x = self.esawc_transform(x)
        with torch.no_grad():
            y = self.encoder_decoder(x.float())

        return self.logits_transform(y)

    def get_output_shape(self, x) -> list:
        """Return the expected shape of the output with `x` in input"""
        return [s // 6 for s in x.shape[-2:]]

    def logits_transform(self, logits) -> torch.Tensor:
        """Transform applied to the logits"""
        return logits.argmax(1).squeeze().cpu()

    def predict_from_domain(self, qb) -> torch.Tensor:
        """Apply translation to a geographical domain.


        Parameters
        ----------
        qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
            Geographical domain


        Returns
        -------
        y: `torch.Tensor` of shape (H', W')
            Tensor of predicted ECOSG+ land cover labels
        """
        if not isinstance(qb, BoundingBox):
            qb = qb.to_tgbox(self.esawc.crs)

        x = self.esawc[qb]
        return self.predict_from_data(x["mask"])


class EsawcToEsgpAsMap(EsawcToEsgp, landcovers.InferenceResults):
    """Map translator with plotting methods inherited from a land cover class

    Allow to visualize the inference results just like a regular map.
    Be careful to provide domains with appropriate size (mutliples of 300 are good choices).

    Used in: scripts/qualitative_evalutation.py
    """

    res = misc.DEFAULT_RESOLUTION_10M

    def __getitem__(self, qb):
        return {"mask": self.predict_from_domain(qb)}


class EsawcToEsgpMembers(EsawcToEsgp):
    """Variation of `EsawcToEsgp` to produce members.

    The prediction methods are the same. Only the logit transform is modified
    to take into account the additional parameter `u` defining the random
    draw of the member.

    Used in scripts/inference_and_merging.py


    Main attributes
    ---------------
    Same as `EsawcToEsgp`, except for

    u: float or None
        Random draw uniformly distributed between 0 and 1. When set to None,
        the translator takes the maximum logit probability and is therefore
        equal to `EsawcToEsgp`.


    Main methods
    ------------
    Same as `EsawcToEsgp`

    logits_transform(logits) -> torch.Tensor
        Transform applied to the logits
        Modified to generate members
    """

    def __init__(
        self,
        checkpoint_path=os.path.join(
            mmt_repopath, "data", "saved_models", "mmt-weights-v2.0.ckpt"
        ),
        device="cuda",
        remove_tmpdirs=True,
        output_dtype="int16",
        always_predict=True,
        u=None,
    ):
        """Constructor.


        Parameters
        ----------
        Same as EsawcToEsgp.__init__, except for

        u: float, default=None
            Random draw uniformly distributed between 0 and 1. When set to None,
            the translator takes the maximum logit probability and is therefore
            equal to `EsawcToEsgp`.
        """
        self.u = u
        super().__init__(
            checkpoint_path, device, remove_tmpdirs, output_dtype, always_predict
        )

    def _logits_to_member0(self, logits) -> torch.Tensor:
        return logits.argmax(1).squeeze().cpu()

    def _logits_to_member(self, logits) -> torch.Tensor:
        assert (
            self.u is not None
        ), "Please provide a u value (in ]0,1[) to generate member"
        proba = logits.softmax(1).squeeze().cpu()
        cdf = proba.cumsum(0) / proba.sum(0)
        labels = (cdf < self.u).sum(0)
        return labels

    def logits_transform(self, logits) -> torch.Tensor:
        """Transform applied to the logits"""
        if self.u is None:
            return self._logits_to_member0(logits)
        else:
            return self._logits_to_member(logits)


class EsawcToEsgpShowEnsemble(EsawcToEsgpAsMap):
    """Same as EsawcToEsgpAsMap but with the merging criterion.

    Allows to visualize the members before exporting the map on a large domain.

    Used in scripts/show_infres_ensemble.py
    """

    def __init__(
        self,
        checkpoint_path=os.path.join(
            mmt_repopath, "data", "saved_models", "mmt-weights-v2.0.ckpt"
        ),
        device="cuda",
        remove_tmpdirs=True,
        output_dtype="int16",
        always_predict=True,
        u=None,
    ):
        super().__init__(
            checkpoint_path, device, remove_tmpdirs, output_dtype, always_predict
        )

        self.u = u

        self.auxmap = landcovers.ScoreECOSGplus(
            transforms=mmt_transforms.ScoreTransform(divide_by=100),
            crs=self.esawc.crs,
            res=self.esawc.res * 6,
        )
        self.auxmap.crs = self.esawc.crs
        self.auxmap.res = self.esawc.res * 6

        self.score_min = self.auxmap.cutoff

        self.bottommap = landcovers.EcoclimapSGplus(
            score_min=self.score_min,
            crs=self.esawc.crs,
            res=self.esawc.res * 6,
        )
        self.bottommap.crs = self.esawc.crs
        self.bottommap.res = self.esawc.res * 6

    def predict_from_domain(self, qb) -> torch.Tensor:
        """Apply translation to a geographical domain.


        Parameters
        ----------
        qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
            Geographical domain


        Returns
        -------
        y: `torch.Tensor` of shape (H', W')
            Tensor of predicted ECOSG+ land cover labels
        """
        if not isinstance(qb, BoundingBox):
            qb = qb.to_tgbox(self.esawc.crs)

        x = self.esawc[qb]
        top = self.predict_from_data(x["mask"])
        aux = self.auxmap[qb]["image"]
        bottom = self.bottommap[qb]["mask"]

        return torch.where(self.criterion(top, aux), top, bottom).squeeze()

    def criterion(self, top, aux):
        """Criterion to use the top map instead of the bottom map"""
        return torch.logical_and(top != 0, aux < self.score_min)

    def _logits_to_member0(self, logits) -> torch.Tensor:
        return logits.argmax(1).squeeze().cpu()

    def _logits_to_member(self, logits) -> torch.Tensor:
        assert (
            self.u is not None
        ), "Please provide a u value (in ]0,1[) to generate member"
        proba = logits.softmax(1).squeeze().cpu()
        cdf = proba.cumsum(0) / proba.sum(0)
        labels = (cdf < self.u).sum(0)
        return labels

    def logits_transform(self, logits) -> torch.Tensor:
        """Transform applied to the logits"""
        if self.u is None:
            return self._logits_to_member0(logits)
        else:
            return self._logits_to_member(logits)


class EsawcToEsgpProba(EsawcToEsgp):
    """Variation of `EsawcToEsgp` to access land cover probabilities instead of classes.

    The prediction methods are the same. Only the logit transform is modified
    to take into account the additional parameter `u` defining the random
    draw of the member.

    Not used in the paper but useful.


    Main attributes
    ---------------
    Same as `EsawcToEsgp`, except for `output_dtype` which default to float32


    Main methods
    ------------
    Same as `EsawcToEsgp`

    logits_transform(logits) -> torch.Tensor
        Transform applied to the logits
        Modified to return probabilities of classes
    """

    def __init__(
        self,
        checkpoint_path=os.path.join(
            mmt_repopath, "data", "saved_models", "mmt-weights-v2.0.ckpt"
        ),
        device="cuda",
        remove_tmpdirs=True,
        output_dtype="float32",
        always_predict=True,
    ):
        super().__init__(
            checkpoint_path, device, remove_tmpdirs, output_dtype, always_predict
        )

    def get_output_shape(self, x):
        """Return the expected shape of the output with `x` in input"""
        return [35] + [s // 6 for s in x.shape[-2:]]

    def logits_transform(self, logits):
        """Transform applied to the logits"""
        return logits.softmax(1).squeeze().cpu()


class MapMerger(_MapTranslator):
    """Merge the inference result map with ECOCLIMAP-SG+ according to a quality criterion

    Once the inference has run on a large domain and is now stored in TIF files,
    ECOSG-ML is the merge of these inference results (one per member) and ECOSG+.
    The labels are taken from ECOSG+ if its quality score is high enough, or from
    the inference results if it is low. The transition between the two is set by
    the threshold `score_lim`. Maps are first set to the same resolution and CRS.


    Main attributes
    ---------------
    score_lim: float, default = None
        Threshold to apply ECOSG+ labels or inference labels.
        When set to done, it is taken equal to the ECOSG+ quality score
        cutoff value (so-called `score_min=0.525`).

    score: `mmt.datasets.landcovers.ScoreECOSGplus`
        Map of the ECOSG+ quality score

    bguess: `mmt.datasets.landcovers.SpecialistLabelsECOSGplus`
        Map of the ECOSG+ best-guess map (without any pixel from ECOSG)

    infres: `mmt.datasets.landcovers.InferenceResults`
        Map of the inference results stored in TIF files

    ecosg: `mmt.datasets.landcovers.EcoclimapSG`
        Map of the ECOSG labels


    Main methods
    ------------
    predict_from_domain(qb) -> torch.Tensor:
        Apply merging criterion to a geographical domain.
        The sketch of the method is as follows:
        ```
        if score > score_lim
            return bguess
        else
            return infres
        unless they are at 0 (missing data)
            return ecosg
        ```
    """

    def __init__(
        self,
        source_map_path,
        device="cpu",
        remove_tmpdirs=True,
        output_dtype="int16",
        score_lim=None,
    ):
        """Constructor. Instanciate the quality score map, the best-guess
        map and the inference results map


        Parameters
        ----------
        Same as _MapTranslator.__init__, except for

        source_map_path: str
            Path to the inference results stored in TIF files

        score_lim: float, default = None
            Threshold to apply ECOSG+ labels or inference labels.
            When set to done, it is taken equal to the ECOSG+ quality score
            cutoff value (so-called `score_min=0.525`).
        """
        super().__init__("merger", device, remove_tmpdirs, output_dtype)

        self.score = landcovers.ScoreECOSGplus(
            transforms=mmt_transforms.ScoreTransform(divide_by=100),
        )
        if score_lim is None:
            self.score_lim = self.score.cutoff
        else:
            self.score_lim = score_lim

        self.bguess = landcovers.SpecialistLabelsECOSGplus(res=self.score.res)
        self.bguess.res = self.score.res

        self.ecosg = landcovers.EcoclimapSG(res=self.score.res)
        self.ecosg.res = self.score.res

        self.infres = landcovers.InferenceResults(
            path=source_map_path, res=self.score.res
        )
        self.infres.res = self.score.res

        self.landcover = self.ecosg

    def predict_from_domain(self, qb) -> torch.Tensor:
        """Apply translation to a geographical domain.


        Parameters
        ----------
        qb: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
            Geographical domain


        Returns
        -------
        y: `torch.Tensor` of shape (H', W')
            Tensor of merged land cover labels
        """
        if not isinstance(qb, BoundingBox):
            qb = qb.to_tgbox(self.landcover.crs)

        x_ecosg = self.ecosg[qb]["mask"].squeeze()

        def safeget(lc, key="mask"):
            try:
                l = lc[qb][key].squeeze()
            except IndexError:
                l = torch.zeros_like(x_ecosg)
            return l

        x_infres = safeget(self.infres)
        x_bguess = safeget(self.bguess)
        x_score = safeget(self.score, key="image")

        x_ = torch.where(x_score > self.score_lim, x_bguess, x_infres)
        x = torch.where(x_ == 0, x_ecosg, x_)

        if (x == 0).sum() > 0:
            # These print should not appear
            print(f"{(x==0).sum()} zeros that shouldn't be there... {x.shape} {qb}")
            print(
                f"Zeros: x_infres {(x_infres==0).sum()}, x_ecosg {(x_ecosg==0).sum()}, x_score {(x_score==0).sum()}, x_bguess {(x_bguess==0).sum()}, x_ {(x_==0).sum()}"
            )

        return x


# FUNCTIONS
# =========


def sample_domain_with_patches(domain, landcover, patch_size, tmp_dir) -> str:
    """Sample create a list of TIF file names that cover the whole domain

    Each file name gives the boundaries of a patch with the following convention:
    'minx-MINXVALUE_maxx-MAXXVALUE_minyMINYVALUE_maxyMAXYVALUE.tif'
    Storing the bounding boxes this way ensure reproducibility in the sampling
    of the domain and allows safe stops and restarts.

    Used in mmt.inference.translators._MapTranslator.predict_from_large_domain


    Parameters
    ----------
    domain: `torchgeo.datasets.utils.BoundingBox` or `mmt.utils.domains.GeoRectangle`
        Geographical domain

    landcover: `torchgeo.datasets.RasterDataset`
        Land cover to sample

    patch_size: int
        Size of the sampling patches (# of pixels)

    tmp_dir: str
        Path to the temporary directory where the inference results will be stored


    Returns
    -------
    patches_definition_file: str
        Path to the file containing the TIF files names to be created.


    Example
    -------
    >>> # Sample Ireland with ECOSG 100-px patches
    >>> from mmt.utils import domains
    >>> from mmt.datasets import landcovers
    >>> from mmt.inference import translators
    >>> landcover = landcovers.EcoclimapSG()
    >>> translators.sample_domain_with_patches(domains.ireland, landcover, 100, "tmp")
    'tmp/patches_definition_file.txt'
    >>> with open('tmp/patches_definition_file.txt', 'r') as f:
    ...     oneline = f.readlines()[0]
    ...     count = len(f.readlines())
    >>> count
    540
    >>> oneline
    'minx-11.1_maxx-10.822222222222221_miny50.9_maxy51.17777777777778.tif\n'
    """
    if not isinstance(domain, BoundingBox):
        qb = domain.to_tgbox(landcover.crs)

    margin = patch_size // 6
    sampler = samplers.GridGeoSampler(
        landcover, size=patch_size, stride=patch_size - margin, roi=qb
    )
    if len(sampler) == 0:
        raise ValueError(
            f"Empty sampler. size={patch_size}, stride={patch_size - margin}, roi={qb}, landcover bounds={landcover.bounds}"
        )

    patches_definition_file = os.path.join(tmp_dir, "patches_definition_file.txt")
    with open(patches_definition_file, "w") as f:
        for iqb in sampler:
            f.write(
                "_".join(
                    [f"{k}{getattr(iqb,k)}" for k in ["minx", "maxx", "miny", "maxy"]]
                )
                + ".tif\n"
            )

    return patches_definition_file

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Transforms
"""
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF

from mmt.utils.misc import rmsuffix


class OneHotHdf5:
    """One-hot encoding of land cover labels (original transform).
    Sample is expected to have keys {"source_data", "target_data"}.
    Transform is stored in keys {"source_one_hot", "target_one_hot"}

    Used in mmt.dataset.landcover_to_landcover.LandcoverToLandcoverDataLoader
    """
    def __init__(
        self,
        nclasses,
        loc="_data",
        dtype=torch.float,
        ind=["source_data", "target_data"],
    ):
        self.nclasses = nclasses
        self.loc = loc
        self.dtype = dtype
        self.ind = ind

    def one_hot_embedding(self, labels, num_classes):
        """Embedding labels to one-hot form.


        Parameters
        ----------
        labels: torch.LongTensor of size (N,)
            Class labels
        
        num_classes: int
            Number of classes


        Returns
        -------
        y: torch.Tensor of size (N, num_classes)
            One-hot encoded labels
        """
        y = torch.eye(num_classes, dtype=self.dtype, device=labels.device)
        return y[labels]

    def __call__(self, sample):
        with torch.no_grad():
            for k in self.ind:
                sample[k.replace("_data", "_one_hot")] = self.one_hot_embedding(
                    sample[k].long(),
                    num_classes=self.nclasses[
                        rmsuffix(sample[k.replace("_data", "_name")])
                    ]
                    + 1,
                ).permute(0, 3, 1, 2,)[0]
        return sample


class CoordEnc:
    """Coordinates encoding (original transform)

    Used in mmt.dataset.landcover_to_landcover.LandcoverToLandcoverDataLoader
    """
    def __init__(self, datasets):
        self.d = 128
        self.d = int(self.d / 2)
        self.d_i = np.arange(0, self.d / 2)
        self.freq = 1 / (10000 ** (2 * self.d_i / self.d))
        self.datasets = datasets

    def __call__(self, sample):
        x, y = sample["coordinate"]
        x, y = x / 10000, y / 10000
        enc = np.zeros(self.d * 2)
        enc[0 : self.d : 2] = np.sin(x * self.freq)
        enc[1 : self.d : 2] = np.cos(x * self.freq)
        enc[self.d :: 2] = np.sin(y * self.freq)
        enc[self.d + 1 :: 2] = np.cos(y * self.freq)
        sample["coordenc"] = enc
        return sample


class RotationTransform:
    """Rotate by one of the given angles.

    Used in mmt.dataset.landcover_to_landcover.LandcoverToLandcoverDataLoader
    """
    def __init__(
        self, angles, use_image=False, keys_to_mod=["source_data", "target_data"]
    ):
        self.angles = angles
        self.use_image = use_image
        self.keys_to_mod = keys_to_mod

    def __call__(self, sample):
        c = random.choice(self.angles)
        for k in self.keys_to_mod:
            sample[k] = TF.rotate(sample[k], c)
        return sample


class FlipTransform:
    """Rotate by one of the given angles.

    Used in mmt.dataset.landcover_to_landcover.LandcoverToLandcoverDataLoader
    """
    def __init__(self, use_image=False, keys_to_mod=["source_data", "target_data"]):
        self.use_image = use_image
        self.keys_to_mod = keys_to_mod

    def __call__(self, sample):
        if random.random() > 0.5:
            for k in self.keys_to_mod:
                sample[k] = TF.hflip(sample[k])
        if random.random() > 0.5:
            for k in self.keys_to_mod:
                sample[k] = TF.vflip(sample[k])
        return sample


# ======================
# _TransformsDictOrTensor
# ======================


class _TransformsDictOrTensor:
    """Abstract class for transforms that can be applied equally to tensor or to dict

    Overwrite the `transform` method and treat the argument as a tensor.


    Examples
    --------
    >>> tdot = _TransformsDictOrTensor(key = "mask")
    >>> tdot(X)
    >>> tdot({"mask":X})
    """

    def __init__(self, key="mask"):
        self.key = key

    def applytodict(transform):
        """Decorator to apply seaminglessly to dict or tensors"""

        def wrappedtransform(self, x):
            if isinstance(x, dict):
                x[self.key] = transform(self, x[self.key])
            else:
                x = transform(self, x)
            return x

        return wrappedtransform

    @applytodict
    def __call__(self, x):
        return self.transform(x)

    def transform(self, x):
        raise NotImplementedError(
            f"Abstract class {self.__class__.__name__}. Inherit the class and overwrite the method"
        )


class OneHotTorchgeo(_TransformsDictOrTensor):
    """One-hot encoding of land cover patches (new transform)
    Sample is expected to have keys {"mask", "image"}. Transform is made in-place

    Used in mmt.inference.translators.EsawcToEsgp
    """
    def __init__(self, nclasses=-1, device="cpu", dtype=torch.float, key="mask"):
        super().__init__(key)
        self.nclasses = nclasses
        self.dtype = dtype
        self.device = device

    def transform(self, sample):
        if sample.ndim == 3:
            perm_idx = (0, 3, 1, 2)
        else:
            perm_idx = (2, 0, 1)

        sample = torch.nn.functional.one_hot(sample, num_classes=self.nclasses)
        sample = (
            sample.permute(*perm_idx)
            .contiguous()
            .to(dtype=self.dtype, device=self.device)
        )

        return sample

class FillMissingWithNeighbors(_TransformsDictOrTensor):
    """Remplace missing data labels by most frequent non-missing neighbors"""

    def __init__(self, missing_label=0, neighboring_size=1, key="mask"):
        super().__init__(key)
        self.missing_label = missing_label
        self.neighboring_size = neighboring_size

    def transform(self, x):
        if x.ndim == 2:
            nx, ny = x.shape
            x2 = x
        elif x.ndim == 3:
            nc, nx, ny = x.shape
            x2 = x[0, :, :]
        elif x.ndim == 4:
            nb, nc, nx, ny = x.shape
            x2 = x[0, 0, :, :]
        else:
            raise ValueError(f"Unable to deal with {x.ndim}-dimensional tensors")

        w = torch.where(x2 == 0)

        for xz, yz in zip(*w):
            x0 = max(0, xz - self.neighboring_size)
            x1 = min(nx, xz + self.neighboring_size + 1)
            y0 = max(0, yz - self.neighboring_size)
            y1 = min(ny, yz + self.neighboring_size + 1)
            xx = x2[x0:x1, y0:y1]
            if len(xx[xx != 0]) > 0:
                v, c = torch.unique(xx[xx != 0], return_counts=True)
                localmode = v[c.argmax()]
            else:
                localmode = 0

            x2[xz, yz] = localmode

        return x


class FloorDivMinus(_TransformsDictOrTensor):
    """Floor division and substraction"""

    def __init__(self, div=10, minus=0, key="mask"):
        super().__init__(key)
        self.div = div
        self.minus = minus

    def transform(self, x):
        return x // self.div - self.minus


class FillMissingWithSea(_TransformsDictOrTensor):
    """Remplace missing data labels by sea label"""

    def __init__(self, missing_label=0, sea_label=1, key="mask"):
        super().__init__(key)
        self.missing_label = missing_label
        self.sea_label = sea_label

    def transform(self, x):
        x[x == self.missing_label] = self.sea_label
        return x


class EsawcTransform(_TransformsDictOrTensor):
    """The transform to apply to ESA WorldCover original labels"""
    
    def __init__(self, key="mask", threshold=94):
        super().__init__(key)
        self.threshold = threshold

    def transform(self, x):
        x[x == 0] = 80  # Replace zeros by sea
        return torch.where(x < self.threshold, x // 10, x // 10 + 1)


class ScoreTransform(_TransformsDictOrTensor):
    """Remove nan and divide by 100"""

    def __init__(self, divide_by=1, key="image"):
        super().__init__(key)
        self.divide_by = divide_by

    def transform(self, x):
        return torch.where(x.isnan(), 0, x) / self.divide_by


# EOF

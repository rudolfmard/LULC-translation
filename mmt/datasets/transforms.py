#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Transforms
"""
import random

import numpy as np
import torch
import torchvision.transforms as tvt
import torchvision.transforms.functional as TF
from pyproj import Transformer


def rmsuffix(s, startchar="-", stopchar="."):
    """Remove suffix between `startchar` and `stopchar`"""
    if startchar in s:
        return s.split(startchar)[0] + stopchar + s.split(stopchar)[1]
    else:
        return s


class OneHot:
    """One-hot encoding of land cover patches (new transform)
    Sample is expected to have keys {"mask", "image"}. Transform is made in-place

    TODO: inherit from TransformsDictOrTensor. Duplicate of ToOneHot?

    Used in mmt.inference.translators.EsawcToEsgp
    """

    def __init__(self, nclasses=-1, device="cpu", dtype=torch.float, key="mask"):
        self.nclasses = nclasses
        self.dtype = dtype
        self.key = key
        self.device = device

    def __call__(self, sample):
        if isinstance(sample, dict):
            if sample[self.key].ndim == 3:
                perm_idx = (0, 3, 1, 2)
            else:
                perm_idx = (2, 0, 1)

            sample[self.key] = torch.nn.functional.one_hot(
                sample[self.key], num_classes=self.nclasses
            )
            sample[self.key] = (
                sample[self.key]
                .permute(*perm_idx)
                .contiguous()
                .to(dtype=self.dtype, device=self.device)
            )
        else:
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


class ToOneHot:
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

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N, #classes].
        """
        # labelr=labels
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


class CoordEnc(object):
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


class ToLonLat:
    """Convert easting and northing from given CRS to regular lat-lon"""

    def __init__(self, source_crs, target_crs="EPSG:4326"):
        raise DeprecationWarning(
            f"{__name__}.{self.__class__.__name__}: This class is deprecated"
        )
        self.target_crs = target_crs
        if hasattr(source_crs, "to_string"):
            self.source_crs = source_crs.to_string()
        else:
            self.source_crs = source_crs

        self.trans = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    def __call__(self, sample):
        if self.target_crs != self.source_crs:
            x, y = sample["coordinate"]
            lon, lat = self.trans.transform(x, y)
            sample["coordinate"] = (lon, lat)

        return sample


class GeolocEncoder:
    """Encode lon/lat coordinates, as per Vaswani et al. (2017).

    Assume lon/lat coordinates are stored in degrees (as in EPSG:4326).
    Default value adapted to better represent variability over the globe
    (n = 360 instead of 10000)"""

    def __init__(self, d=128, n=360):
        raise DeprecationWarning(
            f"{__name__}.{self.__class__.__name__}: This class is deprecated"
        )
        self.d = int(d / 2)
        self.d_i = np.arange(0, self.d / 2)
        self.freq = 1 / (n ** (2 * self.d_i / self.d))

    def __call__(self, sample):
        x, y = sample["coordinate"]
        enc = np.zeros(self.d * 2)
        enc[0 : self.d : 2] = np.sin(x * self.freq)
        enc[1 : self.d : 2] = np.cos(x * self.freq)
        enc[self.d :: 2] = np.sin(y * self.freq)
        enc[self.d + 1 :: 2] = np.cos(y * self.freq)
        sample["coordenc"] = enc
        return sample


class GeolocEncoderPxwise(GeolocEncoder):
    """Encode lon/lat coordinates, as per Vaswani et al. (2017).

    Assume lon/lat coordinates are stored in degrees (as in EPSG:4326) and
    locate the upper-left corner of an image of shape `imshape`. The returned
    tensor will be of shape (d, imshape).
    Default value adapted to better represent variability over the globe
    (n = 360 instead of 10000)"""

    def __init__(self, imshape, res, d=128, n=360):
        raise DeprecationWarning(
            f"{__name__}.{self.__class__.__name__}: This class is deprecated"
        )
        super().__init__(d=d, n=n)

        if hasattr(res, "__len__"):
            self.imshape = imshape
        else:
            self.imshape = (imshape, imshape)

        if hasattr(res, "__len__"):
            self.res = res
        else:
            self.res = (res, res)

    def __call__(self, sample):
        ulx, uly = sample["coordinate"]
        xs = np.linspace(ulx, ulx + self.imshape[0] * self.res[0], self.imshape[0])
        ys = np.linspace(uly, uly - self.imshape[1] * self.res[1], self.imshape[1])
        x, y = np.meshgrid(xs, ys)
        xf = np.expand_dims(x, axis=0) * np.expand_dims(self.freq, axis=[1, 2])
        yf = np.expand_dims(y, axis=0) * np.expand_dims(self.freq, axis=[1, 2])
        enc = np.zeros((self.d * 2, self.imshape[0], self.imshape[1]))
        enc[0 : self.d : 2, ::] = np.sin(xf)
        enc[1 : self.d : 2, ::] = np.cos(xf)
        enc[self.d :: 2, ::] = np.sin(yf)
        enc[self.d + 1 :: 2, ::] = np.cos(yf)
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
# TransformsDictOrTensor
# ======================


class TransformsDictOrTensor:
    """Abstract class for transforms that can be applied equally to tensor or to dict

    Overwrite the `transform` method and treat the argument as a tensor.


    Examples
    --------
    >>> tdot = TransformsDictOrTensor(key = "mask")
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


class FillMissingWithNeighbors(TransformsDictOrTensor):
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


class FloorDivMinus(TransformsDictOrTensor):
    """Floor division and substraction"""

    def __init__(self, div=10, minus=0, key="mask"):
        super().__init__(key)
        self.div = div
        self.minus = minus

    def transform(self, x):
        return x // self.div - self.minus


class FillMissingWithSea(TransformsDictOrTensor):
    """Remplace missing data labels by sea label"""

    def __init__(self, missing_label=0, sea_label=1, key="mask"):
        super().__init__(key)
        self.missing_label = missing_label
        self.sea_label = sea_label

    def transform(self, x):
        x[x == self.missing_label] = self.sea_label
        return x


class EsawcTransform(TransformsDictOrTensor):
    def __init__(self, key="mask", threshold=94):
        super().__init__(key)
        self.threshold = threshold

    def transform(self, x):
        x[x == 0] = 80  # Replace zeros by sea
        return torch.where(x < self.threshold, x // 10, x // 10 + 1)


class ScoreTransform(TransformsDictOrTensor):
    """Remove nan and divide by 100"""

    def __init__(self, divide_by=1, key="image"):
        super().__init__(key)
        self.divide_by = divide_by

    def transform(self, x):
        return torch.where(x.isnan(), 0, x) / self.divide_by


# EOF

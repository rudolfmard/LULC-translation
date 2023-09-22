import torch
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as tvt
import random

class OneHot:
    def __init__(self, nclasses = -1, device = "cpu", dtype=torch.float, key="mask"):
        self.nclasses = nclasses
        self.dtype = dtype
        self.key = key
        self.device = device
    
    def __call__(self, sample):
        if isinstance(sample, dict):
            if sample[self.key].ndim == 3:
                perm_idx = (0,3,1,2)
            else:
                perm_idx = (2,0,1)
                
            sample[self.key] = torch.nn.functional.one_hot(sample[self.key], num_classes = self.nclasses)
            sample[self.key] = sample[self.key].permute(*perm_idx).contiguous().to(dtype = self.dtype, device = self.device)
        else:
            if sample.ndim == 3:
                perm_idx = (0,3,1,2)
            else:
                perm_idx = (2,0,1)
            
            sample = torch.nn.functional.one_hot(sample, num_classes = self.nclasses)
            sample = sample.permute(*perm_idx).contiguous().to(dtype = self.dtype, device = self.device)
        
        return sample

class ToOneHot(object):
    """Convert ndarrays in sample to Tensors."""

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
                    num_classes=self.nclasses[sample[k.replace("_data", "_name")]] + 1,
                ).permute(0, 3, 1, 2,)[0]
        return sample


class CoordEnc(object):
    """Convert ndarrays in sample to Tensors."""
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
    """Rotate by one of the given angles."""
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
    """Rotate by one of the given angles."""
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


class FloorDivMinus:
    """Floor division and substraction"""
    def __init__(self, div = 10, minus = 1):
        self.div = div
        self.minus = minus
    
    def __call__(self, x):
        if isinstance(x, dict):
            x["mask"] = x["mask"] // self.div - self.minus
        else:
            x = x // self.div - self.minus
        return x

class FillMissingWithSea:
    """Remplace missing data labels by sea label"""
    def __init__(self, missing_label = 0, sea_label = 1):
        self.missing_label = missing_label
        self.sea_label = sea_label
    
    def __call__(self, x):
        if isinstance(x, dict):
            x["mask"][x["mask"] == self.missing_label] = self.sea_label
        else:
            x[x == self.missing_label] = self.sea_label
        return x

EsawcTransform = tvt.Compose([FloorDivMinus(10, 0), FillMissingWithSea(0, 8)])

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Dataset and dataloaders for the training of auto-encoders in map translation
"""
import json
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import transforms as mmt_transforms
from mmt.utils.misc import rmsuffix

# Length of one patch side in metres
PATCH_SIZE_METRES = 6000

CMAP_CATALOG = {
    "clc.hdf5": [
        (255, 255, 255),
        (230, 0, 77),
        (255, 0, 0),
        (204, 77, 242),
        (204, 0, 0),
        (230, 204, 204),
        (230, 204, 230),
        (166, 0, 204),
        (166, 77, 0),
        (255, 77, 255),
        (255, 166, 255),
        (255, 230, 255),
        (255, 255, 168),
        (255, 255, 0),
        (230, 230, 0),
        (230, 128, 0),
        (242, 166, 77),
        (230, 166, 0),
        (230, 230, 77),
        (255, 230, 166),
        (255, 230, 77),
        (230, 204, 77),
        (242, 204, 166),
        (128, 255, 0),
        (0, 166, 0),
        (77, 255, 0),
        (204, 242, 77),
        (166, 255, 128),
        (166, 230, 77),
        (166, 242, 0),
        (230, 230, 230),
        (204, 204, 204),
        (204, 255, 204),
        (0, 0, 0),
        (166, 230, 204),
        (166, 166, 255),
        (77, 77, 255),
        (204, 204, 255),
        (230, 230, 255),
        (166, 166, 230),
        (0, 204, 242),
        (128, 242, 230),
        (0, 255, 166),
        (166, 255, 230),
        (230, 242, 255),
    ],
    "oso.hdf5": [
        (255, 255, 255),
        (255, 0, 255),
        (255, 85, 255),
        (255, 170, 255),
        (0, 255, 255),
        (255, 255, 0),
        (208, 255, 0),
        (161, 214, 0),
        (255, 170, 68),
        (214, 214, 0),
        (255, 85, 0),
        (197, 255, 255),
        (170, 170, 97),
        (170, 170, 0),
        (170, 170, 255),
        (85, 0, 0),
        (0, 156, 0),
        (0, 50, 0),
        (170, 255, 0),
        (85, 170, 127),
        (255, 0, 0),
        (255, 184, 2),
        (190, 190, 190),
        (0, 0, 255),
    ],
    "mos.hdf5": [
        (255, 255, 255),
        (132, 202, 157),
        (230, 223, 205),
        (249, 245, 208),
        (211, 238, 251),
        (180, 193, 170),
        (255, 205, 93),
        (192, 43, 64),
        (187, 184, 220),
        (58, 147, 169),
        (0, 0, 0),
        (138, 140, 143),
    ],
    "ocsge_o.hdf5": [
        (255, 255, 255),
        (255, 55, 122),
        (255, 145, 145),
        (255, 255, 153),
        (166, 77, 0),
        (204, 204, 204),
        (0, 204, 242),
        (166, 230, 204),
        (128, 255, 0),
        (0, 166, 0),
        (128, 190, 0),
        (166, 255, 128),
        (230, 128, 0),
        (204, 242, 77),
        (204, 255, 204),
    ],
    "ocsge_u.hdf5": [
        (255, 255, 255),
        (255, 255, 168),
        (0, 128, 0),
        (166, 0, 204),
        (0, 0, 153),
        (230, 0, 77),
        (204, 0, 0),
        (90, 90, 90),
        (230, 204, 230),
        (0, 102, 255),
        (255, 0, 0),
        (255, 75, 0),
        (255, 77, 255),
        (64, 64, 64),
        (240, 240, 40),
    ],
    "cgls.hdf5": [
        (255, 255, 255),
        (46, 128, 21),
        (132, 151, 0),
        (255, 187, 34),
        (255, 255, 76),
        (0, 150, 160),
        (250, 230, 160),
        (180, 180, 180),
        (240, 150, 255),
        (250, 0, 0),
        (240, 240, 240),
        (0, 50, 200),
        (0, 0, 128),
    ],
    "ecosg.hdf5": [
        (0, 0, 0),
        (0, 0, 128),
        (0, 0, 205),
        (0, 0, 255),
        (211, 211, 211),
        (169, 169, 169),
        (255, 250, 250),
        (240, 255, 240),
        (85, 107, 47),
        (154, 205, 50),
        (0, 128, 0),
        (255, 127, 80),
        (160, 82, 45),
        (34, 139, 34),
        (188, 143, 143),
        (205, 133, 63),
        (222, 184, 135),
        (50, 205, 50),
        (255, 215, 0),
        (32, 178, 170),
        (173, 255, 47),
        (189, 183, 107),
        (102, 102, 0),
        (46, 139, 87),
        (138, 2, 0),
        (206, 0, 0),
        (252, 1, 1),
        (255, 90, 0),
        (255, 120, 0),
        (255, 150, 0),
        (255, 180, 0),
        (255, 210, 0),
        (255, 240, 0),
        (128, 128, 128),
    ],
    "esgp.hdf5": [
        (0, 0, 0),
        (0, 0, 128),
        (0, 0, 205),
        (0, 0, 255),
        (211, 211, 211),
        (169, 169, 169),
        (255, 250, 250),
        (240, 255, 240),
        (85, 107, 47),
        (154, 205, 50),
        (0, 128, 0),
        (255, 127, 80),
        (160, 82, 45),
        (34, 139, 34),
        (188, 143, 143),
        (205, 133, 63),
        (222, 184, 135),
        (50, 205, 50),
        (255, 215, 0),
        (32, 178, 170),
        (173, 255, 47),
        (189, 183, 107),
        (102, 102, 0),
        (46, 139, 87),
        (138, 2, 0),
        (206, 0, 0),
        (252, 1, 1),
        (255, 90, 0),
        (255, 120, 0),
        (255, 150, 0),
        (255, 180, 0),
        (255, 210, 0),
        (255, 240, 0),
        (128, 128, 128),
    ],
    "esawc.hdf5": [
        (0, 0, 0),
        (0, 100, 0),
        (255, 187, 34),
        (255, 255, 76),
        (240, 150, 255),
        (250, 0, 0),
        (180, 180, 180),
        (240, 240, 240),
        (0, 100, 200),
        (0, 150, 160),
        (0, 207, 117),
        (250, 230, 160),
    ],
}
LABELS_CATALOG = {
    "clc.hdf5": [
        "no data",
        "cont. urban",
        "disc urban",
        "ind/ om",
        "road/ rail",
        "port",
        "airport",
        "mine",
        "dump",
        "construction",
        "green urban",
        "leisure",
        "non irrigated crops",
        "perm irrigated crops",
        "rice",
        "vineyards",
        "fruit",
        "olive",
        "pastures",
        "mixed crops",
        "complex crops",
        "crops + nature",
        "agro-forestry",
        "broad leaved",
        "conifere",
        "mixed forest",
        "natural grass",
        "moors",
        "sclerophyllous",
        "transi wood-shrub",
        "sand",
        "rocks",
        "sparsely vege",
        "burnt",
        "snow",
        "marshes",
        "peat bogs",
        "salt marshes",
        "salines",
        "intertidal flats",
        "river",
        "lakes",
        "lagoons",
        "estuaries",
        "sea",
    ],
    "oso.hdf5": [
        "no data",
        "dense urban",
        "sparse urban",
        "ind and com",
        "roads",
        "rapeseeds",
        "cereals",
        "protein crops",
        "soy",
        "sunflower",
        "maize",
        "rice",
        "tubers",
        "meadow",
        "orchards",
        "vineyards",
        "Broad-leaved",
        "coniferous",
        "lawn",
        "shrubs",
        "rocks",
        "sand",
        "snow",
        "water",
    ],
    "mos.hdf5": [
        "no data",
        "forest",
        "semi-natural",
        "crops",
        "water",
        "green urban",
        "ind. housing",
        "col. housing",
        "activities",
        "facilities",
        "transport",
        "Mine/dump",
    ],
    "ocsge_o.hdf5": [
        "no data",
        "built",
        "concrete",
        "mineral",
        "mixed materials",
        "bare soil",
        "water",
        "snow",
        "broad-leaved",
        "neadle-leaved",
        "mixed-trees",
        "shrubs",
        "vine",
        "grass",
        "moss",
    ],
    "ocsge_u.hdf5": [
        "no data",
        "farming",
        "forestry",
        "extraction",
        "fishing",
        "house/ind/com",
        "roads",
        "rails",
        "airports",
        "fluvial transport",
        "logistics/storage",
        "public uti networks",
        "transitionnal",
        "abandoned",
        "no-use",
    ],
    "cgls.hdf5": [
        "no data",
        "closed forest",
        "open forest",
        "shrubland",
        "herbaceous",
        "wetland",
        "moss/lichen",
        "bare/sparse",
        "cropland",
        "built-up",
        "snow",
        "water",
        "ocean",
    ],
    "ecosg.hdf5": [
        "0. no data",
        "1. sea and oceans",
        "2. lakes",
        "3. rivers",
        "4. bare land",
        "5. bare rock",
        "6. permanent snow",
        "7. boreal broadleaf deciduous",
        "8. temperate broadleaf deciduous",
        "9. tropical broadleaf deciduous",
        "10. temperate broadleaf evergreen",
        "11. tropical broadleaf evergreen",
        "12. boreal needleleaf evergreen",
        "13. temperate needleleaf evergreen",
        "14. boreal needleleaf deciduous",
        "15. shrubs",
        "16. boreal grassland",
        "17. temperate grassland",
        "18. tropical grassland",
        "19. winter C3 crops",
        "20. summer C3 crops",
        "21. C4 crops",
        "22. flooded trees",
        "23. flooded grassland",
        "24. LCZ1: compact high-rise",
        "25. LCZ2: compact midrise",
        "26. LCZ3: compact low-rise",
        "27. LCZ4: open high-rise",
        "28. LCZ5: open midrise",
        "29: LCZ6: open low-rise",
        "30: LCZ7: lightweight low-rise",
        "31: LCZ8: large low-rise",
        "32: LCZ9: sparsely built",
        "33: LCZ10: heavy industry",
    ],
    "esgp.hdf5": [
        "0. no data",
        "1. sea and oceans",
        "2. lakes",
        "3. rivers",
        "4. bare land",
        "5. bare rock",
        "6. permanent snow",
        "7. boreal broadleaf deciduous",
        "8. temperate broadleaf deciduous",
        "9. tropical broadleaf deciduous",
        "10. temperate broadleaf evergreen",
        "11. tropical broadleaf evergreen",
        "12. boreal needleleaf evergreen",
        "13. temperate needleleaf evergreen",
        "14. boreal needleleaf deciduous",
        "15. shrubs",
        "16. boreal grassland",
        "17. temperate grassland",
        "18. tropical grassland",
        "19. winter C3 crops",
        "20. summer C3 crops",
        "21. C4 crops",
        "22. flooded trees",
        "23. flooded grassland",
        "24. LCZ1: compact high-rise",
        "25. LCZ2: compact midrise",
        "26. LCZ3: compact low-rise",
        "27. LCZ4: open high-rise",
        "28. LCZ5: open midrise",
        "29: LCZ6: open low-rise",
        "30: LCZ7: lightweight low-rise",
        "31: LCZ8: large low-rise",
        "32: LCZ9: sparsely built",
        "33: LCZ10: heavy industry",
    ],
    "esawc.hdf5": [
        "No data",
        "Tree cover",
        "Shrubland",
        "Grassland",
        "Cropland",
        "Built-up",
        "Bare/sparse veg.",
        "Snow and ice",
        "Permanent water bodies",
        "Herbaceous wetland",
        "Mangroves",
        "Moss and lichen",
    ],
}
RESOLUTION_CATALOG = {  # Official resolution in metres
    "clc.hdf5": 100,
    "oso.hdf5": 10,
    "mos.hdf5": 5,
    "ocsge_o.hdf5": 5,
    "ocsge_u.hdf5": 5,
    "cgls.hdf5": 100,
    "ecosg.hdf5": 300,
    "esgp.hdf5": 60,
    "esawc.hdf5": 10,
}



class LandcoverToLandcover(Dataset):
    """Dataset used in the phase 1 of the training (based on  the 
    files "ecosg.hdf5", "esawc.hdf5", "esgp.hdf5", "oso.hdf5", ..., "train_test_val_60.json")
    
    Items of this dataset are pairs of land cover patches (source and target)
    covering the same location. They are extracted from the HDF5 files and
    stored in dict with the following keys:
      * `source_data`: the land cover labels of the source map
      * `target_data`: the land cover labels of the target map
      * `coordinate`: (easting, northing) coordinates of the upper-left corner of the patch
      * `patch_id`: the ID of the returned patch (same for target and source)
      * `source_name`: the file name of the source land cover
      * `target_name`: the file name of the target land cover
    
    
    Example
    -------
    >>> from mmt.datasets import landcover_to_landcover
    >>> ds = landcover_to_landcover.LandcoverToLandcover(
        path="data/hdf5_data",
        source="ecosg.hdf5",
        target="oso.hdf5",
        list_patch_id=['5456', '5457', '5458', '5459']
    )
    >>> ds[0]
    {
        'patch_id': 5459.0,
        'source_data': tensor([[[19, 21, 21, 20, ..., 21, 21,  8]]], device='cuda:0'),
        'target_data': tensor([[[16, 17, 17,  ..., 16, 16, 16]]], device='cuda:0'),
        'coordinate': (213639.6885, 6802786.6972),
        'source_name': 'ecosg.hdf5',
        'target_name': 'oso.hdf5',
    }
    """
    def __init__(
        self,
        path,
        source,
        target,
        list_patch_id,
        mode="train",
        transform=None,
        device="cuda",
    ):
        """Constructor.
        
        
        Parameters
        ----------
        path: str
            Path to the directory containing the HDF5 files
        
        source: str
            File name of the source land cover
        
        target: str
            File name of the target land cover
        
        list_patch_id: list of str
            List of patch to be included in the dataset. Returned patches
            are the intersection between the patches in this list and the ones
            in the JSON files with train/test/val split.
        
        mode: {"train", "test", "val"}
            The subset of data to be considered. Patches IDs are loaded
            from the file `train_test_val_60.json`, also sotred under `path`
            
        transform: callable, optional
            Any transform to apply to the sample
        
        device: {"cuda", "cpu"}
            The device on which the data will be loaded
        
        
        Notes
        -----
        No check is made on the patches ID. Please make sure the IDs
        in `list_patch_id` are also in `self.source_dataset.keys()`
        """
        self.source = source
        self.target = target
        self.source_dataset_path = os.path.join(path, source)
        self.target_dataset_path = os.path.join(path, target)
        self.device = device

        self.list_patch_id = list_patch_id
        with open(os.path.join(path, "train_test_val_60.json"), "r") as fp:
            data = json.load(fp)
        
        self.list_patch_id = list(
            set.intersection(set(self.list_patch_id), set(data[mode]))
        )

        self.transform = transform

    def __len__(self):
        """Length of the dataset (intersection of patch IDs in `list_patch_id` and the file `train_test_val_60.json`)"""
        return len(self.list_patch_id)

    def open_hdf5(self):
        self.source_dataset = h5py.File(
            self.source_dataset_path, "r", swmr=True, libver="latest"
        )
        self.target_dataset = h5py.File(
            self.target_dataset_path, "r", swmr=True, libver="latest"
        )
    
    def close_hdf5(self):
        self.source_dataset.close()
        self.target_dataset.close()

    def __getitem__(self, idx):
        """Fetch an item of the dataset
        
        
        Parameters
        ----------
        idx: int
            Index of the item (between 0 and len(self))
        
        
        Returns
        -------
        sample: dict
            Land cover data of the source and target maps stored in a dict
            with the following keys:
              * `source_data`: the land cover labels of the source map
              * `target_data`: the land cover labels of the target map
              * `coordinate`: (easting, northing) coordinates of the upper-left corner of the patch
              * `patch_id`: the ID of the returned patch (same for target and source)
              * `source_name`: the file name of the source land cover
              * `target_name`: the file name of the target land cover
        """
        if not hasattr(self, "source_dataset"):
            self.open_hdf5()
        
        with torch.no_grad():
            patch_id = self.list_patch_id[idx]
            sample = {"patch_id": float(patch_id)}

            tmp = self.source_dataset.get(patch_id)
            sample["source_data"] = torch.tensor(
                tmp[:].astype(float), dtype=torch.float, device=self.device
            )  # .astype(float)
            tmp2 = self.target_dataset.get(patch_id)
            sample["target_data"] = torch.tensor(
                tmp2[:].astype(float), dtype=torch.float, device=self.device
            )

            s1 = sample["source_data"].shape
            s2 = sample["target_data"].shape
            m1 = torch.clip(sample["source_data"], 0, 1)
            m2 = torch.clip(sample["target_data"], 0, 1)

            s = max(s1[1], s2[1])
            m1 = torch.nn.functional.interpolate(
                m1[None, :], size=(s, s), mode="nearest", recompute_scale_factor=False
            )
            m2 = torch.nn.functional.interpolate(
                m2[None, :], size=(s, s), mode="nearest", recompute_scale_factor=False
            )
            m = m1 * m2

            sample["source_data"] *= torch.nn.functional.interpolate(
                m, size=(s1[1], s1[2]), mode="nearest", recompute_scale_factor=False
            )[0]
            sample["target_data"] *= torch.nn.functional.interpolate(
                m, size=(s2[1], s2[2]), mode="nearest", recompute_scale_factor=False
            )[0]
            sample["source_data"] = sample["source_data"].long()
            sample["target_data"] = sample["target_data"].long()
            sample["coordinate"] = (
                tmp.attrs["x_coor"].astype(float),
                tmp.attrs["y_coor"].astype(float),
            )

            sample["source_name"] = self.source
            sample["target_name"] = self.target

        if self.transform:
            sample = self.transform(sample)
        
        return sample


class LandcoverToLandcoverNoJson(LandcoverToLandcover):
    """Dataset used in the phase 2 of the training (based on  the 
    files "ecosg-train.hdf5", "esawc-train.hdf5", "esgp-train.hdf5")
    
    Items of this dataset are the same as in LandcoverToLandcover. The main
    difference with this class in that the train/test/val split is not made
    in a JSON file but in the file names. Only train subset is extracted.
    
    
    Example
    -------
    >>> from mmt.datasets import landcover_to_landcover
    >>> ds = landcover_to_landcover.LandcoverToLandcoverNoJson(
        path="data/hdf5_data",
        source="ecosg-train.hdf5",
        target="esgp-train.hdf5",
        list_patch_id=['719', '72', '720', '721']
    )
    >>> ds[0]
    {
        'patch_id': 719.0,
        'source_data': tensor([[[12, 19, 19, ..., 19, 19, 20]]], device='cuda:0'),
        'target_data': tensor([[[12, 12, 12,  ..., 17, 19, 20]]], device='cuda:0'),
        'coordinate': (14.079684561057785, 49.34717120826244),
        'source_name': 'ecosg-train.hdf5',
        'target_name': 'esgp-train.hdf5',
    }
    """
    def __init__(
        self,
        path,
        source,
        target,
        list_patch_id,
        mode="train",
        transform=None,
        device="cuda",
    ):
        if "-" in source:
            source.replace("train", mode)
            target.replace("train", mode)

        self.source = source
        self.target = target
        self.source_dataset_path = os.path.join(path, source)
        self.target_dataset_path = os.path.join(path, target)
        self.device = device
        self.list_patch_id = list_patch_id
        self.transform = transform

    def __getitem__(self, idx):
        """Fetch an item of the dataset
        
        
        Parameters
        ----------
        idx: int
            Index of the item (between 0 and len(self))
        
        
        Returns
        -------
        sample: dict
            Land cover data of the source and target maps stored in a dict
            with the following keys:
              * `source_data`: the land cover labels of the source map
              * `target_data`: the land cover labels of the target map
              * `coordinate`: (easting, northing) coordinates of the upper-left corner of the patch
              * `patch_id`: the ID of the returned patch (same for target and source)
              * `source_name`: the file name of the source land cover
              * `target_name`: the file name of the target land cover
        """
        if not hasattr(self, "source_dataset"):
            self.open_hdf5()
        with torch.no_grad():
            patch_id = self.list_patch_id[idx]
            sample = {"patch_id": float(patch_id)}

            src = self.source_dataset.get(patch_id)
            sample["source_data"] = torch.tensor(
                src[:].astype(np.int64), dtype=torch.long, device=self.device
            ).unsqueeze(
                0
            )  # .astype(float)
            trg = self.target_dataset.get(patch_id)
            sample["target_data"] = torch.tensor(
                trg[:].astype(np.int64), dtype=torch.long, device=self.device
            ).unsqueeze(0)

            sample["coordinate"] = (
                src.attrs["x_coor"].astype(float),
                src.attrs["y_coor"].astype(float),
            )

            sample["source_name"] = self.source
            sample["target_name"] = self.target

        if self.transform:
            sample = self.transform(sample)
        return sample


class LandcoverToLandcoverDataLoader:
    """Data loader used in the training of the auto-encoders doing map translation.
    
    While the dataset return samples with one source map and one target map,
    the data loader takes a list of land cover and return Pytorch dataloaders
    for each pair of maps, used alternatively as source or target.
    
    
    Main attributes
    ---------------
    train_loader: dict of `torch.utils.data.Dataloader`
        Nested dict of dataloaders used in training. See examples for the nesting tree.
    
    test_loader: dict of `torch.utils.data.Dataloader`
        Same as train_loader but on the testing set
    
    val_loader: dict of `torch.utils.data.Dataloader`
        Same as train_loader but on the testing set
    
    
    Main methods
    ------------
    plot_samples_per_epoch(
        self,
        inputs,
        targets,
        outputs,
        embedding,
        dataset_src,
        dataset_tgt,
        epoch,
        coordinate,
    ) -> str:
        Plot the source map, the target map, the translation and the embedding.
    
    
    Examples
    --------
    >>> from mmt.datasets import landcover_to_landcover
    >>> from mmt.utils import config as utilconfig
    >>> config = utilconfig.process_config("configs/test_config.yaml")
    ...
    
    >>> # Instanciate the data loaders
    >>> dl = landcover_to_landcover.LandcoverToLandcoverDataLoader(
        config,
        datasets = ['esawc.hdf5', 'esgp.hdf5', 'ecosg.hdf5']
    )
    >>> dl.train_loader
    {
        'esawc.hdf5': {
            'esgp.hdf5': <torch.utils.data.dataloader.DataLoader object at 0x14d936f86c10>,
            'ecosg.hdf5': <torch.utils.data.dataloader.DataLoader object at 0x14d938297310>
        },
        'esgp.hdf5': {
            'esawc.hdf5': <torch.utils.data.dataloader.DataLoader object at 0x14d938297410>,
            'ecosg.hdf5': <torch.utils.data.dataloader.DataLoader object at 0x14d936f85010>
        },
        'ecosg.hdf5': {
            'esawc.hdf5': <torch.utils.data.dataloader.DataLoader object at 0x14d936f85250>,
            'esgp.hdf5': <torch.utils.data.dataloader.DataLoader object at 0x14d936f85310>
        }
    }
    >>> # Access the data
    >>> source = 'esawc.hdf5'
    >>> target = 'esgp.hdf5'
    >>> dl.train_loader[source][target] # This is the dataloader used to provide data in the training of ESAWC to ECOSG+
    <torch.utils.data.dataloader.DataLoader object at 0x14d936f86c10>
    >>>
    >>> # Plot the data
    >>> x = next(iter(dl.train_loader[source][target]))
    >>> figname = dl.plot_samples_per_epoch(
            inputs = x["source_one_hot"],
            targets = x["target_one_hot"],
            outputs = x["target_one_hot"],    # Fake data, just for the demo
            embedding = x["source_one_hot"],    # Fake data, just for the demo
            dataset_src = "esawc.hdf5",
            dataset_tgt = "esgp.hdf5",
            epoch = 42,    # Fake data, just for the demo
            coordinate = x["coordinate"]
        )
    >>> figname # Path to the figure
    """
    def __init__(
        self,
        config,
        datasets,
        dataset_class="LandcoverToLandcover",
        to_one_hot=True,
        pos_enc=False,
        ampli=True,
        num_workers=4,
    ):
        """Constructor.
        
        
        Parameters
        ----------
        config: dict
            The configuration of the experiment
        
        datasets: list of str
            List of land cover maps to include in the training.
        
        dataset_class: {"LandcoverToLandcover", "LandcoverToLandcoverNoJson"}
            The dataset class to use ("LandcoverToLandcover" for phase 1, "LandcoverToLandcoverNoJson" for phase 2)
        
        to_one_hot: bool
            Append one-hot encoding of land cover labels to the sample dict
        
        pos_enc: bool
            Triggers position encoding of patches
        
        ampli: bool
            Triggers augmentation of the dataset by flipping and rotating the patches
        
        num_workers: int
            The number of CPU used to access the data
        """
        self.config = config
        self.datadir = os.path.join(mmt_repopath, "data", "hdf5_data")
        self.device = "cuda" if config.cuda else "cpu"
        self.datasets = [rmsuffix(dataset) for dataset in datasets]

        full_path_datasets = [
            os.path.join(self.datadir, dataset) for dataset in datasets
        ]
        self.input_channels = []
        self.output_channels = []
        self.real_patch_sizes = []
        self.nb_patch_per_dataset = []
        self.n_classes = {}
        id_patch_per_dataset = {}
        for fpds in full_path_datasets:
            with h5py.File(fpds, "r") as f:
                self.real_patch_sizes.append(int(f.attrs["patch_size"]))
                self.input_channels.append(int(f.attrs["numberclasses"]) + 1)
                self.output_channels.append(int(f.attrs["numberclasses"]) + 1)
                self.nb_patch_per_dataset.append(len(f.keys()))
                self.n_classes[rmsuffix(os.path.basename(fpds))] = int(
                    f.attrs["numberclasses"]
                )
                id_patch_per_dataset[fpds] = list(f.keys())

        self.couple_patch_per_dataset = {}
        nested_dict_datasets = {}
        self.total_couple = 0
        for fpsrc in full_path_datasets:
            innerdict_inter = {}
            innerdict_none = {}
            for fptrg in full_path_datasets:
                if fpsrc != fptrg:
                    inter = list(
                        set.intersection(
                            set(id_patch_per_dataset[fpsrc]),
                            set(id_patch_per_dataset[fptrg]),
                        )
                    )
                    if len(inter) > 0:
                        innerdict_inter[os.path.basename(fptrg)] = inter
                        innerdict_none[rmsuffix(os.path.basename(fptrg))] = None
                        self.total_couple += len(inter)

            self.couple_patch_per_dataset[os.path.basename(fpsrc)] = innerdict_inter
            nested_dict_datasets[rmsuffix(os.path.basename(fpsrc))] = innerdict_none

        """At this point we have the following:
        
        couple_patch_per_dataset = {
            'ecosg-train.hdf5':
                {
                    'esawc-train.hdf5': ['79', '3', ...],
                    'esgp-train.hdf5': ['79', '3', ...],
                },
            'esawc-train.hdf5': 
                {
                    'ecosg-train.hdf5': ['79', '3', ...],
                    'esgp-train.hdf5': ['79', '3', ...],
                },
                ...
        } --> will be used to correctly instanciate the DatasetClasses
        
        nested_dict_datasets = {
            'ecosg.hdf5':
                {
                    'esawc.hdf5': None,
                    'esgp.hdf5': None,
                },
            'esawc.hdf5': 
                {
                    'ecosg.hdf5': None,
                    'esgp.hdf5': None,
                },
                ...
        } --> will be used to loop on the datasets
        """
        self.nb_patch_per_dataset = np.array(self.nb_patch_per_dataset)
        self.nb_patch_per_dataset = (
            self.nb_patch_per_dataset / self.nb_patch_per_dataset.sum()
        )

        dic_list_transform = {
            source: {target: [] for target, _ in targetval.items()}
            for source, targetval in nested_dict_datasets.items()
        }
        dic_list_train_transform = {
            source: {target: [] for target, _ in targetval.items()}
            for source, targetval in nested_dict_datasets.items()
        }
        for source, targetval in nested_dict_datasets.items():
            for target, _ in targetval.items():
                if ampli:
                    dic_list_train_transform[source][target].append(
                        mmt_transforms.FlipTransform()
                    )
                    dic_list_train_transform[source][target].append(
                        mmt_transforms.RotationTransform([0, 90, 180, 270])
                    )
                if to_one_hot:
                    dic_list_transform[source][target].append(
                        mmt_transforms.OneHotHdf5(self.n_classes)
                    )
                if pos_enc:
                    dic_list_transform[source][target].append(
                        mmt_transforms.CoordEnc(self.n_classes.keys())
                    )

                dic_list_train_transform[source][target] = Compose(
                    dic_list_train_transform[source][target]
                    + dic_list_transform[source][target]
                )
                dic_list_transform[source][target] = Compose(
                    dic_list_transform[source][target]
                )
        DatasetClass = eval(dataset_class)
        self.train = {
            rmsuffix(source): {
                rmsuffix(target): DatasetClass(
                    self.datadir,
                    source,
                    target,
                    val,
                    mode="train",
                    transform=dic_list_train_transform[rmsuffix(source)][
                        rmsuffix(target)
                    ],
                    device=self.device,
                )
                for target, val in targetval.items()
            }
            for source, targetval in self.couple_patch_per_dataset.items()
        }

        self.valid = {
            rmsuffix(source): {
                rmsuffix(target): DatasetClass(
                    self.datadir,
                    source,
                    target,
                    val,
                    mode="validation",
                    transform=dic_list_transform[rmsuffix(source)][rmsuffix(target)],
                    device=self.device,
                )
                for target, val in targetval.items()
            }
            for source, targetval in self.couple_patch_per_dataset.items()
        }
        self.test = {
            rmsuffix(source): {
                rmsuffix(target): DatasetClass(
                    self.datadir,
                    source,
                    target,
                    val,
                    mode="test",
                    transform=dic_list_transform[rmsuffix(source)][rmsuffix(target)],
                    device=self.device,
                )
                for target, val in targetval.items()
            }
            for source, targetval in self.couple_patch_per_dataset.items()
        }

        if self.device == "cpu":
            pin_memory = True
        else:
            pin_memory = False

        self.train_loader = {
            source: {
                target: DataLoader(
                    val,
                    batch_size=self.config.training.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=num_workers > 0,
                )
                for target, val in targetval.items()
            }
            for source, targetval in self.train.items()
        }
        self.valid_loader = {
            source: {
                target: DataLoader(
                    val,
                    batch_size=self.config.training.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=num_workers > 0,
                )
                for target, val in targetval.items()
            }
            for source, targetval in self.valid.items()
        }
        self.test_loader = {
            source: {
                target: DataLoader(
                    val,
                    batch_size=self.config.training.batch_size,
                    shuffle=True,
                    num_workers=num_workers,
                    pin_memory=pin_memory,
                    persistent_workers=num_workers > 0,
                )
                for target, val in targetval.items()
            }
            for source, targetval in self.test.items()
        }

    def plot_samples_per_epoch(
        self,
        inputs,
        targets,
        outputs,
        embedding,
        dataset_src,
        dataset_tgt,
        epoch,
        coordinate,
        cmap="original",
        figname=None,
    ):
        """Plot the source map, the target map, the translation and the embedding.
        
        
        Parameters
        ----------
        inputs: torch.tensor of shape (B, C, H, W)
            Source land cover labels
        
        targets: torch.tensor of shape (B, C', H', W')
            Target land cover labels
        
        outputs: torch.tensor of shape (B, C', H', W')
            Result of the map translation
        
        embedding: torch.tensor of shape (B, C", H", W")
            Embedding after encoding the source land cover map
        
        dataset_src: str
            Name of the source land cover
        
        dataset_tgt: str
            Name of the target land cover
        
        epoch: int
            Epoch number (used only in the figure name)
        
        coordinate:
            Coordinate of the upper-left corner of the patch (only used in the title of the figure)
        
        cmap: {"default", "orgininal"}, optional
            If "original", uses the colormap defined for the land cover map.
            If "default", uses the "RdBu" colormap.
        
        figname: str, optional
            File name of the figure (will be completed with the directory `self.config.paths.out_dir`)
        
        
        Notes
        -----
        The embeddings are reduced to 3-dimensional tensors (RGB) with PCA.
        Used in MultiLULCAgent.validate()
        """
        
        with torch.no_grad():
            if len(inputs.shape) == 4:
                inputs = torch.argmax(inputs[0], dim=0)
                targets = torch.argmax(targets[0], dim=0)
            else:
                inputs = inputs[0]
                targets = targets[0]
            outputs = torch.argmax(outputs[0], dim=0)
            if figname is None:
                figname = os.path.join(
                    self.config.paths.out_dir,
                    f"Epoch_{epoch}_Source_{dataset_src}_Target_{dataset_tgt}.png",
                )
            else:
                figname = os.path.join(self.config.paths.out_dir, figname)

            # Start figure
            # --------------
            f, ax = plt.subplots(2, 2, figsize=(20, 20))
            # get discrete colormap
            if cmap == "default":
                cmap_src = plt.get_cmap("RdBu", self.n_classes[dataset_src] + 1)
                cmap_tgt = plt.get_cmap("RdBu", self.n_classes[dataset_tgt] + 1)
            elif cmap == "original":
                cmap_src = LinearSegmentedColormap.from_list(
                    dataset_src,
                    np.array(CMAP_CATALOG[dataset_src]) / 255,
                    N=self.n_classes[dataset_src] + 1,
                )
                cmap_tgt = LinearSegmentedColormap.from_list(
                    dataset_tgt,
                    np.array(CMAP_CATALOG[dataset_tgt]) / 255,
                    N=self.n_classes[dataset_tgt] + 1,
                )

            # Source
            # --------
            m1 = ax[0][0].imshow(
                inputs.cpu().long().numpy(),
                cmap=cmap_src,
                vmin=0 - 0.5,
                vmax=self.n_classes[dataset_src] + 0.5,
            )
            ax[0][0].set_title("Source")

            # Target
            # --------
            m2 = ax[0][1].imshow(
                targets.cpu().long().numpy(),
                cmap=cmap_tgt,
                vmin=0 - 0.5,
                vmax=self.n_classes[dataset_tgt] + 0.5,
            )
            ax[0][1].set_title("Target")

            # Translation
            # -------------
            m3 = ax[1][0].imshow(
                outputs.cpu().long().numpy(),
                cmap=cmap_tgt,
                vmin=0 - 0.5,
                vmax=self.n_classes[dataset_tgt] + 0.5,
            )
            ax[1][0].set_title("Translation")

            # Embedding
            # -----------
            fmap_dim = embedding.shape[1]  # nchannel
            n_pix = embedding.shape[2]
            # we use a pca to project the embeddings to a RGB space
            pca = PCA(n_components=3)
            pca.fit(np.eye(fmap_dim))
            # we need to adapt dimension and memory allocation to CPU
            fmap_ = (
                embedding[0]
                .cpu()
                .detach()
                .numpy()
                .squeeze()
                .reshape((fmap_dim, -1))
                .transpose(1, 0)
            )  # yikes
            color_vector = pca.transform(fmap_)
            emb = color_vector.reshape((n_pix, n_pix, 3), order="F").transpose(1, 0, 2)
            m4 = ax[1][1].imshow((emb * 255).astype(np.uint8))
            ax[1][1].set_title("Embedding")

            # Wrap-up
            # ---------
            # tell the colorbar to tick at integers
            f.colorbar(
                m1, ticks=np.arange(0, self.n_classes[dataset_src] + 1), ax=ax[0][0]
            )
            f.colorbar(
                m2, ticks=np.arange(0, self.n_classes[dataset_tgt] + 1), ax=ax[0][1]
            )
            f.colorbar(
                m3, ticks=np.arange(0, self.n_classes[dataset_tgt] + 1), ax=ax[1][0]
            )
            acc = np.round((outputs == targets).sum().item() / targets.numel(), 3)
            f.suptitle(
                f"x={coordinate[0][0].item()}, y={coordinate[1][0].item()}, accuracy={acc}"
            )
        f.savefig(figname)
        plt.close(f)

        return figname

# EOF

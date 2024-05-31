#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Dataset and dataloaders
"""
import json
import os

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import transforms

rmsuffix = transforms.rmsuffix

# Length of one patch side in metres
patch_size_metres = 6000

cmap_dict = {
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
label_dict = {
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
resolution_dict = {  # Official geometric accuracy in metres
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

# cmap_dict.update(
# {
# f"{lcname}-{subset}.hdf5":cmap_dict[lcname + ".hdf5"]
# for lcname in ["esawc", "ecosg", "esgp"]
# for subset in ["train", "test", "val"]
# }
# )
# label_dict.update(
# {
# f"{lcname}-{subset}.hdf5":label_dict[lcname + ".hdf5"]
# for lcname in ["esawc", "ecosg", "esgp"]
# for subset in ["train", "test", "val"]
# }
# )


class EEEmapsDataset(Dataset):
    """Dataset providing ESAWorldCover, EcoclimapSG and EcoclimapSGplus patches (EEE)"""

    def __init__(self, path, mode="train", transform=None):
        raise DeprecationWarning(
            f"{__name__}.{self.__class__.__name__}: This class is deprecated"
        )
        self.path = path
        self.mode = mode
        self.transform = transform
        self.h5f = {
            lc: h5py.File(
                os.path.join(path, f"{lc}-{mode}.hdf5"), "r", swmr=True, libver="latest"
            )
            for lc in ["esawc", "esgp", "ecosg"]
        }

    def __len__(self):
        return len(self.h5f["esawc"])

    def __getitem__(self, idx):
        idx = str(idx)
        sample = {
            lc: torch.Tensor(self.h5f[lc][idx][:]).long()
            for lc in ["esawc", "esgp", "ecosg"]
        }
        sample["coordinate"] = (
            self.h5f["esgp"][idx].attrs["x_coor"],
            self.h5f["esgp"][idx].attrs["y_coor"],
        )
        if self.transform:
            sample = self.transform(sample)

        return sample

    def close_hdf5(self):
        for lc in ["esawc", "esgp", "ecosg"]:
            self.h5f[lc].close()


class LandcoverToLandcover(Dataset):
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
        self.source = source
        # print(self.master_dataset)
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

        # self.list_patch_id =self.list_patch_id[:250]
        self.transform = transform

    def __len__(self):
        return len(self.list_patch_id)

    def open_hdf5(self):
        self.source_dataset = h5py.File(
            self.source_dataset_path, "r", swmr=True, libver="latest"
        )
        self.target_dataset = h5py.File(
            self.target_dataset_path, "r", swmr=True, libver="latest"
        )

    def __getitem__(self, idx):
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
        """
        :param config:
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
                        transforms.FlipTransform()
                    )
                    dic_list_train_transform[source][target].append(
                        transforms.RotationTransform([0, 90, 180, 270])
                    )
                if to_one_hot:
                    dic_list_transform[source][target].append(
                        transforms.ToOneHot(self.n_classes)
                    )
                if pos_enc:
                    dic_list_transform[source][target].append(
                        transforms.CoordEnc(self.n_classes.keys())
                    )
                    # dic_list_transform[source][target].append(
                    # transforms.ToLonLat(source_crs = "EPSG:2154")
                    # )
                    # dic_list_transform[source][target].append(
                    # transforms.GeolocEncoder()
                    # )

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
        title=None,
    ):

        with torch.no_grad():
            if len(inputs.shape) == 4:
                inputs = torch.argmax(inputs[0], dim=0)
                targets = torch.argmax(targets[0], dim=0)
            else:
                inputs = inputs[0]
                targets = targets[0]
            outputs = torch.argmax(outputs[0], dim=0)
            if title is None:
                title = os.path.join(
                    self.config.paths.out_dir,
                    f"Epoch_{epoch}_Source_{dataset_src}_Target_{dataset_tgt}.png",
                )
            else:
                title = os.path.join(self.config.paths.out_dir, title)

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
                    np.array(cmap_dict[dataset_src]) / 255,
                    N=self.n_classes[dataset_src] + 1,
                )
                cmap_tgt = LinearSegmentedColormap.from_list(
                    dataset_tgt,
                    np.array(cmap_dict[dataset_tgt]) / 255,
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
            # f.colorbar(m4,ax=ax[1][1])
            acc = np.round((outputs == targets).sum().item() / targets.numel(), 3)
            f.suptitle(
                f"x={coordinate[0][0].item()}, y={coordinate[1][0].item()}, accuracy={acc}"
            )
        f.savefig(title)
        plt.close(f)

        return imageio.imread(title)

    def finalize(self):
        pass


class LandcoverToLandcoverDataLoaderNewPosenc(LandcoverToLandcoverDataLoader):

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
        """
        :param config:
        """
        raise DeprecationWarning(
            f"{__name__}.{self.__class__.__name__}: This class is deprecated"
        )
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
        self.patches_crs = []
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
                        transforms.FlipTransform()
                    )
                    dic_list_train_transform[source][target].append(
                        transforms.RotationTransform([0, 90, 180, 270])
                    )
                if to_one_hot:
                    dic_list_transform[source][target].append(
                        transforms.ToOneHot(self.n_classes)
                    )
                if pos_enc:
                    # dic_list_transform[source][target].append(
                    # transforms.ToLonLat(source_crs = "EPSG:2154")
                    # )
                    dic_list_transform[source][target].append(
                        transforms.GeolocEncoder()
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


# EOF

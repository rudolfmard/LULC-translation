#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Land cover maps


Class diagram
-------------
torchgeo.datasets.RasterDataset  (-> https://torchgeo.readthedocs.io/en/v0.4.1/api/datasets.html#rasterdataset)
 ├── TorchgeoLandcover
 |   ├── ESAWorldCover
 |   ├── EcoclimapSG
 |   |   ├── SpecialistLabelsECOSGplus
 |   |   ├── InferenceResults
 |   |   └── EcoclimapSGML
 |   └── CompositeMap
 |       ├── EcoclimapSGplus
 |       └── EcoclimapSGMLcomposite
 |   
 ├── ScoreMap
 |   └── ScoreECOSGplus
 |   
 └── ProbaLandcover
     └── InferenceResultsProba
    
OpenStreetMap
"""
import os
import sys
import time
from typing import Any, Dict, Optional

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import rasterio
import torch
import torchgeo.datasets as tgd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from mmt import _repopath_ as mmt_repopath
from mmt.datasets import transforms as mmt_transforms
from mmt.utils import domains, misc

# VARIABLES
# ============

ECOCLIMAPSG_LABELS = """0. no data
1. sea and oceans
2. lakes
3. rivers
4. bare land
5. bare rock
6. permanent snow
7. boreal broadleaf deciduous
8. temperate broadleaf deciduous
9. tropical broadleaf deciduous
10. temperate broadleaf evergreen
11. tropical broadleaf evergreen
12. boreal needleleaf evergreen
13. temperate needleleaf evergreen
14. boreal needleleaf deciduous
15. shrubs
16. boreal grassland
17. temperate grassland
18. tropical grassland
19. winter C3 crops
20. summer C3 crops
21. C4 crops
22. flooded trees
23. flooded grassland
24. LCZ1: compact high-rise
25. LCZ2: compact midrise
26. LCZ3: compact low-rise
27. LCZ4: open high-rise
28. LCZ5: open midrise
29: LCZ6: open low-rise
30: LCZ7: lightweight low-rise
31: LCZ8: large low-rise
32: LCZ9: sparsely built
33: LCZ10: heavy industry""".split(
    "\n"
)

ECOCLIMAPSG_LABELS_IRISH = """0. gan sonraí
1. farraige agus aigéin
2. lochanna
3. aibhneacha
4. talamh lom
5. carraig lom
6. sneachta buan
7. duillsilteach leathanduilleach boreal
8. duillsilteach leathanduilleach measartha
9. duillsilteach leathanduilleach trópaiceach
10. síorghlas leathanduilleach measartha
11. síorghlas leathanduilleach trópaiceach
12. síorghlas snáth-dhuilleog boreal
13. síorghlas snáth-dhuilleog mheasartha
14. duillsilteach snáth-dhuilleog boreal
15. tom
16. féarach boreal
17. féarach measartha
18. féarach trópaiceach
19. barraí C3 geimhridh
20. barra C3 samhraidh
21. C4 barra
22. crainn tuilte
23. féarach faoi thuilte
24. LCZ1: ard-ardú dlúth
25. LCZ2: meán-ardú dlúth
26. LCZ3: íseal-ardú dlúth
27. LCZ4: ard-ardú oscailte
28. LCZ5: meán-ardú oscailte
29: LCZ6: íseal-ardú oscailte
30: LCZ7: meáchan éadrom íseal-ardú
31: LCZ8: íseal-ardú mór
32: LCZ9: tógtha go tearc
33: LCZ10: tionscal trom""".split(
    "\n"
)

ECOCLIMAPSG_CMAP = [
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
]

N_ECOCLIMAPSG_LABELS = len(ECOCLIMAPSG_LABELS)

ECOCLIMAPSG_LABEL_HIERARCHY = {
    "water": [
        "1. sea and oceans",
        "2. lakes",
        "3. rivers",
    ],
    "bareland": [
        "4. bare land",
        "5. bare rock",
    ],
    "snow": [
        "6. permanent snow",
    ],
    "trees": [
        "7. boreal broadleaf deciduous",
        "8. temperate broadleaf deciduous",
        "9. tropical broadleaf deciduous",
        "10. temperate broadleaf evergreen",
        "11. tropical broadleaf evergreen",
        "12. boreal needleleaf evergreen",
        "13. temperate needleleaf evergreen",
        "14. boreal needleleaf deciduous",
    ],
    "shrubs": [
        "15. shrubs",
    ],
    "grassland": [
        "16. boreal grassland",
        "17. temperate grassland",
        "18. tropical grassland",
    ],
    "crops": [
        "19. winter C3 crops",
        "20. summer C3 crops",
        "21. C4 crops",
    ],
    "flooded_veg": [
        "22. flooded trees",
        "23. flooded grassland",
    ],
    "urban": [
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
}


# BASE CLASSES
# ============

class TorchgeoLandcover(tgd.RasterDataset):
    """Abstract class for land cover dataset using TorchGeo.

    This class is a [customised TorchGeo RasterDataset](https://torchgeo.readthedocs.io/en/latest/tutorials/custom_raster_dataset.html).
    The customisation is in the precision of the path where to find the data and several
    attributes that are common to all land covers classes.


    Parameters
    ----------
    transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]]
        A function/transform that takes an input sample and returns a transformed version

    crs: `rasterio.crs.CRS`
        Coordinate reference system in which the land cover will be transformed

    res: float
        Resolution in units of CRS

    tgeo_init: bool
        If False, the Rtree index will not be created. Faster but no access to data (useful e.g. for using plots only)


    Main attributes
    ---------------
    path: str
        Absolute path to the root directory containing the data

    filename_glob: str
        Glob expression used to search for files

    is_image: bool=False
        True if dataset contains imagery, False if dataset contains mask

    separate_files: bool=False
        True if data is stored in a separate file for each band, else False.

    crs: `rasterio.crs.CRS`
        Coordinate reference system in which the land cover is

    labels: list of str
        Labels names

    cmap: list of 3-tuple
        Colormap for each


    Main methods
    ------------
    __getitem__: Return a sample of data
    plot: Plot a sample of data
    """

    path = ""
    filename_glob = "*.tif"
    is_image = False
    element_size = 8  # Bytes per pixel
    separate_files = False
    # crs = None  # Native coordinate reference system
    orig_crs = None  # Native coordinate reference system
    labels = []
    cmap = []

    def __init__(self, crs=None, res=None, transforms=None, tgeo_init=True):
        self.n_labels = len(self.labels)
        if tgeo_init:
            super().__init__(self.path, crs=crs, res=res, transforms=transforms)

        if crs is not None:
            self.crs = crs
        else:
            self.crs = self.orig_crs

        if res is not None:
            self.res = res

    def get_bytes_for_domain(self, qb):
        """Return the size in bytes that would be necessary to load the query domain (does not load anything)"""
        if isinstance(qb, tgd.BoundingBox):
            qdomain = domains.GeoRectangle(qb, fmt="tgbox")

        return misc.get_bytes_for_domain(qdomain, self.res, self.element_size)

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        show_colorbar: bool = True,
        title: Optional[str] = None,
        suptitle: Optional[str] = None,
        figax=None,
    ):
        """Plot a sample of data.


        Parameters
        ----------
        sample: dict
            Sample of data. Labels must be accessible under the 'mask' key.

        show_titles: bool, default=True
            True if a title is diplayed over the figure.

        show_colorbar: bool, default=True
            True if a title is diplayed over the figure.

        title: str
            If provided, the string given here is put as local title of the figure.

        suptitle: str
            If provided, the string given here is put as main title of the figure.


        Returns
        -------
        fig, ax
            Figure and axes of the plot

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> from mmt.datasets import landcovers
        >>> from mmt.utils import domains
        >>> lc = landcovers.EcoclimapSG()
        >>> qb = domains.dublin_city.to_tgbox(lc.crs)
        >>> x = lc[qb]
        >>> lc.plot(x)
        >>> plt.show()
        """
        assert len(self.labels) == len(
            self.cmap
        ), f"The number of labels ({len(self.labels)}) do not match the number of colors ({len(self.cmap)})"

        cmap = ListedColormap(np.array(self.cmap) / 255)
        if figax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        else:
            fig, ax = figax

        label = sample["mask"].squeeze()
        ul = np.unique(label).astype(int)
        nx, ny = label.shape
        im = ax.imshow(label, cmap=cmap, vmin=0, vmax=len(self.labels) - 1)
        if show_titles:
            if title is None:
                title = self.__class__.__name__

            ax.set_title(title)

        if show_colorbar:
            if len(ul) > 2:
                cbar = plt.colorbar(
                    im, values=ul, spacing="uniform", shrink=0.5, ticks=ul
                )
                cbar.ax.set_yticklabels([self.labels[i] for i in ul])
            else:
                for i, l in enumerate(ul):
                    ax.text(int(nx / 3), int(ny / 3) + i * int(ny / 4), self.labels[l])

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig, ax

    def _export_to_dirhdr(self, sample, ofn_dir=None):
        """Export a sample to the SURFEX-readable format DIR/HDR"""
        if ofn_dir is None:
            ofn_dir = os.path.join(
                self.path,
                f"COVER_{self.__class__.__name__}_2024.dir",
            )

        # HDR file
        ofn_hdr = ofn_dir.replace(".dir", ".hdr")
        hdr_dict = {
            "nodata": 0,
            "north": sample["bbox"].maxy,
            "south": sample["bbox"].miny,
            "west": sample["bbox"].minx,
            "east": sample["bbox"].maxx,
            "rows": sample["mask"].shape[-2],
            "cols": sample["mask"].shape[-1],
            "recordtype": "integer 8 bytes",
        }
        with open(ofn_hdr, "w") as hdr:
            hdr.write(os.path.basename(ofn_dir)[:-4] + "\n")
            for k, v in hdr_dict.items():
                hdr.write(f"{k}: {v}\n")

        # DIR file
        with open(ofn_dir, "wb") as f:
            f.write(sample["mask"].squeeze().numpy().astype(np.uint8).tobytes("C"))

        return ofn_dir, ofn_hdr

    def _export_to_netcdf(self, sample, ofn_nc=None):
        """Export a sample to netCDF"""
        if ofn_nc is None:
            ofn_nc = os.path.join(
                self.path,
                f"COVER_{self.__class__.__name__}_2024.nc",
            )

        data = sample["mask"].squeeze().numpy()
        qb = sample["bbox"]

        nx, ny = data.shape
        ncf = nc.Dataset(ofn_nc, "w")
        ncf.createDimension("x", nx)
        ncf.createDimension("y", ny)

        lc = ncf.createVariable("landcover", np.uint8, ("x", "y"))
        lc[:, :] = data[:, :]
        lc.units = "ECOCLIMAP-SG land cover labels"

        ncf.setncatts(
            dict(
                title=f"{self.__class__.__name__} land cover.",
                source="TIF files",
                crs=self.crs.to_string(),
                resolution=self.res,
                bounds=f"lower-left corner = ({qb.minx}, {qb.miny}); upper-right corner = ({qb.maxx}, {qb.maxy})",
                labels="\n".join(self.labels),
                institution="Met Eireann, met.ie",
                history=f"Created the {time.ctime()}",
                contactperson="Thomas Rieutord (thomas.rieutord@met.ie)",
            )
        )
        ncf.close()

        return ofn_nc

    def _export_to_tif(self, sample, ofn_tif=None):
        """Export a sample to GeoTIFF"""
        if ofn_tif is None:
            ofn_tif = os.path.join(
                self.path,
                f"COVER_{self.__class__.__name__}_2024.tif",
            )
        data = sample["mask"].squeeze().numpy()
        width, height = data.shape
        transf = rasterio.transform.from_bounds(
            sample["bbox"].minx,
            sample["bbox"].miny,
            sample["bbox"].maxx,
            sample["bbox"].maxy,
            width,
            height,
        )
        kwargs = {
            "driver": "gTiff",
            "dtype": np.uint8,
            "count": 1,
            "crs": self.crs,
            "transform": transf,
            "width": width,
            "height": height,
        }

        with rasterio.open(ofn_tif, "w", **kwargs) as f:
            f.write(sample["mask"].squeeze().numpy(), 1)

        return ofn_tif

    def _export_to_npy(self, sample, ofn_npy=None):
        """Export a sample to numpy array"""
        if ofn_npy is None:
            ofn_npy = os.path.join(
                self.path,
                f"COVER_{self.__class__.__name__}_2024.npy",
            )

        np.save(ofn_npy, sample["mask"].squeeze().numpy().astype(np.uint8))

        return ofn_npy

    def export(self, sample, ofn):
        """Export a sample.

        The file format is determined by the extension of ofn. Currently, the
        supported formats are 'dir', 'tif', 'nc', and 'npy'.


        Parameters
        ----------
        sample : dict
            Sample to export. The sample must contain 'mask' attribute, and 'mask'
            must be a 2D numpy array.

        ofn : str
            Output filename.


        Returns
        -------
        ofn : str
            Path to the output file.


        Example
        -------
        >>> from mmt.datasets import landcovers
        >>> from mmt.utils import domains
        >>> lc = landcovers.EcoclimapSG()
        >>> qb = domains.dublin_city.to_tgbox(lc.crs)
        >>> x = lc[qb]
        >>> lc.export(x, "dublin_city_ecoclimapsg_2024.dir")
        ('dublin_city_ecoclimapsg_2024.dir', 'dublin_city_ecoclimapsg_2024.hdr')
        >>> lc.export(x, "dublin_city_ecoclimapsg_2024.nc")
        'dublin_city_ecoclimapsg_2024.nc'
        >>> lc.export(x, "dublin_city_ecoclimapsg_2024.tif")
        'dublin_city_ecoclimapsg_2024.tif'
        >>> lc.export(x, "dublin_city_ecoclimapsg_2024.npy")
        'dublin_city_ecoclimapsg_2024.npy'
        """
        if ofn.endswith(".dir"):
            return self._export_to_dirhdr(sample, ofn)
        elif ofn.endswith(".nc"):
            return self._export_to_netcdf(sample, ofn)
        elif ofn.endswith(".tif"):
            return self._export_to_tif(sample, ofn)
        elif ofn.endswith(".npy"):
            return self._export_to_npy(sample, ofn)
        else:
            raise Exception("Unknown file format")


class ScoreMap(tgd.RasterDataset):
    """Abstract class for score dataset using TorchGeo.

    Similar to TorchgeoLandcover, but for real-valued data instead of integer.
    Consequently, they share most of attributes and methods.



    Parameters
    ----------
    cutoff: float
        Threshold for the score colormap (red below, green above)

    transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]]
        A function/transform that takes an input sample and returns a transformed version

    crs: `rasterio.crs.CRS`
        Coordinate reference system in which the land cover will be transformed

    res: float
        Resolution in units of CRS

    tgeo_init: bool
        If False, the Rtree index will not be created. Faster but no access to data (useful e.g. for using plots only)
    """

    path = ""
    filename_glob = "*.tif"
    is_image = True
    element_size = 32  # Bytes per pixel
    separate_files = False
    orig_crs = None

    def __init__(
        self, cutoff=0.525, crs=None, res=None, transforms=None, tgeo_init=True
    ):
        self.cutoff = cutoff
        cmap = LinearSegmentedColormap.from_list(
            "mycmap", [(0.0, "red"), (cutoff, "gainsboro"), (1, "green")]
        )
        self.cmap = [tuple(c[:3]) for c in cmap(np.linspace(0, 1, 100))]
        if tgeo_init:
            super().__init__(self.path, crs=crs, res=res, transforms=transforms)

        if crs is not None:
            self.crs = crs
        else:
            self.crs = self.orig_crs

        if res is not None:
            self.res = res

    def get_bytes_for_domain(self, qb):
        """Return the size in bytes that would be necessary to load the query domain (does not load anything)"""
        if isinstance(qb, tgd.BoundingBox):
            qdomain = domains.GeoRectangle(qb, fmt="tgbox")
        else:
            qdomain = qb

        return misc.get_bytes_for_domain(qdomain, self.res, self.element_size)

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        show_colorbar: bool = True,
        title: Optional[str] = None,
        suptitle: Optional[str] = None,
        figax=None,
    ):
        """Plot a sample of data.


        Parameters
        ----------
        sample: dict
            Sample of data. Labels must be accessible under the 'image' key.

        show_titles: bool, default=True
            True if a title is diplayed over the figure.

        show_colorbar: bool, default=True
            True if a title is diplayed over the figure.

        title: str
            If provided, the string given here is put as local title of the figure.

        suptitle: str
            If provided, the string given here is put as main title of the figure.


        Returns
        -------
        fig, ax
            Figure and axes of the plot

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> from mmt.datasets import landcovers
        >>> from mmt.datasets import transforms as mmt_transforms
        >>> from mmt.utils import domains
        >>> sc = landcovers.ScoreECOSGplus(transforms=mmt_transforms.ScoreTransform(100))
        >>> qb = domains.dublin_city.to_tgbox(sc.crs)
        >>> x = sc[qb]
        >>> sc.plot(x)
        >>> plt.show()
        """
        if figax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        else:
            fig, ax = figax

        x = sample["image"].squeeze()

        im = ax.imshow(x, cmap=ListedColormap(self.cmap))

        if show_titles:
            if title is None:
                title = self.__class__.__name__

            ax.set_title(title)

        if show_colorbar:
            plt.colorbar(im, shrink=0.5)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig, ax


class ProbaLandcover(tgd.RasterDataset):
    """Abstract class for land cover probability dataset using TorchGeo.

    Similar to TorchgeoLandcover, but probabilities data instead of cover classes.
    Consequently, they share most of attributes and methods.


    Parameters
    ----------
    Same as TorchgeoLandcover.


    Notes
    -----
    Not used in the paper, but useful for exploring the data.
    """

    path = ""
    filename_glob = "*.tif"
    is_image = True
    element_size = 32  # Bytes per pixel
    separate_files = False
    orig_crs = None
    labels = []
    cmap = []

    def __init__(self, crs=None, res=None, transforms=None, tgeo_init=True):
        self.n_labels = len(self.labels)
        if tgeo_init:
            super().__init__(self.path, crs=crs, res=res, transforms=transforms)

        if crs is not None:
            self.crs = crs
        else:
            self.crs = self.orig_crs

        if res is not None:
            self.res = res

    def get_bytes_for_domain(self, qb):
        """Return the size in bytes that would be necessary to load the query domain (does not load anything)"""
        if isinstance(qb, tgd.BoundingBox):
            qdomain = domains.GeoRectangle(qb, fmt="tgbox")
        else:
            qdomain = qb

        return misc.get_bytes_for_domain(qdomain, self.res, self.element_size)

    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        title: Optional[str] = None,
        suptitle: Optional[str] = None,
        figax=None,
    ):
        """Plot a sample of data: colors are the sum of labels original colors multiplied by their probability.


        Parameters
        ----------
        sample: dict
            Sample of data. Labels must be accessible under the 'mask' key.

        show_titles: bool, default=True
            True if a title is diplayed over the figure.

        show_colorbar: bool, default=True
            True if a title is diplayed over the figure.

        title: str
            If provided, the string given here is put as local title of the figure.

        suptitle: str
            If provided, the string given here is put as main title of the figure.


        Returns
        -------
        fig, ax
            Figure and axes of the plot

        Example
        -------
        >>> import matplotlib.pyplot as plt
        >>> from mmt.datasets import landcovers
        >>> from mmt.utils import domains
        >>> lc = landcovers.EcoclimapSG()
        >>> qb = domains.dublin_city.to_tgbox(lc.crs)
        >>> x = lc[qb]
        >>> lc.plot(x)
        >>> plt.show()
        """
        assert len(self.labels) == len(
            self.cmap
        ), f"The number of labels ({len(self.labels)}) do not match the number of colors ({len(self.cmap)})"

        cmap = np.array(self.cmap) / 255
        if figax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        else:
            fig, ax = figax

        proba = sample["image"].squeeze().permute(1, 2, 0)

        x = np.matmul(proba[:, :, :-1], cmap)
        im = ax.imshow(x)
        if show_titles:
            if title is None:
                title = self.__class__.__name__

            ax.set_title(title)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig, ax

    def getitem_uq(self, qb):
        """Return the uncertainty quantification values.


        The uncertainty is quantified at each pixel by the ratio between
        the second highest and the highest probability values.

            p = sample["image"][:, x, y].sort(0)
            uq = p[-2]/p[-1]

        For example, with p = [0, ..., 0.05, 0.92], the highest probability label
        is really dominant, therefore we are confident on this label.
        UQ = 0.05/0.92 = 0.0543 is low, which reflects this confidence.

        With p = [0, ..., 0.31, 0.37], the highest probability label
        is almost as likely as the second one, therefore we are not confident on this label.
        UQ = 0.31/0.37 = 0.837 is high, which reflects this uncertainty.
        """
        sample = self[qb]
        proba = sample["image"].squeeze().softmax(0)
        psort, _ = proba.sort(0)

        return psort[-2] / psort[-1]

    def plot_uncertainty(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        show_colorbar: bool = True,
        title: Optional[str] = None,
        suptitle: Optional[str] = None,
        logscale: bool = False,
        figax=None,
    ):
        """Plot an estimate of the uncertainty on the land cover label from the land cover probabilities.
        See `getitem_uq` for definition and properties of the UQ score.


        Parameters
        ----------
        sample: dict
            Sample of data. Probabilities must be accessible under the 'image' key.

        show_titles: bool, default=True
            True if a title is diplayed over the figure.

        suptitle: str
            If provided, the string given here is put as main title of the figure.
        """

        if figax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        else:
            fig, ax = figax

        proba = sample["image"].squeeze().softmax(0)

        psort, _ = proba.sort(0)
        uq = psort[-2] / psort[-1]

        if logscale:
            im = ax.imshow(torch.log(uq), cmap="brg")
        else:
            im = ax.imshow(uq, cmap="brg", vmin=0, vmax=1)

        if show_titles:
            if title is None:
                title = "Uncertainty " + self.__class__.__name__

            ax.set_title(title)

        if show_colorbar:
            cbar = plt.colorbar(im)

        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig, ax

    def generate_member(self, sample, u=None, print_u=False):
        """Generate a land cover member for the given probability sample"""
        if u is None:
            u = torch.rand(1)
            if print_u:
                print("u=", u)

        proba = sample["image"].squeeze()
        cdf = proba.cumsum(0) / proba.sum(0)
        labels = (cdf < u).sum(0)

        return labels


class OpenStreetMap:
    """OpenStreetMap land cover from Cartopy (for plot only).

    The class is init with a level of details and a patch size. It is then
    used to produce Cartopy plots of OSM land cover at given coordinates +/- patch size.
    Coordinates and patch size are expected to be in lon/lon format (EPSG:4326).


    Parameters
    ----------
    details: int, default=3
        Level of details in the map. The higher, the more detailled but the heavier to load

    default_patch_size: float, default=0.05
        Patch size in lon/lat degrees. Overwritten by plot argument, if provided.


    Notes
    -----
    Not used in the paper.
    """

    def __init__(self, details=3, default_patch_size=0.05):
        self.details = details
        self.default_patch_size = default_patch_size
        self.background_image = cimgt.OSM()

    def plot(
        self,
        sample: Dict[str, Any],
        patch_size: Optional[float] = None,
        show_titles: bool = True,
        figax=None,
        rowcolidx=111,
    ):
        """Plot the OpenStreetMap land cover


        Parameters
        ----------
        sample: dict
            Sample with a 'coordinate' or 'bbox' key that will be used to
            specify the location.
            The 'bbox' must have [minx, ..., maxy] attributes
            The 'coordinate' is assumed to point to the upper-left corner and will be completed by `patch_size`
            All location information are expected to be in lon/lat degrees

        patch_size: int, optional
            Patch size to use when the sample only has a 'coordinate' key.
            If not provided, the default patch size set in init is used.
        """

        if figax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        else:
            fig, ax = figax
            ax.set_axis_off()

        if patch_size is None:
            patch_size = self.default_patch_size

        if "bbox" in sample.keys():
            minx = sample["bbox"].minx
            miny = sample["bbox"].miny
            maxx = sample["bbox"].maxx
            maxy = sample["bbox"].maxy
        elif "coordinate" in sample.keys():
            # Assume they correspond to the upper-left corner (image convention)
            minx, maxy = sample["coordinate"]
            maxx = minx + patch_size
            miny = maxy - patch_size
        else:
            raise ValueError("Sample does not have geographical info")

        locextent = [minx, maxx, miny, maxy]
        xticks = np.linspace(locextent[0], locextent[1], 5)
        yticks = np.linspace(locextent[2], locextent[3], 5)

        ax0 = fig.add_subplot(rowcolidx, projection=self.background_image.crs)
        ax0.set_extent(locextent)
        ax0.add_image(self.background_image, self.details)
        ax0.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax0.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax0.set_xticklabels(np.round(xticks, 3))
        ax0.set_yticklabels(np.round(yticks, 3))
        if show_titles:
            ax0.set_title(self.__class__.__name__)

        return fig, ax0


class CompositeMap(TorchgeoLandcover):
    """Composite map built with a bottom map, a top map, and an overwritting criterion.
    The composite map returns the top map if the criterion is met and the bottom if it is not.


    Parameters
    ----------
    topmap: mmt.datasets.landcovers
        Map to use when the criterion is met

    bottommap: mmt.datasets.landcovers
        Map to use when the criterion is not met

    auxmap: mmt.datasets.landcovers, optional
        Auxilary map used in the criterion


    Example
    -------
    >>> from mmt.datasets import transforms as mmt_transforms
    >>> from mmt.datasets import landcovers
    >>> from mmt.utils import domains
    >>> qb = domains.dublin_city.to_tgbox()

    # ECOSG+ as a composite map
    >>> ecosg = landcovers.EcoclimapSG()
    >>> mstar = landcovers.SpecialistLabelsECOSGplus()
    >>> score = landcovers.ScoreECOSGplus(transforms = mmt_transforms.ScoreTransform(divide_by=100))
    >>> compo = landcovers.CompositeMap(topmap = mstar, bottommap = ecosg, auxmap = score)
    >>> x = compo[qb]

    # ECOSG-ML as a composite map
    >>> esgp = landcovers.EcoclimapSGplus()
    >>> infres = landcovers.InferenceResults("path/to/inference_results")
    >>> score = landcovers.ScoreECOSGplus(transforms = mmt_transforms.ScoreTransform(divide_by=100))
    >>> compo = landcovers.CompositeMap(topmap = infres, bottommap = esgp, auxmap = score)
    >>> x = compo[qb]
    """

    labels = ECOCLIMAPSG_LABELS
    cmap = ECOCLIMAPSG_CMAP

    def __init__(self, topmap, bottommap, auxmap=None):
        self.topmap = topmap
        self.bottommap = bottommap
        self.auxmap = auxmap

        if self.auxmap is None:
            self.maps = topmap & bottommap
        else:
            self.maps = topmap & bottommap & auxmap

        self._crs = self.maps.crs
        self._res = self.maps.res
        self.index = self.maps.index
        self.n_labels = len(self.labels)

    def __getitem__(self, qb):
        x = self.maps[qb]
        top = x["mask"][0]
        bottom = x["mask"][1]

        if self.auxmap is None:
            aux = None
        elif self.auxmap.is_image:
            aux = x["image"]
        else:
            aux = x["mask"][2]

        return {
            "mask": torch.where(self.criterion(top, aux), top, bottom),
            "bbox": qb,
            "crs": self.crs,
        }

    def criterion(self, top, aux):
        return top != 0


# CHILD CLASSES
# =============


class EcoclimapSG(TorchgeoLandcover):
    path = os.path.join(mmt_repopath, "data", "tiff_data", "ECOCLIMAP-SG")
    labels = ECOCLIMAPSG_LABELS
    cmap = ECOCLIMAPSG_CMAP
    orig_crs = rasterio.crs.CRS.from_epsg(4326)


class ESAWorldCover(TorchgeoLandcover):
    path = os.path.join(mmt_repopath, "data", "tiff_data", "ESA-WorldCover-2021")
    labels = [
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
    ]
    cmap = [
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
    ]
    orig_crs = rasterio.crs.CRS.from_epsg(4326)


class InferenceResults(EcoclimapSG):
    """ECOSG-like land cover maps (same labels) loaded from a given path"""

    def __init__(self, path, crs=None, res=None, transforms=None, tgeo_init=True):
        self.path = path
        self.n_labels = len(self.labels)
        super().__init__(crs=crs, res=res, transforms=transforms, tgeo_init=tgeo_init)


class InferenceResultsProba(ProbaLandcover):
    """ECOSG-like land cover probability maps (same labels) loaded from a given path"""

    labels = ECOCLIMAPSG_LABELS
    cmap = ECOCLIMAPSG_CMAP
    orig_crs = rasterio.crs.CRS.from_epsg(4326)

    def __init__(self, path, crs=None, res=None, transforms=None, tgeo_init=True):
        self.path = path
        self.n_labels = len(self.labels)
        super().__init__(crs=crs, res=res, transforms=transforms, tgeo_init=tgeo_init)


class SpecialistLabelsECOSGplus(EcoclimapSG):
    """Best-guess map from specialist maps (ECOSG+ before gap-filling with ECOSG)

    See Bessardon et al. (2024), Equation 9
    """

    path = os.path.join(
        mmt_repopath,
        "data",
        "tiff_data",
        "ECOCLIMAP-SG-plus",
        "v2",
        "labels-isl-ecosgp-v2.0",
    )


class ScoreECOSGplus(ScoreMap):
    """Score quantifying the uncertainty on ECOCLIMAP-SG+ labels"""

    path = os.path.join(
        mmt_repopath,
        "data",
        "tiff_data",
        "ECOCLIMAP-SG-plus",
        "v2",
        "score-ecosgp-v2.0",
    )
    orig_crs = rasterio.crs.CRS.from_epsg(4326)


class EcoclimapSGplus(CompositeMap):
    """ECOSG+ as a CompositeMap"""

    def __init__(self, score_min=0.525, crs=None, res=None, tgeo_init=True):
        self.score_min = score_min

        topmap = SpecialistLabelsECOSGplus(crs=crs, res=res, tgeo_init=tgeo_init)
        bottommap = EcoclimapSG(crs=crs, res=res, tgeo_init=tgeo_init)
        auxmap = ScoreECOSGplus(
            transforms=mmt_transforms.ScoreTransform(divide_by=100),
            crs=crs,
            res=res,
            tgeo_init=tgeo_init,
        )
        super().__init__(topmap, bottommap, auxmap)

    def criterion(self, top, aux):
        return torch.logical_and(top != 0, aux > self.score_min)


class EcoclimapSGMLcomposite(CompositeMap):
    """ECOSG-ML as a CompositeMap from InferenceResults

    Allows to modify the score threshold, but takes only the member provided at the inference path.


    Parameters
    ----------
    path_to_infres: str
        Path to the InferenceResults tiff files

    score_lim: float
        Score threshold

    crs: `rasterio.crs.CRS`
        Coordinate reference system in which the land cover will be transformed

    res: float
        Resolution in units of CRS

    tgeo_init: bool
        If False, the Rtree index will not be created. Faster but no access to data (useful e.g. for using plots only)
    """

    def __init__(
        self, path_to_infres, score_lim=0.3, crs=None, res=None, tgeo_init=True
    ):

        assert os.path.isdir(path_to_infres), f"No directory found at {path_to_infres}"

        self.score_lim = score_lim

        topmap = InferenceResults(
            path=path_to_infres, crs=crs, res=res, tgeo_init=tgeo_init
        )
        auxmap = ScoreECOSGplus(
            cutoff=self.score_lim,
            transforms=mmt_transforms.ScoreTransform(divide_by=100),
            crs=crs,
            res=res,
            tgeo_init=tgeo_init,
        )
        bottommap = EcoclimapSGplus(
            score_min=self.score_lim, crs=crs, res=res, tgeo_init=tgeo_init
        )

        super().__init__(topmap, bottommap, auxmap)

    def criterion(self, top, aux):
        return torch.logical_and(top != 0, aux < self.score_lim)


class EcoclimapSGML(EcoclimapSG):
    """ECOSG-ML with all members

    Score threshold fixed at the value which created the TIF files, but easy to switch the members


    Parameters
    ----------
    member: float or int
        Member (integer) or `u` value (float) to use

    crs: `rasterio.crs.CRS`
        Coordinate reference system in which the land cover will be transformed

    res: float
        Resolution in units of CRS

    tgeo_init: bool
        If False, the Rtree index will not be created. Faster but no access to data (useful e.g. for using plots only)
    """

    path = os.path.join(mmt_repopath, "data", "tiff_data", "ECOCLIMAP-SG-ML")
    n_members = 6

    def __init__(self, member=0, crs=None, res=None, transforms=None, tgeo_init=True):
        if member in [0, None]:
            self.u = None
            self._member = 0
        elif member in [1, 0.82]:
            self.u = 0.82
            self._member = 1
        elif member in [2, 0.11]:
            self.u = 0.11
            self._member = 2
        elif member in [3, 0.47]:
            self.u = 0.47
            self._member = 3
        elif member in [4, 0.34]:
            self.u = 0.34
            self._member = 4
        elif member in [5, 0.65]:
            self.u = 0.65
            self._member = 5
        else:
            raise ValueError(f"Unknown member specification {member}")

        self.path = os.path.join(
            self.path, "ecosgml-v2.0-mb" + str(self.member).zfill(3)
        )
        super().__init__(crs=crs, res=res, transforms=transforms, tgeo_init=tgeo_init)

    @property
    def member(self):
        return self._member

    @member.setter
    def member(self, newmember):
        self.path = os.path.dirname(self.path)
        self.__init__(
            member=newmember, crs=self.crs, res=self.res, transforms=self.transforms
        )

    def plot_all_members(
        self,
        qb: tgd.BoundingBox,
        suptitle: Optional[str] = None,
    ):
        """Plot all members on a given domain. Conversely to other landcovers.plot methods, this one takes a BoundingBox as input.


        Parameters
        ----------
        qb: BoundingBox
            Domain to be plotted

        suptitle: str
            If provided, the string given here is put as main title of the figure.


        Returns
        -------
        fig, ax
            Figure and axes of the plot

        Example
        -------
        >>> from mmt.datasets import landcovers
        >>> from mmt.utils import domains
        >>> lc = landcovers.EcoclimapSGML()
        >>> qb = domains.dublin_city.to_tgbox(lc.crs)
        >>> fig, axs = lc.plot_all_members(qb)
        >>> fig.show()
        """
        fig, axs = plt.subplots(2, 3, figsize=(12, 12))
        for mb, ax in enumerate(axs.flatten()):
            self.member = mb
            # self.path = os.path.dirname(self.path)
            # self.__init__(member=mb, crs=self.crs, res=self.res, transforms=self.transforms)
            x = self[qb]
            fig, ax = self.plot(
                x,
                title=f"Member {mb} (u={self.u})",
                show_colorbar=False,
                figax=(fig, ax),
            )

        if suptitle is not None:
            plt.suptitle(suptitle)

        fig.tight_layout()
        return fig, axs


# EOF

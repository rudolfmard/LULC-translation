#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Land cover maps
"""
import os
import numpy as np
import rasterio
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import torchgeo.datasets as tgd
from typing import Any, Dict, Optional

from mmt import _repopath_ as mmt_repopath
from mmt.utils import config as utilconf

config = utilconf.get_config_from_yaml(
    os.path.join(mmt_repopath, "configs", "new_config_template.yaml")
)

ecoclimapsg_labels = """0. no data
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
33: LCZ10: heavy industry""".split("\n")

n_ecoclimapsg_labels = len(ecoclimapsg_labels)


ecoclimapsg_label_hierarchy = {
    "water":[
        "1. sea and oceans",
        "2. lakes",
        "3. rivers",
    ],
    "bareland":[
        "4. bare land",
        "5. bare rock",
    ],
    "snow":[
        "6. permanent snow",
    ],
    "trees":[
        "7. boreal broadleaf deciduous",
        "8. temperate broadleaf deciduous",
        "9. tropical broadleaf deciduous",
        "10. temperate broadleaf evergreen",
        "11. tropical broadleaf evergreen",
        "12. boreal needleleaf evergreen",
        "13. temperate needleleaf evergreen",
        "14. boreal needleleaf deciduous",
    ],
    "shrubs":[
        "15. shrubs",
    ],
    "grassland":[
        "16. boreal grassland",
        "17. temperate grassland",
        "18. tropical grassland",
    ],
    "crops":[
        "19. winter C3 crops",
        "20. summer C3 crops",
        "21. C4 crops",
    ],
    "flooded_veg":[
        "22. flooded trees",
        "23. flooded grassland",
    ],
    "urban":[
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


def parse_version_infos(path):
    """Read all the fields from the file version-info.txt"""
    version_file = os.path.join(path, "version-infos.txt")
    assert os.path.isfile(version_file), f"Missing version file. Please create file {version_file}'"
    infos = {}
    with open(version_file, "r") as f:
        for l in f.readlines():
            k, v = l.split("=")
            infos[k] = v.split('"')[1]
        
    return infos

class TorchgeoLandcover(tgd.RasterDataset):
    """Abstract class for land cover dataset using TorchGeo.
    
    This class is a [customised TorchGeo RasterDataset](https://torchgeo.readthedocs.io/en/latest/tutorials/custom_raster_dataset.html).
    The customisation is in the precision of the path where to find the data and several
    attributes that are common to all land covers classes.
    
    
    Parameters
    ----------
    transforms: Optional[Callable[[dict[str, Any]], dict[str, Any]]]
        A function/transform that takes an input sample and returns a transformed version
    
    
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
    separate_files = False
    crs = None
    labels = []
    cmap = []
    
    def __init__(self, crs = None, res = None, transforms = None, tgeo_init = True):
        self.n_labels = len(self.labels)
        if tgeo_init:
            super().__init__(self.path, crs = crs, res = res, transforms = transforms)
        
    def plot(
        self,
        sample: Dict[str, Any],
        show_titles: bool = True,
        show_colorbar: bool = True,
        title: Optional[str] = None,
        suptitle: Optional[str] = None,
        figax = None
    ):
        """Plot a sample of data.


        Parameters
        ----------
        sample: dict
            Sample of data. Labels must be accessible under the 'mask' key.

        show_titles: bool, default=True
            True if a title is diplayed over the figure.

        suptitle: str
            If provided, the string given here is put as main title of the figure.
        """
        assert len(self.labels) == len(self.cmap), f"The number of labels ({len(self.labels)}) do not match the number of colors ({len(self.cmap)})"
        
        cmap = ListedColormap(np.array(self.cmap)/255)
        if figax is None:
            fig, ax = plt.subplots(1, 1, figsize=(12,12))
        else:
            fig, ax = figax
        
        label = sample["mask"].squeeze()
        ul = np.unique(label).astype(int)
        nx, ny = label.shape
        im = ax.imshow(label, cmap=cmap, vmin=0, vmax=len(self.labels)-1)
        if show_titles:
            if title is None:
                title = self.__class__.__name__
            
            ax.set_title(title)
        
        if show_colorbar:
            if len(ul) > 2:
                cbar = plt.colorbar(im, values=ul, spacing="uniform", shrink=0.5, ticks=ul)
                cbar.ax.set_yticklabels([self.labels[i] for i in ul])
            else:
                for i, l in enumerate(ul):
                    ax.text(int(nx/3), int(ny/3) + i * int(ny/4), self.labels[l])
                
            
        if suptitle is not None:
            plt.suptitle(suptitle)
        
        # fig.tight_layout()
        return fig, ax


class EcoclimapSG(TorchgeoLandcover):
    path = os.path.join(config.paths.data_dir, "tiff_data", "ECOCLIMAP-SG")
    filename_glob = "ECOCLIMAP-SG-Eurat.tif"
    labels = ecoclimapsg_labels
    cmap = [
        (0,0,0),(0,0,128),(0,0,205),(0, 0, 255),(211,211,211),(169,169,169),(255,250,250),
        (240,255,240),(85,107,47),(154,205,50),(0,128,0),(255,127,80),(160,82,45),(34,139,34),
        (188,143,143),(205,133,63),(222,184,135),(50,205,50),(255,215,0),(32,178,170),(173,255,47),
        (189,183,107),(102,102,0),(46,139,87),(138,2,0),(206,0,0),(252,1,1),(255,90,0),(255,120,0),
        (255,150,0),(255,180,0),(255,210,0),(255,240,0),(128,128,128)
    ]
    crs = rasterio.crs.CRS.from_epsg(4326)


class ESAWorldCover(TorchgeoLandcover):
    path = os.path.join(config.paths.data_dir, "tiff_data", "ESA-WorldCover-2021")#, "ESA_WorldCover_10m_2021_v200_60deg_macrotile_N30E000")
    labels = [
        "No data", "Tree cover", "Shrubland", "Grassland", "Cropland", "Built-up",
        "Bare/sparse veg.", "Snow and ice", "Permanent water bodies",
        "Herbaceous wetland", "Mangroves", "Moss and lichen"
    ]
    cmap = [
        (0,0,0), (0,100,0), (255, 187, 34), (255, 255, 76), (240, 150, 255), (250, 0, 0), (180, 180, 180),
        (240, 240, 240), (0, 100, 200), (0, 150, 160), (0, 207, 117), (250, 230, 160)
    ]
    crs = rasterio.crs.CRS.from_epsg(4326)


class EcoclimapSGplus(TorchgeoLandcover):
    path = os.path.join(config.paths.data_dir, "tiff_data", "ECOCLIMAP-SG-plus", f"ecosgp-labels-v{config.versions.ecosgplus}")
    labels = ecoclimapsg_labels
    cmap = [
        (0,0,0),(0,0,128),(0,0,205),(0, 0, 255),(211,211,211),(169,169,169),(255,250,250),
        (240,255,240),(85,107,47),(154,205,50),(0,128,0),(255,127,80),(160,82,45),(34,139,34),
        (188,143,143),(205,133,63),(222,184,135),(50,205,50),(255,215,0),(32,178,170),(173,255,47),
        (189,183,107),(102,102,0),(46,139,87),(138,2,0),(206,0,0),(252,1,1),(255,90,0),(255,120,0),
        (255,150,0),(255,180,0),(255,210,0),(255,240,0),(128,128,128)
    ]
    crs = rasterio.crs.CRS.from_string(parse_version_infos(path)["crs"])
    
    def get_version(self):
        """Read the version from the file version-info.txt"""
        version_file = os.path.join(self.path, "version-infos.txt")
        assert os.path.isfile(version_file), f"Missing version file. Please create file {version_file} with a line 'version=xxx'"
        with open(version_file, "r") as f:
            for l in f.readlines():
                if "version=" in l:
                    return l.split('"')[1]
                
    def export_to_dirhdr(self, sample, ofn_dir = None):
        """Export a sample to the SURFEX-readable format DIR/HDR
        """
        if ofn_dir is None:
            ofn_dir = os.path.join(self.path, f"COVER_{self.__class__.__name__}_2023_v{self.get_version()}.dir")
            
        # HDR file
        ofn_hdr = ofn_dir.replace(".dir", ".hdr")
        hdr_dict = {
            "nodata":0,
            "north":sample["bbox"].maxy,
            "south":sample["bbox"].miny,
            "west":sample["bbox"].minx,
            "east":sample["bbox"].maxx,
            "rows":sample["mask"].shape[-2],
            "cols":sample["mask"].shape[-1],
            "recordtype": "integer 8 bytes",
        }
        with open(ofn_hdr, "w") as hdr:
            hdr.write(os.path.basename(ofn_dir)[:-4] + "\n")
            for k,v in hdr_dict.items():
                hdr.write(f"{k}: {v}\n")
            
        
        # DIR file
        with open(ofn_dir, "wb") as f:
            f.write(sample["mask"].squeeze().numpy().astype(np.uint8).tobytes("F"))
        
        return ofn_dir, ofn_hdr
    

class QualityFlagsECOSGplus(EcoclimapSGplus):
    path = os.path.join(config.paths.data_dir, "tiff_data", "ECOCLIMAP-SG-plus", f"ecosgp-qflags-v{config.versions.ecosgplus}")
    labels = [
        "0. no data",
        "1. high agreement score + ECOSG",
        "2. high agreement score",
        "3. low agreement score + ECOSG",
        "4. low agreement score", 
        "5. no mode match + ECOSG",
        "6. no mode match"
    ]
    cmap = [(0, 0, 0), (134, 218, 134), (60, 182, 60), (255, 191, 191), (255, 127, 127), (255, 64, 64), (255, 0, 0)]
    

class InferenceResults(EcoclimapSGplus):
    
    def __init__(self, path, crs = None, res = None, transforms = None, tgeo_init = True):
        self.path = path
        self.n_labels = len(self.labels)
        super().__init__(crs = crs, res = res, transforms = transforms, tgeo_init = tgeo_init)

class EcoclimapSGML(EcoclimapSGplus):
    path = os.path.join(config.paths.data_dir, "tiff_data", "ECOCLIMAP-SG-ML", "tif", f"ECOCLIMAP-SG-ML-v{config.versions.ecosgml}")

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
    """
    def __init__(self, details = 3, default_patch_size = 0.05):
        self.details = details
        self.default_patch_size = default_patch_size
        self.background_image = cimgt.OSM()
    
    def plot(
        self,
        sample: Dict[str, Any],
        patch_size: Optional[float] = None,
        show_titles: bool = True,
        figax = None,
        rowcolidx = 111
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
            fig, ax = plt.subplots(1, 1, figsize=(12,12))
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
        ax0.set_xticks(xticks, crs = ccrs.PlateCarree())
        ax0.set_yticks(yticks, crs = ccrs.PlateCarree())
        ax0.set_xticklabels(np.round(xticks,3))
        ax0.set_yticklabels(np.round(yticks,3))
        if show_titles:
            ax0.set_title(self.__class__.__name__)
        
        return fig, ax0
    

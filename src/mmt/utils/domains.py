#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multiple land-cover/land-use Maps Translation (MMT)

Module for geographical domains utilities

Global docstrings reviews:
  * 22 Nov. 2022 (Thomas Rieutord)
  * 05 Jun. 2024 (Thomas Rieutord)
"""

import json

import numpy as np
import pyproj
import rasterio
import shapely.geometry
import utm
from torchgeo.datasets.utils import BoundingBox as TgeoBoundingBox


class GeoRectangle:
    """Class to specify lon-lat rectangles defining geographical areas
    with versatile interface


    Parameters
    ----------
    coords: list or str
        Coordinates of the corner points of the rectangle.

    fmt: str
        Key identifying the format of the coordinates. Possible values are
        - "tlbr": top-left, bottom-right
            coords should be an array-like of shape (2,2) of the type `coords = [[x1,y1], [x3,y3]]`
        - "itlbr": inversed top-left, bottom-right
            coords should be an array-like of shape (2,2) of the type `coords = [[y1,x1], [y3,x3]]`
        - "bltr": bottom-left, top-right
            coords should be an array-like of shape (2,2) of the type `coords = [[x2,x4], [y2,y4]]`
        - "llmm": longitude-latitude minimum-maximum
            coords should be an array-like of shape (4,) of the type `coords = [x2, x4, y2, y4]`
        - "json": coords is a path to a GeoJson file
            the JSON file must contain points coords under ["geometry"]["coordinates"][0] or ["coordinates"][0]
        - "points": from a set of points (take the smaller bounding box containing all points)
            coords is a list of x,y tuples
        - "tgbox": from a torchgeo.datasets.utils.BoundingBox
            coords is a TgeoBoundingBox
        - "epsg": from coordinates in a different CRS (provided in the opt `crs` argument)
            coords is a 4-tuple (left, bottom, right, top)

    crs: `rasterio.crs.CRS` or None
        Coordinate reference system in which the coordinates are given.
        Only used if fmt="epsg".

    Attributes
    ----------
    min_longitude: float
        Minimum longitude (lowest easting)

    max_longitude: float
        Maximum longitude (highest easting)

    min_latitude: float
        Minimum latitude (lowest northing)

    max_latitude: float
        Maximum latitude (hightest northing)


    Notes
    -----
    Convention in use to name the corners of the rectangle

        |       |
      --1-------4--
        |       |
        |       |
      --2-------3--
        |       |

    x1 = x2 = min_longitude = upper_left_longitude = bottom_west_longitude
    x3 = x4 = max_longitude = lower_right_longitude = top_east_longitude
    y1 = y4 = max_latitude = upper_left_latitude = top_east_latitude
    y3 = y2 = min_latitude = lower_right_latitude = bottom_west_latitude
    """

    def __init__(self, coords, fmt="llmm", crs=None):

        if fmt == "tlbr":
            self.min_longitude = coords[0][0]
            self.max_latitude = coords[0][1]
            self.max_longitude = coords[1][0]
            self.min_latitude = coords[1][1]
        if fmt == "itlbr":
            self.max_latitude = coords[0][0]
            self.min_longitude = coords[0][1]
            self.min_latitude = coords[1][0]
            self.max_longitude = coords[1][1]
        elif fmt == "bltr":
            self.min_longitude = coords[0][0]
            self.max_longitude = coords[0][1]
            self.min_latitude = coords[1][0]
            self.max_latitude = coords[1][1]
        elif fmt == "llmm":
            self.min_longitude = coords[0]
            self.max_longitude = coords[1]
            self.min_latitude = coords[2]
            self.max_latitude = coords[3]
        elif fmt == "lbrt":
            self.min_longitude = coords[0]
            self.max_longitude = coords[2]
            self.min_latitude = coords[1]
            self.max_latitude = coords[3]
        elif fmt == "json":
            geojson = json.loads(coords)
            try:
                points = geojson["geometry"]["coordinates"][0]
            except:
                points = geojson["coordinates"][0]
            self._init_from_boundingbox(points)
        elif fmt == "points":
            self._init_from_boundingbox(coords)
        elif fmt == "tgbox":
            assert isinstance(
                coords, TgeoBoundingBox
            ), "With fmt='tgbox', the coords arg must be a torchgeo.BoundingBox"
            self.min_longitude = coords.minx
            self.max_longitude = coords.maxx
            self.min_latitude = coords.miny
            self.max_latitude = coords.maxy
        elif fmt == "epsg":
            if crs is None:
                raise ValueError(
                    "Please provide in 'crs' the rasterio.crs.CRS of your coordinates when using fmt='epsg'"
                )

            xmin, ymin, xmax, ymax = rasterio.warp.transform_bounds(
                crs, rasterio.crs.CRS.from_epsg(4326), *coords
            )
            self.min_longitude = xmin
            self.min_latitude = ymin
            self.max_longitude = xmax
            self.max_latitude = ymax
        else:
            raise ValueError(f"Unknown format {fmt}")

        if (
            self.min_longitude > self.max_longitude
            or self.min_latitude > self.max_latitude
        ):
            raise ValueError("Incoherent min/max values for lon/lat. Please the inputs")

    def _init_from_boundingbox(self, points):
        """Extract the bounding box from a list of points"""
        minlat = 90
        maxlat = -90
        minlon = 180
        maxlon = -180
        for point in points:
            lon, lat = point
            minlat = min(minlat, lat)
            maxlat = max(maxlat, lat)
            minlon = min(minlon, lon)
            maxlon = max(maxlon, lon)

        self.min_longitude = minlon
        self.max_longitude = maxlon
        self.min_latitude = minlat
        self.max_latitude = maxlat

        return self

    def to_llmm(self):
        """Export to the longitude-latitude minimum-maximum format"""
        return (
            self.min_longitude,
            self.max_longitude,
            self.min_latitude,
            self.max_latitude,
        )

    def to_llmm_utm(self):
        """Export to the longitude-latitude minimum-maximum format"""
        xmin, ymin, _, _ = utm.from_latlon(self.min_latitude, self.min_longitude)
        xmax, ymax, _, _ = utm.from_latlon(self.max_latitude, self.max_longitude)
        return xmin, xmax, ymin, ymax

    def to_bltr(self):
        """Export to the bottom-left, top-right format"""
        return (
            (self.min_longitude, self.min_latitude),
            (self.max_longitude, self.max_latitude),
        )

    def to_lbrt(self):
        """Export to the (left, bottom, right, top) format"""
        return (
            self.min_longitude,
            self.min_latitude,
            self.max_longitude,
            self.max_latitude,
        )

    def to_lbrt_utm(self):
        """Export to the (left, bottom, right, top) format for UTM coordinates"""
        xmin, ymin, _, _ = utm.from_latlon(self.min_latitude, self.min_longitude)
        xmax, ymax, _, _ = utm.from_latlon(self.max_latitude, self.max_longitude)
        return xmin, ymin, xmax, ymax

    def to_tlbr(self):
        """Export to the top-left, bottom-right format"""
        return (
            (self.min_longitude, self.max_latitude),
            (self.max_longitude, self.min_latitude),
        )

    def to_mmll(self):
        """Export to the minimum-maximum longitude-latitude format"""
        return (
            self.min_longitude,
            self.min_latitude,
            self.max_longitude,
            self.max_latitude,
        )

    def to_epsg(self, crs):
        """Export to the minimum-maximum longitude-latitude format in the given CRS

        Parameter
        ---------
        crs: `rasterio.crs.CRS` target coordinate reference system
        """
        xmin, ymin, xmax, ymax = rasterio.warp.transform_bounds(
            rasterio.crs.CRS.from_epsg(4326),
            crs,
            self.min_longitude,
            self.min_latitude,
            self.max_longitude,
            self.max_latitude,
        )
        return xmin, ymin, xmax, ymax

    def to_geojson(self, path=None):
        """Return a GeoJson-like string.
        If a path is provided, the string is written into this file.
        """

        geojson = (
            """{
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        "coordinates": [
            ["""
            + f"""
                [
                    {self.min_longitude},
                    {self.max_latitude}
                ],
                [
                    {self.max_longitude},
                    {self.max_latitude}
                ],
                [
                    {self.max_longitude},
                    {self.min_latitude}
                ],
                [
                    {self.min_longitude},
                    {self.min_latitude}
                ],
                [
                    {self.min_longitude},
                    {self.max_latitude}
                ]
            """
            + """]
        ]
    }
}
            """
        )
        if path is not None:
            with open(path, "w") as f:
                f.write(geojson)

        return geojson

    def to_aoi(self):
        "Export to pyproj.aoi.AreaOfInterest`"
        return pyproj.aoi.AreaOfInterest(
            west_lon_degree=self.min_longitude,
            south_lat_degree=self.min_latitude,
            east_lon_degree=self.max_longitude,
            north_lat_degree=self.max_latitude,
        )

    def to_sbox(self):
        """Export to shapely.geometry.box"""
        return shapely.geometry.box(*self.to_mmll())

    def to_tgbox(self, dst_crs=rasterio.crs.CRS.from_epsg(4326), tmin=0, tmax=1e12):
        """Export to torchgeo.datasets.BoundingBox"""
        xmin, ymin, xmax, ymax = self.to_epsg(dst_crs)
        return TgeoBoundingBox(xmin, xmax, ymin, ymax, tmin, tmax)

    def get_utm_crs(self, raise_exception=True):
        """Return the list of Universal Transerve Mercator coordinate
        reference system covering the area.
        """
        utm_crs_list = pyproj.database.query_utm_crs_info(
            datum_name="WGS 84", area_of_interest=self.to_aoi()
        )

        if len(utm_crs_list) > 1:
            if raise_exception:
                raise Exception("Area of interest covers more than one UTM tile")
            else:
                print("WARNING: area of interest covers more than one UTM tile")

        return pyproj.CRS.from_epsg(utm_crs_list[0].code)

    def to_str(self, collate_char="_", n_digits=6):
        return collate_char.join(
            [
                f"xmin{np.round(self.min_longitude, n_digits)}",
                f"xmax{np.round(self.max_longitude, n_digits)}",
                f"ymin{np.round(self.min_latitude, n_digits)}",
                f"ymax{np.round(self.max_latitude, n_digits)}",
            ]
        )

    def enlarge(self, factor):
        return enlarge_domain(self, factor)

    def central_point(self):
        return get_central_point(self)

    def centred_fixed_size(self, n_px, res):
        return centred_fixed_size(self, n_px, res)

    def __str__(self):
        return ", ".join(
            [
                attr + "=" + str(getattr(self, attr))
                for attr in [
                    "min_longitude",
                    "max_longitude",
                    "min_latitude",
                    "max_latitude",
                ]
            ]
        )

    def __repr__(self):
        return str(self.__class__) + " " + self.__str__()


# SOME USEFUL FUNCTIONS
# =======================
def bbox_from_bboxs(bboxs) -> GeoRectangle:
    """Give the smallest bounding box encompassing multiple bounding boxes"""
    pts = []
    for b in bboxs:
        pts += list(b.to_bltr())

    return GeoRectangle(pts, fmt="points")


def centred_fixed_size(domain, n_px, res) -> GeoRectangle:
    """Return a `GeoRectangle` with the same center as the one provided and
    with `n_px` pixels in each direction.
    """
    cp = domain.central_point()
    xmin = cp[0] - (n_px / 2) * res
    xmax = cp[0] + (n_px / 2) * res
    ymin = cp[1] - (n_px / 2) * res
    ymax = cp[1] + (n_px / 2) * res
    return GeoRectangle([xmin, xmax, ymin, ymax])


def enlarge_domain(domain, factor=1) -> GeoRectangle:
    """Create a larger domain than the one provided. Suitable only for small domains.


    Parameters
    ----------
    domain: `wopt.domains.GeoRectangle`
        Geographical domain

    factor: float>0
        Enlargement factor (coefficient applied to the size of the domain on both dimensions)


    Returns
    -------
    enlarged_domain: `wopt.domains.GeoRectangle`
        Enlarged geographical domain
    """
    xmin, xmax, ymin, ymax = domain.to_llmm()
    dx = abs(xmax - xmin)
    dy = abs(ymax - ymin)
    enlarged_llmm = (
        max(-180, xmin - factor * dx),
        min(180, xmax + factor * dx),
        max(-90, ymin - factor * dy),
        min(90, ymax + factor * dy),
    )
    return GeoRectangle(enlarged_llmm, fmt="llmm")


def get_central_point(domain) -> tuple:
    """Return the coordinates of the points at the center of the rectangle"""
    x = (domain.min_longitude + domain.max_longitude) / 2
    y = (domain.min_latitude + domain.max_latitude) / 2
    return x, y


# SOME USEFUL DOMAINS (alphabetical order)
# ===================
# Easily draw geoJson: https://geojson.io/d
# Easily access to bounding boxes: http://bboxfinder.com/

alps_and_carpates = GeoRectangle([2, 27, 41.3, 51])
antelao_glacier = GeoRectangle(
    [12.19188, 12.2922, 46.4055, 46.4767]
)  # Alpine valley with town (San Vito de Cadore) and glacier (Antelao, 3263m), Italy
bakar_bay_croatia = GeoRectangle(
    [14.5263, 14.5864, 45.266, 45.3126]
)  # Coastline, forest, heavy industry in Croatia
bigearthnet_fulldom = GeoRectangle(
    [-9.00023345437725, 36.956956702083396, 31.598439091981028, 68.02168200047284]
)
breznik_slovenia = GeoRectangle(
    [15.1475, 15.1643, 45.5126, 45.5266]
)  # Forest and village
burren = GeoRectangle([-9.15985, -9.0469, 53.0871, 53.1504])  # Bare rock
canaries_morocco = GeoRectangle(
    [-18.457031, -7.338867, 24.567108, 33.174342]
)  # Track missing data handling around islands and in Sahara
dublin_city = GeoRectangle(
    [-6.3139, -6.2209, 53.3321, 53.3784]
)  # City with airport and heavy industry, complex coastline, estuary
dublin_county = GeoRectangle(
    '{"type": "Polygon", "coordinates": [[[-6.5468919, 53.1782636], [-6.0439503, 53.1782636], [-6.0439503, 53.509983], [-6.5468919, 53.509983], [-6.5468919, 53.1782636]]]}',
    fmt="json",
)
elhichria_tunisia = GeoRectangle(
    [9.390907, 9.509354, 34.832687, 34.921408]
)  # Low quality score
elmenia_algeria = GeoRectangle([2.44, 3.46, 30.01, 30.97])  # Sahara small town and lake
eurat = GeoRectangle(
    [-32, 42, 20, 72]
)  # Europe-Atlantic domain -> TARGET DOMAIN FOR THE LAND COVER MAP
europe = GeoRectangle([-20, 30, 32, 70])  # Hand cut of the European continent
faroe_islands = GeoRectangle([-8.11, -5.88, 61.30, 62.54])  # Faroe islands
fin_south_deode = GeoRectangle(
    [15.5, 32, 57, 65]
)  # forecast domain = [16.33, 31.03, 57.59, 64.07]
florida_mangrove = GeoRectangle([-81.5995, -81.5664, 25.9753, 26.0097])  # Mangroves
freastern_eurat = GeoRectangle(
    [8.24, 42, 20, 72]
)  # Part of EURAT east of France mainland
geneva = GeoRectangle(
    [6.092434, 6.202126, 46.163782, 46.239702]
)  # Large town, river to lake transition
ireland25 = GeoRectangle([-16, 4, 45, 61])  # HARMONIE-AROME domain for NWP
ireland = GeoRectangle([-11.1, -4.8, 50.9, 55.6])  # Hand cut of the Island of Ireland
irl750 = GeoRectangle([-15.5, -5, 49.5, 57.5])  # HECTOR domain for NWP
istanbul_power_plant = GeoRectangle([28.689, 28.703, 40.971, 40.983])  # Heavy industry
iso_kihdinluoto = GeoRectangle(
    TgeoBoundingBox(
        minx=21.122670101126037,
        maxx=21.239336767792704,
        miny=60.483903274933496,
        maxy=60.60056994160016,
        mint=0.0,
        maxt=1000000000000.0,
    ),
    fmt="tgbox",
)  # Tiny islands in the South of Finland
iziaslav_ukraine = GeoRectangle(
    [26.711884, 26.902084, 50.037297, 50.178657]
)  # Low score
jokioinen_crops = GeoRectangle(
    [23.4278, 23.4913, 60.7896, 60.8204]
)  # Northern crops and forest
large_sahara = GeoRectangle(
    [-2, 21, 20, 34]
)  # Large portion of Sahara, mostly over Algeria and Lybia
lecce_italy = GeoRectangle(
    [18.091049, 18.277130, 40.254377, 40.412973]
)  # Low quality score
london_city = GeoRectangle([-0.117073, -0.067978, 51.498485, 51.531280])  # LCZ1, river
mars_cevennes = GeoRectangle([3.5389, 3.5836, 43.9913, 44.0259])  # Mars en Cevennes
montpellier_agglo = GeoRectangle([3.63, 4.14, 43.38, 43.82])  # Sea artefact
nanterre = GeoRectangle([2.166, 2.272, 48.861, 48.936])  # LCZ1, river
ngambe_jungle = GeoRectangle([10.5992, 10.6443, 4.2156, 4.2581])  # Cameroon jungle
paname = GeoRectangle([-1.5, 6, 46, 50.5])  # NWP domain for the PANAME experiment
plouguernau_estuary = GeoRectangle(
    [-4.583359, -4.492722, 48.567861, 48.633022]
)  # Estuary, small town and small island. France, Brittany
portugese_crops = GeoRectangle(
    [-7.0984782206142825, -7.0781569753000895, 40.759403727839384, 40.77504037344005],
    fmt="llmm",
)  # Portugese crops and forest
reunion_crops = GeoRectangle(
    [55.6638, 55.7004, -21.0505, -21.0150]
)  # Reunion crops and forest
reunion_island = GeoRectangle([55.1958, 55.8723, -21.4286, -20.8180])
rhone_avignon = GeoRectangle([4.64, 4.95, 43.82, 44.11])  # Stitching artefact
rural_andalousia = GeoRectangle(
    [-4.293423, -4.235573, 36.836493, 36.885387]
)  # Low quality score
savage_canary_island = GeoRectangle(
    [-15.8935, -15.8517, 30.1319, 30.1677]
)  # Small island
snaefell_glacier = GeoRectangle(
    [-15.5828, -15.4863, 64.7673, 64.8113]
)  # Ice, moss and lichen
subotica_serbia = GeoRectangle([19.620895, 19.814529, 46.020330, 46.176502])
toulouse = GeoRectangle([1.336, 1.508, 43.554, 43.674])  # City and river
tripoli_sahara = GeoRectangle([11, 15, 30, 33])  # Small portion of Sahara, near Tripoli
waterville_kerry_124x133 = GeoRectangle([-10.16, -10.141, 51.83, 51.841])
waterville_kerry = GeoRectangle(
    [-10.2018, -10.1518, 51.8020, 51.8466]
)  # Small town, lake, sea

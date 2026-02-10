import logging
import math
import os

import numpy as np
import ee
import rasterio
import rasterio.env
import rasterio.windows

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

class Downloader:
    def __init__(self, output_path="./ae_embeddings", tile_size=120, dequantize=False):
        self.output_path = output_path
        self.tile_size = tile_size
        self.dequantize = dequantize
        
        self.bucket = "s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/"

        self.gee_ae_collection = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
        self.tge_labs_aef_file_endings = ["-0000000000-0000000000.tiff",
                                          "-0000000000-0000008192.tiff",
                                          "-0000008192-0000000000.tiff",
                                          "-0000008192-0000008192.tiff"
                                          ]  # Each GEE asset is split into 4 files in the S3 bucket.

        ee.Initialize()


    def download(self, lat, lon, year, location_id):
        """Download a tile of embeddings around lat,lon for a given year.
        Saves the result to the output directory.
        """
        logger.info(f"Processing {location_id}.")
        utm_data = self._get_utm_zone(lat, lon)
        gee_asset_id = self._get_asset_id(lat, lon, year)

        filenames = self._construct_candidate_file_names(year, str(utm_data["zone"]) + utm_data["hemisphere"], gee_asset_id)

        target_file = None
        for file in filenames:
            if self._check_file_for_point(utm_data["easting"], utm_data["northing"], file):
                # Found the file that contains the point of interest.
                target_file = file
                logger.info(f"Found target file {file}.")
                break

        logger.info("Reading file...")
        data, profile, centered = self._read_file(utm_data["easting"], utm_data["northing"], target_file)

        if self.dequantize:
            data = self._dequantize(data)
            profile["dtype"] = "float32"

        logger.info("Saving data...")
        self._save_tile(data, profile, year, location_id, centered)
        logger.info(f"Finished obtaining data for {location_id}.")

    
    def _get_utm_zone(self, lat, lon):
        """Get a location's UTM zone and the coordinates in the UTM's CRS."""

        point = ee.Geometry.Point(lon, lat)

        # Compute zone number
        zone = int(math.floor((lon + 180) / 6) + 1)

        # EPSG codes:
        # Northern: 32601–32660
        # Southern: 32701–32760
        if lat >= 0:
            epsg = 32600 + zone
            hemi = "N"
        else:
            epsg = 32700 + zone
            hemi = "S"

        proj = ee.Projection(f"EPSG:{epsg}")

        # Transform point
        utm_point = point.transform(proj, 1)
        coords = utm_point.coordinates().getInfo()

        return {
                "zone": zone,  # E.g., 10
                "hemisphere": hemi,  # [N, S]
                "epsg": f"EPSG:{epsg}",
                "easting": coords[0],
                "northing": coords[1],
        }
            

    def _get_asset_id(self, lat, lon, year):
        """Get the asset which contains the location in the GEE AlphaEarth Embedding collection.
        E.g., `x02qcrn30k70b9ql6`."""

        point = ee.Geometry.Point(lon, lat)
        collection = ee.ImageCollection(self.gee_ae_collection)

        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"

        filtered = (
            collection
            .filterDate(start_date, end_date)
            .filterBounds(point)
        )

        image = filtered.first()
        info = image.getInfo()

        if info is None:
            return None

        return info["id"].split("/")[-1]
    

    def _construct_candidate_file_names(self, year, utm_zone, file_id):
        """Create a path like `s3://us-west-2.opendata.source.coop/tge-labs/aef/v1/annual/2018/10N/x02qcrn30k70b9ql6-0000000000-0000000000.tiff`."""
        return [os.path.join(self.bucket, str(year), utm_zone, file_id + suffix) for suffix in self.tge_labs_aef_file_endings]


    def _check_file_for_point(self, utm_x, utm_y, filepath):
        """Check if a given file on S3 contains a point."""

        env_options = {
            'AWS_NO_SIGN_REQUEST': 'YES',
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tiff,.vrt',
            'GDAL_HTTP_TIMEOUT': '60'
        }
        vrt_path = filepath.replace('.tiff', '.vrt')
                
        with rasterio.env.Env(**env_options):
            with rasterio.open(vrt_path) as src:
                # Check if point is in this file's bounds
                return (src.bounds.left <= utm_x <= src.bounds.right and
                    src.bounds.bottom <= utm_y <= src.bounds.top)


    def _read_file(self, utm_x, utm_y, filename):
        """Read a window from the file on S3."""

        env_options = {
            'AWS_NO_SIGN_REQUEST': 'YES',
            'GDAL_DISABLE_READDIR_ON_OPEN': 'EMPTY_DIR',
            'CPL_VSIL_CURL_ALLOWED_EXTENSIONS': '.tiff,.vrt',
            'GDAL_HTTP_TIMEOUT': '300',
            'GDAL_HTTP_MAX_RETRY': '3'
        }
        with rasterio.env.Env(**env_options):
            with rasterio.open(filename.replace("tiff", "vrt")) as src:
                row, col = src.index(utm_x, utm_y)
                half_size = self.tile_size // 2
                    
                col_off = int(col) - half_size
                row_off = int(row) - half_size
                centered = True


                if col_off < 0 or src.width - self.tile_size < col_off:
                    logger.info(f"Adjusting colum offset: {col_off=}, {src.width - self.tile_size=}")
                    centered = False
                if row_off < 0 or src.height - self.tile_size < row_off:
                    logger.info(f"Adjusting row offset: {row_off=}, {src.height - self.tile_size=}")
                    centered = False
                    
                col_off = max(0, min(col_off, src.width - self.tile_size))
                row_off = max(0, min(row_off, src.height - self.tile_size))
                
                window = rasterio.windows.Window(col_off, row_off, self.tile_size, self.tile_size)
                data = src.read(window=window)

                transform = src.window_transform(window)
                profile = src.profile.copy()
                profile.update({
                    "height": self.tile_size,
                    "width": self.tile_size,
                    "transform": transform,
                    "driver": "GTiff",
                })


        return data, profile, centered


    def _dequantize(self, data):
        """Formula from documentation: ((values / 127.5) ** 2) * sign(values)
        NoData value (-128) is converted to NaN.
        """
        # Identify NoData pixels (-128)
        nodata_mask = data == -128
        
        # De-quantize: ((v / 127.5)^2) * sign(v)
        values_float = data.astype(np.float32)
        result = ((values_float / 127.5) ** 2) * np.sign(values_float)
        
        # Set NoData to NaN
        result[nodata_mask] = np.nan
        
        return result


    def _save_tile(self, data, profile, year, location_id, msg=""):
        output_dir = os.path.join(self.output_path, str(year))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        output_file = output_dir + "/" + location_id + f"_centered_{msg}" + ".tiff"

        with rasterio.open(output_file, "w", **profile) as dst:
            dst.write(data)

        logger.info(f"Saved data to {output_file}.")
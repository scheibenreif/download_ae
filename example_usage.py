"""
Simple usage example for Alpha Earth downloader.
"""

import tqdm
import pandas as pd

from ae_downloader import Downloader


year = 2022  # Use embeddings from this year.
tile_size = 120  # Height and width of the downloaded data (10m GSD).
skip_if_present = True  # Skip a location if the corresponding file is already downloaded.

# Load the locations of air pollution monitoring stations.
station_data = pd.read_csv("../global_air_pollution_datalake/station_data.csv")

# Use some sample locations.
# locations = {"san_francisco": (37.7749, -122.4194),
#              "new_york": (40.7128, -74.0060),
#              "los_angeles": (34.0522, -118.2437),
#              "bregenz": (47.5049, 9.7256),
#             }

if __name__ == "__main__":
    downloader = Downloader(output_path="./ae_embeddings/air_pollution/", tile_size=tile_size, skip_if_present=skip_if_present)

    # for location_id, (lat, lon) in tqdm.tqdm(locations.items()):
    for idx, (location_id, lat, lon) in tqdm.tqdm(station_data.iterrows(), total=station_data.shape[0]):
        # try:
        downloader.download(lat, lon, year, location_id)
        # except Exception as e:
        #     print(e)
        #     continue

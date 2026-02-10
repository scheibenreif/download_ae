"""
Simple usage example for Alpha Earth downloader.
"""

import numpy as np
import tqdm
import pandas as pd

from ae_downloader import Downloader


year = 2022
locations = {"san_francisco": (37.7749, -122.4194),
             "new_york": (40.7128, -74.0060),
             "los_angeles": (34.0522, -118.2437),
             "bregenz": (47.50489881092864, 9.725558000922499),
            }

station_data = pd.read_csv("../global_air_pollution_datalake/station_data.csv")
downloader = Downloader(output_path="./ae_embeddings/air_pollution/", tile_size=5)

# for location_id, (lat, lon) in tqdm.tqdm(locations.items()):
for idx, (location_id, lat, lon) in tqdm.tqdm(station_data.iterrows(), total=station_data.shape[0]):
    try:
        downloader.download(lat, lon, year, location_id)
    except Exception as e:
        print(e)
        continue

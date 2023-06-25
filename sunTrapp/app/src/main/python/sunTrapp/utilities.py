import rasterio
from pyproj import Transformer
import datetime

def open_tif_file(fileName):
    # Data from https://environment.data.gov.uk/DefraDataDownload/?Mode=survey
    dataset = rasterio.open(fileName)
    data = dataset.read(1)
    transform = dataset.transform
    return data, transform

def get_centre_indexes(loc, transform, upsampling):

    transformer_toOS = Transformer.from_crs("EPSG:4326", "EPSG:27700")
    idx_raw = transformer_toOS.transform(loc[0],loc[1])
    idx = (idx_raw[1], -idx_raw[0])
    y, x = rasterio.transform.rowcol(transform, -idx[1], idx[0])
    y*=upsampling
    x*=upsampling

    return x, y, idx_raw


def generate_time_stamps(start_time, end_time, n):
    start = datetime.datetime.strptime(start_time, "%H:%M:%S")
    end = datetime.datetime.strptime(end_time, "%H:%M:%S")
    
    time_diff = (end - start) / (n + 1)
    
    time_stamps = []
    current_time = start
    
    for _ in range(n):
        current_time += time_diff
        time_stamps.append(current_time.strftime("%H:%M:%S"))
    
    return time_stamps

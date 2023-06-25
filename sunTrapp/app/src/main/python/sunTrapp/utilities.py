import rasterio
from pyproj import Transformer

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
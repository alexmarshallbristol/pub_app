import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from pyproj import Transformer
from affine import Affine

# data from https://environment.data.gov.uk/DefraDataDownload/?Mode=survey

tif_file = "DSM_ST5570_P_10631_20190117_20190117.tif"
tif_dataset = rasterio.open(tif_file)
tif_data = tif_dataset.read(1)

transform = tif_dataset.transform
crs = tif_dataset.crs
print(crs)

height, width = tif_data.shape


idx = tif_dataset.index(0, 0, precision=1E-1) # top left 
print(idx)

transformer_toGPS = Transformer.from_crs("EPSG:27700", "EPSG:4326")
idx_GPS = transformer_toGPS.transform(-idx[1],idx[0])
print(idx_GPS)

transformer_toOS = Transformer.from_crs("EPSG:4326", "EPSG:27700")
idx = transformer_toOS.transform(idx_GPS[0],idx_GPS[1])
idx = (idx[1], -idx[0])
print(idx)


affine = Affine(*transform)

# pub = [51.467334, -2.586485] # cadbury
# pub = [51.468803, -2.593292] # cat and wheel
pub = [51.453291, -2.609596] # hope and anchor

transformer_toOS = Transformer.from_crs("EPSG:4326", "EPSG:27700")
idx = transformer_toOS.transform(pub[0],pub[1])
idx = (idx[1], -idx[0])
print(idx)

y_point, x_point = rasterio.transform.rowcol(transform, -idx[1], idx[0])
print(x_point, y_point)

N_pixels = 45 # number of metres around point
tif_data = tif_data[y_point-N_pixels:y_point+N_pixels,x_point-N_pixels:x_point+N_pixels]

# Visualize the TIFF data
plt.figure(figsize=(10, 10))
# plt.imshow(tif_data, cmap='gray',norm=LogNorm())
plt.imshow(tif_data)
plt.colorbar()
plt.title('TIFF Data')

# plt.plot(x_point, y_point, 'ro', markersize=1)

plt.plot(N_pixels, N_pixels, 'ro', markersize=8)

plt.show()

# Close the TIFF dataset
tif_dataset.close()


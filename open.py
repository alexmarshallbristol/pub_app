import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from pyproj import Transformer
from affine import Affine
import ephem
from math import degrees
from PIL import Image
import imageio
import os
from scipy.ndimage import gaussian_filter
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import glob
import moviepy.editor as mp


# speed by code - vectorise
# make better output plots - height of sun cartoon etc
# stitch two tif files together
# garden anlysis - sell app to rightmove lol








sun_color = '#FFD700'  # Brighter yellow color for sun
shade_color = '#121110'  # Darker blue color for shade

colors = [shade_color,sun_color]
cmap = LinearSegmentedColormap.from_list('sun_shade_cmap', colors)


# data from https://environment.data.gov.uk/DefraDataDownload/?Mode=survey

tif_file = "DSM_ST5570_P_10631_20190117_20190117.tif"
tif_dataset = rasterio.open(tif_file)
tif_data = tif_dataset.read(1)

transform = tif_dataset.transform
crs = tif_dataset.crs

height, width = tif_data.shape

tif_data_full = tif_data.copy()

# idx = tif_dataset.index(0, 0, precision=1E-1) # top left 

# transformer_toGPS = Transformer.from_crs("EPSG:27700", "EPSG:4326")
# idx_GPS = transformer_toGPS.transform(-idx[1],idx[0])

# transformer_toOS = Transformer.from_crs("EPSG:4326", "EPSG:27700")
# idx = transformer_toOS.transform(idx_GPS[0],idx_GPS[1])
# idx = (idx[1], -idx[0])


# pub = [51.467334, -2.586485] # cadbury
# pub = [51.468803, -2.593292] # cat and wheel
pub = [51.453291, -2.609596] # hope and anchor
# pub = [51.455028, -2.590344] # castle park

pub_name = "Hope and Anchor (large winter)"

output_gif = f"output_{pub_name.replace(' ','_')}"

def compute_sun_direction(date, time, latitude, longitude):
    observer = ephem.Observer()
    observer.lat = str(latitude)
    observer.lon = str(longitude)
    observer.date = date + ' ' + time

    sun = ephem.Sun()
    sun.compute(observer)

    azimuth = degrees(sun.az)
    altitude = degrees(sun.alt)

    return azimuth, altitude

def compute_angle_from_north(origin_x, origin_y, points_x, points_y):

    distance_x = (points_x-origin_x)
    distance_y = (points_y-origin_y)

    angles = np.degrees(np.arctan2(distance_x,distance_y))+90
    angles = angles % 360

    return angles


def select_elements_within_angle(array, angle, cone=5, N_surrounding_pixels=50):

    array[np.where(array<0)] += 360

    adjusted_angle = angle % 360

    lower_bound = adjusted_angle - cone
    upper_bound = adjusted_angle + cone

    if lower_bound < 0:
        where = np.where(np.logical_and(array > upper_bound, array < 360 + lower_bound))
    elif upper_bound > 360:
        where = np.where(np.logical_and(array < lower_bound, array > upper_bound - 360))
    else:
        where = np.where(np.logical_or(array < lower_bound, array > upper_bound))

    row_to_remove = [N_surrounding_pixels, N_surrounding_pixels]
    to_delete = np.where((where[0] == row_to_remove[0]) & (where[1] == row_to_remove[1]))
    where = np.delete(where, to_delete, axis=1)
    where = (np.asarray(where[0]), np.asarray(where[1]))

    return where


def compute_angle_from_horizontal(origin_x, origin_y, points_x, points_y, current_height, height):

    distance_x = points_x-origin_x
    distance_y = points_y-origin_y

    distance = np.sqrt(distance_x**2 + distance_y**2)

    angles = np.degrees(np.arctan2(height-current_height,distance))

    return angles


def _is_sunny(y_point, x_point, date, time, longitude, latitude, blur=True, N_surrounding_pixels=50):

    tif_data = tif_data_full[y_point-N_surrounding_pixels:y_point+N_surrounding_pixels,x_point-N_surrounding_pixels:x_point+N_surrounding_pixels]


    tif_data = np.expand_dims(tif_data, -1)

    rows = np.arange(N_surrounding_pixels*2).reshape(N_surrounding_pixels*2, 1)
    cols = np.arange(N_surrounding_pixels*2).reshape(1, N_surrounding_pixels*2)
    meshgrid_rows, meshgrid_cols = np.meshgrid(rows, cols)
    result_array = np.stack((meshgrid_cols,meshgrid_rows), axis=2)

    tif_data = np.concatenate((tif_data,result_array),axis=-1)

    azimuth, altitude = compute_sun_direction(date, time, latitude, longitude)

    angles = compute_angle_from_north(N_surrounding_pixels, N_surrounding_pixels, tif_data[:,:,1], tif_data[:,:,2])
    where = select_elements_within_angle(angles, azimuth, 5, N_surrounding_pixels)

    tif_data = tif_data[:,:,0]
    tif_data[where] = 0
    tif_data = np.expand_dims(tif_data, -1)
    tif_data = np.concatenate((tif_data,result_array),axis=-1)

    # if blur:
    #     tif_data[:,:,0] = gaussian_filter(tif_data[:,:,0], sigma=0.5)

    angle_to_horizontal = compute_angle_from_horizontal(N_surrounding_pixels, N_surrounding_pixels, tif_data[:,:,1], tif_data[:,:,2], tif_data[N_surrounding_pixels,N_surrounding_pixels,0], tif_data[:,:,0])

    if np.amax(angle_to_horizontal) < altitude:
        sunny = 1 # True
    else:
        sunny = 0 # False

    return sunny, azimuth




transformer_toOS = Transformer.from_crs("EPSG:4326", "EPSG:27700")
idx = transformer_toOS.transform(pub[0],pub[1])
idx = (idx[1], -idx[0])
y_point_gps, x_point_gps = rasterio.transform.rowcol(transform, -idx[1], idx[0])

print(y_point_gps, x_point_gps)

date = '2023/06/21'  # Date format: yyyy/mm/dd
# date = '2023/01/21'  # Date format: yyyy/mm/dd
latitude = f'{pub[0]}'   # Latitude of the location
longitude = f'{pub[1]}'  # Longitude of the location

time = '12:00:00'   # Time format: hh:mm:ss
time_iter = 0
for hour in range(10,21):
    for minute in ['00', 15, 30, 45]:
    # for minute in ['00', '05', 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
        time_iter += 1
        time = f'{hour}:{minute}:00'   # Time format: hh:mm:ss
        
        print(time)

        steps = 100 # size of area to compute shadows

        is_sunny_array = np.zeros(np.shape(tif_data_full))

        for y_point_i in range(y_point_gps-steps, y_point_gps+steps+1):
            for x_point_i in range(x_point_gps-steps, x_point_gps+steps+1):
                is_sunny, azimuth = _is_sunny(y_point_i, x_point_i, date, time, longitude, latitude)
                is_sunny_array[y_point_i][x_point_i] = is_sunny



        N_pixels = 125 # number of metres around point to plot
        tif_data_i = tif_data_full[y_point_gps-N_pixels:y_point_gps+N_pixels,x_point_gps-N_pixels:x_point_gps+N_pixels]
        is_sunny_array_i = is_sunny_array[y_point_gps-N_pixels:y_point_gps+N_pixels,x_point_gps-N_pixels:x_point_gps+N_pixels]

        is_sunny_array_i_blur = gaussian_filter(is_sunny_array_i, sigma=0.5)

        
        is_sunny, azimuth = _is_sunny(y_point_gps, x_point_gps, date, time, longitude, latitude)

        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(f"{pub_name} {time} {date}", fontsize=14)

        ax = plt.subplot(1,2,1)
        plt.imshow(tif_data_i, norm=LogNorm())
        rect = patches.Rectangle((N_pixels-steps-0.5, N_pixels-steps-0.5), steps*2+1, steps*2+1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        angle_radians = np.radians(azimuth)
        line_length = N_pixels*4
        x_endpoint = -line_length * np.cos(angle_radians)
        y_endpoint = line_length * np.sin(angle_radians)
        ax.plot([N_pixels-0.5, N_pixels-0.5 + y_endpoint], [N_pixels-0.5, N_pixels-0.5 + x_endpoint], color='red')
        plt.xlim(0-0.5, N_pixels*2-0.5)
        plt.ylim(N_pixels*2-0.5,-0.5)

        ax = plt.subplot(1,2,2)
        plt.imshow(tif_data_i, norm=LogNorm())
        plt.imshow(is_sunny_array_i_blur, cmap=cmap, alpha=0.5)
        rect = patches.Rectangle((N_pixels-steps-0.5, N_pixels-steps-0.5), steps*2+1, steps*2+1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        plt.tight_layout()        
        plt.savefig(f'temp/plot_{time_iter}.png')
        plt.close('all')


image_files = glob.glob('temp/plot_*.png')
image_idx_max = 0
for image in image_files:
    image_idx = image[10:]
    image_idx = image_idx[:-4]
    if int(image_idx) > int(image_idx_max): image_idx_max = image_idx
image_files = []
for image_idx in range(1,int(image_idx_max)):
    image_files.append(f'temp/plot_{image_idx}.png')


# Create a list to store the image frames
frames = []

# Read and append each image to the frames list
for image_file in image_files:
    image = Image.open(image_file)
    frames.append(image)

# Save frames as an animated GIF
frames[0].save(output_gif+".gif", format='GIF', append_images=frames[1:], save_all=True, duration=200, loop=0)
clip = mp.VideoFileClip(output_gif+".gif")
clip.write_videofile(f"{output_gif}.mp4")

os.system('rm temp/plot_*.png')
os.system('rm *.gif')


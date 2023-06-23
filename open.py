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
import scipy
from PIL import ImageFilter
import time
import datetime
# speed by code - vectorise
# make better output plots - height of sun cartoon etc
# stitch two tif files together
# garden anlysis - sell app to rightmove lol
# need a better selection of points - maybe in cone and +1
# smoothing/upsampling algo






sun_color = '#FFD700'  # Brighter yellow color for sun
shade_color = '#121110'  # Darker blue color for shade

colors = [shade_color,sun_color]
cmap = LinearSegmentedColormap.from_list('sun_shade_cmap', colors)


# data from https://environment.data.gov.uk/DefraDataDownload/?Mode=survey

# tif_file = "tif/DSM_ST5570_P_10631_20190117_20190117.tif"
tif_file = "tif/DSM_ST6070_P_10631_20190117_20190117.tif"
# tif_file = "tif/homefarm.tif"
# tif_file = "tif/perann.tif"
tif_dataset = rasterio.open(tif_file)
tif_data = tif_dataset.read(1)

transform = tif_dataset.transform
crs = tif_dataset.crs

height, width = tif_data.shape


upsampling = 2
tif_data_plain_upsampling = scipy.ndimage.zoom(tif_data, upsampling, order=0)
tif_data = scipy.ndimage.zoom(tif_data, upsampling, order=1)

tif_data_full = tif_data.copy()


def smooth_image(image_array, size=3):
    _min = np.amin(image_array)
    _max = np.amax(image_array)
    image_array += -_min
    image_array *= (1./(_max-_min))
    image = Image.fromarray(np.uint8(image_array*255))
    image = image.filter(ImageFilter.ModeFilter(size=size))
    image_array = np.array(image)
    image_array = ((image_array*_max)+_min)/255
    return image_array


# pub = [51.453291, -2.609596]
# transformer_toOS = Transformer.from_crs("EPSG:4326", "EPSG:27700")
# idx = transformer_toOS.transform(pub[0],pub[1])
# idx = (idx[1], -idx[0])
# y_point_gps, x_point_gps = rasterio.transform.rowcol(transform, -idx[1], idx[0])
# y_point_gps*=upsampling
# x_point_gps*=upsampling

# N_pixels = 25
# tif_data_i = tif_data_full[y_point_gps-N_pixels:y_point_gps+N_pixels,x_point_gps-N_pixels:x_point_gps+N_pixels]

# plt.figure(figsize=(10,6))

# ax = plt.subplot(1,2,1)
# plt.imshow(tif_data_i, norm=LogNorm())

# tif_data_i = smooth_image(tif_data_i)
# ax = plt.subplot(1,2,2)
# plt.imshow(tif_data_i, norm=LogNorm())

# plt.tight_layout()        
# plt.show()
# quit()




# idx = tif_dataset.index(0, 0, precision=1E-1) # top left 

# transformer_toGPS = Transformer.from_crs("EPSG:27700", "EPSG:4326")
# idx_GPS = transformer_toGPS.transform(-idx[1],idx[0])

# transformer_toOS = Transformer.from_crs("EPSG:4326", "EPSG:27700")
# idx = transformer_toOS.transform(idx_GPS[0],idx_GPS[1])
# idx = (idx[1], -idx[0])


# pub = [51.467334, -2.586485] # cadbury
# pub = [51.468803, -2.593292] # cat and wheel
# pub = [51.453291, -2.609596] # hope and anchor
# pub = [51.455028, -2.590344] # castle park
# pub = [51.220550, -0.341143] # home farm
# pub = [51.220670, -0.334087] # tennis club
# pub = [50.341347, -5.159108] # chris' gafff
# pub = [51.458621, -2.601988] # uni
pub = [51.45795467636116, -2.548439358008857] # seneca
pub_name = "Seneca"

output_gif = f"output_{pub_name.replace(' ','_')}"


# from scipy.spatial import ConvexHull

# def convexhull(p):
# 	p = np.array(p)
# 	hull = ConvexHull(p)
# 	return p[hull.vertices,:]

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

    distance_x = (points_x-origin_x)/upsampling
    distance_y = (points_y-origin_y)/upsampling

    angles = np.degrees(np.arctan2(distance_x,distance_y))+90
    angles = angles % 360

    return angles


def select_elements_within_angle(array, angle, cone=5, N_surrounding_pixels=50):

    array[np.where(array<0)] += 360

    adjusted_angle = angle % 360

    lower_bound = adjusted_angle - cone
    upper_bound = adjusted_angle + cone

    if lower_bound < 0:
        where = np.where(np.logical_or(array <= upper_bound, array >= 360 + lower_bound))
    elif upper_bound > 360:
        where = np.where(np.logical_or(array >= lower_bound, array <= upper_bound - 360))
    else:
        where = np.where(np.logical_and(array >= lower_bound, array <= upper_bound))

    # row_to_remove = [N_surrounding_pixels, N_surrounding_pixels]
    # to_delete = np.where((where[0] == row_to_remove[0]) & (where[1] == row_to_remove[1]))
    # where = np.delete(where, to_delete, axis=1)
    # where = (np.asarray(where[0]), np.asarray(where[1]))

    return where


def select_elements_without_angle(array, angle, cone=5, N_surrounding_pixels=50):

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

    distance = np.sqrt(distance_x**2 + distance_y**2)/upsampling

    angles = np.degrees(np.arctan2(height-current_height,distance))

    return angles


def _is_sunny(y_point, x_point, date, time, longitude, latitude, blur=True, N_surrounding_pixels=50, make_plots=False):
    
    tif_data = tif_data_full[y_point-N_surrounding_pixels:y_point+N_surrounding_pixels,x_point-N_surrounding_pixels:x_point+N_surrounding_pixels]

    if make_plots:
        plt.imshow(tif_data)
        plt.savefig('A')
        plt.close('all')

    tif_data = np.expand_dims(tif_data, -1)

    rows = np.arange(N_surrounding_pixels*2).reshape(N_surrounding_pixels*2, 1)
    cols = np.arange(N_surrounding_pixels*2).reshape(1, N_surrounding_pixels*2)
    meshgrid_rows, meshgrid_cols = np.meshgrid(rows, cols)
    result_array = np.stack((meshgrid_cols,meshgrid_rows), axis=2)

    tif_data = np.concatenate((tif_data,result_array),axis=-1)

    azimuth, altitude = compute_sun_direction(date, time, latitude, longitude)

    angles = compute_angle_from_north(N_surrounding_pixels, N_surrounding_pixels, tif_data[:,:,1], tif_data[:,:,2])

    if make_plots:
        plt.imshow(angles)
        plt.savefig('B')
        plt.close('all')
        

    # where = select_elements_without_angle(angles, azimuth, 5, N_surrounding_pixels)
    where = select_elements_without_angle(angles, azimuth, 10, N_surrounding_pixels)

    tif_data = tif_data[:,:,0]
    tif_data[where] = 0

    if make_plots:
        plt.imshow(tif_data)
        plt.savefig('C')
        plt.close('all')

    tif_data = np.expand_dims(tif_data, -1)
    tif_data = np.concatenate((tif_data,result_array),axis=-1)

    # if blur:
    #     tif_data[:,:,0] = gaussian_filter(tif_data[:,:,0], sigma=0.5)

    angle_to_horizontal = compute_angle_from_horizontal(N_surrounding_pixels, N_surrounding_pixels, tif_data[:,:,1], tif_data[:,:,2], tif_data[N_surrounding_pixels,N_surrounding_pixels,0], tif_data[:,:,0])

    if make_plots:
        plt.imshow(angle_to_horizontal)
        plt.savefig('D')
        plt.close('all')

    if np.amax(angle_to_horizontal) < altitude:
        sunny = 1 # True
    else:
        sunny = 0 # False

    return sunny, azimuth

def compute_distance(origin_x, origin_y, points_x, points_y):

    distance_x = points_x-origin_x
    distance_y = points_y-origin_y

    distance = np.sqrt(distance_x**2 + distance_y**2)/upsampling

    return distance



transformer_toOS = Transformer.from_crs("EPSG:4326", "EPSG:27700")
idx = transformer_toOS.transform(pub[0],pub[1])
idx = (idx[1], -idx[0])
y_point_gps, x_point_gps = rasterio.transform.rowcol(transform, -idx[1], idx[0])
print(y_point_gps, x_point_gps)
y_point_gps*=upsampling
x_point_gps*=upsampling
print(y_point_gps, x_point_gps, 'upsampled')
# boundaries = np.asarray([[51.45798598418041, -2.5484111551876887], [51.457932504048706, -2.548385674201808], [51.4579266546555, -2.5484647993684906],[51.457972196339945, -2.548487598145332]])

boundaries = np.asarray([[51.45796107488286, -2.5484832507789275], [51.45792472509282, -2.5484691691814665], [51.45793141011384, -2.548386691253483],[51.45798530806013, -2.5484101605825846], [51.457976116166826, -2.5484557581362663],[51.45796441739083, -2.5484503937181864]])

print(np.shape(boundaries))

def no_floor(values):
    return values
boundaries_pixels = np.empty((0,2))
for boundary_point in boundaries:
    idx = transformer_toOS.transform(boundary_point[0],boundary_point[1])
    idx = (idx[1], -idx[0])
    y_point_gps_boundary, x_point_gps_boundary = rasterio.transform.rowcol(transform, -idx[1], idx[0], op=no_floor)
    y_point_gps_boundary*=upsampling
    x_point_gps_boundary*=upsampling
    y_point_gps_boundary+= -y_point_gps
    x_point_gps_boundary+= -x_point_gps
    boundaries_pixels = np.append(boundaries_pixels, [[y_point_gps_boundary, x_point_gps_boundary]], axis=0)

# plt.scatter(boundaries_pixels[:,0], boundaries_pixels[:,1])
# plt.scatter(0, 0)
# plt.show()

date = '2023/06/22'  # Date format: yyyy/mm/dd
# date = '2023/01/21'  # Date format: yyyy/mm/dd
latitude = f'{pub[0]}'   # Latitude of the location
longitude = f'{pub[1]}'  # Longitude of the location

steps = 15*upsampling # size of area to compute shadows
buffer = int(steps/4.)
boundaries_pixels = steps+buffer+boundaries_pixels

time_iter = 0
for hour in range(6,21):
    # for minute in ['00']:
    for minute in ['00', 15, 30, 45]:
    # for minute in ['00', '05', 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
        time_iter += 1
        time_string = f'{hour}:{minute}:00'   # Time format: hh:mm:ss

# for time_string in ["13:50:00"]:
        
        print(time_string)



        L_shadow = 50


        yx_points = np.empty((0,4))
        for y_point_idx, y_point_gps_i in enumerate(range(y_point_gps-steps, y_point_gps+steps+1)):
            for x_point_idx, x_point_gps_i in enumerate(range(x_point_gps-steps, x_point_gps+steps+1)):
                yx_points = np.append(yx_points, [[y_point_gps_i, x_point_gps_i, y_point_idx, x_point_idx]], axis=0)
        
        yx_points = yx_points.astype(int)

        tif_data_samples = np.asarray([tif_data_full[coords[0]-L_shadow:coords[0]+L_shadow,coords[1]-L_shadow:coords[1]+L_shadow] for coords in yx_points])

        rows = np.arange(L_shadow*2).reshape(L_shadow*2, 1)
        cols = np.arange(L_shadow*2).reshape(1, L_shadow*2)
        meshgrid_rows, meshgrid_cols = np.meshgrid(rows, cols)
        result_array = np.stack((meshgrid_cols,meshgrid_rows), axis=2)
        angles = compute_angle_from_north(L_shadow, L_shadow, result_array[:,:,0], result_array[:,:,1])

        azimuth, altitude = compute_sun_direction(date, time_string, latitude, longitude)
        where = select_elements_without_angle(angles, azimuth, 10, L_shadow)

        distance_to_centre = compute_distance(L_shadow, L_shadow, result_array[:,:,0], result_array[:,:,1])
        tif_data_samples[:,where[0],where[1]] = 0

        tif_data_samples_diff = tif_data_samples-np.expand_dims(np.expand_dims(tif_data_samples[:,L_shadow,L_shadow],1),1)

        angle = np.degrees(np.arctan2(tif_data_samples_diff,np.expand_dims(distance_to_centre,0)))

        angle_max = np.amax(angle, axis=(1,2))

        where = np.where(angle_max>altitude) # where in dark
        is_sunny = np.ones(np.shape(angle_max))
        is_sunny[where] = 0.
        is_sunny = is_sunny.reshape((steps*2+1,steps*2+1))

        is_sunny = smooth_image(is_sunny, size=4)

        # tif_data_i = tif_data_full[y_point_gps-steps:y_point_gps+steps+1,x_point_gps-steps:x_point_gps+steps+1]
        # plt.imshow(tif_data_i, norm=LogNorm())
        # plt.imshow(is_sunny, cmap=cmap, alpha=0.5)        
        # plt.tight_layout()        
        # plt.show()




        is_sunny_pad = np.pad(is_sunny, buffer)
        tif_data_i = tif_data_plain_upsampling[y_point_gps-steps-buffer:y_point_gps+steps+1+buffer,x_point_gps-steps-buffer:x_point_gps+steps+1+buffer]

        plt.figure(figsize=(12,4))

        ax = plt.subplot(1,3,1)
        plt.imshow(tif_data_i, norm=LogNorm())
        # plt.fill(boundaries_pixels[:,1], boundaries_pixels[:,0], c='w',alpha=0.5)
        boundaries_pixels_swap = boundaries_pixels.copy()
        boundaries_pixels_swap[:,1] = boundaries_pixels[:,0]
        boundaries_pixels_swap[:,0] = boundaries_pixels[:,1]
        poly = plt.Polygon(boundaries_pixels_swap, ec="w", fill=False)
        path = poly.get_path()


        rows = np.arange(steps*2+buffer*2+1).reshape(steps*2+buffer*2+1, 1)
        cols = np.arange(steps*2+buffer*2+1).reshape(1, steps*2+buffer*2+1)
        meshgrid_rows, meshgrid_cols = np.meshgrid(rows, cols)
        result_array = np.stack((meshgrid_cols,meshgrid_rows), axis=2)
        result_array = result_array.reshape(((steps*2+buffer*2+1)**2,2))
        result_array_swap = result_array.copy()
        result_array_swap[:,1] = result_array[:,0]
        result_array_swap[:,0] = result_array[:,1]
        where_contained = path.contains_points(result_array_swap)
        # plt.scatter(result_array[where_contained][:,0], result_array[where_contained][:,1])

        is_sunny_pad_garden = np.expand_dims(is_sunny_pad, -1)
        is_sunny_pad_garden = is_sunny_pad_garden.reshape(((steps*2+buffer*2+1)**2,1))
        is_sunny_pad_garden = np.concatenate((is_sunny_pad_garden,result_array_swap),-1)
        # is_sunny_pad_garden = is_sunny_pad_garden[where_contained]

        # quit()
        # plt.scatter()

        ax.add_patch(poly)
        rect = patches.Rectangle((buffer-0.5, buffer-0.5), steps*2+1, steps*2+1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect) 
        plt.text(0.01, 0.99, f'Time: {time_string}', horizontalalignment='left',verticalalignment='top', transform=ax.transAxes, c='w')
        plt.text(0.01, 0.94, f'Date: {date}', horizontalalignment='left',verticalalignment='top', transform=ax.transAxes, c='w')
        plt.yticks([],[])   
        plt.xticks([],[]) 
        # plt.colorbar()

        ax = plt.subplot(1,3,2)
        plt.imshow(tif_data_i, norm=LogNorm())
        plt.imshow(is_sunny_pad, cmap=cmap, alpha=0.5, vmin=0, vmax=1)    
        # plt.colorbar()
        rect = patches.Rectangle((buffer-0.5, buffer-0.5), steps*2+1, steps*2+1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect) 
        plt.fill(boundaries_pixels[:,1], boundaries_pixels[:,0], c='w',alpha=0.25)
        plt.yticks([],[])   
        plt.xticks([],[])   



        is_sunny_pad_garden = is_sunny_pad_garden[where_contained]

        ax = plt.subplot(1,3,3)
        plt.scatter(is_sunny_pad_garden[:,1], is_sunny_pad_garden[:,2], c=is_sunny_pad_garden[:,0], vmin=0, vmax=1)
        plt.gca().invert_yaxis()
        plt.yticks([],[])   
        plt.xticks([],[])   


        plt.subplots_adjust(hspace=0,wspace=0)
        plt.tight_layout()        
        plt.savefig(f'temp/plot_{time_iter}.png')
        # plt.show()
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
frames[0].save(output_gif+".gif", format='GIF', append_images=frames[1:], save_all=True, duration=200*2, loop=0)
clip = mp.VideoFileClip(output_gif+".gif")
clip.write_videofile(f"mp4/{output_gif}.mp4")

# os.system('rm temp/plot_*.png')
os.system('rm *.gif')


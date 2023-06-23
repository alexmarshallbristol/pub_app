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

tif_file = "tif/DSM_ST5570_P_10631_20190117_20190117.tif"
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
    image_array = (image_array*_max)+_min
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
pub = [51.453291, -2.609596] # hope and anchor
# pub = [51.455028, -2.590344] # castle park
# pub = [51.220550, -0.341143] # home farm
# pub = [51.220670, -0.334087] # tennis club
# pub = [50.341347, -5.159108] # chris' gafff
# pub = [51.458621, -2.601988] # uni
pub_name = "University"

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




transformer_toOS = Transformer.from_crs("EPSG:4326", "EPSG:27700")
idx = transformer_toOS.transform(pub[0],pub[1])
idx = (idx[1], -idx[0])
y_point_gps, x_point_gps = rasterio.transform.rowcol(transform, -idx[1], idx[0])
y_point_gps*=upsampling
x_point_gps*=upsampling
print(y_point_gps, x_point_gps)


date = '2023/06/22'  # Date format: yyyy/mm/dd
# date = '2023/01/21'  # Date format: yyyy/mm/dd
latitude = f'{pub[0]}'   # Latitude of the location
longitude = f'{pub[1]}'  # Longitude of the location

# time = '12:00:00'   # Time format: hh:mm:ss
# time_iter = 0
# for hour in range(10,21):
#     for minute in ['00', 15, 30, 45]:
#     # for minute in ['00', '05', 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]:
#         time_iter += 1
#         time = f'{hour}:{minute}:00'   # Time format: hh:mm:ss

for time_string in ["13:50:00"]:
        
        print(time_string)

        # steps = 100 # size of area to compute shadows
        steps = 25*upsampling # size of area to compute shadows
        # steps = 3*upsampling # size of area to compute shadows

        L_shadow = 50

        time_A = time.time()
        is_sunny_array = np.zeros(np.shape(tif_data_full))
        for y_point_idx, y_point_gps_i in enumerate(range(y_point_gps-steps, y_point_gps+steps+1)):
            for x_point_idx, x_point_gps_i in enumerate(range(x_point_gps-steps, x_point_gps+steps+1)):
                if y_point_idx == 0 and x_point_idx == 0:
                    is_sunny, azimuth = _is_sunny(y_point_gps_i, x_point_gps_i, date, time_string, longitude, latitude, make_plots=True)
                else:
                    is_sunny, azimuth = _is_sunny(y_point_gps_i, x_point_gps_i, date, time_string, longitude, latitude)
                is_sunny_array[y_point_gps_i][x_point_gps_i] = is_sunny
        time_B = time.time()
        print(f'{time_B-time_A:.3f}')

        is_sunny_array = is_sunny_array[y_point_gps-L_shadow:y_point_gps+L_shadow,x_point_gps-L_shadow:x_point_gps+L_shadow]


        time_A = time.time()


        yx_points = np.empty((0,4))
        for y_point_idx, y_point_gps_i in enumerate(range(y_point_gps-steps, y_point_gps+steps)):
            for x_point_idx, x_point_gps_i in enumerate(range(x_point_gps-steps, x_point_gps+steps)):
                yx_points = np.append(yx_points, [[y_point_gps_i, x_point_gps_i, y_point_idx, x_point_idx]], axis=0)
        
        yx_points = yx_points.astype(int)
        print(np.shape(yx_points))

        rows = np.arange(steps*2+1).reshape(steps*2+1, 1)-steps
        cols = np.arange(steps*2+1).reshape(1, steps*2+1)-steps
        meshgrid_rows, meshgrid_cols = np.meshgrid(rows, cols)
        result_array = np.stack((meshgrid_cols,meshgrid_rows), axis=2)
        print(np.shape(result_array))
        result_array = result_array.reshape((np.shape(result_array)[0]*np.shape(result_array)[1],np.shape(result_array)[2]))
        print(np.shape(result_array))
        result_array_gps = result_array.copy()
        result_array_gps[:,0] += y_point_gps
        result_array_gps[:,1] += x_point_gps
        result_array_overall = np.concatenate((result_array_gps, result_array),axis=1)


        # for i in yx_points:
            # print(i)
        # quit()

        
        
        tif_data_samples = np.asarray([tif_data_full[coords[0]-L_shadow:coords[0]+L_shadow,coords[1]-L_shadow:coords[1]+L_shadow] for coords in yx_points])

        print(np.shape(tif_data_samples))
        plt.imshow(tif_data_samples[0])
        plt.savefig('A2')
        plt.close('all')

        rows = np.arange(L_shadow*2).reshape(L_shadow*2, 1)
        cols = np.arange(L_shadow*2).reshape(1, L_shadow*2)
        meshgrid_rows, meshgrid_cols = np.meshgrid(rows, cols)
        result_array = np.stack((meshgrid_cols,meshgrid_rows), axis=2)
        angles = compute_angle_from_north(L_shadow, L_shadow, result_array[:,:,0], result_array[:,:,1])
        plt.imshow(angles)
        plt.savefig('B2')
        plt.close('all')
        azimuth, altitude = compute_sun_direction(date, time_string, latitude, longitude)
        # where = select_elements_within_angle(angles, azimuth, 10, L_shadow)
        where = select_elements_without_angle(angles, azimuth, 10, L_shadow)

        def compute_distance(origin_x, origin_y, points_x, points_y):

            distance_x = points_x-origin_x
            distance_y = points_y-origin_y

            distance = np.sqrt(distance_x**2 + distance_y**2)/upsampling

            return distance

        heights = tif_data_full[y_point_gps-L_shadow:y_point_gps+L_shadow,x_point_gps-L_shadow:x_point_gps+L_shadow]

        # plt.imshow(heights)
        # plt.savefig('H2')
        # plt.close('all')

        distance_to_centre = compute_distance(L_shadow, L_shadow, result_array[:,:,0], result_array[:,:,1])
        # plt.imshow(distance_to_centre)
        # plt.savefig('Dist2')
        # plt.close('all')

        # HERE ALL GOOD

        # distance_to_centre = distance_to_centre[where]
        # tif_data_samples = tif_data_samples[:,where[1],where[0]]
        # tif_data_samples[:,where] = 0
        tif_data_samples[:,where[0],where[1]] = 0

        # plt.imshow(tif_data_samples[0])
        # plt.savefig('E2')
        # plt.close('all')
        # quit()


        # height_at_this_point = heights[result_array_overall[:,2], result_array_overall[:,3]]
        height_at_this_point = heights.reshape(((steps*2)**2))
        # print(np.shape(heights))
        # print(np.shape(result_array_overall))
        # print(np.shape(height_at_this_point))
        # print(np.shape(tif_data_samples))
        # quit()

        # tif_data_samples_diff = tif_data_samples-np.expand_dims(np.expand_dims(height_at_this_point,1),1)
        tif_data_samples_diff = tif_data_samples-np.expand_dims(np.expand_dims(tif_data_samples[:,steps,steps],1),1)

        print(np.shape(tif_data_samples_diff), np.shape(distance_to_centre))
        angle = np.degrees(np.arctan2(tif_data_samples_diff,np.expand_dims(distance_to_centre,0)))

        print(np.shape(angle))
        angle_max = np.amax(angle, axis=(1,2))
        print(np.shape(angle_max))

        where = np.where(angle_max>altitude) # where in dark
        is_sunny = np.ones(np.shape(angle_max))
        is_sunny[where] = 0.
        is_sunny = is_sunny.reshape((steps*2,steps*2))
        print(np.shape(is_sunny))

        plt.subplot(1,2,1)
        plt.imshow(is_sunny_array)
        plt.subplot(1,2,2)
        plt.imshow(is_sunny)
        plt.savefig("output.png")
        plt.show()

        print(np.sum(is_sunny_array))
        print(np.sum(is_sunny))

        quit()


        # for coords in yx_points:
        #     array_i = tif_data_full[coords[0]-L_shadow:coords[0]+L_shadow,coords[1]-L_shadow:coords[1]+L_shadow]
        #     # print(np.shape(array_i))
        #     array_i[where] = 0.
        #     # print(np.shape(array_i))
        #     plt.imshow(array_i)
        #     plt.show()
        #     quit()

        # quit()

        tif_data_samples = np.asarray([[coords[2], coords[3], np.amax(tif_data_full[coords[0]-L_shadow:coords[0]+L_shadow,coords[1]-L_shadow:coords[1]+L_shadow][where])] for coords in yx_points])
        # print(np.shape(tif_data_samples))

        

        where = np.where(tif_data_samples[:,2]>altitude)
        tif_data_samples[:,2] = np.ones(np.shape(tif_data_samples[:,2]))
        tif_data_samples[where] = 0


        # print(tif_data_samples[0])
        square_array = np.zeros((np.max(tif_data_samples[:, 0].astype(int))+1,np.max(tif_data_samples[:, 0].astype(int))+1))
        print(np.shape(square_array))
        square_array[tif_data_samples[:, 0].astype(int), tif_data_samples[:, 2].astype(int)] = tif_data_samples[:,2]

        print(np.shape(square_array))

        plt.imshow(square_array)
        plt.show()

        # print(where)
        quit()
        angles = np.expand_dims(angles, 0)
        angles = np.repeat(angles, np.shape(tif_data_samples)[0], axis=0)

        print(np.shape(tif_data_samples), np.shape(angles))






        quit()

        is_sunny, azimuth = _is_sunny_vec(y_point_i, x_point_i, date, time_string, longitude, latitude)
        is_sunny_array[y_point_i][x_point_i] = is_sunny
        time_B = time.time()
        print(f'{time_B-time_A:.3f}')
        quit()



        N_pixels = 30*upsampling # number of metres around point to plot
        # tif_data_i = tif_data_full[y_point_gps-N_pixels:y_point_gps+N_pixels,x_point_gps-N_pixels:x_point_gps+N_pixels]
        tif_data_i = tif_data_plain_upsampling[y_point_gps-N_pixels:y_point_gps+N_pixels,x_point_gps-N_pixels:x_point_gps+N_pixels]
        is_sunny_array_i = is_sunny_array[y_point_gps-N_pixels:y_point_gps+N_pixels,x_point_gps-N_pixels:x_point_gps+N_pixels]

        # is_sunny_array_i = gaussian_filter(is_sunny_array_i, sigma=0.5)
        is_sunny_array_i = smooth_image(is_sunny_array_i, size=4)

        
        is_sunny, azimuth = _is_sunny(y_point_gps, x_point_gps, date, time_string, longitude, latitude, make_plots=True)

        fig = plt.figure(figsize=(10, 6))
        fig.suptitle(f"{pub_name} {time_string} {date}", fontsize=14)

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
        plt.imshow(is_sunny_array_i, cmap=cmap, alpha=0.5)
        rect = patches.Rectangle((N_pixels-steps-0.5, N_pixels-steps-0.5), steps*2+1, steps*2+1, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)

        plt.tight_layout()        
        # plt.savefig(f'temp/plot_{time_iter}.png')
        plt.savefig(f'test_mode4x2.png')
        plt.close('all')
        quit()

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
clip.write_videofile(f"mp4/{output_gif}.mp4")

# os.system('rm temp/plot_*.png')
os.system('rm *.gif')


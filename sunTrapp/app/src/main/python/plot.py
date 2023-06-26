import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import io
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import ephem
from os.path import dirname, join
import pickle
from pyproj import Transformer
from affine import Affine
from PIL import Image
import imageio
from com.chaquo.python import Python
import os
from math import degrees
from PIL import ImageFilter
from matplotlib.colors import LogNorm
from matplotlib.colors import LinearSegmentedColormap


import sunTrapp.satellite
import sunTrapp.shadow_computer
import sunTrapp.utilities
import sunTrapp.image_tools
import sunTrapp.plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
import scipy.ndimage

sun_color = '#FFD700'  # Brighter yellow color for sun
shade_color = '#121110'  # Darker blue color for shade
colors = [shade_color,sun_color]
cmap = LinearSegmentedColormap.from_list('sun_shade_cmap', colors)


# data from https://environment.data.gov.uk/DefraDataDownload/?Mode=survey

def plot_func(latitude_in, longitude_in):

	loc = [float(longitude_in), float(latitude_in)]

	max_shadow_length = 75
	compute_size = [30, 30] # area 2N (x, y)
	compute_size[0] = int(compute_size[0]*(410/350))
	edge_buffer = 5
	output = f"plot"
	upsampling = 1
	date = '2023/06/22'

	###
	start_time = "16:00:00"
	end_time = "18:00:00"
	time_steps = 3
	time_string = sunTrapp.utilities.generate_time_stamps(start_time, end_time, time_steps)
	###

	#############################################
	print(f"shadow size: {(compute_size[0]*2+1)*upsampling}, {(compute_size[1]*2+1)*upsampling}")

	data, idx_raw = sunTrapp.utilities.get_organised_data(loc, compute_size, edge_buffer, max_shadow_length, upsampling, app=True, file_app=__file__)

	if upsampling != 1.:
		max_shadow_length *= upsampling
		compute_size[0] *= upsampling
		compute_size[1] *= upsampling
		edge_buffer *= upsampling

	x_idx = int(max_shadow_length+compute_size[0]+edge_buffer)
	y_idx = int(max_shadow_length+compute_size[1]+edge_buffer)

	data, data_raw = sunTrapp.image_tools.upsample(data, upsampling)

	cropped_satellite_image = sunTrapp.satellite.get_cropped_satellite_image(loc, idx_raw, compute_size, upsampling, app=True, file_app=__file__)


	if time_string.__class__ != list: time_string = [time_string]

	f = io.BytesIO()
	image_files = []

	for time_itr, time_string_i in enumerate(time_string):

		print(time_itr, time_string_i)

		shadow_calculator = sunTrapp.shadow_computer.shadow_computer(date, time_string_i, loc[0], loc[1], max_shadow_length, upsampling)
		shadows, region_of_interest = shadow_calculator.compute(data, x_idx, y_idx, compute_size, edge_buffer, upsampling)

		shadows = sunTrapp.image_tools.smooth_image(shadows)

		filename = str(join(dirname(__file__), f"{output}_{time_itr}.png"))
		
		overlay = cropped_satellite_image
		width = 8
		height = width * (np.shape(shadows)[0]/np.shape(shadows)[1])

		plt.figure(figsize=(width,height))
		ax = plt.gca()
		if overlay is not None:
			shadows = scipy.ndimage.zoom(shadows, np.shape(overlay)[0]/np.shape(shadows)[0], order=0)
			plt.imshow(overlay)
		plt.imshow(shadows, cmap=cmap, alpha=0.35, vmin=0, vmax=1)
		plt.yticks([],[])   
		plt.xticks([],[]) 
		if time_string is not None:
			plt.text(0.01, 0.97, f'Time: {time_string}', fontsize=20, horizontalalignment='left',verticalalignment='top', transform=ax.transAxes, c='w')
		if date is not None:
			plt.text(0.01, 0.90, f'Date: {date}', fontsize=20, horizontalalignment='left',verticalalignment='top', transform=ax.transAxes, c='w')
		plt.subplots_adjust(hspace=0,wspace=0)
		plt.tight_layout()        
		plt.savefig(join(dirname(__file__), f"{output}_{time_itr}.png"), transparent=True)
		plt.close('all')
		
		# sunTrapp.plotting.publish_plot(shadows, filename, cropped_satellite_image, time_string=time_string_i, date=date, show=False)

		# plt.imshow(shadows, cmap=cmap, alpha=0.5, vmin=0, vmax=1)    
		# plt.yticks([],[])   
		# plt.xticks([],[])   
		# ax = plt.gca()
		# plt.text(0.05, 0.95, latitude_in, c='w',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
		# plt.text(0.05, 0.85, longitude_in, c='w',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
		# plt.savefig(f, format="png")
		# plt.savefig(join(dirname(__file__), f"{output}_{time_itr}.png"))
		# plt.close()

		image_files.append(Image.open(join(dirname(__file__), f"{output}_{time_itr}.png")))
	
	context = Python.getPlatform().getApplication()
	file_path = os.path.join(context.getFilesDir().getAbsolutePath(), f"gif_file_{latitude_in}_{longitude_in}.gif")
		
	image_files[0].save(
		file_path,
		format="GIF",
		append_images=image_files[1:],
		save_all=True,
		duration=500,  # Time delay between frames in milliseconds
		loop=0,  # Number of loops (0 means infinite loop)
	)

	return f.getvalue()





# image_files = glob.glob(f"{output}_*.png")
# image_idx_max = 0
# for image in image_files:
# 	image_idx = image[len(output)+1:]
# 	image_idx = image_idx[:-4]
# 	if int(image_idx) > int(image_idx_max): image_idx_max = image_idx
# image_files = []
# for image_idx in range(1,int(image_idx_max)):
# 	image_files.append(f'{output}_{image_idx}.png')

# # Create a list to store the image frames
# frames = []

# # Read and append each image to the frames list
# for image_file in image_files:
# 	image = Image.open(image_file)
# 	frames.append(image)

# # Save frames as an animated GIF
# frames[0].save("gif/"+output_gif+".gif", format='GIF', append_images=frames[1:], save_all=True, duration=200*2, loop=0)
# clip = mp.VideoFileClip("gif/"+output_gif+".gif")
# clip.write_videofile(f"mp4/{output_gif}.mp4")












































# sun_color = '#FFD700'  # Brighter yellow color for sun
# shade_color = '#121110'  # Darker blue color for shade
# colors = [shade_color,sun_color]
# cmap = LinearSegmentedColormap.from_list('sun_shade_cmap', colors)

# # import sunTrapp.shadow_computer as shadow_computer
# # print(shadow_computer)


# def transform_rowcol(transformer, x_coords, y_coords):
# 	# inv_transformer = transformer.inverse
# 	inv_transformer = ~transformer
# 	row_col_tuples = []
# 	for x, y in zip([x_coords], [y_coords]):
# 		row, col = inv_transformer * (x, y)
# 		row, col = int(row), int(col)
# 		row_col_tuples.append((row, col))
# 	row_col_tuples = row_col_tuples[0]
# 	return row_col_tuples[1], row_col_tuples[0]

# def compute_angle_from_north(origin_x, origin_y, points_x, points_y, upsampling=1.):

# 	distance_x = (points_x-origin_x)/upsampling
# 	distance_y = (points_y-origin_y)/upsampling

# 	angles = np.degrees(np.arctan2(distance_x,distance_y))+90
# 	angles = angles % 360

# 	return angles

# def compute_sun_direction(date, time, latitude, longitude):
# 	observer = ephem.Observer()
# 	observer.lat = str(latitude)
# 	observer.lon = str(longitude)
# 	observer.date = date + ' ' + time

# 	sun = ephem.Sun()
# 	sun.compute(observer)

# 	azimuth = degrees(sun.az)
# 	altitude = degrees(sun.alt)

# 	return azimuth, altitude

# def select_elements_without_angle(array, angle, cone=5, N_surrounding_pixels=50):

# 	array[np.where(array<0)] += 360

# 	adjusted_angle = angle % 360

# 	lower_bound = adjusted_angle - cone
# 	upper_bound = adjusted_angle + cone

# 	if lower_bound < 0:
# 		where = np.where(np.logical_and(array > upper_bound, array < 360 + lower_bound))
# 	elif upper_bound > 360:
# 		where = np.where(np.logical_and(array < lower_bound, array > upper_bound - 360))
# 	else:
# 		where = np.where(np.logical_or(array < lower_bound, array > upper_bound))

# 	row_to_remove = [N_surrounding_pixels, N_surrounding_pixels]
# 	to_delete = np.where((where[0] == row_to_remove[0]) & (where[1] == row_to_remove[1]))
# 	where = np.delete(where, to_delete, axis=1)
# 	where = (np.asarray(where[0]), np.asarray(where[1]))

# 	return where

# def compute_distance(origin_x, origin_y, points_x, points_y, upsampling=1.):

# 	distance_x = points_x-origin_x
# 	distance_y = points_y-origin_y

# 	distance = np.sqrt(distance_x**2 + distance_y**2)/upsampling

# 	return distance

# def smooth_image(image_array, size=3):
# 	_min = np.amin(image_array)
# 	_max = np.amax(image_array)
# 	image_array += -_min
# 	image_array *= (1./(_max-_min))
# 	image = Image.fromarray(np.uint8(image_array*255))
# 	image = image.filter(ImageFilter.ModeFilter(size=size))
# 	image_array = np.array(image)
# 	image_array = ((image_array*_max)+_min)/255
# 	return image_array

# def plot_func(latitude_in, longitude_in):

# 	# plot tif file
# 	filename_npy = join(dirname(__file__), "DSM_ST5570_P_10631_20190117_20190117.npy")
# 	tif_data = np.load(filename_npy,allow_pickle=True)
# 	transformer_toGPS = Transformer.from_crs("EPSG:27700", "EPSG:4326")
# 	filename_transformer = join(dirname(__file__), "DSM_ST5570_P_10631_20190117_20190117_transformer.pkl")
# 	with open(filename_transformer, 'rb') as f:
# 		transformer = pickle.load(f)

# 	tif_data_full = tif_data.copy()

# 	upsampling = 1.

# 	pub = [float(longitude_in), float(latitude_in)] # hope and anchor
		
# 	transformer_toOS = Transformer.from_crs("EPSG:4326", "EPSG:27700")
# 	idx = transformer_toOS.transform(pub[0],pub[1])
# 	idx = (idx[1], -idx[0])
# 	y_point_gps, x_point_gps = transform_rowcol(transformer, -idx[1], idx[0])
# 	print(y_point_gps, x_point_gps)

# 	date = '2023/06/22'  # Date format: yyyy/mm/dd
# 	latitude = f'{pub[0]}'   # Latitude of the location
# 	longitude = f'{pub[1]}'  # Longitude of the location
	
# 	steps = 15*upsampling # size of area to compute shadows
# 	buffer = int(steps/4.)

# 	time_iter = 0

# 	minute = 36

# 	f = io.BytesIO()
# 	image_files = []
# 	# for hour in [15,16,17]:
# 	for hour in range(6,21):

# 		time_iter += 1

# 		time_string = f'{hour}:{minute}:00'

# 		L_shadow = 50

# 		yx_points = np.empty((0,4))
# 		for y_point_idx, y_point_gps_i in enumerate(range(int(y_point_gps-steps), int(y_point_gps+steps+1))):
# 			for x_point_idx, x_point_gps_i in enumerate(range(int(x_point_gps-steps), int(x_point_gps+steps+1))):
# 				yx_points = np.append(yx_points, [[y_point_gps_i, x_point_gps_i, y_point_idx, x_point_idx]], axis=0)
		
# 		yx_points = yx_points.astype(int)

# 		tif_data_samples = np.asarray([tif_data_full[coords[0]-L_shadow:coords[0]+L_shadow,coords[1]-L_shadow:coords[1]+L_shadow] for coords in yx_points])

# 		rows = np.arange(L_shadow*2).reshape(L_shadow*2, 1)
# 		cols = np.arange(L_shadow*2).reshape(1, L_shadow*2)
# 		meshgrid_rows, meshgrid_cols = np.meshgrid(rows, cols)
# 		result_array = np.stack((meshgrid_cols,meshgrid_rows), axis=2)
# 		angles = compute_angle_from_north(L_shadow, L_shadow, result_array[:,:,0], result_array[:,:,1], upsampling=upsampling)

# 		azimuth, altitude = compute_sun_direction(date, time_string, latitude, longitude)
# 		where = select_elements_without_angle(angles, azimuth, 10, L_shadow)

# 		distance_to_centre = compute_distance(L_shadow, L_shadow, result_array[:,:,0], result_array[:,:,1], upsampling=upsampling)
# 		tif_data_samples[:,where[0],where[1]] = 0

# 		tif_data_samples_diff = tif_data_samples-np.expand_dims(np.expand_dims(tif_data_samples[:,L_shadow,L_shadow],1),1)

# 		angle = np.degrees(np.arctan2(tif_data_samples_diff,np.expand_dims(distance_to_centre,0)))

# 		angle_max = np.amax(angle, axis=(1,2))

# 		where = np.where(angle_max>altitude) # where in dark
# 		is_sunny = np.ones(np.shape(angle_max))
# 		is_sunny[where] = 0.
# 		is_sunny = is_sunny.reshape((int(steps*2+1),int(steps*2+1)))

# 		is_sunny = smooth_image(is_sunny, size=4)


# 		is_sunny_pad = np.pad(is_sunny, buffer)
# 		tif_data_i = tif_data[int(y_point_gps-steps-buffer):int(y_point_gps+steps+1+buffer),int(x_point_gps-steps-buffer):int(x_point_gps+steps+1+buffer)]
		
		
# 		plt.imshow(tif_data_i, norm=LogNorm())
# 		plt.imshow(is_sunny_pad, cmap=cmap, alpha=0.5, vmin=0, vmax=1)    
# 		plt.yticks([],[])   
# 		plt.xticks([],[])   
# 		ax = plt.gca()
# 		plt.text(0.05, 0.95, latitude_in, c='w',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
# 		plt.text(0.05, 0.85, longitude_in, c='w',horizontalalignment='left',verticalalignment='top', transform=ax.transAxes)
# 		plt.savefig(f, format="png")
# 		plt.savefig(join(dirname(__file__), f"output_{time_iter}.png"))
# 		plt.close()
# 		image_files.append(Image.open(join(dirname(__file__), f"output_{time_iter}.png")))

# 	# f = io.BytesIO()
# 	# plt.imshow(I)
# 	# plt.savefig(f, format="png")
# 	# plt.savefig(join(dirname(__file__), "output.png"))
# 	# return f.getvalue()








	
# 	# f = io.BytesIO()
# 	# image_files = []
# 	# for i in range(5):
# 	# 	data = np.random.normal(0,1,(1000,2))
# 	# 	plt.hist2d(data[:, 0], data[:, 1], bins=25, range=[[-5, 5], [-5, 5]])
# 	# 	plt.savefig(join(dirname(__file__), f"output_{i}.png"))
# 	# 	plt.savefig(f, format="png")
# 	# 	plt.close()
# 	# 	image_files.append(Image.open(join(dirname(__file__), f"output_{i}.png")))

# 	context = Python.getPlatform().getApplication()
# 	file_path = os.path.join(context.getFilesDir().getAbsolutePath(), f"gif_file_{latitude_in}_{longitude_in}.gif")
		
# 	image_files[0].save(
# 		file_path,
# 		format="GIF",
# 		append_images=image_files[1:],
# 		save_all=True,
# 		duration=500,  # Time delay between frames in milliseconds
# 		loop=0,  # Number of loops (0 means infinite loop)
# 	)

# 	return f.getvalue()





# 	# # plot tif file
# 	# filename_npy = join(dirname(__file__), "DSM_ST5570_P_10631_20190117_20190117.npy")
# 	# I = np.load(filename_npy,allow_pickle=True)

# 	# transformer_toGPS = Transformer.from_crs("EPSG:27700", "EPSG:4326")

# 	# filename_transformer = join(dirname(__file__), "DSM_ST5570_P_10631_20190117_20190117_transformer.pkl")
# 	# with open(filename_transformer, 'rb') as f:
# 	# 	transformer = pickle.load(f)

# 	# f = io.BytesIO()
# 	# plt.imshow(I)
# 	# plt.savefig(f, format="png")
# 	# plt.savefig(join(dirname(__file__), "output.png"))
# 	# return f.getvalue()

	

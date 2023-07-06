from pyproj import Transformer
import datetime
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
import geopy.distance
import sunTrapp.image_tools
from math import radians, sin, cos, sqrt, atan2
import pickle
from os.path import dirname, join
from pyproj import Transformer

def no_floor(values):
	return values

def convert_boundaries_GPS_to_relative_pixels(loc, boundaries, upsampling, transform, centre_indexes):

	transformer_toOS = Transformer.from_crs("EPSG:4326", "EPSG:27700")
	boundaries_pixels = np.empty((0,2))
	for boundary_point in boundaries:
		idx = transformer_toOS.transform(boundary_point[0],boundary_point[1])
		idx = (idx[1], -idx[0])
		y_point_gps_boundary, x_point_gps_boundary = transform_rowcol(transform, -idx[1], idx[0])#, op=no_floor)
		y_point_gps_boundary*=upsampling
		x_point_gps_boundary*=upsampling
		y_point_gps_boundary+= -centre_indexes[0]*upsampling
		x_point_gps_boundary+= -centre_indexes[1]*upsampling
		boundaries_pixels = np.append(boundaries_pixels, [[y_point_gps_boundary, x_point_gps_boundary]], axis=0)

	return boundaries_pixels




def remove_outer_regions(array, buffer=2):
		# Find the indices of the non-zero elements along each axis
		non_zero_rows = np.nonzero(np.any(array != -1, axis=1))[0]
		non_zero_cols = np.nonzero(np.any(array != -1, axis=0))[0]

		# Slice the array to remove the featureless outer regions
		result = array[non_zero_rows[0]-buffer:non_zero_rows[-1]+1+buffer, non_zero_cols[0]-buffer:non_zero_cols[-1]+1+buffer]
		
		return result





def check_coordinate_within_bounds(coordinate, bounds):
	top_left = bounds['top_left']
	top_right = bounds['top_right']
	bottom_left = bounds['bottom_left']
	bottom_right = bounds['bottom_right']

	if (coordinate[0] >= bottom_left[0] and coordinate[0] <= top_left[0] and
			coordinate[1] >= top_left[1] and coordinate[1] <= top_right[1]):
		return True
	else:
		return False
		
def get_organised_data(loc, compute_size, edge_buffer, max_shadow_length, upsampling, app=False, file_app=None):

	image_height = ((compute_size[1]+edge_buffer+max_shadow_length)*2+1)*upsampling
	image_width = ((compute_size[0]+edge_buffer+max_shadow_length)*2+1)*upsampling
	print(f"full image size: {image_width}, {image_height}")

	if app:
		with open(join(dirname(file_app), "bounds.json"), "r") as json_file:
			bounds = json.load(json_file)
	else:
		with open("tif/bounds.json", "r") as json_file:
			bounds = json.load(json_file)


	file = False
	for file_str in bounds:

		if check_coordinate_within_bounds(loc, bounds[file_str]):
			file = file_str
			file_info = bounds[file_str]
			break
	
	if app: data, transform = open_tif_file_app(file, file_app)
	else: data, transform, dataset = open_tif_file(file)
	x_idx, y_idx, idx_raw = get_centre_indexes(loc, transform, 1., app)
	centre_indexes = (y_idx, x_idx)


	top = y_idx - (int(image_height/2.)+2)
	bottom = y_idx + (int(image_height/2.)+2)
	right = x_idx + (int(image_height/2.)+2)
	left = x_idx - (int(image_height/2.)+2)

	top_left = transform * (left, top)  # Multiply transform with pixel indices
	top_right = transform * (right, top)
	bottom_left = transform * (left, bottom)
	bottom_right = transform * (right, bottom)

	transformer_toGPS = Transformer.from_crs("EPSG:27700", "EPSG:4326")
	top_left = transformer_toGPS.transform(top_left[0], top_left[1])
	top_right = transformer_toGPS.transform(top_right[0], top_right[1])
	bottom_left = transformer_toGPS.transform(bottom_left[0], bottom_left[1])
	bottom_right = transformer_toGPS.transform(bottom_right[0], bottom_right[1])

	corners = {}
	corners["top_left"] = check_coordinate_within_bounds(top_left, bounds[file])
	corners["top_right"] = check_coordinate_within_bounds(top_right, bounds[file])
	corners["bottom_left"] = check_coordinate_within_bounds(bottom_left, bounds[file])
	corners["bottom_right"] = check_coordinate_within_bounds(bottom_right, bounds[file])

	#   A
	# D   B
	#   C
	padding_size = 5050
	shape_data = np.shape(data)
	x_idx += padding_size
	y_idx += padding_size
	data_pad = np.pad(data, padding_size)

	if corners["top_left"] == False and corners["top_right"] == False:

		for file_str_new in bounds:
			file_info = bounds[file_str]
			file_info_new = bounds[file_str_new]

			distance = geopy.distance.geodesic(file_info["top_left"], file_info_new["bottom_left"]).m

			if distance < 50:
				
				if app: data_new, transform_new = open_tif_file_app(file_str_new, file_app)
				else: data_new, transform_new, dataset_new = open_tif_file(file_str_new)

				up_down_start = padding_size-np.shape(data_new)[0] # top left
				left_right_start = padding_size # top left
				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

		if corners["bottom_left"] == False:

				distance = geopy.distance.geodesic(file_info["top_left"], file_info_new["bottom_right"]).m

				if distance < 50:
				
					if app: data_new, transform_new = open_tif_file_app(file_str_new, file_app)
					else: data_new, transform_new, dataset_new = open_tif_file(file_str_new)

					up_down_start = padding_size-np.shape(data_new)[0] # top left
					left_right_start = padding_size-np.shape(data_new)[1] # top left
					data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new



	if corners["top_right"] == False and corners["bottom_right"] == False:
		
		for file_str_new in bounds:
			file_info = bounds[file_str]
			file_info_new = bounds[file_str_new]

			distance = geopy.distance.geodesic(file_info["top_right"], file_info_new["top_left"]).m

			if distance < 50:
				
				if app: data_new, transform_new = open_tif_file_app(file_str_new, file_app)
				else: data_new, transform_new, dataset_new = open_tif_file(file_str_new)

				up_down_start = padding_size # top left
				left_right_start = padding_size+shape_data[1] # top left
				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

			if corners["top_left"] == False:

				distance = geopy.distance.geodesic(file_info["top_right"], file_info_new["bottom_left"]).m

				if distance < 50:
					
					if app: data_new, transform_new = open_tif_file_app(file_str_new, file_app)
					else: data_new, transform_new, dataset_new = open_tif_file(file_str_new)

					up_down_start = padding_size-np.shape(data_new)[0] # top left
					left_right_start = padding_size+shape_data[1] # top left
					data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new


	if corners["bottom_left"] == False and corners["bottom_right"] == False:

		for file_str_new in bounds:
			file_info = bounds[file_str]
			file_info_new = bounds[file_str_new]

			distance = geopy.distance.geodesic(file_info["bottom_right"], file_info_new["top_right"]).m

			if distance < 50:
				
				if app: data_new, transform_new = open_tif_file_app(file_str_new, file_app)
				else: data_new, transform_new, dataset_new = open_tif_file(file_str_new)

				up_down_start = padding_size+shape_data[0] # top left
				left_right_start = padding_size # top left
				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

			if corners["top_right"] == False:

				distance = geopy.distance.geodesic(file_info["bottom_right"], file_info_new["top_left"]).m

				if distance < 50:
				
					if app: data_new, transform_new = open_tif_file_app(file_str_new, file_app)
					else: data_new, transform_new, dataset_new = open_tif_file(file_str_new)

					up_down_start = padding_size+shape_data[0] # top left
					left_right_start = padding_size+np.shape(data_new)[1] # top left
					data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new


	if corners["bottom_left"] == False and corners["top_left"] == False:

		for file_str_new in bounds:
			file_info = bounds[file_str]
			file_info_new = bounds[file_str_new]

			distance = geopy.distance.geodesic(file_info["bottom_left"], file_info_new["bottom_right"]).m

			if distance < 50:
				
				if app: data_new, transform_new = open_tif_file_app(file_str_new, file_app)
				else: data_new, transform_new, dataset_new = open_tif_file(file_str_new)

				up_down_start = padding_size # top left
				left_right_start = padding_size-np.shape(data_new)[1] # top left
				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

			if corners["bottom_right"] == False:

				distance = geopy.distance.geodesic(file_info["bottom_left"], file_info_new["top_right"]).m

				if distance < 50:
				
					if app: data_new, transform_new = open_tif_file_app(file_str_new, file_app)
					else: data_new, transform_new, dataset_new = open_tif_file(file_str_new)

					up_down_start = padding_size+shape_data[0] # top left
					left_right_start = padding_size-np.shape(data_new)[1] # top left
					data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

	data = data_pad[int(y_idx-max_shadow_length-compute_size[1]-edge_buffer):int(y_idx+max_shadow_length+compute_size[1]+edge_buffer+1),int(x_idx-max_shadow_length-compute_size[0]-edge_buffer):int(x_idx+max_shadow_length+compute_size[0]+edge_buffer+1)]

	data[np.where(data<-1E3)] = -99
	data = sunTrapp.image_tools.replace_zeros(data, value=-99)
	
	return data, idx_raw, transform, centre_indexes


def index_tif_files(fileNames, json_filepath):

	tif_files = glob.glob(fileNames)
	
	bounds = {}

	for file in tif_files:

		tif_i_data, tif_i_transform, tif_i_dataset = open_tif_file(file)

		# Get the raster dimensions
		width = tif_i_dataset.width
		height = tif_i_dataset.height

		# Calculate the GPS coordinates of the four corners
		top_left = tif_i_transform * (0, 0)  # Multiply transform with pixel indices
		top_right = tif_i_transform * (width, 0)
		bottom_left = tif_i_transform * (0, height)
		bottom_right = tif_i_transform * (width, height)

		transformer_toGPS = Transformer.from_crs("EPSG:27700", "EPSG:4326")
		top_left = transformer_toGPS.transform(top_left[0], top_left[1])
		top_right = transformer_toGPS.transform(top_right[0], top_right[1])
		bottom_left = transformer_toGPS.transform(bottom_left[0], bottom_left[1])
		bottom_right = transformer_toGPS.transform(bottom_right[0], bottom_right[1])
		
		bounds[file] = {}
		bounds[file]["top_left"] = top_left
		bounds[file]["top_right"] = top_right
		bounds[file]["bottom_left"] = bottom_left
		bounds[file]["bottom_right"] = bottom_right
		bounds[file]["width"] = width
		bounds[file]["height"] = height

	with open(json_filepath, "w") as json_file:
		json.dump(bounds, json_file)


def copy_npy(fileNames, save_filepath):

	tif_files = glob.glob(fileNames)
	
	bounds = {}

	for file in tif_files:
		
		tif_i_data, tif_i_transform, tif_i_dataset = open_tif_file(file)

		file_root = file[4:-4]
		np.save(f'{save_filepath}/{file_root}.npy',tif_i_data)
		with open(f'{save_filepath}/{file_root}_transformer.pkl', 'wb') as f:
		    pickle.dump(tif_i_transform, f)

def open_tif_file_app(fileName, file_app):
	# Data from https://environment.data.gov.uk/DefraDataDownload/?Mode=survey
	fileName = fileName[4:-4]
	filename_npy = join(dirname(file_app), fileName+".npy")
	data = np.load(filename_npy,allow_pickle=True)
	filename_transformer = join(dirname(file_app), fileName+"_transformer.pkl")
	with open(filename_transformer, 'rb') as f:
		transform = pickle.load(f)
	return data, transform

def open_tif_file(fileName):
	import rasterio # hack - avoid importing in app
	# Data from https://environment.data.gov.uk/DefraDataDownload/?Mode=survey
	dataset = rasterio.open(fileName)
	data = dataset.read(1)
	transform = dataset.transform
	return data, transform, dataset

def transform_rowcol(transformer, x_coords, y_coords, no_floor=True):
	# inv_transformer = transformer.inverse
	inv_transformer = ~transformer
	row_col_tuples = []
	for x, y in zip([x_coords], [y_coords]):
		row, col = inv_transformer * (x, y)
		if no_floor == False:
			row, col = int(row), int(col)
		row_col_tuples.append((row, col))
	row_col_tuples = row_col_tuples[0]
	return row_col_tuples[1], row_col_tuples[0]

def get_centre_indexes(loc, transform, upsampling, app):
	
	if app:
		transformer_toOS = Transformer.from_crs("EPSG:4326", "EPSG:27700")
		idx_raw = transformer_toOS.transform(loc[0],loc[1])
		idx = (idx_raw[1], -idx_raw[0])
		y, x = transform_rowcol(transform, -idx[1], idx[0])
		y*=upsampling
		x*=upsampling
	else:
		import rasterio # hack - avoid importing in app
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

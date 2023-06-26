import rasterio
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

def get_organised_data(loc, compute_size, edge_buffer, max_shadow_length, upsampling):

	image_height = ((compute_size[1]+edge_buffer+max_shadow_length)*2+1)*upsampling
	image_width = ((compute_size[0]+edge_buffer+max_shadow_length)*2+1)*upsampling
	print(f"full image size: {image_width}, {image_height}")


	with open("tif/bounds.json", "r") as json_file:
		bounds = json.load(json_file)

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
		

	file = False
	for file_str in bounds:
		# print(check_coordinate_within_bounds(loc, data[file_str]), file_str)
		if check_coordinate_within_bounds(loc, bounds[file_str]):
			file = file_str
			file_info = bounds[file_str]
			break

	data, transform, dataset = open_tif_file(file)
	x_idx, y_idx, idx_raw = get_centre_indexes(loc, transform, 1.)



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

				data_new, transform_new, dataset_new = open_tif_file(file_str_new)

				up_down_start = padding_size-np.shape(data_new)[0] # top left
				left_right_start = padding_size # top left
				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

		if corners["bottom_left"] == False:

				distance = geopy.distance.geodesic(file_info["top_left"], file_info_new["bottom_right"]).m

				if distance < 50:
				
					data_new, transform_new, dataset_new = open_tif_file(file_str_new)
					
					up_down_start = padding_size-np.shape(data_new)[0] # top left
					left_right_start = padding_size-np.shape(data_new)[1] # top left
					data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new



	if corners["top_right"] == False and corners["bottom_right"] == False:
		
		for file_str_new in bounds:
			file_info = bounds[file_str]
			file_info_new = bounds[file_str_new]

			distance = geopy.distance.geodesic(file_info["top_right"], file_info_new["top_left"]).m

			if distance < 50:
				
				data_new, transform_new, dataset_new = open_tif_file(file_str_new)

				up_down_start = padding_size # top left
				left_right_start = padding_size+shape_data[1] # top left
				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

			if corners["top_left"] == False:

				distance = geopy.distance.geodesic(file_info["top_right"], file_info_new["bottom_left"]).m

				if distance < 50:
					
					data_new, transform_new, dataset_new = open_tif_file(file_str_new)
					
					up_down_start = padding_size-np.shape(data_new)[0] # top left
					left_right_start = padding_size+shape_data[1] # top left
					data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new


	if corners["bottom_left"] == False and corners["bottom_right"] == False:

		for file_str_new in bounds:
			file_info = bounds[file_str]
			file_info_new = bounds[file_str_new]

			distance = geopy.distance.geodesic(file_info["bottom_right"], file_info_new["top_right"]).m

			if distance < 50:
				
				data_new, transform_new, dataset_new = open_tif_file(file_str_new)
				
				up_down_start = padding_size+shape_data[0] # top left
				left_right_start = padding_size # top left
				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

			if corners["top_right"] == False:

				distance = geopy.distance.geodesic(file_info["bottom_right"], file_info_new["top_left"]).m

				if distance < 50:
				
					data_new, transform_new, dataset_new = open_tif_file(file_str_new)
					
					up_down_start = padding_size+shape_data[0] # top left
					left_right_start = padding_size+np.shape(data_new)[1] # top left
					data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new


	if corners["bottom_left"] == False and corners["top_left"] == False:

		for file_str_new in bounds:
			file_info = bounds[file_str]
			file_info_new = bounds[file_str_new]

			distance = geopy.distance.geodesic(file_info["bottom_left"], file_info_new["bottom_right"]).m

			if distance < 50:
				
				data_new, transform_new, dataset_new = open_tif_file(file_str_new)
				
				up_down_start = padding_size # top left
				left_right_start = padding_size-np.shape(data_new)[1] # top left
				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

			if corners["bottom_right"] == False:

				distance = geopy.distance.geodesic(file_info["bottom_left"], file_info_new["top_right"]).m

				if distance < 50:
				
					data_new, transform_new, dataset_new = open_tif_file(file_str_new)
					
					up_down_start = padding_size+shape_data[0] # top left
					left_right_start = padding_size-np.shape(data_new)[1] # top left
					data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

	data = data_pad[int(y_idx-max_shadow_length-compute_size[1]-edge_buffer):int(y_idx+max_shadow_length+compute_size[1]+edge_buffer+1),int(x_idx-max_shadow_length-compute_size[0]-edge_buffer):int(x_idx+max_shadow_length+compute_size[0]+edge_buffer+1)]
	if upsampling != 1.:
		max_shadow_length *= upsampling
		compute_size *= upsampling
		edge_buffer *= upsampling
	x_idx = int(max_shadow_length+compute_size[0]+edge_buffer)
	y_idx = int(max_shadow_length+compute_size[1]+edge_buffer)

	data[np.where(data<-1E3)] = -99
	data = sunTrapp.image_tools.replace_zeros(data, value=-99)
	
	return data, x_idx, y_idx, idx_raw


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



def open_tif_file(fileName):
	# Data from https://environment.data.gov.uk/DefraDataDownload/?Mode=survey
	dataset = rasterio.open(fileName)
	data = dataset.read(1)
	transform = dataset.transform
	return data, transform, dataset

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

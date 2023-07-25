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
from PIL import Image


import sunTrapp.satellite
import sunTrapp.shadow_computer
import sunTrapp.utilities
import sunTrapp.image_tools
import sunTrapp.plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from PIL import Image
import os
import glob
import requests
import zipfile
import json
from OSGridConverter import latlong2grid
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import Select
import time
import pickle



def no_floor(values):
	return values

def convert_boundaries_GPS_to_relative_pixels(loc, boundaries, upsampling, transform, centre_indexes):

	print('centre_indexes', centre_indexes)
	transformer_toOS = Transformer.from_crs("EPSG:4326", "EPSG:27700")
	boundaries_pixels = np.empty((0,2))
	for boundary_point in boundaries:
		print('\n')
		print(boundary_point)
		idx = transformer_toOS.transform(boundary_point[0],boundary_point[1])
		print(idx)
		idx = (idx[1], -idx[0])
		y_point_gps_boundary, x_point_gps_boundary = transform_rowcol(transform, -idx[1], idx[0])#, op=no_floor)
		print(y_point_gps_boundary, x_point_gps_boundary)
		y_point_gps_boundary*=upsampling
		x_point_gps_boundary*=upsampling
		print(y_point_gps_boundary, x_point_gps_boundary)
		y_point_gps_boundary+= -centre_indexes[0]*upsampling
		x_point_gps_boundary+= -centre_indexes[1]*upsampling
		print(y_point_gps_boundary, x_point_gps_boundary)
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




def get_organised_data_from_png(loc, compute_size, edge_buffer, max_shadow_length, upsampling, app=False, file_app=None):

	image_height = ((compute_size[1]+edge_buffer+max_shadow_length)*2+1)*upsampling
	image_width = ((compute_size[0]+edge_buffer+max_shadow_length)*2+1)*upsampling
	print(f"full image size: {image_width}, {image_height}")

	try:
		if app:
			with open(join(dirname(file_app), "bounds.json"), "r") as json_file:
				bounds = json.load(json_file)
		else:
			with open("tif/bounds.json", "r") as json_file:
				bounds = json.load(json_file)
	except:
		bounds = None

	file = False
	if bounds:
		for file_str in bounds:

			if check_coordinate_within_bounds(loc, bounds[file_str]):
				file = file_str
				file_info = bounds[file_str]
				break
	
	if file == False:
		print("Downloading file...")
		# return None, None, None, None
		file = download_png(loc[0], loc[1], output_location=file_app.replace('BUFFER',''), plot=False)

	if app: data, transform = open_tif_file_app_png(file, file_app)
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

	if not bounds:
		if app:
			with open(join(dirname(file_app), "bounds.json"), "r") as json_file:
				bounds = json.load(json_file)
		else:
			with open("tif/bounds.json", "r") as json_file:
				bounds = json.load(json_file)

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
				
				if app: data_new, transform_new = open_tif_file_app_png(file_str_new, file_app)
				else: data_new, transform_new, dataset_new = open_tif_file(file_str_new)

				up_down_start = padding_size-np.shape(data_new)[0] # top left
				left_right_start = padding_size # top left
				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

		if corners["bottom_left"] == False:

				distance = geopy.distance.geodesic(file_info["top_left"], file_info_new["bottom_right"]).m

				if distance < 50:
				
					if app: data_new, transform_new = open_tif_file_app_png(file_str_new, file_app)
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
				
				if app: data_new, transform_new = open_tif_file_app_png(file_str_new, file_app)
				else: data_new, transform_new, dataset_new = open_tif_file(file_str_new)

				up_down_start = padding_size # top left
				left_right_start = padding_size+shape_data[1] # top left
				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

			if corners["top_left"] == False:

				distance = geopy.distance.geodesic(file_info["top_right"], file_info_new["bottom_left"]).m

				if distance < 50:
					
					if app: data_new, transform_new = open_tif_file_app_png(file_str_new, file_app)
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
				
				if app: data_new, transform_new = open_tif_file_app_png(file_str_new, file_app)
				else: data_new, transform_new, dataset_new = open_tif_file(file_str_new)

				up_down_start = padding_size+shape_data[0] # top left
				left_right_start = padding_size # top left
				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

			if corners["top_right"] == False:

				distance = geopy.distance.geodesic(file_info["bottom_right"], file_info_new["top_left"]).m

				if distance < 50:
				
					if app: data_new, transform_new = open_tif_file_app_png(file_str_new, file_app)
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
				
				if app: data_new, transform_new = open_tif_file_app_png(file_str_new, file_app)
				else: data_new, transform_new, dataset_new = open_tif_file(file_str_new)

				up_down_start = padding_size # top left
				left_right_start = padding_size-np.shape(data_new)[1] # top left
				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

			if corners["bottom_right"] == False:

				distance = geopy.distance.geodesic(file_info["bottom_left"], file_info_new["top_right"]).m

				if distance < 50:
				
					if app: data_new, transform_new = open_tif_file_app_png(file_str_new, file_app)
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
	

def index_tif_file(file, PIL=False):
	
	bounds = {}

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

	return bounds



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

def open_tif_file_app_png(fileName, file_app):
	# Data from https://environment.data.gov.uk/DefraDataDownload/?Mode=survey
	fileName = fileName[:-4]
	filename_npy = join(dirname(file_app), fileName+".npy")
	filename_png = join(dirname(file_app), fileName+".png")


	loaded_image = Image.open(filename_png)
	loaded_array = np.array(loaded_image)
	min_max_value = np.load(filename_npy)
	min_value = min_max_value[0]
	max_value = min_max_value[1]

	# Renormalize the loaded array using the recorded minimum and maximum values
	data = (loaded_array.astype(np.float32) / 255.0) * (max_value - min_value) + min_value
	
	filename_transformer = join(dirname(file_app), fileName+".pkl")
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

def open_tif_file_PIL(fileName):
	dataset = Image.open(fileName)
	data = np.asarray(dataset)
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





def get_grid_coord(latitude, longitude):

	g=latlong2grid(latitude, longitude)
	g=str(g)

	grid_string = ''
	grid_string += g[:2]
	grid_string += g.split(' ')[1][0]+g.split(' ')[2][0]
	if int(g.split(' ')[2][1:]) < 5000: grid_string += 'S'
	else: grid_string += 'N'
	if int(g.split(' ')[1][1:]) < 5000: grid_string += 'W'
	else: grid_string += 'E'

	return grid_string


def extract_tif_files(zip_path):
	with zipfile.ZipFile(zip_path, 'r') as zip_ref:
		for file_name in zip_ref.namelist():
			if file_name.endswith('.tif'):
				zip_ref.extract(file_name)
		folder = zip_ref.infolist()[0].filename.split('/')[0]
	file_name_no_tif = file_name.split('/')[-1][:-4]
	os.system(f'mv {file_name} {file_name_no_tif}.tif')
	os.system(f'rm -r {folder}')
	return file_name_no_tif, file_name_no_tif+".tif"

def download_png(latitude, longitude, output_location= "/Users/am13743/Desktop/pub_gardens/tifs_as_png/", plot=False):

	url = get_file_name_from_latlong(latitude, longitude, viewer=False)

	try:
		r = requests.get(url, allow_redirects=True)
		open('download.zip', 'wb').write(r.content)
		file_name_no_tif, file_name = extract_tif_files("download.zip")
		os.system('rm download.zip')
	except:
		print("File not found")
		return 

	tif_i_data, tif_i_transform, tif_i_dataset = sunTrapp.utilities.open_tif_file(file_name)
	print("getting bounds")
	bounds = sunTrapp.utilities.index_tif_file(file_name, PIL=True)

	if not os.path.isfile(f"{output_location}/bounds.json"):
		with open(f"{output_location}/bounds.json", 'w') as file:
			json.dump(bounds, file)
	else:
		with open(f"{output_location}/bounds.json", 'r') as file:
			data = json.load(file)
		if list(bounds.keys())[0] not in data.keys():
			data[list(bounds.keys())[0] ] = bounds[list(bounds.keys())[0] ]
		with open(f"{output_location}/bounds.json", 'w') as file:
			json.dump(data, file, indent=4)

	# print("rm tif file")
	# os.system(f'rm {file_name}')



	min_other_than_1E3 = np.amin(tif_i_data[np.where(tif_i_data>-1E3)])
	tif_i_data[np.where(tif_i_data<-1E3)] = min_other_than_1E3-10

	array = tif_i_data

	# Record the minimum and maximum values
	min_value = np.min(array)
	max_value = np.max(array)

	# Normalize the array values to the range [0, 1]
	normalized_array = (array.astype(np.float32) - min_value) / (max_value-min_value)

	# Convert the normalized array to a grayscale image
	image = Image.fromarray((normalized_array * 255).astype(np.uint8), mode='L')

	if plot:
		plt.imshow(image, norm=LogNorm())
		plt.show()


	# Save the image as a PNG file

	with open(f'{output_location}/{file_name_no_tif}.pkl', 'wb') as handle:
		pickle.dump(tif_i_transform, handle, protocol=pickle.HIGHEST_PROTOCOL)

	image.save(f'{output_location}/{file_name_no_tif}.png')
	np.save(f'{output_location}/{file_name_no_tif}.npy',np.asarray([min_value, max_value]))
	os.system(f'ls -lh {output_location}/{file_name_no_tif}.png')     

	return file_name_no_tif+".tif"

def get_file_name_from_latlong(latitude, longitude, viewer=False):


	location = f"X:{longitude:.6f} Y:{latitude:.6f}"

	if viewer:
		driver = webdriver.Chrome() 
	else:
		op = webdriver.ChromeOptions()
		op.add_argument('headless')
		op.add_argument('window-size=1920x1080');
		driver = webdriver.Chrome(options=op)

	driver.get('https://environment.data.gov.uk/DefraDataDownload/?Mode=survey')  # Open Google Maps
	print("Page loading...")

	while 1 == 1:
		time.sleep(1)
		try:
			draw_button = driver.find_element(By.ID, 'polygon')
			time.sleep(3)
			draw_button.click()
			break
		except:
			pass

	print("Finding element")
	# Find the search box and enter the GPS location
	while 1 == 1:
		time.sleep(1)
		try:
			search_box = driver.find_element(By.ID, 'esri_dijit_Search_0_input')
			search_box.send_keys(location)
			search_box.send_keys(Keys.RETURN)
			break
		except:
			pass


	print("Selecting draw button")
	while 1 == 1:
		time.sleep(1)
		try:
			draw_button.click()
			break
		except:
			pass
	

	print("Clicking")
	# Create an ActionChains object
	actions = ActionChains(driver)

	# Define the coordinates for the points of the triangle
	triangle_points = [
		( -80, 50),  # 4 o'clock position
		( -70, 60),  # 4 o'clock position
		( -80, 50),  # 8 o'clock position
	]

	# Move the cursor to each point and click to draw the triangle
	for point_idx, point in enumerate(triangle_points):
		
		if point_idx == len(triangle_points) - 1:
			print("Double click...")
			actions.move_to_element_with_offset(driver.find_element(By.ID, 'map_graphics_layer'), point[0], point[1]).double_click().perform()            
		else:
			print("Click...")
			actions.move_to_element_with_offset(driver.find_element(By.ID, 'map_graphics_layer'), point[0], point[1]).click().perform()
		time.sleep(1)  # Wait a second between clicks (adjust as needed)


	print("Clicking through to downloads...")    
	while 1 == 1:
		time.sleep(1)
		try:
			draw_button = driver.find_element(By.ID, '0_projectImage')
			draw_button.click()
			break
		except:
			pass
	
	time.sleep(2)
	print("Selecting option from dropdown")    
	while 1 == 1:
		time.sleep(1)
		try:
			dropdown_element = driver.find_element(By.ID, 'productSelect')
			dropdown = Select(dropdown_element)
			for option in dropdown.options:
				if "National LIDAR Programme DSM" in option.text:
					value = option.text            
			dropdown.select_by_value(value);
			break
		except:
			pass

	while 1 == 1:
		time.sleep(1)
		try:
			dropdown_element = driver.find_element(By.ID, 'resolutionSelect')
			dropdown = Select(dropdown_element)
			for option in dropdown.options:
				if "1M" in option.text:
					value = option.text            
			dropdown.select_by_value(value);
			break
		except:
			pass


	print("Extracting url")    
	while 1 == 1:
		time.sleep(1)
		try:
			link_element = driver.find_element(By.CSS_SELECTOR, "a[href*='https://environment.data.gov.uk/UserDownloads/interactive/']")
			link_url = link_element.get_attribute('href')
			break
		except:
			pass
	
	print(f"url: {link_url}")

	return link_url
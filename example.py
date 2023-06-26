import sys
sys.path.append("/Users/am13743/Desktop/pub_gardens/sunTrapp/app/src/main/python/")
import sunTrapp.satellite
import sunTrapp.shadow_computer
import sunTrapp.utilities
import sunTrapp.image_tools
import sunTrapp.plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# data from https://environment.data.gov.uk/DefraDataDownload/?Mode=survey

# # # one off:
sunTrapp.utilities.index_tif_files("tif/*.tif", "tif/bounds.json")
## OPTIONS

# tif_fileName = "tif/homefarm.tif"
# loc = [51.220550, -0.341143]

tif_fileName = "tif/DSM_ST5570_P_10631_20190117_20190117.tif"
# loc = [51.453291, -2.609596] # hope and anchor
# loc = [51.47214797493697, -2.5777949681918355]
# loc = [51.476459748752944, -2.539813990044241]#star
loc = [51.47626161552586, -2.53695255376283]

max_shadow_length = 75
compute_size = [50, 50] # area 2N (x, y)
edge_buffer = 5
output = f"temp/plot"
output_gif = "university"
upsampling = 1
date = '2023/06/22'

##
time_string = "16:50:00"
debug = True
show = True
###

# ###
# show = False
# debug = False
# start_time = "15:00:00"
# end_time = "23:00:00"
# time_steps = 10
# time_string = sunTrapp.utilities.generate_time_stamps(start_time, end_time, time_steps)
# ###

avoid_satellite = False

#############################################
print(f"shadow size: {(compute_size[0]*2+1)*upsampling}, {(compute_size[1]*2+1)*upsampling}")


data, x_idx, y_idx, idx_raw = sunTrapp.utilities.get_organised_data(loc, compute_size, edge_buffer, max_shadow_length, upsampling)

# plt.imshow(data, norm=LogNorm())
# plt.show()
# quit()


# import json
# from pyproj import Transformer
# import geopy.distance

# with open("tif/bounds.json", "r") as json_file:
# 	bounds = json.load(json_file)

# def check_coordinate_within_bounds(coordinate, bounds):
# 	top_left = bounds['top_left']
# 	top_right = bounds['top_right']
# 	bottom_left = bounds['bottom_left']
# 	bottom_right = bounds['bottom_right']

# 	if (coordinate[0] >= bottom_left[0] and coordinate[0] <= top_left[0] and
# 			coordinate[1] >= top_left[1] and coordinate[1] <= top_right[1]):
# 		return True
# 	else:
# 		return False

# from math import radians, sin, cos, sqrt, atan2
	
	
# print('\n\n')
# file = False
# for file_str in bounds:
# 	# print(check_coordinate_within_bounds(loc, data[file_str]), file_str)
# 	if check_coordinate_within_bounds(loc, bounds[file_str]):
# 		file = file_str
# 		file_info = bounds[file_str]
# 		break

# print(file)
# print(file_info)


# data, transform, dataset = sunTrapp.utilities.open_tif_file(file)
# x_idx, y_idx, idx_raw = sunTrapp.utilities.get_centre_indexes(loc, transform, 1.)




# print(x_idx, y_idx) # y_idx is the vertical thats too short 

# top = y_idx - (int(image_height/2.)+2)
# bottom = y_idx + (int(image_height/2.)+2)
# right = x_idx + (int(image_height/2.)+2)
# left = x_idx - (int(image_height/2.)+2)

# top_left = transform * (left, top)  # Multiply transform with pixel indices
# top_right = transform * (right, top)
# bottom_left = transform * (left, bottom)
# bottom_right = transform * (right, bottom)

# transformer_toGPS = Transformer.from_crs("EPSG:27700", "EPSG:4326")
# top_left = transformer_toGPS.transform(top_left[0], top_left[1])
# top_right = transformer_toGPS.transform(top_right[0], top_right[1])
# bottom_left = transformer_toGPS.transform(bottom_left[0], bottom_left[1])
# bottom_right = transformer_toGPS.transform(bottom_right[0], bottom_right[1])

# corners = {}
# corners["top_left"] = check_coordinate_within_bounds(top_left, bounds[file])
# corners["top_right"] = check_coordinate_within_bounds(top_right, bounds[file])
# corners["bottom_left"] = check_coordinate_within_bounds(bottom_left, bounds[file])
# corners["bottom_right"] = check_coordinate_within_bounds(bottom_right, bounds[file])

# #   A
# # D   B
# #   C

# shape_data = np.shape(data)
# x_idx += 5100
# y_idx += 5100
# data_pad = np.pad(data, 5100)

# if corners["top_left"] == False and corners["top_right"] == False:

# 	for file_str_new in bounds:
# 		file_info = bounds[file_str]
# 		file_info_new = bounds[file_str_new]

# 		distance = geopy.distance.geodesic(file_info["top_left"], file_info_new["bottom_left"]).m

# 		if distance < 50:

# 			data_new, transform_new, dataset_new = sunTrapp.utilities.open_tif_file(file_str_new)

# 			up_down_start = 5100-np.shape(data_new)[0] # top left
# 			left_right_start = 5100 # top left
# 			data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

# 	if corners["bottom_left"] == False:

# 			distance = geopy.distance.geodesic(file_info["top_left"], file_info_new["bottom_right"]).m

# 			if distance < 50:
			
# 				data_new, transform_new, dataset_new = sunTrapp.utilities.open_tif_file(file_str_new)
				
# 				up_down_start = 5100-np.shape(data_new)[0] # top left
# 				left_right_start = 5100-np.shape(data_new)[1] # top left
# 				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new



# if corners["top_right"] == False and corners["bottom_right"] == False:
	
# 	for file_str_new in bounds:
# 		file_info = bounds[file_str]
# 		file_info_new = bounds[file_str_new]

# 		distance = geopy.distance.geodesic(file_info["top_right"], file_info_new["top_left"]).m

# 		if distance < 50:
			
# 			data_new, transform_new, dataset_new = sunTrapp.utilities.open_tif_file(file_str_new)

# 			up_down_start = 5100 # top left
# 			left_right_start = 5100+shape_data[1] # top left
# 			data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

# 		if corners["top_left"] == False:

# 			distance = geopy.distance.geodesic(file_info["top_right"], file_info_new["bottom_left"]).m

# 			if distance < 50:
				
# 				data_new, transform_new, dataset_new = sunTrapp.utilities.open_tif_file(file_str_new)
				
# 				up_down_start = 5100-np.shape(data_new)[0] # top left
# 				left_right_start = 5100+shape_data[1] # top left
# 				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new


# if corners["bottom_left"] == False and corners["bottom_right"] == False:

# 	for file_str_new in bounds:
# 		file_info = bounds[file_str]
# 		file_info_new = bounds[file_str_new]

# 		distance = geopy.distance.geodesic(file_info["bottom_right"], file_info_new["top_right"]).m

# 		if distance < 50:
			
# 			data_new, transform_new, dataset_new = sunTrapp.utilities.open_tif_file(file_str_new)
			
# 			up_down_start = 5100+shape_data[0] # top left
# 			left_right_start = 5100 # top left
# 			data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

# 		if corners["top_right"] == False:

# 			distance = geopy.distance.geodesic(file_info["bottom_right"], file_info_new["top_left"]).m

# 			if distance < 50:
			
# 				data_new, transform_new, dataset_new = sunTrapp.utilities.open_tif_file(file_str_new)
				
# 				up_down_start = 5100+shape_data[0] # top left
# 				left_right_start = 5100+np.shape(data_new)[1] # top left
# 				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new



# if corners["bottom_left"] == False and corners["top_left"] == False:

# 	for file_str_new in bounds:
# 		file_info = bounds[file_str]
# 		file_info_new = bounds[file_str_new]

# 		distance = geopy.distance.geodesic(file_info["bottom_left"], file_info_new["bottom_right"]).m

# 		if distance < 50:
			
# 			data_new, transform_new, dataset_new = sunTrapp.utilities.open_tif_file(file_str_new)
			
# 			up_down_start = 5100 # top left
# 			left_right_start = 5100-np.shape(data_new)[1] # top left
# 			data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new

# 		if corners["bottom_right"] == False:

# 			distance = geopy.distance.geodesic(file_info["bottom_left"], file_info_new["top_right"]).m

# 			if distance < 50:
			
# 				data_new, transform_new, dataset_new = sunTrapp.utilities.open_tif_file(file_str_new)
				
# 				up_down_start = 5100+shape_data[0] # top left
# 				left_right_start = 5100-np.shape(data_new)[1] # top left
# 				data_pad[up_down_start:up_down_start+np.shape(data_new)[0],left_right_start:left_right_start+np.shape(data_new)[1]] = data_new








# # if y_idx < int(image_height/2.)+2: # 2 for safety


# data = data_pad[int(y_idx-max_shadow_length-compute_size[1]-edge_buffer):int(y_idx+max_shadow_length+compute_size[1]+edge_buffer+1),int(x_idx-max_shadow_length-compute_size[0]-edge_buffer):int(x_idx+max_shadow_length+compute_size[0]+edge_buffer+1)]
# if upsampling != 1.:
#     max_shadow_length *= upsampling
#     compute_size *= upsampling
#     edge_buffer *= upsampling
# x_idx = int(max_shadow_length+compute_size[0]+edge_buffer)
# y_idx = int(max_shadow_length+compute_size[1]+edge_buffer)

# data[np.where(data<-1E3)] = -99
# data = sunTrapp.image_tools.replace_zeros(data, value=-99)

# # plt.imshow(data, vmin=0.1, vmax=70, norm=LogNorm())
# plt.imshow(data, norm=LogNorm())
# plt.show()
# # data[np.where(data<-1E3)] = -99
# # data = sunTrapp.image_tools.replace_zeros(data, value=-99)

# # print(np.shape(data))
# # plt.imshow(data)
# # plt.show()

# quit()




# # data, transform, dataset = sunTrapp.utilities.open_tif_file(tif_fileName)

# # x_idx, y_idx, idx_raw = sunTrapp.utilities.get_centre_indexes(loc, transform, 1.)

# # data = data[int(y_idx-max_shadow_length-compute_size[1]-edge_buffer):int(y_idx+max_shadow_length+compute_size[1]+edge_buffer+1),int(x_idx-max_shadow_length-compute_size[0]-edge_buffer):int(x_idx+max_shadow_length+compute_size[0]+edge_buffer+1)]
# # if upsampling != 1.:
# # 	max_shadow_length *= upsampling
# # 	compute_size *= upsampling
# # 	edge_buffer *= upsampling
# # x_idx = int(max_shadow_length+compute_size[0]+edge_buffer)
# # y_idx = int(max_shadow_length+compute_size[1]+edge_buffer)

# # data[np.where(data<-1E3)] = -99
# # data = sunTrapp.image_tools.replace_zeros(data, value=-99)

# # print(np.shape(data))
# # plt.imshow(data)
# # plt.show()
# # quit()





data, data_raw = sunTrapp.image_tools.upsample(data, upsampling)

cropped_satellite_image = sunTrapp.satellite.get_cropped_satellite_image(loc, idx_raw, compute_size, upsampling)


if time_string.__class__ != list: time_string = [time_string]

for time_itr, time_string_i in enumerate(time_string):

	print(time_itr, time_string_i)

	shadow_calculator = sunTrapp.shadow_computer.shadow_computer(date, time_string_i, loc[0], loc[1], max_shadow_length, upsampling)
	shadows, region_of_interest = shadow_calculator.compute(data, x_idx, y_idx, compute_size, edge_buffer, upsampling)

	shadows = sunTrapp.image_tools.smooth_image(shadows)

	if debug:
		sunTrapp.plotting.debug_plot(shadows, output, region_of_interest, cropped_satellite_image, time_string=time_string, date=date, show=show)
	else:
		sunTrapp.plotting.publish_plot(shadows, f"{output}_{time_itr}.png", cropped_satellite_image, time_string=time_string_i, date=date, show=False)

   
if debug: quit()


import glob
from PIL import Image
import moviepy.editor as mp

image_files = glob.glob(f"{output}_*.png")
image_idx_max = 0
for image in image_files:
	image_idx = image[len(output)+1:]
	image_idx = image_idx[:-4]
	if int(image_idx) > int(image_idx_max): image_idx_max = image_idx
image_files = []
for image_idx in range(1,int(image_idx_max)):
	image_files.append(f'{output}_{image_idx}.png')

# Create a list to store the image frames
frames = []

# Read and append each image to the frames list
for image_file in image_files:
	image = Image.open(image_file)
	frames.append(image)

# Save frames as an animated GIF
frames[0].save("gif/"+output_gif+".gif", format='GIF', append_images=frames[1:], save_all=True, duration=200*2, loop=0)
clip = mp.VideoFileClip("gif/"+output_gif+".gif")
clip.write_videofile(f"mp4/{output_gif}.mp4")





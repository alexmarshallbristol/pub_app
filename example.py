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
upsampling = 2
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

data, idx_raw = sunTrapp.utilities.get_organised_data(loc, compute_size, edge_buffer, max_shadow_length, upsampling)

if upsampling != 1.:
    max_shadow_length *= upsampling
    compute_size[0] *= upsampling
    compute_size[1] *= upsampling
    edge_buffer *= upsampling

x_idx = int(max_shadow_length+compute_size[0]+edge_buffer)
y_idx = int(max_shadow_length+compute_size[1]+edge_buffer)

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





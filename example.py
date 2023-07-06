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
import datetime

from matplotlib.colors import LinearSegmentedColormap
sun_color = '#fcd303'  # Brighter yellow color for sun
shade_color = '#403606'  # Darker blue color for shade
colors = [shade_color,sun_color]
integrated_map_cmap = LinearSegmentedColormap.from_list('sun_shade_cmap', colors)
integrated_map_cmap.set_under('white')

# data from https://environment.data.gov.uk/DefraDataDownload/?Mode=survey

# # # one off:
# sunTrapp.utilities.index_tif_files("tif/*.tif", "tif/bounds.json")

## OPTIONS

# tif_fileName = "tif/homefarm.tif"
# loc = [51.220550, -0.341143]

tif_fileName = "tif/DSM_ST5570_P_10631_20190117_20190117.tif"
# loc = [51.453291, -2.609596] # hope and anchor
# loc = [51.46136560299646, -2.5539305734162956]

# # Seneca
# loc = [51.457953359949684, -2.5484364453772]
# boundaries = np.asarray([[51.457983442517744, -2.5484116349449013],[51.457931215824445, -2.548386824512603],[51.4579224417341, -2.5484659496750686],[51.457962551847594, -2.5484833840328998],[51.457966729982395, -2.5484525386305825],[51.457978010944416, -2.5484545502872558]])

# # 74
# loc = [51.46113076318816, -2.554846755961126]
# boundaries = np.asarray([[51.461136194386285, -2.554917834500688], [51.46115959030936, -2.5547649485854027], [51.46112449642028, -2.5547475142266425], [51.46109942934012, -2.5549037529032286]])


# 74
loc = [51.45842207161029, -2.53821568964564]
boundaries = np.asarray([[51.4584765066667, -2.5382468911433276], [51.45846795259071, -2.538164519189432], [51.45837541293884, -2.5381832400880446], [51.45838474467694, -2.5382668601018477]])




max_shadow_length = 75
compute_size = [10, 10] # area 2N (x, y)
edge_buffer = 5
# output = f"temp/plot"
output = f"example"
output_gif = "NiaandBens"
upsampling = 2
date = '2023/06/22'

# ##
# time_string = "15:50:00"
# debug = True
# show = True
# ###


###
show = False
debug = False
start_time = "04:00:00"
end_time = "23:00:00"
time_steps = 50
time_string = sunTrapp.utilities.generate_time_stamps(start_time, end_time, time_steps)
###

avoid_satellite = False

#############################################
print(f"shadow size: {(compute_size[0]*2+1)*upsampling}, {(compute_size[1]*2+1)*upsampling}")

data, idx_raw, transform, centre_indexes = sunTrapp.utilities.get_organised_data(loc, compute_size, edge_buffer, max_shadow_length, upsampling)


####################
print(idx_raw)
print(centre_indexes)
print('\n')

boundaries_pixels = sunTrapp.utilities.convert_boundaries_GPS_to_relative_pixels(loc, boundaries, upsampling, transform, centre_indexes)
####################




if upsampling != 1.:
    max_shadow_length *= upsampling
    compute_size[0] *= upsampling
    compute_size[1] *= upsampling
    edge_buffer *= upsampling

x_idx = int(max_shadow_length+compute_size[0]+edge_buffer)
y_idx = int(max_shadow_length+compute_size[1]+edge_buffer)

data, data_raw = sunTrapp.image_tools.upsample(data, upsampling)

# cropped_satellite_image = sunTrapp.satellite.get_cropped_satellite_image(loc, idx_raw, compute_size, upsampling)

if time_string.__class__ != list: time_string = [time_string]

mean_sun_fractions_vs_offset = np.empty((0,4))

# for sun_azimuth_offset in [0, 60, 120, 180, 240, 300]:
for sun_azimuth_offset in np.linspace(0, 359, 40):


	sun_fractions = np.empty(0)
	sun_fractions_labels = []

	for time_itr, time_string_i in enumerate(time_string):

		print(time_itr, time_string_i)

		shadow_calculator = sunTrapp.shadow_computer.shadow_computer(date, time_string_i, loc[0], loc[1], max_shadow_length, upsampling, sun_azimuth_offset=sun_azimuth_offset)
		shadows, region_of_interest = shadow_calculator.compute(data, x_idx, y_idx, compute_size, edge_buffer, upsampling)
		

		# shadow_calculator = sunTrapp.shadow_computer.shadow_computer_fast(data, max_shadow_length, upsampling, edge_buffer)
		# shadows, region_of_interest = shadow_calculator.compute(data, x_idx, y_idx, compute_size, edge_buffer, upsampling)

		# shadow_calculator = sunTrapp.shadow_computer.shadow_computer_Bresenhams(date, time_string_i, loc[0], loc[1], max_shadow_length, upsampling)
		# shadows, region_of_interest = shadow_calculator.compute(data, x_idx, y_idx, compute_size, edge_buffer, upsampling)

		shadows = sunTrapp.image_tools.smooth_image(shadows)

		sun_frac, garden_map = sunTrapp.plotting.analyse_garden(shadows, boundaries_pixels)

		if time_itr == 0:
			integrated_map = garden_map
		else:
			integrated_map += garden_map

		sun_fractions = np.append(sun_fractions, sun_frac)
		sun_fractions_labels.append(time_string_i)


	integrated_map = integrated_map/np.amax(integrated_map)

	plt.figure(figsize=(8,4))

	plt.subplot(1,2,1)
	plt.imshow(integrated_map,vmin=np.amin(integrated_map[integrated_map>0]), vmax=1, cmap=integrated_map_cmap)
	plt.yticks([],[])   
	plt.xticks([],[]) 
	plt.colorbar()

	plt.subplot(1,2,2)
	plt.plot(np.arange(np.shape(sun_fractions)[0]), sun_fractions)
	plt.xticks(np.arange(np.shape(sun_fractions)[0]), sun_fractions_labels, rotation=90) 

	plt.savefig(f'offset={sun_azimuth_offset}.png')
	plt.close('all')

	mean_sun_score = np.mean(sun_fractions)

	# print(np.asarray(sun_fractions_labels))
	# quit()
	# greater_than_16 = np.where(np.array(sun_fractions_labels, dtype='datetime64') > np.datetime64('16:00:00'))
	# print(greater_than_16)

	evening_times = [time for time in sun_fractions_labels if datetime.datetime.strptime(time, '%H:%M:%S') > datetime.datetime.strptime('15:00:00', '%H:%M:%S')]
	morning_times = [time for time in sun_fractions_labels if datetime.datetime.strptime(time, '%H:%M:%S') < datetime.datetime.strptime('11:00:00', '%H:%M:%S')]


	where_evening = np.isin(np.asarray(sun_fractions_labels), evening_times)
	where_morning = np.isin(np.asarray(sun_fractions_labels), morning_times)

	mean_sun_score_morning = np.mean(sun_fractions[where_morning])
	mean_sun_score_evening = np.mean(sun_fractions[where_evening])
	mean_sun_fractions_vs_offset = np.append(mean_sun_fractions_vs_offset, [[sun_azimuth_offset, mean_sun_score, mean_sun_score_morning, mean_sun_score_evening]], axis=0)

mean_sun_fractions_vs_offset[:,1] = mean_sun_fractions_vs_offset[:,1]/np.amax(mean_sun_fractions_vs_offset[:,1])
mean_sun_fractions_vs_offset[:,2] = mean_sun_fractions_vs_offset[:,2]/np.amax(mean_sun_fractions_vs_offset[:,2])
mean_sun_fractions_vs_offset[:,3] = mean_sun_fractions_vs_offset[:,3]/np.amax(mean_sun_fractions_vs_offset[:,3])
plt.plot(mean_sun_fractions_vs_offset[:,0], mean_sun_fractions_vs_offset[:,1], label='total')
plt.plot(mean_sun_fractions_vs_offset[:,0], mean_sun_fractions_vs_offset[:,2], label='morning')
plt.plot(mean_sun_fractions_vs_offset[:,0], mean_sun_fractions_vs_offset[:,3], label='evening')
plt.legend()
plt.savefig(f'mean_sun_vs_offset.png')
plt.close('all')

print(mean_sun_fractions_vs_offset[0])

print(f'Total: {mean_sun_fractions_vs_offset[0][1]*10:.1f}/10')
print(f'Morning: {mean_sun_fractions_vs_offset[0][2]*10:.1f}/10')
print(f'Evening: {mean_sun_fractions_vs_offset[0][3]*10:.1f}/10')
quit()

if debug:
	# sunTrapp.plotting.debug_plot(shadows, output, region_of_interest, region_of_interest, time_string=time_string, date=date, show=show)
	sunTrapp.plotting.debug_plot_boundaries(shadows, boundaries_pixels, output, region_of_interest, cropped_satellite_image, time_string=time_string, date=date, show=show)
else:
	sunTrapp.plotting.publish_plot(shadows, f"{output}_{time_itr}.png", cropped_satellite_image, time_string=time_string_i, date=date, show=False)
quit()
   
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





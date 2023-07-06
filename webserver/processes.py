import requests 
import matplotlib.pyplot as plt
import time
import io
import numpy as np
import os
import glob
import datetime
import sunTrapp.satellite
import sunTrapp.shadow_computer
import sunTrapp.utilities
import sunTrapp.image_tools
import sunTrapp.plotting
from matplotlib.colors import LinearSegmentedColormap
import scipy

sun_color = '#FFD700'  # Brighter yellow color for sun
shade_color = '#121110'  # Darker blue color for shade
colors = [shade_color,sun_color]
cmap = LinearSegmentedColormap.from_list('sun_shade_cmap', colors)

__file__ = '/Users/am13743/Desktop/pub_gardens/sunTrapp/app/src/main/python/BUFFER'

def query_google_maps_search(display_string):
	
	api_key = "AIzaSyBbngN_VCGUbLyOBYpn1FepIDJYCsmr-GA"
	address = display_string+", bristol"
	url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
	response = requests.get(url)
	data = response.json()
	if data["status"] == "OK":
		location = data["results"][0]["geometry"]["location"]
		latitude = location["lat"]
		longitude = location["lng"]
		loc = [float(latitude), float(longitude)]
		print(f"Latitude: {latitude}, Longitude: {longitude}")
	else:
		print("Geocoding failed. Status:", data["status"])
		return [0., 0.]

	display_string = f'{latitude:.6f} {longitude:.6f}'
	return loc


def run_process(input_location_string, API=False):

	root_dir = os.getcwd()
	files = glob.glob(os.path.join(root_dir, f'images/*.png'))
	for file in files:
		os.remove(file)
	timestamp = int(time.time())


	latlong = query_google_maps_search(input_location_string)

	loc = [float(latlong[0]), float(latlong[1])]

	max_shadow_length = 25
	compute_size = [35, 35] # area 2N (x, y)
	compute_size[0] = int(compute_size[0]*(410/350))
	edge_buffer = 5
	output = f"plot"
	upsampling = 2

	current_time = datetime.datetime.now()
	current_date = datetime.date.today()
	hourText = str(current_time.hour)
	minuteText = str(current_time.minute)
	dayText = str(current_date.day)
	monthText = str(current_date.month)

	# time_string = f"{hourText}:{minuteText}:00"
	# date = f'2023/{monthText}/{dayText}'
	time_string = f"15:30:00"
	date = f'2023/{monthText}/{dayText}'

	print(f"shadow size: {(compute_size[0]*2+1)*upsampling}, {(compute_size[1]*2+1)*upsampling}")

	data, idx_raw, transform, centre_indexes = sunTrapp.utilities.get_organised_data(loc, compute_size, edge_buffer, max_shadow_length, upsampling, app=True, file_app=__file__)

	if upsampling != 1.:
		max_shadow_length *= upsampling
		compute_size[0] *= upsampling
		compute_size[1] *= upsampling
		edge_buffer *= upsampling

	x_idx = int(max_shadow_length+compute_size[0]+edge_buffer)
	y_idx = int(max_shadow_length+compute_size[1]+edge_buffer)

	data, data_raw = sunTrapp.image_tools.upsample(data, upsampling)


	cropped_satellite_image = sunTrapp.satellite.get_cropped_satellite_image(loc, idx_raw, compute_size, upsampling, app=True, file_app=__file__, timestamp=timestamp)


	file_path = f'images/satellite_image_cropped{timestamp}.png'

	### ok 
	shadow_calculator = sunTrapp.shadow_computer.shadow_computer(date, time_string, loc[0], loc[1], max_shadow_length, upsampling)
	shadows, region_of_interest = shadow_calculator.compute(data, x_idx, y_idx, compute_size, edge_buffer, upsampling)

	shadows = sunTrapp.image_tools.smooth_image(shadows)

	file_path = f'images/sun{timestamp}.png'

	overlay = cropped_satellite_image
	width = 8
	height = width * (np.shape(shadows)[0]/np.shape(shadows)[1])

	plt.figure(figsize=(width,height))
	ax = plt.axes((0, 0, 1, 1))
	# ax = plt.gca()
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
	if input_location_string is not None:
		plt.text(0.01, 0.83, f'Search: {input_location_string}', fontsize=20, horizontalalignment='left',verticalalignment='top', transform=ax.transAxes, c='w')
	
	plt.subplots_adjust(hspace=0,wspace=0)
	plt.tight_layout()        
	plt.savefig(file_path, transparent=True, pad_inches=0)
	plt.close('all')

	return file_path
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
from pyproj import Transformer
from PIL import Image


sun_color = '#FFD700'  # Brighter yellow color for sun
shade_color = '#121110'  # Darker blue color for shade
colors = [shade_color,sun_color]
cmap = LinearSegmentedColormap.from_list('sun_shade_cmap', colors)

__file__ = '/Users/am13743/Desktop/pub_gardens/sunTrapp/app/src/main/python/BUFFER'

def query_google_maps_search(display_string):
		
	api_key = "AIzaSyBbngN_VCGUbLyOBYpn1FepIDJYCsmr-GA"
	address = display_string+", bristol"
	url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address}&key={api_key}"
	try:
		response = requests.get(url)
	except:
		return [0., 0.]
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

class process_runner():

	shadows = None
	image_GPS_corners = {}
	image_fileName = ''
	garden_fileName = ''
	idx_raw = []
	compute_size = []	
	timestamp = ''

	def run_process(self, input_location_string, API=False):

		root_dir = os.getcwd()
		files = glob.glob(os.path.join(root_dir, f'images/*.png'))
		for file in files:
			os.remove(file)
		self.timestamp = int(time.time())

		latlong = query_google_maps_search(input_location_string)

		if latlong == [0., 0.]: return f'graphics/no_response.png'

		self.loc = [float(latlong[0]), float(latlong[1])]

		max_shadow_length = 15
		# self.compute_size = [35, 35] # area 2N (x, y)
		self.compute_size = [15, 15] # area 2N (x, y)
		self.compute_size[0] = int(self.compute_size[0]*(410/350))
		edge_buffer = 5
		output = f"plot"
		self.upsampling = 2

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

		print(f"shadow size: {(self.compute_size[0]*2+1)*self.upsampling}, {(self.compute_size[1]*2+1)*self.upsampling}")

		data, self.idx_raw, self.transform, self.centre_indexes = sunTrapp.utilities.get_organised_data(self.loc, self.compute_size, edge_buffer, max_shadow_length, self.upsampling, app=True, file_app=__file__)

		if self.upsampling != 1.:
			max_shadow_length *= self.upsampling
			self.compute_size[0] *= self.upsampling
			self.compute_size[1] *= self.upsampling
			edge_buffer *= self.upsampling

		x_idx = int(max_shadow_length+self.compute_size[0]+edge_buffer)
		y_idx = int(max_shadow_length+self.compute_size[1]+edge_buffer)

		data, data_raw = sunTrapp.image_tools.upsample(data, self.upsampling)


		cropped_satellite_image, self.image_GPS_corners = sunTrapp.satellite.get_cropped_satellite_image(self.loc, self.idx_raw, self.compute_size, self.upsampling, app=True, file_app=__file__, timestamp=self.timestamp)


		file_path = f'images/satellite_image_cropped{self.timestamp}.png'

		### ok 
		shadow_calculator = sunTrapp.shadow_computer.shadow_computer(date, time_string, self.loc[0], self.loc[1], max_shadow_length, self.upsampling)
		self.shadows, region_of_interest = shadow_calculator.compute(data, x_idx, y_idx, self.compute_size, edge_buffer, self.upsampling)

		self.shadows = sunTrapp.image_tools.smooth_image(self.shadows)

		self.image_fileName = f'images/sun{self.timestamp}.png'

		overlay = cropped_satellite_image
		self.width = 8
		self.height = self.width * (np.shape(self.shadows)[0]/np.shape(self.shadows)[1])

		plt.figure(figsize=(self.width,self.height))
		ax = plt.axes((0, 0, 1, 1))
		# ax = plt.gca()
		if overlay is not None:
			self.shadows = scipy.ndimage.zoom(self.shadows, np.shape(overlay)[0]/np.shape(self.shadows)[0], order=0)
			plt.imshow(overlay)
		plt.imshow(self.shadows, cmap=cmap, alpha=0.35, vmin=0, vmax=1)
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
		plt.savefig(self.image_fileName, transparent=True, pad_inches=0)
		plt.close('all')

		return self.image_fileName

	def run_garden_analysis(self, boundaries_pixels):

		sun_image = Image.open(self.image_fileName)
		image_width = sun_image.width
		image_height = sun_image.height

		for idx in range(np.shape(boundaries_pixels)[0]):
			boundaries_pixels[idx][0] = (boundaries_pixels[idx][0]/image_width)*np.shape(self.shadows)[1]
			boundaries_pixels[idx][1] = (boundaries_pixels[idx][1]/image_height)*np.shape(self.shadows)[0]

		sun_frac, garden_map = sunTrapp.plotting.analyse_garden(self.shadows, boundaries_pixels)

		self.garden_fileName = f'images/garden{self.timestamp}.png'

		plt.figure(figsize=(self.width,self.height))

		plt.imshow(garden_map)
		plt.yticks([],[])   
		plt.xticks([],[]) 

		plt.subplots_adjust(hspace=0,wspace=0)
		plt.tight_layout()        
		plt.savefig(self.garden_fileName, transparent=True, pad_inches=0)
		plt.close('all')

		return self.garden_fileName
	


	def plot_polygon(self, coordinates_file):

		coordinates_array = np.loadtxt(coordinates_file, delimiter=',')

		if np.shape(coordinates_array)[0] > 2:
			fileName = self.run_garden_analysis(coordinates_array)	
			return fileName
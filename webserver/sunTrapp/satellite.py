import requests
import math
from pyproj import Transformer
import numpy as np
import geopy.distance
from PIL import Image
from os.path import dirname, join

def get_satellite_image(latitude, longitude, zoom=20, size=(640, 640), api_key="YOUR_API_KEY", app=None, file_app=None, timestamp=''):
	url = "https://maps.googleapis.com/maps/api/staticmap"
	params = {
		"center": f"{latitude},{longitude}",
		"zoom": zoom,
		"size": f"{size[0]}x{size[1]}",
		"maptype": "satellite",
		"key": "AIzaSyBbngN_VCGUbLyOBYpn1FepIDJYCsmr-GA",
	}
	response = requests.get(url, params=params)
	if response.status_code == 200:
		with open(f"images/satellite_image{timestamp}.png", "wb") as file:
			file.write(response.content)
		# print("Satellite image saved successfully.")
	else:
		print("Error retrieving satellite image.")

def getPointLatLng(latitude, longitude, x, y, zoom, image_size):
	parallelMultiplier = math.cos(latitude * math.pi / 180)
	degreesPerPixelX = 360 / math.pow(2, zoom + 8)
	degreesPerPixelY = 360 / math.pow(2, zoom + 8) * parallelMultiplier
	pointLat = latitude - degreesPerPixelY * ( y - image_size / 2)
	pointLng = longitude + degreesPerPixelX * ( x  - image_size / 2)

	return (pointLat, pointLng)



def get_cropped_satellite_image(loc, idx_raw, N_pixels, upsampling, app=None, file_app=None, timestamp=''):

	N_pixels = np.asarray(N_pixels)*(1./upsampling)
	
	transformer_toGPS = Transformer.from_crs("EPSG:27700", "EPSG:4326")
	idx_GPS = transformer_toGPS.transform(idx_raw[0]+N_pixels[0],idx_raw[1]+N_pixels[1])
	idx_GPS_top = (idx_GPS[0], idx_GPS[1])

	idx_GPS = transformer_toGPS.transform(idx_raw[0]-N_pixels[0],idx_raw[1]-N_pixels[1])
	idx_GPS_bottom = (idx_GPS[0], idx_GPS[1])

	# Example usage
	required_bounds = {
		"south": np.amin([idx_GPS_top[0], idx_GPS_bottom[0]]),
		"west": np.amin([idx_GPS_top[1], idx_GPS_bottom[1]]),
		"north": np.amax([idx_GPS_top[0], idx_GPS_bottom[0]]),
		"east": np.amax([idx_GPS_top[1], idx_GPS_bottom[1]])
	}

	# print(required_bounds)
	# {'south': 51.45094186123943, 'west': -2.6078918259586525, 'north': 51.45121412892701, 'east': -2.6074061502428316}

	# Example usage
	latitude = loc[0]
	longitude = loc[1]
	image_size = 640
	zoom = 23

	for zoom_out in range(40):
		zoom = zoom - 1
		NE = getPointLatLng(latitude, longitude, image_size, 0, zoom, image_size)
		SW = getPointLatLng(latitude, longitude, 0, image_size, zoom, image_size)
		NW = getPointLatLng(latitude, longitude, 0, 0, zoom, image_size)
		SE = getPointLatLng(latitude, longitude, image_size, image_size, zoom, image_size)

		image_bounds = {
			"south": SE[0],
			"west": SW[1],
			"north": NE[0],
			"east": SE[1]
		}

		zoom_out = False
		for bound_i in image_bounds.keys():
			if bound_i == "north":
				if image_bounds[bound_i] < required_bounds[bound_i]: zoom_out = True # zoom out
			if bound_i == "east":
				if image_bounds[bound_i] < required_bounds[bound_i]: zoom_out = True # zoom out
		if zoom_out:
			continue
		else:
			break



	get_satellite_image(latitude, longitude, zoom=zoom, size=(image_size, image_size), app=app, file_app=file_app, timestamp=timestamp)

	image_centre = (latitude, longitude)

	# image_right_edge = (latitude, image_bounds["east"])
	# required_right_edge = (latitude, required_bounds["east"])
	# scale_horizontal = geopy.distance.geodesic(image_centre, required_right_edge).m/geopy.distance.geodesic(image_centre, image_right_edge).m

	image_top_edge = (image_bounds["north"], longitude)
	required_top_edge = (required_bounds["north"], longitude)
	scale_vertical = geopy.distance.geodesic(image_centre, required_top_edge).m/geopy.distance.geodesic(image_centre, image_top_edge).m


	img=Image.open(f'images/satellite_image{timestamp}.png')

	frac_vertical = scale_vertical
	frac_horizontal = frac_vertical*(N_pixels[0]/N_pixels[1])
	left = img.size[0]*((1-frac_horizontal)/2)
	upper = img.size[1]*((1-frac_vertical)/2)
	right = img.size[0]-((1-frac_horizontal)/2)*img.size[0]
	bottom = img.size[1]-((1-frac_vertical)/2)*img.size[1]
	cropped_img = img.crop((left, upper, right, bottom))

	cropped_img.save(f"images/satellite_image_cropped{timestamp}.png")


	return cropped_img, required_bounds
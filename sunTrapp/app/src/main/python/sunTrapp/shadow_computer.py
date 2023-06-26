import numpy as np
from math import degrees
import ephem

class shadow_computer:

	def __init__(self, date, time_string, latitude, longitude, max_shadow_length, upsampling):

		self.upsampling = upsampling
		self.max_shadow_length = max_shadow_length
		# rows = np.arange(int(self.max_shadow_length*2+1)).reshape(int(self.max_shadow_length*2)+1, 1)
		# cols = np.arange(int(self.max_shadow_length*2)+1).reshape(1, int(self.max_shadow_length*2)+1)
		rows = np.arange(int(self.max_shadow_length*2+1)).reshape(int(self.max_shadow_length*2)+1, 1)
		cols = np.arange(int(self.max_shadow_length*2)+1).reshape(1, int(self.max_shadow_length*2)+1)
		meshgrid_rows, meshgrid_cols = np.meshgrid(rows, cols)
		result_array = np.stack((meshgrid_cols,meshgrid_rows), axis=2)
		angles = self.compute_angle_from_north(self.max_shadow_length, self.max_shadow_length, result_array[:,:,0], result_array[:,:,1])
		azimuth, self.altitude = self.compute_sun_direction(date, time_string, latitude, longitude)
		self.mask = self.select_elements_within_angle(angles, azimuth, 10, self.max_shadow_length)
		self.distance_to_centre = self.compute_distance(self.max_shadow_length, self.max_shadow_length, result_array[:,:,0], result_array[:,:,1])

	def compute_distance(self, origin_x, origin_y, points_x, points_y):

		distance_x = points_x-origin_x
		distance_y = points_y-origin_y

		distance = np.sqrt(distance_x**2 + distance_y**2)/self.upsampling

		return distance

	def select_elements_within_angle(self, array, angle, cone=5, N_surrounding_pixels=50):

		array[np.where(array<0)] += 360

		adjusted_angle = angle % 360

		lower_bound = adjusted_angle - cone
		upper_bound = adjusted_angle + cone

		if lower_bound < 0:
			where = np.where(np.logical_or(array <= upper_bound, array >= 360 + lower_bound))
		elif upper_bound > 360:
			where = np.where(np.logical_or(array >= lower_bound, array <= upper_bound - 360))
		else:
			where = np.where(np.logical_and(array >= lower_bound, array <= upper_bound))

		return where

	def compute_angle_from_north(self, origin_x, origin_y, points_x, points_y):

		distance_x = (points_x-origin_x)/self.upsampling
		distance_y = (points_y-origin_y)/self.upsampling

		angles = np.degrees(np.arctan2(distance_x,distance_y))+90
		angles = angles % 360

		return angles

	def compute_sun_direction(self, date, time, latitude, longitude):
		observer = ephem.Observer()
		observer.lat = str(latitude)
		observer.lon = str(longitude)
		observer.date = date + ' ' + time

		sun = ephem.Sun()
		sun.compute(observer)

		azimuth = degrees(sun.az)
		altitude = degrees(sun.alt)

		return azimuth, altitude

	def convolve2D_mask(self, padded_image, distance_matrix, mask):
		kernel_height, kernel_width = distance_matrix.shape

		pad_height = kernel_height // 2
		pad_width = kernel_width // 2

		image_height, image_width = np.shape(padded_image)[1] - 2*pad_height, np.shape(padded_image)[0] - 2*pad_height

		# print(image_height, image_width)
		# quit()
		result = np.zeros((image_width, image_height))
		distance_matrix = distance_matrix[mask]
		for j in range(pad_height, image_height + pad_height):
			for i in range(pad_width, image_width + pad_width):
				region_heights = padded_image[i - pad_height : i + pad_height + 1, j - pad_width : j + pad_width + 1]
				# print(np.shape(region_heights))
				# quit()
				region_heights = region_heights[mask]
				convolution = np.degrees(np.nanmax(np.arctan2((region_heights - padded_image[i, j]), distance_matrix)))
				result[i - pad_height, j - pad_width] = convolution

		return result
	
	def compute(self, region_of_interest, x_idx, y_idx, compute_size, edge_buffer, upsampling):
		
		region_of_interest = region_of_interest[int(edge_buffer):int(-edge_buffer),int(edge_buffer):int(-edge_buffer)]

		is_sunny_angels = self.convolve2D_mask(region_of_interest, self.distance_to_centre, self.mask)
		is_sunny = np.ones(np.shape(is_sunny_angels))
		is_sunny[np.where(is_sunny_angels>self.altitude)] = 0.

		region_of_interest = region_of_interest[int(y_idx-compute_size[1]-edge_buffer):int(y_idx+compute_size[1]-edge_buffer+1*upsampling),int(x_idx-compute_size[0]-edge_buffer):int(x_idx+compute_size[0]-edge_buffer+1*upsampling)]
	
		return is_sunny, region_of_interest


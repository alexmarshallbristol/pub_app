import numpy as np
from math import degrees
import ephem
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import math




class shadow_computer:

	def __init__(self, date, time_string, latitude, longitude, max_shadow_length, upsampling, sun_azimuth_offset=0):

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
		if sun_azimuth_offset != 0:
			azimuth += sun_azimuth_offset
			azimuth = azimuth % 360		
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

























class shadow_computer_Bresenhams:

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
		self.azimuth, self.altitude = self.compute_sun_direction(date, time_string, latitude, longitude)
		self.mask = self.select_elements_within_angle(angles, self.azimuth, 10, self.max_shadow_length)
		self.distance_to_centre = self.compute_distance(self.max_shadow_length, self.max_shadow_length, result_array[:,:,0], result_array[:,:,1])


	def normalize_to_255(self, image):
		image_min = np.amin(image)
		image_max = np.amax(image)
		image = image - image_min
		image = image/(image_max-image_min)
		print(np.amax(image))
		image = image*255
		image = np.floor(image).astype(np.int)
		return image

	def make_3D_blocks(self, image):
		array_3d = np.zeros((np.shape(image)[0], np.shape(image)[1], 255))
		
		for x in range(np.shape(image)[0]):
			for y in range(np.shape(image)[1]):
				height = image[x, y]  # Assuming you have loaded your image as a 2D NumPy array called 'image'
				array_3d[x, y, :height] = 1
			
		return array_3d
		

	def compute_first_pixel(self, azimuthal, altitude, array_shape, current_pos):
		x = np.cos(azimuthal) * np.cos(altitude)
		y = np.sin(azimuthal) * np.cos(altitude)
		z = np.sin(altitude)

		length = np.sqrt(x**2 + y**2 + z**2)
		x /= length
		y /= length
		z /= length

		step_sizes = np.array([1, 1, 1])  # Adjust the step sizes as needed

		current_pos = np.array(current_pos)
		while np.all(current_pos >= 0) and np.all(current_pos < array_shape):
			current_pos += step_sizes

		return current_pos - step_sizes


	def loop_pixels_Bresenham(self, padded_image, padded_image_3D, distance_matrix, mask):
		kernel_height, kernel_width = distance_matrix.shape

		pad_height = kernel_height // 2
		pad_width = kernel_width // 2

		image_height, image_width = np.shape(padded_image)[1] - 2*pad_height, np.shape(padded_image)[0] - 2*pad_height
		

		first_pixel = self.compute_first_pixel(self.azimuth, self.altitude, np.shape(padded_image_3D), [254, 254, 254])
		print(first_pixel)
		quit()

		# print(image_height, image_width)
		# quit()
		result = np.zeros((image_width, image_height))
		distance_matrix = distance_matrix[mask]
		for j in range(pad_height, image_height + pad_height):
			for i in range(pad_width, image_width + pad_width):
				print(i, j)

				first_pixel = self.compute_first_pixel(self.azimuth, self.altitude, np.shape(padded_image_3D), [i,j,padded_image[i, j]])
				print(first_pixel)


				# region_heights = padded_image[i - pad_height : i + pad_height + 1, j - pad_width : j + pad_width + 1]
				# # print(np.shape(region_heights))
				# # quit()
				# region_heights = region_heights[mask]
				# convolution = np.degrees(np.nanmax(np.arctan2((region_heights - padded_image[i, j]), distance_matrix)))
				# result[i - pad_height, j - pad_width] = convolution

		quit()
		# self.bresenham3D(region_of_interest_3D, 0,0,0, 25,25,25)


		return result
	


	def compute(self, region_of_interest, x_idx, y_idx, compute_size, edge_buffer, upsampling):

		region_of_interest = region_of_interest[int(edge_buffer):int(-edge_buffer),int(edge_buffer):int(-edge_buffer)]

		print(np.shape(region_of_interest))

		# plt.figure(figsize=(8,8))
		# plt.subplot(2,2,1)
		# plt.imshow(region_of_interest)
		# plt.colorbar()

		region_of_interest = self.normalize_to_255(region_of_interest)

		# plt.subplot(2,2,2)
		# plt.imshow(region_of_interest)
		# plt.colorbar()

		

		region_of_interest_3D = self.make_3D_blocks(region_of_interest)

		# print(np.where())

		# pixels = self.bresenham3D(region_of_interest_3D, 0,0,0, 25,25,25)

		# marked_tracks = np.zeros_like(region_of_interest_3D)
		# pixels = np.array(pixels)
		# marked_tracks[pixels[:, 0], pixels[:, 1], pixels[:, 2]] = 1

		# print(marked_tracks[0])
		# marked_tracks_sum = np.sum(marked_tracks,axis=2)
		# plt.imshow(marked_tracks_sum)
		# plt.show()	

		print(self.azimuth, self.altitude)
		print(np.shape(region_of_interest))

		if self.azimuth>225 and self.azimuth<315:
			middle = (np.shape(region_of_interest)[0]/2, np.shape(region_of_interest)[1]/2)
			angle_i = self.azimuth-270

			O = (np.shape(region_of_interest)[0]/2) * np.tan(math.radians(angle_i))
			first_pixel = (0, int((np.shape(region_of_interest)[1]/2)-O))

			O = ((np.shape(region_of_interest)[0]/2)-self.max_shadow_length) * np.tan(math.radians(angle_i))
			first_pixel_of_interest = (0, int((np.shape(region_of_interest)[1]/2)-O))

			print(first_pixel)
			print(first_pixel_of_interest)

			pixels = self.bresenham3D(region_of_interest_3D, 0,0,0, 25,25,25)

			quit()



		# is_sunny_angels = self.loop_pixels_Bresenham(region_of_interest, region_of_interest_3D, self.distance_to_centre, self.mask)


		quit()


	def bresenham3D(self, data, x1, y1, z1, x2, y2, z2):

		dx = abs(x2 - x1)
		dy = abs(y2 - y1)
		dz = abs(z2 - z1)
		xs = 1 if x2 > x1 else -1
		ys = 1 if y2 > y1 else -1
		zs = 1 if z2 > z1 else -1

		pixels = []

		if dx >= dy and dx >= dz:
			p1 = 2 * dy - dx
			p2 = 2 * dz - dx
			while x1 != x2:
				pixels.append((x1, y1, z1))
				# mark the pixel at (x1, y1, z1)
				# here, you can perform your desired operation on the pixel
				x1 += xs
				if p1 >= 0:
					y1 += ys
					p1 -= 2 * dx
				if p2 >= 0:
					z1 += zs
					p2 -= 2 * dx
				p1 += 2 * dy
				p2 += 2 * dz
		elif dy >= dx and dy >= dz:
			p1 = 2 * dx - dy
			p2 = 2 * dz - dy
			while y1 != y2:
				pixels.append((x1, y1, z1))
				# mark the pixel at (x1, y1, z1)
				# here, you can perform your desired operation on the pixel
				y1 += ys
				if p1 >= 0:
					x1 += xs
					p1 -= 2 * dy
				if p2 >= 0:
					z1 += zs
					p2 -= 2 * dy
				p1 += 2 * dx
				p2 += 2 * dz
		else:
			p1 = 2 * dy - dz
			p2 = 2 * dx - dz
			while z1 != z2:
				pixels.append((x1, y1, z1))
				# mark the pixel at (x1, y1, z1)
				# here, you can perform your desired operation on the pixel
				z1 += zs
				if p1 >= 0:
					y1 += ys
					p1 -= 2 * dz
				if p2 >= 0:
					x1 += xs
					p2 -= 2 * dz
				p1 += 2 * dy
				p2 += 2 * dx
		# mark the last pixel at (x2, y2, z2)
		# here, you can perform your desired operation on the pixel
		pixels.append((x2, y2, z2))
		return pixels





		# plt.subplot(2,2,3)
		# plt.imshow(region_of_interest[:,:,128])
		# plt.colorbar()

		# plt.subplot(2,2,4)
		# plt.imshow(region_of_interest[:,:,170])
		# plt.colorbar()


		# plt.show()





		quit()

		is_sunny_angels = self.convolve2D_mask(region_of_interest, self.distance_to_centre, self.mask)
		is_sunny = np.ones(np.shape(is_sunny_angels))
		is_sunny[np.where(is_sunny_angels>self.altitude)] = 0.

		region_of_interest = region_of_interest[int(y_idx-compute_size[1]-edge_buffer):int(y_idx+compute_size[1]-edge_buffer+1*upsampling),int(x_idx-compute_size[0]-edge_buffer):int(x_idx+compute_size[0]-edge_buffer+1*upsampling)]


		return is_sunny, region_of_interest
	




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
	
















# class shadow_computer_fast:
	
# 	def __init__(self, region_of_interest, max_shadow_length, upsampling, edge_buffer):

# 		self.upsampling = upsampling
# 		self.max_shadow_length = max_shadow_length

# 		rows = np.arange(int(self.max_shadow_length*2+1)).reshape(int(self.max_shadow_length*2)+1, 1)
# 		cols = np.arange(int(self.max_shadow_length*2)+1).reshape(1, int(self.max_shadow_length*2)+1)
# 		meshgrid_rows, meshgrid_cols = np.meshgrid(rows, cols)
# 		result_array = np.stack((meshgrid_cols,meshgrid_rows), axis=2)
# 		angles = self.compute_angle_from_north(self.max_shadow_length, self.max_shadow_length, result_array[:,:,0], result_array[:,:,1])
# 		distance_matrix = self.compute_distance(self.max_shadow_length, self.max_shadow_length, result_array[:,:,0], result_array[:,:,1])

# 		padded_image = region_of_interest[int(edge_buffer):int(-edge_buffer),int(edge_buffer):int(-edge_buffer)]

# 		# print(np.shape(padded_image))
# 		# plt.imshow(padded_image)
# 		# plt.show()
# 		# quit()
# 		kernel_height, kernel_width = distance_matrix.shape

# 		pad_height = kernel_height // 2
# 		pad_width = kernel_width // 2

# 		image_height, image_width = np.shape(padded_image)[1] - 2*pad_height, np.shape(padded_image)[0] - 2*pad_height

# 		# ########
# 		# # Attempt 1
# 		# #######
# 		# print(image_height, image_width)

# 		# azimuthal_angle_steps = 20
# 		# max_feature_altitude_az = np.empty((azimuthal_angle_steps, image_height, image_width))
# 		# for az_idx, azimuthal_angle in enumerate(np.linspace(0,360, azimuthal_angle_steps)):
# 		# 	print(az_idx)
# 		# 	mask = self.select_elements_within_angle(angles, azimuthal_angle, cone=5)
# 		# 	for j in range(pad_height, image_height + pad_height):
# 		# 		for i in range(pad_width, image_width + pad_width):
# 		# 			region_heights = padded_image[i - pad_height : i + pad_height + 1, j - pad_width : j + pad_width + 1]
# 		# 			distance_matrix_i = distance_matrix[mask]
# 		# 			region_heights_i = region_heights[mask]
# 		# 			print(np.shape(region_heights_i), np.shape(distance_matrix_i))
# 		# 			print(np.shape(mask))
# 		# 			quit()
# 		# 			max_feature_altitude = np.degrees(np.nanmax(np.arctan2((region_heights_i - padded_image[i, j]), distance_matrix_i)))
					
# 		# 			max_feature_altitude_az[az_idx,i - pad_height, j - pad_width] = max_feature_altitude

# 		# print(azimuthal_angle_steps)
# 		# quit()



# 		########
# 		# Attempt 2
# 		#######
# 		print(image_height, image_width)

# 		azimuthal_angle_steps = 20


# 		max_mask_length = 0
# 		masks = []
# 		for az_idx, azimuthal_angle in enumerate(np.linspace(0,360, azimuthal_angle_steps)):
# 			mask = self.select_elements_within_angle(angles, azimuthal_angle, cone=5)
# 			masks.append(mask)
# 			if np.shape(mask)[1] > max_mask_length: max_mask_length = np.shape(mask)[1]

# 		distance_matrix_az = np.empty((azimuthal_angle_steps, max_mask_length))
# 		region_heights_az = np.empty((azimuthal_angle_steps, max_mask_length))

# 		for j in range(pad_height, image_height + pad_height):
# 			for i in range(pad_width, image_width + pad_width):

# 				region_heights = padded_image[i - pad_height : i + pad_height + 1, j - pad_width : j + pad_width + 1]
# 				pixel_height = padded_image[i, j]
# 				region_heights += -pixel_height

# 				for az_idx, azimuthal_angle in enumerate(np.linspace(0,360, azimuthal_angle_steps)):

# 					mask = masks[az_idx]
# 					distance_matrix_az[az_idx] = np.concatenate((distance_matrix[mask],np.zeros(max_mask_length-np.shape(mask)[1])))
# 					region_heights_az[az_idx] = np.concatenate((region_heights[mask],np.zeros(max_mask_length-np.shape(mask)[1])))

# 				pixel_max_altitude_array = np.degrees(np.nanmax(np.arctan2((region_heights_az), distance_matrix_az),axis=1))

# 				# x = np.concatenate((np.linspace(0,360, azimuthal_angle_steps)[:-1]-360,np.linspace(0,360, azimuthal_angle_steps),np.linspace(0,360, azimuthal_angle_steps)+360))
# 				# y = np.concatenate((pixel_max_altitude_array,pixel_max_altitude_array,pixel_max_altitude_array))
# 				# print(x)
# 				x = np.linspace(0,360, azimuthal_angle_steps)
# 				y = pixel_max_altitude_array
# 				cs = CubicSpline(x, y)

# 				plt.plot(np.linspace(0,360, azimuthal_angle_steps), pixel_max_altitude_array)
# 				plt.plot(np.linspace(0,360, azimuthal_angle_steps), cs(np.linspace(0,360, azimuthal_angle_steps),2))
# 				# if i == 7+pad_width:
# 				plt.show()
# 				quit()
# 		quit()
		
# 		max_feature_altitude = np.degrees(np.nanmax(np.arctan2((region_heights_az), distance_matrix_az)))
					

# 		for az_idx, azimuthal_angle in enumerate(np.linspace(0,360, azimuthal_angle_steps)):
# 			print(az_idx)
# 			mask = self.select_elements_within_angle(angles, azimuthal_angle, cone=10)
# 			for j in range(pad_height, image_height + pad_height):
# 				for i in range(pad_width, image_width + pad_width):
# 					region_heights = padded_image[i - pad_height : i + pad_height + 1, j - pad_width : j + pad_width + 1]
# 					distance_matrix_i = distance_matrix[mask]
# 					region_heights_i = region_heights[mask]
# 					max_feature_altitude = np.degrees(np.nanmax(np.arctan2((region_heights_i - padded_image[i, j]), distance_matrix_i)))
					
# 					max_feature_altitude_az[az_idx,i - pad_height, j - pad_width] = max_feature_altitude

# 		print(azimuthal_angle_steps)


# 		quit()


# 		# rows = np.arange(int(self.max_shadow_length*2+1)).reshape(int(self.max_shadow_length*2)+1, 1)
# 		# cols = np.arange(int(self.max_shadow_length*2)+1).reshape(1, int(self.max_shadow_length*2)+1)
# 		rows = np.arange(int(self.max_shadow_length*2+1)).reshape(int(self.max_shadow_length*2)+1, 1)
# 		cols = np.arange(int(self.max_shadow_length*2)+1).reshape(1, int(self.max_shadow_length*2)+1)
# 		meshgrid_rows, meshgrid_cols = np.meshgrid(rows, cols)
# 		result_array = np.stack((meshgrid_cols,meshgrid_rows), axis=2)
# 		angles = self.compute_angle_from_north(self.max_shadow_length, self.max_shadow_length, result_array[:,:,0], result_array[:,:,1])
# 		azimuth, self.altitude = self.compute_sun_direction(date, time_string, latitude, longitude)
# 		self.mask = self.select_elements_within_angle(angles, azimuth, 10)
# 		self.distance_to_centre = self.compute_distance(self.max_shadow_length, self.max_shadow_length, result_array[:,:,0], result_array[:,:,1])

# 	def compute_distance(self, origin_x, origin_y, points_x, points_y):

# 		distance_x = points_x-origin_x
# 		distance_y = points_y-origin_y

# 		distance = np.sqrt(distance_x**2 + distance_y**2)/self.upsampling

# 		return distance

# 	def select_elements_within_angle(self, array, angle, cone=5):

# 		array[np.where(array<0)] += 360

# 		adjusted_angle = angle % 360

# 		lower_bound = adjusted_angle - cone
# 		upper_bound = adjusted_angle + cone

# 		if lower_bound < 0:
# 			where = np.where(np.logical_or(array <= upper_bound, array >= 360 + lower_bound))
# 		elif upper_bound > 360:
# 			where = np.where(np.logical_or(array >= lower_bound, array <= upper_bound - 360))
# 		else:
# 			where = np.where(np.logical_and(array >= lower_bound, array <= upper_bound))

# 		return where

# 	def compute_angle_from_north(self, origin_x, origin_y, points_x, points_y):

# 		distance_x = (points_x-origin_x)/self.upsampling
# 		distance_y = (points_y-origin_y)/self.upsampling

# 		angles = np.degrees(np.arctan2(distance_x,distance_y))+90
# 		angles = angles % 360

# 		return angles

# 	def compute_sun_direction(self, date, time, latitude, longitude):
# 		observer = ephem.Observer()
# 		observer.lat = str(latitude)
# 		observer.lon = str(longitude)
# 		observer.date = date + ' ' + time

# 		sun = ephem.Sun()
# 		sun.compute(observer)

# 		azimuth = degrees(sun.az)
# 		altitude = degrees(sun.alt)

# 		return azimuth, altitude

# 	def convolve2D_mask(self, padded_image, distance_matrix, mask):
# 		kernel_height, kernel_width = distance_matrix.shape

# 		pad_height = kernel_height // 2
# 		pad_width = kernel_width // 2

# 		image_height, image_width = np.shape(padded_image)[1] - 2*pad_height, np.shape(padded_image)[0] - 2*pad_height

# 		# print(image_height, image_width)
# 		# quit()
# 		result = np.zeros((image_width, image_height))
# 		distance_matrix = distance_matrix[mask]
# 		for j in range(pad_height, image_height + pad_height):
# 			for i in range(pad_width, image_width + pad_width):
# 				region_heights = padded_image[i - pad_height : i + pad_height + 1, j - pad_width : j + pad_width + 1]
# 				# print(np.shape(region_heights))
# 				# quit()
# 				region_heights = region_heights[mask]
# 				convolution = np.degrees(np.nanmax(np.arctan2((region_heights - padded_image[i, j]), distance_matrix)))
# 				result[i - pad_height, j - pad_width] = convolution

# 		return result
	
# 	def compute(self, region_of_interest, x_idx, y_idx, compute_size, edge_buffer, upsampling):

# 		region_of_interest = region_of_interest[int(edge_buffer):int(-edge_buffer),int(edge_buffer):int(-edge_buffer)]

# 		is_sunny_angels = self.convolve2D_mask(region_of_interest, self.distance_to_centre, self.mask)
# 		is_sunny = np.ones(np.shape(is_sunny_angels))
# 		is_sunny[np.where(is_sunny_angels>self.altitude)] = 0.

# 		region_of_interest = region_of_interest[int(y_idx-compute_size[1]-edge_buffer):int(y_idx+compute_size[1]-edge_buffer+1*upsampling),int(x_idx-compute_size[0]-edge_buffer):int(x_idx+compute_size[0]-edge_buffer+1*upsampling)]


# 		return is_sunny, region_of_interest


















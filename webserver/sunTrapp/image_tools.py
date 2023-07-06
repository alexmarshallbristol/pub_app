import scipy.ndimage
import numpy as np
from PIL import Image
from PIL import ImageFilter
from scipy.spatial.distance import cdist

def upsample(data, upsampling):
	data_raw = scipy.ndimage.zoom(data, upsampling, order=0)
	data = scipy.ndimage.zoom(data, upsampling, order=2)
	return data, data_raw

def smooth_image(image_array, size=3):
	_min = np.amin(image_array)
	_max = np.amax(image_array)
	image_array += -_min
	image_array *= (1./(_max-_min))
	image = Image.fromarray(np.uint8(image_array*255))
	image = image.filter(ImageFilter.ModeFilter(size=size))
	image_array = np.array(image)
	image_array = ((image_array*_max)+_min)/255
	return image_array

def replace_zeros(array, value=0):
    # Create a copy of the input array
    replaced_array = np.copy(array)

    # Find the indices of zero values in the array
    zero_indices = np.argwhere(array == value)

    # Iterate over each zero index
    for zero_index in zero_indices:
        # Compute the Euclidean distance between the zero index and all non-zero indices
        distances = cdist([zero_index], np.argwhere(array != value))

        # Find the index of the closest non-zero value
        closest_index = np.unravel_index(np.argmin(distances), array.shape)

        # Replace the zero value with the closest non-zero value
        replaced_array[zero_index[0], zero_index[1]] = array[closest_index[0], closest_index[1]]

    return replaced_array






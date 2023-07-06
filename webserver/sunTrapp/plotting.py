import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib.colors import LogNorm
import sunTrapp.utilities

sun_color = '#FFD700'  # Brighter yellow color for sun
shade_color = '#121110'  # Darker blue color for shade

colors = [shade_color,sun_color]
cmap = LinearSegmentedColormap.from_list('sun_shade_cmap', colors)

def publish_plot(shadows, filename, overlay=None, time_string=None, date=None, show=False):
	
	width = 8
	height = width * (np.shape(shadows)[0]/np.shape(shadows)[1])

	plt.figure(figsize=(width,height))
	ax = plt.axes((0, 0, 1, 1))
	# ax = plt.gca()
	if overlay is not None:
		shadows = scipy.ndimage.zoom(shadows, np.shape(overlay)[0]/np.shape(shadows)[0], order=0)
		plt.imshow(overlay)
	plt.imshow(shadows, cmap=cmap, alpha=0.35, vmin=0, vmax=1)
	# plt.yticks([],[])   
	# plt.xticks([],[]) 
	plt.axis('off')
	if time_string is not None:
		plt.text(0.01, 0.97, f'Time: {time_string}', fontsize=20, horizontalalignment='left',verticalalignment='top', transform=ax.transAxes, c='w')
	if date is not None:
		plt.text(0.01, 0.90, f'Date: {date}', fontsize=20, horizontalalignment='left',verticalalignment='top', transform=ax.transAxes, c='w')
	plt.subplots_adjust(hspace=0,wspace=0)
	plt.tight_layout()        
	plt.savefig(filename, transparent=True, pad_inches=0)
	if show:
		plt.show()
	plt.close('all')


def debug_plot(shadows, filename, region_of_interest, satellite, time_string=None, date=None, show=False):

	width = 4
	height = width * (np.shape(shadows)[0]/np.shape(shadows)[1])

	plt.figure(figsize=(width*3,height*2))

	plt.subplot(2,3,1)
	plt.imshow(region_of_interest, norm=LogNorm())
	# plt.colorbar()

	plt.subplot(2,3,2)
	plt.imshow(satellite)
	# plt.colorbar()

	plt.subplot(2,3,3)
	plt.imshow(shadows, cmap=cmap, vmin=0, vmax=1)
	# plt.colorbar()

	plt.subplot(2,3,4)
	ax = plt.gca()
	plt.imshow(region_of_interest, vmin=0.1)
	plt.imshow(shadows, cmap=cmap, alpha=0.35, vmin=0, vmax=1)
	plt.yticks([],[])   
	plt.xticks([],[]) 
	
	plt.subplot(2,3,5)
	ax = plt.gca()
	shadows = scipy.ndimage.zoom(shadows, np.shape(satellite)[0]/np.shape(shadows)[0], order=0)
	plt.imshow(satellite)
	plt.imshow(shadows, cmap=cmap, alpha=0.35, vmin=0, vmax=1)
	plt.yticks([],[])   
	plt.xticks([],[]) 

	plt.subplots_adjust(hspace=0,wspace=0)
	plt.tight_layout()        
	plt.savefig(filename, transparent=True)
	if show:
		plt.show()
	plt.close('all')


def analyse_garden(shadows, boundaries_pixels_in):

	boundaries_pixels = boundaries_pixels_in.copy()
	boundaries_pixels[:,0] = boundaries_pixels[:,0]+(np.shape(shadows)[0]/2)
	boundaries_pixels[:,1] = boundaries_pixels[:,1]+(np.shape(shadows)[1]/2)
	boundaries_pixels_swap = boundaries_pixels.copy()
	boundaries_pixels_swap[:,1] = boundaries_pixels[:,0]
	boundaries_pixels_swap[:,0] = boundaries_pixels[:,1]

	poly = plt.Polygon(boundaries_pixels_swap, ec="w", fill=False)
	path = poly.get_path()

	rows = np.arange(np.shape(shadows)[0]).reshape(np.shape(shadows)[0], 1)
	cols = np.arange(np.shape(shadows)[1]).reshape(1, np.shape(shadows)[1])
	meshgrid_rows, meshgrid_cols = np.meshgrid(rows, cols)
	result_array = np.stack((meshgrid_cols,meshgrid_rows), axis=2)
	result_array = result_array.reshape((np.shape(result_array)[0]*np.shape(result_array)[1],2))
	result_array_swap = result_array.copy()
	result_array_swap[:,1] = result_array[:,0]
	result_array_swap[:,0] = result_array[:,1]

	where_contained = path.contains_points(result_array_swap)

	is_sunny_pad_garden = np.expand_dims(shadows, -1)
	save_shape = np.shape(is_sunny_pad_garden)
	is_sunny_pad_garden = is_sunny_pad_garden.reshape((np.shape(result_array)[0],1))
	
	is_sunny_pad_garden_only = is_sunny_pad_garden[where_contained]

	is_sunny_pad_garden[~where_contained] = -1
	is_sunny_pad_garden = is_sunny_pad_garden.reshape(save_shape)

	is_sunny_pad_garden_cut = sunTrapp.utilities.remove_outer_regions(is_sunny_pad_garden)

	# plt.imshow(is_sunny_pad_garden_cut)
	# plt.show()

	fraction_sunny = np.sum(is_sunny_pad_garden_only)/np.shape(is_sunny_pad_garden_only)[0]

	return fraction_sunny, is_sunny_pad_garden_cut


def debug_plot_boundaries(shadows, boundaries_pixels, filename, region_of_interest, satellite, time_string=None, date=None, show=False):

	width = 4
	height = width * (np.shape(shadows)[0]/np.shape(shadows)[1])
	n_w = 4
	n_h = 2
	plt.figure(figsize=(width*n_w,height*n_h))

	plt.subplot(n_h,n_w,1)
	plt.imshow(region_of_interest, norm=LogNorm())
	# plt.colorbar()

	plt.subplot(n_h,n_w,2)
	plt.imshow(satellite)
	# plt.colorbar()

	plt.subplot(n_h,n_w,3)
	plt.imshow(shadows, cmap=cmap, vmin=0, vmax=1)
	# plt.colorbar()

	plt.subplot(n_h,n_w,4)
	ax = plt.gca()
	plt.imshow(region_of_interest, vmin=0.1)
	plt.imshow(shadows, cmap=cmap, alpha=0.35, vmin=0, vmax=1)
	plt.yticks([],[])   
	plt.xticks([],[]) 
	
	plt.subplot(n_h,n_w,5)
	ax = plt.gca()
	zoom = np.shape(satellite)[0]/np.shape(shadows)[0]
	shadows = scipy.ndimage.zoom(shadows, zoom, order=0)
	plt.imshow(satellite)
	plt.imshow(shadows, cmap=cmap, alpha=0.35, vmin=0, vmax=1)
	plt.yticks([],[])   
	plt.xticks([],[]) 

	boundaries_pixels *= zoom
	boundaries_pixels[:,0] = boundaries_pixels[:,0]+(np.shape(shadows)[0]/2)
	boundaries_pixels[:,1] = boundaries_pixels[:,1]+(np.shape(shadows)[1]/2)
	boundaries_pixels_swap = boundaries_pixels.copy()
	boundaries_pixels_swap[:,1] = boundaries_pixels[:,0]
	boundaries_pixels_swap[:,0] = boundaries_pixels[:,1]

	poly = plt.Polygon(boundaries_pixels_swap, ec="w", fill=False)
	path = poly.get_path()
	ax = plt.gca()
	ax.add_patch(poly)


	plt.subplot(n_h,n_w,6)

	rows = np.arange(np.shape(shadows)[0]).reshape(np.shape(shadows)[0], 1)
	cols = np.arange(np.shape(shadows)[1]).reshape(1, np.shape(shadows)[1])
	meshgrid_rows, meshgrid_cols = np.meshgrid(rows, cols)
	result_array = np.stack((meshgrid_cols,meshgrid_rows), axis=2)
	result_array = result_array.reshape((np.shape(result_array)[0]*np.shape(result_array)[1],2))
	result_array_swap = result_array.copy()
	result_array_swap[:,1] = result_array[:,0]
	result_array_swap[:,0] = result_array[:,1]

	where_contained = path.contains_points(result_array_swap)

	is_sunny_pad_garden = np.expand_dims(shadows, -1)
	is_sunny_pad_garden = is_sunny_pad_garden.reshape((np.shape(result_array)[0],1))
	is_sunny_pad_garden = np.concatenate((is_sunny_pad_garden,result_array_swap),-1)

	is_sunny_pad_garden = is_sunny_pad_garden[where_contained]
	
	# plt.imshow(shadows)
	plt.imshow(satellite)
	plt.scatter(is_sunny_pad_garden[:,1], is_sunny_pad_garden[:,2], c=is_sunny_pad_garden[:,0], vmin=-1, vmax=2,alpha=0.5, cmap='Blues')
	ax = plt.gca()
	poly = plt.Polygon(boundaries_pixels_swap, ec="w", fill=False)
	ax.add_patch(poly)



	plt.subplot(n_h,n_w,7)
	plt.scatter(is_sunny_pad_garden[:,1], is_sunny_pad_garden[:,2], c=is_sunny_pad_garden[:,0], vmin=0, vmax=1)
	plt.gca().invert_yaxis()

	plt.subplots_adjust(hspace=0,wspace=0)
	plt.tight_layout()        
	plt.savefig(filename, transparent=True)
	if show:
		plt.show()
	plt.close('all')



import matplotlib.pyplot as plt
import scipy.ndimage
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from matplotlib.colors import LogNorm

sun_color = '#FFD700'  # Brighter yellow color for sun
shade_color = '#121110'  # Darker blue color for shade

colors = [shade_color,sun_color]
cmap = LinearSegmentedColormap.from_list('sun_shade_cmap', colors)

def publish_plot(shadows, filename, overlay=None, time_string=None, date=None, show=False):
	
	width = 8
	height = width * (np.shape(shadows)[0]/np.shape(shadows)[1])

	plt.figure(figsize=(width,height))
	ax = plt.gca()
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
	plt.subplots_adjust(hspace=0,wspace=0)
	plt.tight_layout()        
	plt.savefig(filename, transparent=True)
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



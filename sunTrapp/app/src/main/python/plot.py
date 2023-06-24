import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import io
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import ephem
from os.path import dirname, join
import pickle
from pyproj import Transformer
from affine import Affine
from PIL import Image
import imageio
from com.chaquo.python import Python
import os

def transform_rowcol(transformer, x_coords, y_coords):
    inv_transformer = transformer.inverse
    row_col_tuples = []
    for x, y in zip(x_coords, y_coords):
        row, col = inv_transformer * (x, y)
        row, col = int(row), int(col)
        row_col_tuples.append((row, col))
    return row_col_tuples[0], row_col_tuples[1]

def plot_func():
	# # plot image
	# f = io.BytesIO()
	# rand = np.random.normal(0,1,(1000,2))
	# plt.hist2d(rand[:,0], rand[:,1], bins=25, range=[[-5,5],[-5,5]])
	# plt.savefig(f, format="png")
	# return f.getvalue()

	context = Python.getPlatform().getApplication()
	
	f = io.BytesIO()
	image_files = []
	for i in range(5):
		data = np.random.normal(0,1,(1000,2))
		plt.hist2d(data[:, 0], data[:, 1], bins=25, range=[[-5, 5], [-5, 5]])
		plt.savefig(join(dirname(__file__), f"output_{i}.png"))
		plt.savefig(f, format="png")
		plt.close()
		image_files.append(Image.open(join(dirname(__file__), f"output_{i}.png")))

	# gif_file = io.BytesIO()
	# imageio.mimsave(gif_file, image_files, format="GIF", duration=0.5)
	file_path = os.path.join(context.getFilesDir().getAbsolutePath(), "gif_file.gif")
        
	image_files[0].save(
		file_path,
		format="GIF",
		append_images=image_files[1:],
		save_all=True,
		duration=100,  # Time delay between frames in milliseconds
		loop=0,  # Number of loops (0 means infinite loop)
	)
	# print(file_path)
	# with open(file_path, 'rb') as file:
	# 	gif_bytes = file.read()

	# return gif_bytes
	return f.getvalue()





	# # plot tif file
	# filename_npy = join(dirname(__file__), "DSM_ST5570_P_10631_20190117_20190117.npy")
	# I = np.load(filename_npy,allow_pickle=True)

	# transformer_toGPS = Transformer.from_crs("EPSG:27700", "EPSG:4326")

	# filename_transformer = join(dirname(__file__), "DSM_ST5570_P_10631_20190117_20190117_transformer.pkl")
	# with open(filename_transformer, 'rb') as f:
	# 	transformer = pickle.load(f)

	# f = io.BytesIO()
	# plt.imshow(I)
	# plt.savefig(f, format="png")
	# plt.savefig(join(dirname(__file__), "output.png"))
	# return f.getvalue()

	

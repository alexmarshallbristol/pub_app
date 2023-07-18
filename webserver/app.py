from flask import Flask, render_template, request, send_file
from PIL import Image
from flask import jsonify
import io
import processes
import numpy as np
import os
import glob

app = Flask(__name__)

coordinates_file = 'coordinates.txt'
process_runner = processes.process_runner()


@app.route('/record_coordinates', methods=['POST'])
def record_coordinates():
	data = request.get_json()
	x = data['x']
	y = data['y']

	# Append the coordinates to the text file
	with open(coordinates_file, 'a') as f:
		f.write(f'{x},{y}\n')

	print(f'Recorded coordinates: {x}, {y}')


	return jsonify({'message': 'Coordinates recorded successfully'})

@app.route('/processPolygon', methods=['POST'])
def processPolygon():
	root_dir = os.getcwd()
	files = glob.glob(os.path.join(root_dir, f'images/garden*.png'))
	for file in files:
		os.remove(file)
	timestamp = process_runner.get_new_timestamp()
	publish, file_name = process_runner.plot_polygon(coordinates_file, timestamp)
	return jsonify(file_path=file_name)

@app.route('/generated_image')
def generated_image():
	file_path = request.args.get('file_path')
	return render_template('generated_image.html', file_path=file_path)


@app.route('/clear_coordinates', methods=['POST'])
def clear_coordinates():
	with open(coordinates_file, 'w') as f:
		f.write('')  # Clear the contents of the file

	print('Cleared coordinates file')

	return jsonify({'message': 'Coordinates file cleared successfully'})



@app.route('/')
def home():
	return render_template('index.html')



@app.route('/api/process', methods=['GET'])
def api_process():
	input_location_string = request.args.get('location')
	
	if input_location_string:
	
		file_path = process_runner.run_process(input_location_string)

		image = Image.open(file_path)
		image_buffer = io.BytesIO()
		image.save(image_buffer, format='PNG')
		image_buffer.seek(0)
		
		# Send the image as a response
		return send_file(image_buffer, mimetype='image/png')

	return 'No input text provided'

@app.route('/process', methods=['POST'])
def process():
	input_location_string = request.form['location']
	clear_coordinates()
	file_path = process_runner.run_process(input_location_string)
	return render_template('index.html', file_path=file_path)

@app.route('/images/<path:filename>')
def serve_image(filename):
	return send_file(f'{filename}')


if __name__ == '__main__':
	# host = '0.0.0.0'  # Set to '0.0.0.0' to make the server accessible externally
	# port = 8080  # Set to the desired port number
	# app.run(host=host, port=port)
	app.run(debug=True)
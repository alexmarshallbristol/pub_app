from flask import Flask, render_template, request, send_file
from PIL import Image
from flask import jsonify
import io
import processes
import numpy as np
import os
import glob
from celery import Celery
from celery.result import AsyncResult
from celery import states
import time

app = Flask(__name__)
app.config["CELERY_BROKER_URL"] = "redis://localhost:6379"

celery = Celery(app.name, broker=app.config["CELERY_BROKER_URL"], backend='rpc://')
celery.conf.update(app.config)



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
# http://127.0.0.1:5000/api/process?location=50.911794092395404, -0.9654510884959012






@celery.task()
def __celery__run_process(input_location_string):
    try:
        __celery__run_process.update_state(state=states.STARTED)
        file_path = process_runner.run_process(input_location_string)
        __celery__run_process.update_state(state=states.SUCCESS, meta={'file_path': file_path})
        return file_path
    except Exception as e:
        __celery__run_process.update_state(state=states.FAILURE, meta={'error_message': str(e)})
        raise

@app.route('/process', methods=['POST'])
def process():
    input_location_string = request.form['location']
    clear_coordinates()
    task = __celery__run_process.apply_async(args=[input_location_string], retry=True)
	
    # Return the task ID to the client so it can query the status later
    return render_template('index.html', task_id=task.id)







@app.route('/images/<path:filename>')
def serve_image(filename):
	return send_file(f'{filename}')


if __name__ == '__main__':
	app.run(debug=True)
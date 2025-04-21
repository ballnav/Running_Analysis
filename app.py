import os
import csv
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading
import tempfile
import uuid

from video_processing import process_video

app = Flask(__name__)

# Configure upload folder - use env var for Render or default to local
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
# Create the upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Use temp directory for results to avoid permissions issues
RESULTS_FOLDER = os.environ.get('RESULTS_FOLDER', os.path.join(tempfile.gettempdir(), 'running_analysis_results'))
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Set maximum file size (10MB)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Store processing tasks
processing_tasks = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<path:filename>')
def result_file(filename):
    return send_from_directory(RESULTS_FOLDER, filename)

def process_video_task(video_path, task_id):
    try:
        # Update task status
        processing_tasks[task_id]['status'] = 'processing'
        
        # Process the video
        result = process_video(video_path, output_dir=RESULTS_FOLDER)
        
        if not result:
            processing_tasks[task_id]['status'] = 'failed'
            processing_tasks[task_id]['error'] = 'Video processing failed'
            return
        
        # Read the analysis results from the detailed CSV
        analysis_results = []
        with open(result['detailed_csv'], mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                analysis_results.append(row)
        
        # Read the summary data
        summary_data = {}
        try:
            with open(result['summary_csv'], mode='r') as summary_file:
                csv_reader = csv.DictReader(summary_file)
                summary_data = next(csv_reader)  # Get the first row
        except (FileNotFoundError, StopIteration):
            pass  # Handle missing summary file gracefully
        
        # Update task with results
        processing_tasks[task_id]['status'] = 'completed'
        processing_tasks[task_id]['result'] = {
            'analysis_results': analysis_results,
            'summary_data': summary_data,
            'output_video': result['output_video'].replace(RESULTS_FOLDER, '/results')
        }
    except Exception as e:
        processing_tasks[task_id]['status'] = 'failed'
        processing_tasks[task_id]['error'] = str(e)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Create a unique ID for this task
        task_id = str(uuid.uuid4())
        
        # Generate a unique filename to avoid collisions
        filename = secure_filename(file.filename)
        unique_filename = f"{task_id}_{filename}"
        video_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Save the uploaded video
        file.save(video_path)
        
        # Create task entry
        processing_tasks[task_id] = {
            'status': 'pending',
            'video_path': video_path
        }
        
        # Start processing in a background thread
        thread = threading.Thread(target=process_video_task, args=(video_path, task_id))
        thread.daemon = True
        thread.start()
        
        # Return the task ID immediately
        return jsonify({
            'task_id': task_id,
            'status': 'pending'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/task/<task_id>', methods=['GET'])
def check_task(task_id):
    if task_id not in processing_tasks:
        return jsonify({'error': 'Task not found'}), 404
    
    task = processing_tasks[task_id]
    response = {'status': task['status']}
    
    if task['status'] == 'completed':
        response['result'] = task['result']
    elif task['status'] == 'failed':
        response['error'] = task.get('error', 'Unknown error')
    
    return jsonify(response)

# Health check endpoint for Render
@app.route('/health')
def health():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Use PORT environment variable if set (for Render)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)

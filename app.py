import os
import csv
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Import the processing function from your actual module
# Adjust this import to match your project structure
from video_processing import process_video  # Or whatever your actual module is

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index13.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save the uploaded video with secure filename
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)

        # Process the video and get results
        result = process_video(video_path)

        if not result:
            return jsonify({'error': 'Video processing failed'}), 500

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

        # Return both the detailed analysis and summary data along with the video path
        return jsonify({
            'analysis_results': analysis_results,
            'summary_data': summary_data,
            'output_video': result['output_video']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

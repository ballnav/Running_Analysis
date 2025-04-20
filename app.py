import os
import csv
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

from video_processing import process_video  # ← ใช้ชื่อโมดูลที่ถูกต้อง

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')  # ← เปลี่ยนชื่อไฟล์ HTML


@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)

        result = process_video(video_path)

        if not result:
            return jsonify({'error': 'Video processing failed'}), 500

        analysis_results = []
        with open(result['detailed_csv'], mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                analysis_results.append(row)

        summary_data = {}
        try:
            with open(result['summary_csv'], mode='r') as summary_file:
                csv_reader = csv.DictReader(summary_file)
                summary_data = next(csv_reader)
        except (FileNotFoundError, StopIteration):
            pass

        return jsonify({
            'analysis_results': analysis_results,
            'summary_data': summary_data,
            'output_video': result['output_video']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)

import os
import csv
from flask import Flask, request, render_template, jsonify
from count import process_video  # Assuming 'count' is the correct module name

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index10.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Save the uploaded video file
        file = request.files['video']
        video_path = 'uploaded_video.mp4'
        file.save(video_path)

        # Process the video using the process_video function
        process_video(video_path)

        # Read the contents of 'count2.csv'
        analysis_results = []
        with open('count2.csv', mode='r') as count2_file:
            count2_reader = csv.DictReader(count2_file)
            for row in count2_reader:
                analysis_results.append(row)

        # Read the contents of 'gait_summary.csv'
        summary = []
        with open('gait_summary.csv', mode='r') as gait_summary_file:
            gait_summary_reader = csv.DictReader(gait_summary_file)
            for row in gait_summary_reader:
                summary.append(row)

        # Return both sets of results as JSON
        return jsonify({
            'analysis_results': analysis_results,
            'summary': summary
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Adjusted error code to 500 for server errors

if __name__ == '__main__':
    app.run(debug=True)

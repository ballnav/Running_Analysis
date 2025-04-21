import csv
import time
import cv2
import mediapipe as mp
import os
import glob
import datetime
import logging
import numpy as np

from calculate_angles import calculate_angle, calculate_trunk_lean, get_text_color, evaluate_angle, \
    count_conditions, get_point, evaluate_each_body
from running_phase import RunningGaitCycleCounter

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_video(video_path, output_dir="data"):
    try:
        logger.info(f"Starting processing for video: {video_path}")
        
        # Extract base filename to use for output files
        base_filename = os.path.splitext(os.path.basename(video_path))[0]

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create a video-specific folder for this analysis with timestamp to avoid overwriting
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_output_dir = os.path.join(output_dir, f"{base_filename}_{timestamp}")

        # Ensure unique folder by adding counter if needed
        folder_counter = 0
        original_video_output_dir = video_output_dir
        while os.path.exists(video_output_dir):
            folder_counter += 1
            video_output_dir = f"{original_video_output_dir}_{folder_counter}"

        os.makedirs(video_output_dir)

        # Define output file paths
        csv_detailed_filename = os.path.join(video_output_dir, f"{base_filename}_detailed.csv")
        csv_summary_filename = os.path.join(video_output_dir, f"{base_filename}_summary.csv")
        output_video_path = os.path.join(video_output_dir, f"{base_filename}_analyzed.mp4")

        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Open video using OpenCV
        cap = cv2.VideoCapture(video_path)

        # Check if video opened successfully
        if not cap.isOpened():
            logger.error(f"Could not open video {video_path}")
            return None

        # Get video properties for output video
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")

        # Initialize video writer with proper codec for web compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # compatible with most browsers
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Define CSV field names
        fieldnames = ['Frame', 'Running Gait Cycle', 'Sub Phase', '% Cycle', 'Trunk Lean', 'Front Knee Angle',
                    'Back Knee Angle', 'Front Hip Angle', 'Angle Each Body %', 'Result']
        fieldnames_summary = [
            'cycleCount', 'totalFrames',
            'trunkLeanValue', 'trunkLeanPercentage', 'trunkLeanRes',
            'frontKneeValue', 'frontKneePercentage', 'frontKneeRes',
            'backKneeValue', 'backKneePercentage', 'backKneeRes',
            'hipValue', 'hipPercentage', 'hipRes',
            'angleScore', 'angleRes',
            'GoodScore', 'GoodPercentage',
            'SatisfactoryScore', 'SatisfactoryPercentage',
            'Should_ImproveScore', 'Should_ImprovePercentage'
        ]
        
        # Initialize gait counter
        gait_counter = RunningGaitCycleCounter()

        # Data for summary
        total_frames_processed = 0
        total_gait_cycles = 0
        total_trunk_lean = 0
        total_front_knee_angle = 0
        total_back_knee_angle = 0
        total_front_hip_angle = 0
        total_angle_each_body = 0

        total_good = 0
        total_satisfactory = 0
        total_should_improve = 0

        logger.info(f"Processing video: {os.path.basename(video_path)}")
        logger.info(f"Saving results to: {video_output_dir}")

        # Process 1 frame every `stride` frames to speed up processing
        # Adjust this value based on your performance needs
        stride = max(1, int(fps / 10))  # Process 10 frames per second
        
        with open(csv_detailed_filename, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            frame_count = 0
            while cap.isOpened():
                ret, image = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                # Process every `stride` frames to improve performance
                if frame_count % stride != 0 and frame_count != 1:  # Always process the first frame
                    continue
                
                # Convert to RGB for mediapipe
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Process with mediapipe
                results = pose.process(image_rgb)

                # Initialize variables with default values
                trunk_lean = 0
                right_angle_knee = 0
                left_angle_knee = 0
                right_angle_hip = 0
                AngleEach = 0
                right_subphase = "Unknown"
                res_AngleEach = "Unknown"
                rknee_optimal = (0, 0)
                lknee_optimal = (0, 0)
                hip_optimal = (0, 0)
                trunk_optimal = (4, 20)

                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    left_ankle = (
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
                    left_knee = (
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
                    left_hip = (
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
                    left_shoulder = (
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
                    right_ankle = (
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
                    right_knee = (
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
                    right_hip = (
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
                    right_shoulder = (
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)

                    left_angle_knee = calculate_angle(left_hip, left_knee, left_ankle)
                    right_angle_knee = calculate_angle(right_hip, right_knee, right_ankle)
                    right_angle_hip = calculate_angle(right_knee, right_hip, right_shoulder)
                    trunk_lean = calculate_trunk_lean(right_hip, right_shoulder)
                    right_subphase = gait_counter.get_subphase(right_ankle, right_knee, right_hip)
                    left_subphase = gait_counter.get_subphase(left_ankle, left_knee, left_hip)
                    running_phase = gait_counter.get_running_phase(
                        right_ankle, right_knee, right_hip, 
                        left_ankle, left_knee, left_hip
                    )
                    gait_counter.process_phases(
                        right_ankle, right_knee, right_hip, 
                        left_ankle, left_knee, left_hip
                    )

                    trunk_optimal = (4, 20)
                    res_trunk = evaluate_angle(trunk_lean, trunk_optimal)
                    point_trunk = get_point(trunk_lean, trunk_optimal)

                    hip_optimal_map = {
                        ("Initial Contact", "Initial Contact"): (15, 65),
                        ("Midstance", "Midstance"): (10, 60),
                        ("Terminal Stance", "Terminal Stance"): (25, 75),
                        ("Preswing", "Preswing"): (15, 65),
                        ("Initial Swing", "Initial Swing"): (20, 70),
                        ("Midswing", "Midswing"): (30, 80),
                        ("Terminal Swing", "Terminal Swing"): (5, 55)
                    }
                    hip_optimal = hip_optimal_map.get((right_subphase, right_subphase), (0, 0))
                    res_righthip = evaluate_angle(right_angle_hip, hip_optimal)
                    point_righthip = get_point(right_angle_hip, hip_optimal)

                    knee_optimal_map = {
                        ("Initial Contact", "Initial Contact"): (0, 40),
                        ("Midstance", "Midstance"): (5, 45),
                        ("Terminal Stance", "Terminal Stance"): (35, 75),
                        ("Preswing", "Preswing"): (25, 65),
                        ("Initial Swing", "Initial Swing"): (55, 95),
                        ("Midswing", "Midswing"): (25, 65),
                        ("Terminal Swing", "Terminal Swing"): (95, 135)
                    }

                    rknee_optimal = knee_optimal_map.get((right_subphase, right_subphase), (0, 0))
                    lknee_optimal = knee_optimal_map.get((left_subphase, left_subphase), (0, 0))
                    res_rightknee = evaluate_angle(right_angle_knee, rknee_optimal)
                    res_leftknee = evaluate_angle(left_angle_knee, lknee_optimal)
                    point_rightknee = get_point(right_angle_knee, rknee_optimal)
                    point_leftknee = get_point(left_angle_knee, lknee_optimal)

                    # Calculate Angle Each Body %
                    Sum = point_trunk + point_righthip + point_rightknee + point_leftknee
                    AngleEach = (Sum / 4) * 100
                    res_AngleEach = evaluate_each_body(AngleEach, (70, 100))

                    total_subphases = sum(gait_counter.subphase_counts.values())
                    subphase_percentages = {
                        phase: (count / total_subphases) * 100 
                        for phase, count in gait_counter.subphase_counts.items()
                    } if total_subphases > 0 else {}

                    # Count conditions
                    conditions = [AngleEach]
                    condition_counts = count_conditions(conditions, (70, 100))
                    total_good += condition_counts["Good"]
                    total_satisfactory += condition_counts["Satisfactory"]
                    total_should_improve += condition_counts["Should Improve"]

                    # Write row data to CSV
                    row_data = {
                        'Frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                        'Running Gait Cycle': gait_counter.get_cycle_count(),
                        'Sub Phase': right_subphase,
                        '% Cycle': f'{round(subphase_percentages.get(right_subphase, 0), 2)}%',
                        'Trunk Lean': f'{round(trunk_lean, 2)} ({res_trunk})',
                        'Front Knee Angle': f'{round(right_angle_knee, 2)} ({res_rightknee})',
                        'Back Knee Angle': f'{round(left_angle_knee, 2)} ({res_leftknee})',
                        'Front Hip Angle': f'{round(right_angle_hip, 2)} ({res_righthip})',
                        'Angle Each Body %': f'{round(AngleEach, 2)} % ',
                        'Result': res_AngleEach,
                    }

                    writer.writerow(row_data)

                    # Update summary data
                    total_frames_processed += 1
                    total_gait_cycles = gait_counter.get_cycle_count()
                    total_trunk_lean += point_trunk
                    total_front_knee_angle += point_rightknee
                    total_back_knee_angle += point_leftknee
                    total_front_hip_angle += point_righthip
                    total_angle_each_body += AngleEach

                # Add analysis overlays to the frame
                cv2.putText(image, f'Trunk Lean: {trunk_lean:.2f} ', (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                          get_text_color(trunk_lean, trunk_optimal), 2, cv2.LINE_AA)
                cv2.putText(image, f'Front Knee Angle: {right_angle_knee:.2f} ', (50, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                          get_text_color(right_angle_knee, rknee_optimal), 2, cv2.LINE_AA)
                cv2.putText(image, f'Back Knee Angle: {left_angle_knee:.2f} ', (50, 90),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                          get_text_color(left_angle_knee, lknee_optimal), 2, cv2.LINE_AA)
                cv2.putText(image, f'Front Hip Angle: {right_angle_hip:.2f} ', (50, 110),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                          get_text_color(right_angle_hip, hip_optimal), 2, cv2.LINE_AA)
                cv2.putText(image, f"Sub Phase: {right_subphase}", (50, 190), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                          (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Total running gait cycles: {gait_counter.get_cycle_count()}", 
                          (50, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Angle Each Body % : {AngleEach:.2f} %", (50, 230), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # Draw skeleton on the frame if landmarks are detected
                if results and results.pose_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Write the frame to output video
                out.write(image)

                # No need for cv2.imshow in server environment
                # No need for sleep delays in server environment
                
                # Progress logging for long videos
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames of {total_frames} ({frame_count/total_frames*100:.1f}%)")

        # Release resources
        cap.release()
        out.release()
        
        # Skip cv2.destroyAllWindows() as it's not needed in server environment
        
        # Generate summary data after video processing
        if total_frames_processed > 0:
            logger.info(f"Generating summary for {total_frames_processed} processed frames")
            
            acc_trunk_lean = total_trunk_lean / total_frames_processed * 100
            acc_front_knee_angle = total_front_knee_angle / total_frames_processed * 100
            acc_back_knee_angle = total_back_knee_angle / total_frames_processed * 100
            acc_front_hip_angle = total_front_hip_angle / total_frames_processed * 100
            avg_angle_each_body = total_angle_each_body / total_frames_processed

            sum_acc = ((total_good + total_satisfactory) - total_should_improve)
            avg_good = total_good / total_frames_processed * 100
            avg_satisfactory = total_satisfactory / total_frames_processed * 100
            avg_should_improve = total_should_improve / total_frames_processed * 100
            sum_total = (sum_acc / total_frames_processed) * 100

            res_total = evaluate_each_body(sum_total, (70, 100))
            res_acc_trunk = evaluate_each_body(acc_trunk_lean, (70, 100))
            res_acc_front_knee = evaluate_each_body(acc_front_knee_angle, (70, 100))
            res_acc_back_knee = evaluate_each_body(acc_back_knee_angle, (70, 100))
            res_acc_hip = evaluate_each_body(acc_front_hip_angle, (70, 100))
            res_acc_each = evaluate_each_body(avg_angle_each_body, (65, 100))

            # Create summary data matching JavaScript expectations
            summary_data = {
                'cycleCount': total_gait_cycles,
                'totalFrames': total_frames_processed,

                # Trunk metrics
                'trunkLeanValue': f'{round(total_trunk_lean, 2)}',
                'trunkLeanPercentage': f'{round(acc_trunk_lean, 2)} %',
                'trunkLeanRes': res_acc_trunk,

                # Front knee metrics
                'frontKneeValue': f'{round(total_front_knee_angle, 2)}',
                'frontKneePercentage': f'{round(acc_front_knee_angle, 2)} %',
                'frontKneeRes': res_acc_front_knee,

                # Back knee metrics
                'backKneeValue': f'{round(total_back_knee_angle, 2)}',
                'backKneePercentage': f'{round(acc_back_knee_angle, 2)} %',
                'backKneeRes': res_acc_back_knee,

                # Hip metrics
                'hipValue': f'{round(total_front_hip_angle, 2)}',
                'hipPercentage': f'{round(acc_front_hip_angle, 2)} %',
                'hipRes': res_acc_hip,

                # Angle metrics
                'angleScore': f'{round(avg_angle_each_body, 2)}',
                'angleRes': res_acc_each,

                # Performance distribution
                'GoodScore': total_good,
                'GoodPercentage': f'{round(avg_good, 2)}%',
                'SatisfactoryScore': total_satisfactory,
                'SatisfactoryPercentage': f'{round(avg_satisfactory, 2)}%',
                'Should_ImproveScore': total_should_improve,
                'Should_ImprovePercentage': f'{round(avg_should_improve, 2)}%'
            }

            # Write summary results to CSV
            with open(csv_summary_filename, mode='w', newline='') as csv_summary_file:
                writer = csv.DictWriter(csv_summary_file, fieldnames=fieldnames_summary)
                writer.writeheader()
                writer.writerow(summary_data)

            logger.info(f"✓ Completed processing {os.path.basename(video_path)}")
            logger.info(f"  - {total_frames_processed} frames analyzed")
            logger.info(f"  - {total_gait_cycles} gait cycles detected")
            logger.info(f"  - Files saved to {video_output_dir}")

            return {
                "filename": os.path.basename(video_path),
                "detailed_csv": csv_detailed_filename,
                "summary_csv": csv_summary_filename,
                "output_video": output_video_path,
                "cycles": total_gait_cycles,
                "frames": total_frames_processed
            }
        else:
            logger.error(f"✗ Error processing {os.path.basename(video_path)}: No frames processed")
            return None
            
    except Exception as e:
        logger.exception(f"Error processing video {video_path}: {str(e)}")
        return None


def process_multiple_videos(video_dir, output_dir="output", file_types=None):
    """
    Process multiple videos in a directory
    
    Args:
        video_dir (str): Directory containing videos to process
        output_dir (str): Directory to save output files
        file_types (list): List of file extensions to process
        
    Returns:
        list: Results from processing each video
    """
    if file_types is None:
        file_types = ['.mp4', '.avi', '.mov']

    # Get all video files in the directory
    video_files = []
    for file_type in file_types:
        video_files.extend(glob.glob(os.path.join(video_dir, f'*{file_type}')))

    # Create main output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each video
    results = []
    total_videos = len(video_files)

    if total_videos == 0:
        logger.warning(f"No video files found in {video_dir} with extensions {file_types}")
        return []

    logger.info(f"Found {total_videos} videos to process")

    # Create a summary CSV for all videos with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_videos_summary_file = os.path.join(output_dir, f"all_videos_summary_{timestamp}.csv")
    summary_fieldnames = ["Filename", "Gait Cycles", "Frames", "Output Path"]

    with open(all_videos_summary_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_fieldnames)
        writer.writeheader()

        for i, video_path in enumerate(video_files):
            logger.info(f"\nProcessing video {i + 1}/{total_videos}: {os.path.basename(video_path)}")
            result = process_video(video_path, output_dir)

            if result:
                results.append(result)
                writer.writerow({
                    "Filename": result["filename"],
                    "Gait Cycles": result["cycles"],
                    "Frames": result["frames"],
                    "Output Path": os.path.dirname(result["output_video"])
                })

    logger.info(f"\nAll processing complete! {len(results)}/{total_videos} videos successfully processed")
    logger.info(f"Summary file created: {all_videos_summary_file}")

    return results


# Example usage
if __name__ == "__main__":
    video_directory = "input_videos"  # Change this to your directory with videos
    results = process_multiple_videos(video_directory)

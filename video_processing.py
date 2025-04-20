import csv
import time
import cv2
import mediapipe as mp

from calculate_angles3 import calculate_angle, calculate_trunk_lean, get_text_color,evaluate_angle, count_conditions, get_point, evaluate_each_body
from running_phase2 import RunningGaitCycleCounter


def process_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    csv_filename = 'count3.csv'
    fieldnames = ['Frame', 'Running Gait Cycle', 'Sub Phase', '% Cycle', 'Trunk Lean', 'Front Knee Angle', 'Back Knee Angle',
                  'Front Hip Angle', 'Angle Each Body %', 'Result']
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
    gait_counter = RunningGaitCycleCounter()

    # Data for summary
    total_frames = 0
    total_gait_cycles = 0
    total_trunk_lean = 0
    total_front_knee_angle = 0
    total_back_knee_angle = 0
    total_front_hip_angle = 0
    total_angle_each_body = 0

    total_good = 0
    total_satisfactory = 0
    total_should_improve = 0

    detailed_data = []  # To store detailed frame-by-frame data
    summary_data = {}  # To store summary data

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                break
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y)
                left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y)
                left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y)
                left_shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y)
                right_ankle = (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
                right_knee = (landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
                right_hip = (landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
                right_shoulder = (landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)

                left_angle_knee = calculate_angle(left_hip, left_knee, left_ankle)
                right_angle_knee = calculate_angle(right_hip, right_knee, right_ankle)
                right_angle_hip = calculate_angle(right_knee, right_hip, right_shoulder)
                trunk_lean = calculate_trunk_lean(right_hip, right_shoulder)
                right_subphase = gait_counter.get_subphase(right_ankle, right_knee, right_hip)
                left_subphase = gait_counter.get_subphase(left_ankle, left_knee, left_hip)
                running_phase = gait_counter.get_running_phase(right_ankle, right_knee, right_hip, left_ankle, left_knee, left_hip)
                gait_counter.process_phases(right_ankle, right_knee, right_hip, left_ankle, left_knee, left_hip)

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
                point_rightknee = get_point(right_angle_knee, rknee_optimal)
                res_leftknee = evaluate_angle(left_angle_knee, lknee_optimal)
                point_leftknee = get_point(left_angle_knee, lknee_optimal)

                # Calculate Angle Each Body %
                Sum = point_trunk + point_righthip + point_rightknee + point_leftknee
                AngleEach = (Sum / 4) * 100
                res_AngleEach = evaluate_each_body(AngleEach, (70, 101))

                total_subphases = sum(gait_counter.subphase_counts.values())
                subphase_percentages = {phase: (count / total_subphases) * 100 for phase, count in gait_counter.subphase_counts.items()}

                # นับเงื่อนไข
                conditions = [AngleEach]
                condition_counts = count_conditions(conditions, (70, 100))
                total_good += condition_counts["Good"]
                total_satisfactory += condition_counts["Satisfactory"]
                total_should_improve += condition_counts["Should Improve"]

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
                total_frames += 1
                total_gait_cycles = gait_counter.get_cycle_count()
                total_trunk_lean += point_trunk
                total_front_knee_angle += point_rightknee
                total_back_knee_angle += point_leftknee
                total_front_hip_angle += point_righthip
                total_angle_each_body += AngleEach

            cv2.putText(image, f'Trunk Lean: {trunk_lean:.2f} ', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        get_text_color(trunk_lean, trunk_optimal), 2, cv2.LINE_AA)
            cv2.putText(image, f'Front Knee Angle: {right_angle_knee:.2f} ', (50, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, get_text_color(right_angle_knee, rknee_optimal), 2, cv2.LINE_AA)
            cv2.putText(image, f'Back Knee Angle: {left_angle_knee:.2f} ', (50, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, get_text_color(left_angle_knee, lknee_optimal), 2, cv2.LINE_AA)
            cv2.putText(image, f'Front Hip Angle: {right_angle_hip:.2f} ', (50, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, get_text_color(right_angle_hip, hip_optimal), 2, cv2.LINE_AA)
            cv2.putText(image, f"Sub Phase: {right_subphase}", (50, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Total running gait cycles: {gait_counter.get_cycle_count()}", (50, 210),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(image, f"Angle Each Body % : {AngleEach:.2f} %", (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)

            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


    cap.release()
    cv2.destroyAllWindows()

    # Summary after video processing
    if total_frames > 0:
        acc_trunk_lean = total_trunk_lean / total_frames * 100
        acc_front_knee_angle = total_front_knee_angle / total_frames * 100
        acc_back_knee_angle = total_back_knee_angle / total_frames * 100
        acc_front_hip_angle = total_front_hip_angle / total_frames * 100
        avg_angle_each_body = total_angle_each_body / total_frames

        sum_acc = ((total_good + total_satisfactory) - total_should_improve)
        avg_good = total_good / total_frames * 100
        avg_satisfactory = total_satisfactory / total_frames * 100
        avg_should_improve = total_should_improve / total_frames * 100
        sum_total = (sum_acc / 100) * 100

        res_total = evaluate_each_body(sum_total, (70, 100))
        res_acc_trunk = evaluate_each_body(acc_trunk_lean, (70, 100))
        res_acc_front_knee = evaluate_each_body(acc_front_knee_angle, (70, 100))
        res_acc_back_knee = evaluate_each_body(acc_back_knee_angle, (70, 100))
        res_acc_hip = evaluate_each_body(acc_front_hip_angle, (70, 100))
        res_acc_each = evaluate_each_body(avg_angle_each_body, (60, 100))

        # Create summary data matching JavaScript expectations
        summary_data = {
            'cycleCount': total_gait_cycles,
            'totalFrames': total_frames,

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

            # Performance distribution - use exact keys expected by JS
            'GoodScore': total_good,
            'GoodPercentage': f'{round(avg_good, 2)}%',
            'SatisfactoryScore': total_satisfactory,
            'SatisfactoryPercentage': f'{round(avg_satisfactory, 2)}%',
            'Should_ImproveScore': total_should_improve,  # Changed format to match JS
            'Should_ImprovePercentage': f'{round(avg_should_improve, 2)}%'  # Changed format to match JS
        }

        # Write summary results to CSV
        with open('gait_summary.csv', mode='w', newline='') as csv_summary_file:
            writer = csv.DictWriter(csv_summary_file, fieldnames=fieldnames_summary)
            writer.writeheader()
            writer.writerow(summary_data)

    cap.release()
    cv2.destroyAllWindows()


# Example usage
video_path = 'your_video_path.mp4'
process_video(video_path)

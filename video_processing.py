import csv
import mediapipe as mp

from calculate_angles3 import calculate_angle, calculate_trunk_lean, evaluate_angle, count_conditions, get_point, evaluate_each_body
from running_phase2 import RunningGaitCycleCounter


def process_video(video_path):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = mp.ImageCapture(video_path)  # ถ้าไม่ใช้ OpenCV ต้องใช้วิธีโหลด frame แทน

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

    # ตัวแปรสำหรับค่าสรุป
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

    summary_data = {}

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        frame_number = 0
        for frame in cap:  # สมมติว่ามีวิธีวนผ่าน frame ได้จาก video_path โดยไม่ใช้ OpenCV
            frame_number += 1
            image_rgb = frame  # ต้องเป็น image ที่แปลงเป็น RGB แล้ว
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # [คำนวณตำแหน่ง และมุมต่าง ๆ เหมือนเดิม...]

                # [คำนวณคะแนน และผลการประเมินเหมือนเดิม...]

                row_data = {
                    'Frame': frame_number,
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

                # สะสมข้อมูลสำหรับสรุป
                total_frames += 1
                total_gait_cycles = gait_counter.get_cycle_count()
                total_trunk_lean += point_trunk
                total_front_knee_angle += point_rightknee
                total_back_knee_angle += point_leftknee
                total_front_hip_angle += point_righthip
                total_angle_each_body += AngleEach

                # นับเงื่อนไข
                conditions = [AngleEach]
                condition_counts = count_conditions(conditions, (70, 100))
                total_good += condition_counts["Good"]
                total_satisfactory += condition_counts["Satisfactory"]
                total_should_improve += condition_counts["Should Improve"]

    # Summary CSV
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

        summary_data = {
            'cycleCount': total_gait_cycles,
            'totalFrames': total_frames,
            'trunkLeanValue': f'{round(total_trunk_lean, 2)}',
            'trunkLeanPercentage': f'{round(acc_trunk_lean, 2)} %',
            'trunkLeanRes': res_acc_trunk,
            'frontKneeValue': f'{round(total_front_knee_angle, 2)}',
            'frontKneePercentage': f'{round(acc_front_knee_angle, 2)} %',
            'frontKneeRes': res_acc_front_knee,
            'backKneeValue': f'{round(total_back_knee_angle, 2)}',
            'backKneePercentage': f'{round(acc_back_knee_angle, 2)} %',
            'backKneeRes': res_acc_back_knee,
            'hipValue': f'{round(total_front_hip_angle, 2)}',
            'hipPercentage': f'{round(acc_front_hip_angle, 2)} %',
            'hipRes': res_acc_hip,
            'angleScore': f'{round(avg_angle_each_body, 2)}',
            'angleRes': res_acc_each,
            'GoodScore': total_good,
            'GoodPercentage': f'{round(avg_good, 2)}%',
            'SatisfactoryScore': total_satisfactory,
            'SatisfactoryPercentage': f'{round(avg_satisfactory, 2)}%',
            'Should_ImproveScore': total_should_improve,
            'Should_ImprovePercentage': f'{round(avg_should_improve, 2)}%'
        }

        with open('gait_summary.csv', mode='w', newline='') as csv_summary_file:
            writer = csv.DictWriter(csv_summary_file, fieldnames=fieldnames_summary)
            writer.writeheader()
            writer.writerow(summary_data)


# ตัวอย่างเรียกใช้งาน
video_path = 'your_video_path.mp4'
process_video(video_path)

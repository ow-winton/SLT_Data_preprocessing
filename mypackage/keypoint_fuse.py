import cv2
import mediapipe as mp
import csv
import numpy as np

import cv2
import numpy as np
import mediapipe as mp
import os
import csv

def mp_draw2raw(input_dir, output_dir):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False,
                        min_detection_confidence=0.5)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理每个jpg文件
    for img_file in os.listdir(input_dir):
        if img_file.endswith('.jpg'):
            image_path = os.path.join(input_dir, img_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"图像未加载: {image_path}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                output_path = os.path.join(output_dir, img_file)
                cv2.imwrite(output_path, image)
                print(f"Processed image saved to {output_path}")
            else:
                print("未检测到任何关键点。")

### 方法二：在纯黑背景上绘制关键点

def mp_draw2blk_bg(input_dir, output_dir):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False,
                        min_detection_confidence=0.5)

    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 处理每个jpg文件
    for img_file in os.listdir(input_dir):
        if img_file.endswith('.jpg'):
            image_path = os.path.join(input_dir, img_file)
            image = cv2.imread(image_path)
            if image is None:
                print(f"图像未加载: {image_path}")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            if results.pose_landmarks:
                height, width, _ = image.shape
                black_image = np.zeros((height, width, 3), dtype=np.uint8)
                mp_drawing.draw_landmarks(
                    black_image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )
                output_path = os.path.join(output_dir, img_file)
                cv2.imwrite(output_path, black_image)
                print(f"Processed image saved to {output_path}")
            else:
                print("未检测到任何关键点。")

# -*- coding: utf-8 -*-
import os
import numpy as np
import os
import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from Displayer import Displayer


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/pose_landmarker_heavy.task")
    parser.add_argument("--image", type=str, default="images/girl.jpg")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    base_options = python.BaseOptions(model_asset_path=args.model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options, output_segmentation_masks=True
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    disp = Displayer()

    image = mp.Image.create_from_file(args.image)

    detection_result = detector.detect(image)

    image = image.numpy_view()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image_copy = np.copy(image)
    image_copy = draw_landmarks_on_image(image_copy, detection_result)

    for _ in range(3):
        disp.show(np.hstack((image, image_copy)))
        time.sleep(1)

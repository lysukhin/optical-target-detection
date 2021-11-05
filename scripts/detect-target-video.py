import cv2

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from optard.aruco import ArucoTagsDetector
from optard.detection import compute_target_position, compute_target_position_with_perspective
from optard.vis import show


video_name = "media/video/test_00.mp4"


def main():
    detector = ArucoTagsDetector()

    reader = cv2.VideoCapture(video_name)
    fps = int(round(reader.get(cv2.CAP_PROP_FPS)))
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(num_frames):
        _, image = reader.read()
        corners, ids = detector.run(image)
        # target_position = compute_target_position(corners, ids)
        target_position = compute_target_position_with_perspective(corners, ids)
        show(image, corners, ids, target_position, fps=fps)


if __name__ == "__main__":
    main()

import cv2

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from optard.aruco import ArucoTagsDetector
from optard.detection import compute_target_position, compute_target_position_with_perspective
from optard.vis import show


image_name = "media/images/test_00.jpg"


def main():
    image = cv2.imread(image_name)

    detector = ArucoTagsDetector()
    corners, ids = detector.run(image)

    target_position = compute_target_position(corners, ids)
    print("Target position", target_position)

    compute_target_position_with_perspective(corners, ids)

    show(image, corners, ids, target_position)


if __name__ == "__main__":
    main()

import cv2

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from optard.aruco import ArucoTagsDetector
from optard.vis import show


image_name = "images/test_02.jpg"

    
def main():
    image = cv2.imread(image_name)

    detector = ArucoTagsDetector()
    corners, ids = detector.run(image)

    num_detected_tags = len(corners)
    print("Found tags:", num_detected_tags)
    show(image, corners, ids)
    

if __name__ == "__main__":
    main()

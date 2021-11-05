import cv2
from argparse import ArgumentParser

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from optard.aruco import ArucoTagsDetector
from optard.detection import compute_target_position_with_perspective
from optard.vis import show


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--input-filename", "-i", help="Path to input image filename.", 
                        default="media/images/test_00.jpg")
    parser.add_argument("--output-filename", "-o", help="Path to output image filename.")
    return parser.parse_args()


def main(args):
    image = cv2.imread(args.input_filename)

    detector = ArucoTagsDetector()
    corners, ids = detector.run(image)

    target_position = compute_target_position_with_perspective(corners, ids)
    print("Target position", target_position)

    image = show(image, corners, ids, target_position)

    if args.output_filename is not None:
        cv2.imwrite(args.output_filename, image)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

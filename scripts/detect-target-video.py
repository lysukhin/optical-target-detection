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
    parser.add_argument("--input-filename", "-i", help="Path to input video filename.", 
                        default="media/video/test_00.mp4")
    parser.add_argument("--output-filename", "-o", help="Path to output video filename.")
    return parser.parse_args()


def main(args):
    detector = ArucoTagsDetector()

    reader = cv2.VideoCapture(args.input_filename)
    fps = int(round(reader.get(cv2.CAP_PROP_FPS)))
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output_filename, fourcc, fps, (w, h))

    for i in range(num_frames):
        _, image = reader.read()
        corners, ids = detector.run(image)
        target_position = compute_target_position_with_perspective(corners, ids)
        image = show(image, corners, ids, target_position, fps=fps)
        writer.write(image)
    
    writer.release()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)

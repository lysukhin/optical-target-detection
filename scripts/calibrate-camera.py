"""
https://aliyasineser.medium.com/opencv-camera-calibration-e9a48bdd1844
https://learnopencv.com/camera-calibration-using-opencv/
"""

import cv2
import tqdm
import pickle
import numpy as np


video_name = "calibration/honor-8x_video_1080p_60fps/chessboard.mp4"
output_filename = "calibration/honor-8x_video_1080p_60fps/calib.pkl"
chessboard_size = (9, 6)

num_frames_to_use = 24
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def main():

    objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
    objp[:, :2] = np.mgrid[0: chessboard_size[0], 0: chessboard_size[1]].T.reshape(-1, 2)
    # objp = objp * square_size ???
    print(objp)

    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    reader = cv2.VideoCapture(video_name)
    num_frames = int(round(reader.get(cv2.CAP_PROP_FRAME_COUNT)))
    for i in tqdm.trange(0, num_frames, num_frames // num_frames_to_use):
        _, image = reader.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(image, chessboard_size, None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[::-1], None, None)
    print(ret)
    print(mtx)
    print(dist)
    with open(output_filename, "wb") as fp:
        pickle.dump(dict(K=mtx, D=dist), fp)


if __name__ == "__main__":
    main()

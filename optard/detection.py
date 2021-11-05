"""
https://learnopencv.com/homography-examples-using-opencv-python-c/
"""
import cv2
import numpy as np


def compute_target_position_with_perspective(corners, ids):

    from .template import image_width, image_height, margin, tag_image_size

    if ids is None:
        return None
    ids = [i[0] for i in ids]

    if len(corners) < 2:
        # TODO: add warning
        return None

    tag_points = np.asarray([
        [0.,                0.],
        [tag_image_size,    0.],
        [tag_image_size,    tag_image_size],
        [0.,                tag_image_size]
    ], dtype=np.float32)
    source_points_all = np.concatenate([tag_points]*4)
    source_points_all[4:12, 0] += image_width - 2 * margin - tag_image_size
    source_points_all[8:16, 1] += image_height - 2 * margin - tag_image_size

    source_points = []
    target_points = []
    for i in range(len(ids)):
        corner_id = ids[i]
        if corner_id > 3:
            # TODO: add warning
            continue

        corner = corners[i].reshape((4, 2))
        target_points.extend(corner)
        source_points.extend(source_points_all[corner_id*4: (corner_id+1)*4])

    source_points = np.asarray(source_points, dtype=np.float32)
    target_points = np.asarray(target_points, dtype=np.float32)

    transform_mat, status = cv2.findHomography(source_points, target_points)

    source_center_point = np.array([
        [image_width / 2 - margin, image_height / 2 - margin]
    ], dtype=np.float32)
    target_center_point = cv2.perspectiveTransform(source_center_point[None, :, :], transform_mat)
    
    target_position = target_center_point.ravel().tolist()
    return target_position

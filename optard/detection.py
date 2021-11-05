import cv2
import numpy as np


def compute_tags_centers(corners):
    centers = np.asarray([
        np.mean(corner.reshape((4, 2)), axis=0).tolist() for corner in corners
    ])
    return centers
    

def compute_target_position(corners, ids):
    if ids is None:
        return None

    ids = list(ids)
    centers = compute_tags_centers(corners)

    num_tags = len(centers)
    if num_tags == 4:
        # Use all corners
        position = np.mean(centers, axis=0).round().astype(np.int).tolist()
        return position
    elif num_tags > 1:
        # Use diagonals
        if 0 in ids and 2 in ids:
            center_0 = centers[ids.index(0)]
            center_2 = centers[ids.index(2)]
            position = [
                int(round((center_0[0] + center_2[0]) / 2)),
                int(round((center_0[1] + center_2[1]) / 2)),
            ]
            return position
        elif 1 in ids and 3 in ids:
            center_1 = centers[ids.index(1)]
            center_3 = centers[ids.index(3)]
            position = [
                int(round((center_1[0] + center_3[0]) / 2)),
                int(round((center_1[1] + center_3[1]) / 2)),
            ]
            return position
        else:
            # Can't use two adjacent corners (or not?..)
            return None
    else:
        # Can't use only 1 corner
        return None


def compute_target_position_with_perspective(corners, ids):

    from .template import image_width, image_height, margin, tag_image_size

    if ids is None:
        return None
    ids = list(ids)

    if len(corners) != 4:
        # FIXME: after debug
        return None

    tag_points = np.asarray([
        [0.,                0.],
        [tag_image_size,    0.],
        [tag_image_size,    tag_image_size],
        [0.,                tag_image_size]
    ], dtype=np.float32)
    source_points = np.concatenate([tag_points]*4)
    source_points[4:12, 0] += image_width - 2 * margin - tag_image_size
    source_points[8:16, 1] += image_height - 2 * margin - tag_image_size
    # print("source points", source_points)

    target_points = []
    for i in range(4):
        corner = corners[ids.index(i)].reshape((4, 2))
        target_points.extend(corner)
    target_points = np.asarray(target_points, dtype=np.float32)
    # print("target points", target_points)
    
    transform_mat, status = cv2.findHomography(source_points, target_points)
    # print("transform_mat", transform_mat)

    source_center_point = np.array([
        [image_width / 2 - margin, image_height / 2 - margin]
    ], dtype=np.float32)
    print("source center point", source_center_point)

    target_center_point = cv2.perspectiveTransform(source_center_point[None, :, :], transform_mat)
    print("transformed center point", target_center_point)

    center_point = target_center_point.ravel().tolist()
    return center_point

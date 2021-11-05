"""
https://docs.opencv.org/master/d5/dae/tutorial_aruco_detection.html
https://www.pyimagesearch.com/2020/12/21/detecting-aruco-markers-with-opencv-and-python/
"""
import cv2
import numpy as np


aruco_dictionary_name = 0   # cv2.aruco.DICT_4x4_50
aruco_dictionary = cv2.aruco.Dictionary_get(aruco_dictionary_name)

aruco_tag_image_size = 200
aruco_tag_border_bits = 1

aruco_detection_params = cv2.aruco.DetectorParameters_create()


def create_tags_images(dictionary=aruco_dictionary, 
                       num_ids=4, 
                       tag_image_size=aruco_tag_image_size, 
                       tag_border_bits=aruco_tag_border_bits):
    tags_images = []
    for i in range(num_ids):
        image = np.zeros(shape=(tag_image_size, tag_image_size), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, i, tag_image_size, image, tag_border_bits)
        tags_images.append(image)
    return tags_images


class ArucoTagsDetector:

    def __init__(self, 
                 dictionary=aruco_dictionary, 
                 detection_params=aruco_detection_params):
        self.dictionary = dictionary
        self.detection_params = detection_params
    
    def run(self, image):
        corners, ids, rejected = cv2.aruco.detectMarkers(
            image, 
            self.dictionary, 
            parameters=self.detection_params
        )
        return corners, ids

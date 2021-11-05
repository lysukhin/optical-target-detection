import cv2
import numpy as np

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from optard.aruco import create_tags_images
from optard.template import image_height, image_width, tag_image_size, margin


output_filename = "images/template.png"


def main():
    image = np.ones(shape=(image_height, image_width), dtype=np.uint8) * 255
    tags_images = create_tags_images(num_ids=4, tag_image_size=tag_image_size)

    # top left, ID = 0
    image[margin: margin + tag_image_size, margin: margin + tag_image_size] = tags_images[0]

    # top right, ID = 1
    image[margin: margin + tag_image_size, -margin - tag_image_size: - margin] = tags_images[1]

    # bottom right, ID = 2
    image[- margin - tag_image_size: - margin, -margin - tag_image_size: - margin] = tags_images[2]

    # bottom left, ID = 3
    image[- margin - tag_image_size: - margin, margin: margin + tag_image_size] = tags_images[3]

    # axis
    x_mid =  int(round(image_width / 2))
    y_mid =  int(round(image_height / 2))
    cv2.line(image, 
             (x_mid, 0), 
             (x_mid, image_height - 1), 
             (0, 0, 0), 2)
    cv2.line(image,
             (0, y_mid),
             (image_width - 1, y_mid),
             (0, 0, 0), 2)
    
    # target box
    cv2.rectangle(image, 
                  (x_mid - tag_image_size // 2, y_mid - tag_image_size // 2),
                  (x_mid + tag_image_size // 2, y_mid + tag_image_size // 2),
                  (0, 0, 0), 2)

    cv2.imwrite(output_filename, image)


if __name__ == "__main__":
    main()

import cv2


crop_size = 100
preview_size = 200


def show(image, corners, ids, target_position=None, max_height=480, fps=0):
    h, w = image.shape[:2]
    k = max_height / h
    if k < 1:
        image = cv2.resize(image, dsize=None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        corners = [cs * k for cs in corners]
        if target_position is not None:
            target_position[0] = int(round(target_position[0] * k))
            target_position[1] = int(round(target_position[1] * k))
 

    if target_position is not None:
        cv2.circle(image, target_position, 3, (255, 0, 0), 1)
        x, y = target_position
        crop = image[y - crop_size // 2: y + crop_size // 2, x - crop_size // 2: x + crop_size // 2].copy()
        
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    if target_position is not None:
        image[-preview_size:, -preview_size:] = cv2.resize(crop, dsize=(preview_size, preview_size))

    cv2.imshow("results", image)
    if fps > 0:
        cv2.waitKey(1000 // fps)
    else:
        cv2.waitKey()
    
    return image

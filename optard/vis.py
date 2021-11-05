import cv2


crop_size = 100
preview_size = 200
info_pos = (0.2, 0.1)


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
        
        text_pos = (int(info_pos[0] * w * k), int(info_pos[1] * h * k))
        text = f"Target: x = {x:4d}, y = {y:4d}"
        cv2.putText(image, text, text_pos, cv2.FONT_HERSHEY_DUPLEX, 1, (64, 255, 255), 1)


    cv2.aruco.drawDetectedMarkers(image, corners, ids)
    if target_position is not None:
        try:
            image[-preview_size:, -preview_size:] = cv2.resize(crop, dsize=(preview_size, preview_size))
        except cv2.error as e:
            print(f"cv2.error: {e}")

    cv2.imshow("results", image)
    if fps > 0:
        cv2.waitKey(1000 // fps)
    else:
        cv2.waitKey()
    
    return image

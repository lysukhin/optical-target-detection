# Optical target detection

![gif](media/video/test_00-result.gif)

Simple target point detection using template with ArUco markers.

Contains:
* Template with ArUco markers creation (see image [here](media/images/template.png))
* Template detection (using OpenCV-contib `aruco` module)
* Template center (`target`) position estimation using `findHomography()`
* KalmanFilter-based point tracking

### Requirements
```
pip install numpy opencv-python opencv-contrib-python tqdm filterpy
```


### Usage

* Create template file (ready to print):
```bash
python scripts/create-template.py
```

* Run detection in single image:
```bash
python scripts/detect-target.py [-i IMAGE_NAME -o OUTPUT_NAME]
```

* Run detection on video:
```bash
python scripts/detect-target-video.py [-i VIDEO_NAME -o OUTPUT_NAME]
```

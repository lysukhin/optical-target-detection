# Optical target detection

Simple target point detection (and tracking as `#TODO`) using template with ArUco markers (see image [here](media/images/template.png)).

### Requirements
```
pip install numpy opencv-python opencv-contrib-python tqdm
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
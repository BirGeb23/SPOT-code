# Spot: 360° Camera

Uses the Spot CAM's 360° camera to find people **all around Spot**, so Spot can turn to face someone beside or behind it. Faces and hands are too small in the 360° view, so it detects whole bodies (MediaPipe Object Detector); the PTZ camera still reads gestures (`../Gesture/`).

| File | What it does |
|---|---|
| `spot_360_feed.py` | Live 360° view with each person's direction (0° = front, +90° = Spot's left, ±180° = behind). `--survey` calibrates; `--skeleton` draws full skeletons; `--record` saves video. Spot doesn't move. |
| `spot_search360.py` | Used by the gesture programs with `--search-360`: when nobody is in control for 3 s, look around, turn in place to the closest person (corrected by the survey), then fine-turn once the PTZ sees them. |

## Setup

Copy these files together with `../Gesture/*.py` into the SDK's spot_cam folder, and download two more models there:
```
cp ~/SPOT-code/Gesture/*.py ~/SPOT-code/360_Camera/*.py ~/Spot_SDK/spot-sdk/python/examples/spot_cam/
cd ~/Spot_SDK/spot-sdk/python/examples/spot_cam
curl -L -o efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite
curl -L -o pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

## 1. Survey (once per robot / camera mount)

Open space, nobody else in view. Tape 8 marks **2 m from the centre of Spot's body**: front (where the PTZ points when the gesture program starts), front-left, left (Spot's own left), back-left, behind, back-right, right, front-right. The laptop calls out each spot ("Go to front… Hold still…"):
```
python3 spot_360_feed.py 192.168.80.3 --survey
```
It works out the calibration from the marks and saves `spot_360_calibration.json` (front + direction) and `spot_360_survey.json` (per-direction correction), then prints a table. Good results are within ~15° everywhere; if it warns about errors over 45°, the marks were mixed up (usually front/back) and the survey isn't used.

## 2. Live view (no motion)
```
python3 spot_360_feed.py 192.168.80.3                 # directions of everyone around Spot
python3 spot_360_feed.py 192.168.80.3 --skeleton      # plus full skeletons
python3 spot_360_feed.py 192.168.80.3 --record 30     # also save 30 s to pano_rec.mp4
```
Keys: `q` quit (switches the stream back to the PTZ), `s` snapshot.

## 3. Turn to face people
```
python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 2 --search-360 --pano-skeleton --drive real --no-walk --speed-scale 1.0
```
Watch for `[360] Found …`, `[360] Survey correction …`, `[SPOT] Turn done: turned X of Y deg`, and `fine turn …`. During a search a "Spot 360 search" window shows what it found (centre of that window = behind Spot). It won't turn for someone already within ±25° of the front, and waits 8 s between searches.

## Known limits

- A person **directly behind** Spot is hard to detect (the robot blocks part of the view); standing ~3 m away helps.
- The 360° stream is 1280×720 for the full circle (~3.6 px per degree): good for finding bodies, too coarse for faces, tags and gestures.
- The people detector takes ~200 ms per scan on a laptop.

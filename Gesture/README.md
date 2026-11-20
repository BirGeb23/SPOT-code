# Spot: Gesture Recognition

Drives Boston Dynamics Spot with **hand gestures**. Everything runs on a **laptop** connected to Spot's Wi-Fi: the laptop reads the Spot CAM PTZ video, runs MediaPipe, and sends movement commands through the Spot SDK. Nothing is installed on the robot.

| Gesture | Intent | Spot's movement | Spot says |
|---|---|---|---|
| Thumb up | `ENGAGE` | Leans forward, turns to face you, walks closer (stops at ~1.5 m) | "Hi there! Nice to see you." |
| Pointing up | `CHECK_IN` | One "I see you" nod per streak, slight lean, turns to face | "I see you. How are you doing?" |
| Open palm | `CAUTION` | Holds still; one slow look left and right if held | "Okay. I'll wait right here." |
| Thumb down | `DE_ESCALATE` | Leans back, backs away ~0.45 m | "Okay. I'll give you some space." |
| Anything else | `NEUTRAL` | Neutral stance; small bob after 30 s idle | (silent) |

## Files

| File | Part | What it does |
|---|---|---|
| `spot_gesture_basic.py` | **Program** | Show a gesture, Spot moves. Follows the person nearest the centre of the view. |
| `spot_gesture_tag.py` | **Program** | AprilTag version: only the person wearing the tag (AprilTag 36h11) controls Spot. `--skeleton` adds the skeleton check. |
| `spot_gestures.py` | Gesture | Shared pipeline: Spot CAM stream, face detection/tracking, gesture crop that follows the person, hold check, gesture → intent, distance/direction calibration. |
| `spot_tag_control.py` | Gesture | Tag logic: the controller is the face above the tag; hiding the tag >1.5 s stops control; handing the tag over hands over control. |
| `spot_skeleton.py` | Gesture | Full-body skeleton check: a hand only counts if it is at the end of the tag wearer's own arm. |
| `spot_sound.py` | Sound | The spoken phrases (Mac text-to-speech, played on the Spot CAM speaker). |
| `spot_adapter.py` | Movement | Intent → Spot posture, turning and walking, with speed/distance limits; clears behavior faults at startup. Mock mode prints instead of moving. |
| `spot_intent_test.py` | Tool | Keyboard tester: keys 1–5 send each intent (no camera). |

## Setup (laptop, once)

1. Spot SDK and a virtual environment:
   ```
   git clone https://github.com/boston-dynamics/spot-sdk.git ~/Spot_SDK/spot-sdk
   python3 -m venv ~/spot_venv
   source ~/spot_venv/bin/activate
   pip install bosdyn-client bosdyn-mission bosdyn-choreography-client
   pip install -r ~/Spot_SDK/spot-sdk/python/examples/spot_cam/requirements.txt
   pip install mediapipe opencv-contrib-python
   ```
2. Copy the code into the SDK's Spot CAM example folder (the programs use its `webrtc.py` video stream). Include `360_Camera/` if you'll use `--search-360`:
   ```
   cp ~/SPOT-code/Gesture/*.py ~/SPOT-code/360_Camera/*.py ~/Spot_SDK/spot-sdk/python/examples/spot_cam/
   ```
3. Download the models into that folder (needs internet, so before switching to Spot's Wi-Fi):
   ```
   cd ~/Spot_SDK/spot-sdk/python/examples/spot_cam
   curl -L -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
   curl -L -o blaze_face_full_range.tflite https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/latest/blaze_face_full_range.tflite
   curl -L -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
   ```
4. Connect to Spot's Wi-Fi (Spot is at `192.168.80.3`); the programs ask for Spot's username and password. Set up an E-stop (tablet, or the SDK's `estop` example).
5. **Calibrate distance and direction once** (stand where it tells you). Without `spot_calibration.json`, Spot won't walk or turn:
   ```
   python3 spot_gesture_tag.py 192.168.80.3 --calibrate
   ```

## Running

Run from the spot_cam folder with the venv active. Go in this order: intents only → mock → real without walking → real with walking. **E-stop operator ready and area clear for real motion.**

```
cd ~/Spot_SDK/spot-sdk/python/examples/spot_cam
source ~/spot_venv/bin/activate

# AprilTag version (tag ID 2)
python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 2                                   # intents printed only
python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 2 --drive mock                      # Spot commands printed, no motion
python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 2 --drive real --no-walk --speed-scale 1.0
python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 2 --drive real --speed-scale 1.0

# With the skeleton check
python3 spot_gesture_tag.py 192.168.80.3 --show --tag-ids 2 --skeleton --drive mock

# Basic version (no tag): same flags, without --tag-ids / --skeleton
python3 spot_gesture_basic.py 192.168.80.3 --show --drive mock
```

Useful options:
- `--speed-scale 1.0`: use it; Spot ignores the very slow speeds produced at half speed.
- `--search-360` (+ `--pano-skeleton`): turn to people beside/behind Spot with the 360° camera (see `../360_Camera/README.md`).
- Sound: `--volume 100`, `--voice Samantha`, `--sound-style tones`, `--no-sound`. Test the phrases alone: `python3 spot_sound.py 192.168.80.3`.
- `--skeleton-every 2`: run the skeleton check every other frame if it feels laggy.
- Preview keys: `q` quit (Spot stops and sits cleanly; prefer it over Ctrl+C), `r` reset who's in control.

**Keyboard tester** (no camera): `python3 spot_intent_test.py --mock` (no robot), or `python3 spot_intent_test.py 192.168.80.3 --no-walk --speed-scale 1.0` (real postures), then keys 1–5.

## Troubleshooting

- **`calibration not found` / Spot won't walk or turn:** run `--calibrate` in the spot_cam folder.
- **`BehaviorFaultError`:** a fault from a lease timeout or fall. The adapter clears clearable faults at startup (asks first for a fall); otherwise clear it on the tablet.
- **Lease error:** release control on the tablet first.
- **Stream shows PTZ + 360 stacked:** the program switches to `mech_full` automatically; `--crop-ptz` is a fallback.
- **`No module named aiortc` / `bosdyn`:** wrong environment; `source ~/spot_venv/bin/activate`.

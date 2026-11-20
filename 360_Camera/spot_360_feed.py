#!/usr/bin/env python3
"""
============================================================================
spot_360_feed.py  —  Step 1 of the 360° work: live 360° feed + where people are
============================================================================
Spot does NOT move in this script. It:

  1. Switches the Spot CAM stream to the 360° panorama ("pano_full") and
     shows it live (the black bars above/below are cut off automatically).
  2. Finds people all around Spot with MediaPipe's Object Detector (whole
     bodies: faces and hands are too small in the 360° stream). The
     panorama is cut into overlapping tiles so people stay big enough for
     the detector, and the left/right edges are joined so someone standing
     right on the seam isn't split in two.
  3. Converts each person's position into a direction around Spot:
        0 = straight ahead, +90 = Spot's left, -90 = Spot's right, ±180 = behind.
     That's what step 2 (turning to face someone behind Spot) will use.

One-time direction calibration (saved to spot_360_calibration.json):
    Stand ~2 m DIRECTLY IN FRONT of Spot, alone, and press  f
    Then stand ~2 m at Spot's LEFT side, alone, and press   l

Needs in this folder (Spot SDK examples/spot_cam): webrtc.py from the SDK,
spot_gestures.py, and the detector model:
    curl -L -o efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite

Usage:
    python3 spot_360_feed.py 192.168.80.3                 # live, preview window
    python3 spot_360_feed.py 192.168.80.3 --record 20     # also save 20 s to pano_rec.mp4
    python3 spot_360_feed.py --video pano_rec.mp4         # replay a recording
    python3 spot_360_feed.py 192.168.80.3 --survey        # guided accuracy survey (8 spots)
    python3 spot_360_feed.py 192.168.80.3 --skeleton      # also draw each person's full skeleton
Keys: q quit, f set front, l set left, s save a snapshot.
When you quit, the stream is switched back to the PTZ (mech_full).
============================================================================
"""

import argparse
import json
import os
import sys
import threading
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from spot_gestures import (HERE, LatestFrame, connect_robot, set_ptz_only_view,
                           start_spot_stream)

CAL_FILE = os.path.join(HERE, "spot_360_calibration.json")
SURVEY_FILE = os.path.join(HERE, "spot_360_survey.json")
# Survey stops: (name, true direction in degrees; + = Spot's left)
SURVEY_STOPS = [("FRONT", 0), ("FRONT-LEFT", 45), ("LEFT", 90), ("BACK-LEFT", 135),
                ("BEHIND", 180), ("BACK-RIGHT", -135), ("RIGHT", -90), ("FRONT-RIGHT", -45)]
SURVEY_WALK_S = 8.0       # time to walk to the next spot
SURVEY_MEASURE_S = 4.0    # time measured at each spot
PANO_SCREEN = "pano_full"
BAR_LEVEL = 8            # rows darker than this (mean) are the black bars
TILES = 6                # tiles around the 360° (60° each) ...
TILE_OVERLAP = 0.35      # ... each widened by this fraction on both sides
MIN_SCORE = 0.35         # person detection confidence
MIN_HEIGHT_FRAC = 0.12   # ignore "people" shorter than this fraction of the band
DETECT_EVERY = 3         # run the detector every N frames
MERGE_DEG = 6.0          # detections closer than this (degrees) are the same person


# ---------------------------------------------------------------------------
# Panorama geometry
# ---------------------------------------------------------------------------
def find_band(frame):
    """(top, bottom) rows of the picture, without the black bars."""
    rows = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean(axis=1)
    idx = np.where(rows > BAR_LEVEL)[0]
    if len(idx) == 0:
        return 0, frame.shape[0]
    return int(idx[0]), int(idx[-1]) + 1


def load_cal():
    try:
        with open(CAL_FILE) as f:
            c = json.load(f)
        return float(c["front_x"]), int(c.get("sign", 1))
    except Exception:
        return None, 1


def save_cal(front_x, sign):
    with open(CAL_FILE, "w") as f:
        json.dump({"front_x": front_x, "sign": sign}, f, indent=2)
    print(f"[CAL] Saved {CAL_FILE}: front at {front_x:.3f} of the width, sign {sign:+d}")


def bearing_deg(x_frac, front_x, sign):
    """Image column (0..1 of the width) -> degrees around Spot, in (-180, 180]."""
    a = (x_frac - front_x) * 360.0 * sign
    return (a + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# People detection all around
# ---------------------------------------------------------------------------
class PeopleFinder:
    def __init__(self, model_path):
        self.det = vision.ObjectDetector.create_from_options(
            vision.ObjectDetectorOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE,
                score_threshold=MIN_SCORE, category_allowlist=["person"]))
        self.ms = 0.0

    def find(self, band):
        """[(x_centre_frac, x0, y0, x1, y1, score)] for people in the band;
        x0/x1 may run past the right edge (wrap-around), x_centre_frac is 0..1."""
        h, w = band.shape[:2]
        pad = int(w / TILES * (1 + TILE_OVERLAP))
        ext = np.concatenate([band, band[:, :pad]], axis=1)      # join the seam
        step = w / TILES
        half = step * (0.5 + TILE_OVERLAP)
        found = []
        t = time.time()
        for k in range(TILES):
            c = step * (k + 0.5)
            x0 = int(max(0, c - half)) if k else 0
            x1 = int(min(ext.shape[1], c + half + (step * TILE_OVERLAP if k == 0 else 0)))
            tile = np.ascontiguousarray(cv2.cvtColor(ext[:, x0:x1], cv2.COLOR_BGR2RGB))
            res = self.det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=tile))
            for d in res.detections:
                b = d.bounding_box
                if b.height < MIN_HEIGHT_FRAC * h:
                    continue
                bx0, bx1 = x0 + b.origin_x, x0 + b.origin_x + b.width
                found.append((((bx0 + bx1) / 2.0 % w) / w, bx0, b.origin_y, bx1,
                              b.origin_y + b.height, d.categories[0].score))
        self.ms = 1000 * (time.time() - t)
        return merge(found)

    def close(self):
        self.det.close()


# Full-body skeleton (MediaPipe Pose Landmarker, 33 points) on each person found
POSE_BONES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16),          # shoulders, arms
              (11, 23), (12, 24), (23, 24),                              # torso
              (23, 25), (25, 27), (24, 26), (26, 28),                    # legs
              (27, 29), (29, 31), (27, 31), (28, 30), (30, 32), (28, 32),  # feet
              (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # hands
              (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)]  # face
POSE_PAD = 0.15           # extra margin around each person box for the pose model
POSE_MIN_VIS = 0.5


class SkeletonDrawer:
    """Runs the Pose Landmarker on a crop around each detected person and
    returns their skeletons in panorama coordinates."""

    def __init__(self, model_path):
        self.pose = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.IMAGE, num_poses=1,
                min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4))
        self.ms = 0.0
        self.found = 0

    def skeletons(self, band, people):
        h, w = band.shape[:2]
        ext = np.concatenate([band, band[:, :w // 2]], axis=1)   # people across the seam
        out = []
        t = time.time()
        for _, x0, y0, x1, y1, _ in people:
            bw, bh = x1 - x0, y1 - y0
            cx0 = int(max(0, x0 - POSE_PAD * bw))
            cx1 = int(min(ext.shape[1], x1 + POSE_PAD * bw))
            cy0 = int(max(0, y0 - POSE_PAD * bh))
            cy1 = int(min(h, y1 + POSE_PAD * bh))
            if cx1 - cx0 < 10 or cy1 - cy0 < 20:
                continue
            crop = np.ascontiguousarray(cv2.cvtColor(ext[cy0:cy1, cx0:cx1], cv2.COLOR_BGR2RGB))
            res = self.pose.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=crop))
            if res.pose_landmarks:
                cw, ch = cx1 - cx0, cy1 - cy0
                out.append([(cx0 + lm.x * cw, cy0 + lm.y * ch,
                             1.0 if getattr(lm, "visibility", None) is None else lm.visibility)
                            for lm in res.pose_landmarks[0]])
        self.ms = 1000 * (time.time() - t)
        self.found = len(out)
        return out

    @staticmethod
    def draw(view, skel):
        w = view.shape[1]
        for shift in (0, -w):                   # also draw the part wrapped past the seam
            for a, b in POSE_BONES:
                if skel[a][2] >= POSE_MIN_VIS and skel[b][2] >= POSE_MIN_VIS:
                    cv2.line(view, (int(skel[a][0] + shift), int(skel[a][1])),
                             (int(skel[b][0] + shift), int(skel[b][1])), (255, 0, 255), 2)
            for x, y, v in skel:
                if v >= POSE_MIN_VIS:
                    cv2.circle(view, (int(x + shift), int(y)), 2, (0, 255, 255), -1)

    def close(self):
        self.pose.close()


def merge(found):
    """Same person seen in two overlapping tiles -> keep the best one."""
    out = []
    for p in sorted(found, key=lambda p: -p[5]):
        if all(abs(((p[0] - q[0] + 0.5) % 1.0) - 0.5) * 360 > MERGE_DEG for q in out):
            out.append(p)
    return sorted(out, key=lambda p: p[0])


def say(text):
    """Announce on the laptop (Mac text-to-speech) and print."""
    print(f"\n>>> {text}")
    try:
        import subprocess
        subprocess.Popen(["say", text])
    except Exception:
        pass


def circ_mean_frac(xs):
    a = np.asarray(xs) * 2 * np.pi
    return float((np.arctan2(np.sin(a).mean(), np.cos(a).mean()) / (2 * np.pi)) % 1.0)


def fit_calibration(raw):
    """raw: [(true_deg, x_frac)]. Finds the front column and direction (sign)
    that best match the marks. Returns (front_x, sign, rms_error_deg)."""
    best = None
    for sign in (1, -1):
        front = circ_mean_frac([(x - t / (360.0 * sign)) % 1.0 for t, x in raw])
        res = [(bearing_deg(x, front, sign) - t + 180) % 360 - 180 for t, x in raw]
        rms = float(np.sqrt(np.mean(np.square(res))))
        if best is None or rms < best[2]:
            best = (front, sign, rms)
    return best


def run_survey(next_frame, finder, top, bottom):
    """Guided survey: the person stands at 8 marked spots around Spot (2 m from
    its centre). From where they appear in the 360 image, it works out the
    calibration itself (front column + direction; no f/l keys needed), saves
    it to spot_360_calibration.json, and saves the remaining per-spot errors
    to spot_360_survey.json (used to correct turns)."""
    raw = []                                   # (name, true_deg, x_frac or None, n)
    say("Survey starting. Stand alone, two metres from Spot's centre, at each spot I call.")
    for name, true_deg in SURVEY_STOPS:
        say(f"Go to {name.replace('-', ' ').lower()}.")
        end = time.time() + SURVEY_WALK_S
        while time.time() < end:
            if next_frame() is None:
                return
        say("Hold still.")
        xs = []
        end = time.time() + SURVEY_MEASURE_S
        while time.time() < end:
            f = next_frame()
            if f is None:
                return
            band = f[top:bottom]
            # Roll half a turn so someone BEHIND Spot isn't split by the image seam.
            people = finder.find(np.roll(band, band.shape[1] // 2, axis=1))
            if len(people) == 1:
                xs.append((people[0][0] - 0.5) % 1.0)
        if xs:
            raw.append((name, true_deg, circ_mean_frac(xs), len(xs)))
            print(f"    {name:12s} seen at {raw[-1][2]:.3f} of the image width ({len(xs)} readings)")
        else:
            raw.append((name, true_deg, None, 0))
            print(f"    {name:12s} NOT DETECTED (0 frames with exactly one person)")

    good = [(t, x) for _, t, x, _ in raw if x is not None]
    if len(good) < 4:
        say("Survey failed. Fewer than four spots were detected.")
        return
    front_x, sign, rms = fit_calibration(good)
    save_cal(front_x, sign)
    results = []
    print("\n    spot          true   measured   error")
    for name, t, x, n in raw:
        meas = None if x is None else bearing_deg(x, front_x, sign)
        results.append({"name": name, "true": t, "measured": meas, "n": n})
        if meas is None:
            print(f"    {name:12s} {t:+5d}      -        not detected")
        else:
            err = (meas - t + 180) % 360 - 180
            print(f"    {name:12s} {t:+5d}   {meas:+7.1f}   {err:+6.1f}")
    with open(SURVEY_FILE, "w") as f:
        json.dump({"front_x": front_x, "sign": sign, "rms_deg": rms, "stops": results}, f, indent=2)
    print(f"\n[SURVEY] Calibration fitted from the marks: front at {front_x:.3f}, sign {sign:+d}, "
          f"average error {rms:.0f} deg. Saved {SURVEY_FILE}.")
    if rms > 25:
        print("[SURVEY] WARNING: the marks don't fit well (average error over 25 deg). "
              "Check the marks are 45 deg apart, 2 m from Spot's centre, and nobody else is in view.")
    say("Survey done.")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hostname", nargs="?")
    ap.add_argument("--video", help="Replay a recorded panorama .mp4 instead of the live stream")
    ap.add_argument("--model", default=os.path.join(HERE, "efficientdet_lite0.tflite"))
    ap.add_argument("--record", type=float, default=0,
                    help="Also save this many seconds of the panorama to pano_rec.mp4")
    ap.add_argument("--no-show", action="store_true", help="No preview window (print only)")
    ap.add_argument("--skeleton", action="store_true",
                    help="Also draw each person's full skeleton (MediaPipe Pose Landmarker)")
    ap.add_argument("--pose-model", default=os.path.join(HERE, "pose_landmarker_full.task"),
                    help="Pose model for --skeleton (full recommended: people are small here)")
    ap.add_argument("--survey", action="store_true",
                    help="Guided survey at 8 marked spots around Spot: fits the calibration itself")
    args = ap.parse_args()
    if not args.video and not args.hostname:
        sys.exit("Give the robot IP, or --video FILE.mp4")
    if not os.path.exists(args.model) or os.path.getsize(args.model) == 0:
        sys.exit(f"Missing detector model {args.model}. Download it with:\n  curl -L -o "
                 "efficientdet_lite0.tflite https://storage.googleapis.com/mediapipe-models/"
                 "object_detector/efficientdet_lite0/float16/latest/efficientdet_lite0.tflite")

    finder = PeopleFinder(args.model)
    drawer = None
    if args.skeleton:
        if not os.path.exists(args.pose_model):
            sys.exit(f"Missing pose model {args.pose_model}. Download it with:\n  curl -L -o "
                     "pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/"
                     "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task")
        drawer = SkeletonDrawer(args.pose_model)
    skels = []
    shutdown = threading.Event()
    robot, cap, latest = None, None, LatestFrame()
    if args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        print(f"[SRC] Connecting to Spot at {args.hostname} ...")
        robot = connect_robot(args.hostname)
        set_ptz_only_view(robot, PANO_SCREEN)
        start_spot_stream(robot, args.hostname, latest, shutdown)

    def next_frame():
        if cap is not None:
            ok, f = cap.read()
            return f if ok else None
        return latest.get(timeout=20.0)

    frame = next_frame()
    if frame is None:
        sys.exit("No video frame received (check Spot's Wi-Fi and the stream).")
    top, bottom = find_band(frame)
    H, W = frame.shape[:2]
    print(f"[SRC] Stream {W}x{H}; panorama band rows {top}-{bottom} "
          f"({W / 360:.1f} px per degree)")
    front_x, sign = load_cal()
    if front_x is None:
        print("[CAL] No spot_360_calibration.json yet: directions are relative to the image "
              "centre. Stand in front of Spot and press f, then at its left and press l.")

    if args.survey:
        try:
            run_survey(lambda: next_frame(), finder, top, bottom)
        finally:
            shutdown.set()
            finder.close()
            if robot is not None:
                set_ptz_only_view(robot, "mech_full")
        return

    writer = None
    if args.record > 0:
        writer = cv2.VideoWriter("pano_rec.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))  # ~real loop rate
        print(f"[REC] Saving {args.record:.0f} s to pano_rec.mp4")
    t0 = last_print = time.time()
    frames, people = 0, []
    try:
        while frame is not None:
            frames += 1
            if writer is not None:
                if time.time() - t0 <= args.record:
                    writer.write(frame)
                else:
                    writer.release()
                    writer = None
                    print("[REC] Done: pano_rec.mp4")
            band = frame[top:bottom]
            if frames % DETECT_EVERY == 1:
                people = finder.find(band)
                if drawer is not None:
                    skels = drawer.skeletons(band, people)
            fx = 0.5 if front_x is None else front_x
            now = time.time()
            if now - last_print >= 1.0:
                txt = ", ".join(f"{bearing_deg(p[0], fx, sign):+.0f} deg" for p in people)
                sk = (f", skeletons {drawer.found}/{len(people)} (pose {drawer.ms:.0f} ms)"
                      if drawer is not None else "")
                print(f"[{now - t0:6.1f}s] {len(people)} person(s): {txt or '-'}   "
                      f"(detector {finder.ms:.0f} ms{sk})")
                last_print = now

            if not args.no_show:
                view = band.copy()
                bh, bw = view.shape[:2]
                for a in range(-180, 181, 45):                 # direction ticks
                    x = int(((fx + a / 360.0 * sign) % 1.0) * bw)
                    cv2.line(view, (x, 0), (x, 12), (255, 255, 255), 1)
                    cv2.putText(view, "FRONT" if a == 0 else f"{a:+d}", (x + 3, 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 255, 255) if a == 0 else (255, 255, 255), 1)
                for xc, x0, y0, x1, y1, sc in people:
                    for shift in (0, -bw):                     # draw wrapped boxes on both sides
                        cv2.rectangle(view, (int(x0 + shift), int(y0)),
                                      (int(x1 + shift), int(y1)), (0, 255, 0), 2)
                    cv2.putText(view, f"{bearing_deg(xc, fx, sign):+.0f} deg",
                                (int(xc * bw) - 25, int(y0) - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                for skel in skels:
                    SkeletonDrawer.draw(view, skel)
                cv2.imshow("Spot 360 (q quit, f front, l left, s snapshot)", view)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key in (ord("f"), ord("l")):
                    if len(people) != 1:
                        print(f"[CAL] Need exactly 1 person in view (found {len(people)}).")
                    elif key == ord("f"):
                        front_x = people[0][0]
                        save_cal(front_x, sign)
                    else:
                        if front_x is None:
                            print("[CAL] Press f (front) first.")
                        else:
                            raw = bearing_deg(people[0][0], front_x, 1)
                            sign = 1 if raw > 0 else -1
                            save_cal(front_x, sign)
                            print(f"[CAL] Left side now reads "
                                  f"{bearing_deg(people[0][0], front_x, sign):+.0f} deg")
                if key == ord("s"):
                    name = f"pano_snap_{int(time.time())}.jpg"
                    cv2.imwrite(name, frame)
                    print(f"[SNAP] {name}")
            frame = next_frame()
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        shutdown.set()
        finder.close()
        if drawer is not None:
            drawer.close()
        if robot is not None:
            try:
                set_ptz_only_view(robot, "mech_full")
            except Exception as e:
                print(f"[CAM] Could not switch back to mech_full: {e}")
        if not args.no_show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
============================================================================
spot_search360.py  —  Step 2 of the 360° work: turn to face someone beside/behind
============================================================================
Used by spot_gesture_basic.py / spot_gesture_tag.py with --search-360.

When nobody is in control in the PTZ view for SEARCH_AFTER_S seconds:
  1. The stream is switched to the 360° panorama (pano_full).
  2. People all around Spot are found with MediaPipe's Object Detector
     (spot_360_feed.PeopleFinder), over a few frames.
  3. The stream is switched back to the PTZ (mech_full).
  4. Spot turns in place toward the CLOSEST person (tallest in the image),
     using the direction from spot_360_calibration.json. Turning is a
     "coarse" turn; once the person is in the PTZ view, the normal face
     tracking does the fine alignment and gestures work as usual.

People within FRONT_IGNORE_DEG of straight ahead are not turned to (they are
already in front; the PTZ probably can't see their face because they're
facing away). After a search, the next one waits SEARCH_COOLDOWN_S.

A new gesture/intent interrupts the turn (same "interrupt & replace" rule as
every other movement). --no-turn: searches and prints, but never turns.
Step 3 (later): use the tag to pick the right person in a crowd.
============================================================================
"""

import math
import os
import time

import cv2
import numpy as np

from spot_360_feed import (PeopleFinder, find_band, load_cal, bearing_deg, PANO_SCREEN,
                           SURVEY_FILE, SkeletonDrawer)

ALIGN_MIN_DEG = 8.0
MAX_SURVEY_ERR_DEG = 45.0  # a survey error bigger than this = mixed-up marks, don't use it        # after a 360 turn, fine-turn if the PTZ face is off by more


def load_correction():
    """From spot_360_survey.json: a function measured-bearing -> true-bearing
    (piecewise linear around the circle). None if there's no usable survey."""
    import json
    try:
        with open(SURVEY_FILE) as f:
            stops = [st for st in json.load(f)["stops"] if st["measured"] is not None]
    except Exception:
        return None
    if len(stops) < 3:
        return None
    meas = np.array([st["measured"] for st in stops], dtype=float)
    err = np.array([(st["measured"] - st["true"] + 180) % 360 - 180 for st in stops])
    if np.max(np.abs(err)) > MAX_SURVEY_ERR_DEG:
        worst = stops[int(np.argmax(np.abs(err)))]
        print(f"[360] IGNORING spot_360_survey.json: {worst['name']} is off by "
              f"{float(np.max(np.abs(err))):.0f} deg, which means the marks or the f/l "
              "calibration were mixed up (e.g. front/back swapped). Redo the survey.")
        return None
    order = np.argsort(meas)
    meas, err = meas[order], err[order]
    xs = np.concatenate([meas - 360, meas, meas + 360])
    es = np.concatenate([err, err, err])

    def correct(b):
        t = b - float(np.interp(b, xs, es))
        return (t + 180) % 360 - 180
    return correct

SEARCH_AFTER_S = 3.0       # nobody in control this long -> look around
SEARCH_COOLDOWN_S = 8.0    # wait this long after a search before the next one
FRONT_IGNORE_DEG = 25.0    # don't turn for people already roughly in front
SWITCH_WAIT_S = 1.2        # time for the stream to change view
SEARCH_FRAMES = 3          # panorama frames to look at


class Search360:
    def __init__(self, robot, adapter, motion, model_path, allow_turn=True, show=False,
                 pose_model=None):
        self.robot = robot
        self.adapter = adapter
        self.motion = motion
        self.allow_turn = allow_turn
        self.show = show
        self.last_result = "no search yet"
        self.front_x, self.sign = load_cal()
        if self.front_x is None:
            raise RuntimeError("spot_360_calibration.json not found: run spot_360_feed.py "
                               "and press f (front) and l (left) first")
        if not os.path.exists(model_path):
            raise RuntimeError(f"missing detector model {model_path}")
        self.finder = PeopleFinder(model_path)
        self.drawer = None
        if pose_model:
            if os.path.exists(pose_model):
                self.drawer = SkeletonDrawer(pose_model)
                print(f"[360] Full-body skeletons drawn in the search window "
                      f"({os.path.basename(pose_model)})")
            else:
                print(f"[360] No skeletons: missing {pose_model}")
        from bosdyn.client.spot_cam.compositor import CompositorClient
        self.comp = robot.ensure_client(CompositorClient.default_service_name)
        self.correct = load_correction()
        print("[360] Direction correction from spot_360_survey.json: "
              + ("ON" if self.correct else "none (run spot_360_feed.py --survey)"))
        self.align_until = 0.0
        self.lost_since = None
        self.next_allowed = 0.0
        self.searches = self.turns = 0
        print(f"[360] Search ON: after {SEARCH_AFTER_S:.0f} s with nobody in control, "
              f"look around and {'turn to' if allow_turn else 'report (no turning)'} "
              "the closest person")

    def _switch(self, screen, next_frame):
        self.comp.set_screen(screen)
        end = time.time() + SWITCH_WAIT_S
        while time.time() < end:            # throw away frames of the old view
            next_frame()

    def status(self, now):
        """One line for the status print: why the search is / isn't running."""
        if self.lost_since is None:
            return "360: someone in control, no search"
        wait = max(SEARCH_AFTER_S - (now - self.lost_since), self.next_allowed - time.time())
        return f"360: search in {max(wait, 0):.0f} s (last: {self.last_result})"

    def _people_in(self, frame):
        """People in a panorama frame. The image is rolled by half a turn
        first, so the seam is IN FRONT of Spot (where the PTZ looks) and
        someone directly behind Spot sits whole in the middle of the image."""
        top, bottom = find_band(frame)
        band = frame[top:bottom]
        w = band.shape[1]
        rolled = np.roll(band, w // 2, axis=1)
        out = []
        for p in self.finder.find(rolled):
            x = (p[0] - 0.5) % 1.0                  # back to the original image
            out.append((bearing_deg(x, self.front_x, self.sign),
                        (p[4] - p[2]) / max(band.shape[0], 1), p[5], p))
        if self.show:
            view = rolled.copy()
            if self.drawer is not None:
                raw_people = [o[3] for o in out]
                for skel in self.drawer.skeletons(rolled, raw_people):
                    SkeletonDrawer.draw(view, skel)
            for b, hf, sc, (xc, x0, y0, x1, y1, _) in out:
                for shift in (0, -w):
                    cv2.rectangle(view, (int(x0 + shift), int(y0)), (int(x1 + shift), int(y1)),
                                  (0, 255, 0), 2)
                cv2.putText(view, f"{b:+.0f} deg {sc:.2f}", (int(xc * w) - 40, max(15, int(y0) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(view, "BEHIND", (w // 2 - 30, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.imshow("Spot 360 search (centre = behind Spot)", view)
            cv2.waitKey(1)
        return [o[:3] for o in out]

    def maybe_search(self, now, in_control, next_frame):
        """Call every frame. Returns True if a search just ran (the caller
        should drop gesture state, since the video switched views)."""
        if in_control:
            self.lost_since = None
            return False
        if self.lost_since is None:
            self.lost_since = now
        if now - self.lost_since < SEARCH_AFTER_S or time.time() < self.next_allowed:
            return False

        self.searches += 1
        print(f"[360] Nobody in control for {SEARCH_AFTER_S:.0f} s: looking around...")
        people = []
        try:
            self._switch(PANO_SCREEN, next_frame)
            for _ in range(SEARCH_FRAMES):
                frame, _ = next_frame()
                if frame is None:
                    break
                people += self._people_in(frame)
        finally:
            self._switch("mech_full", next_frame)
        self.next_allowed = time.time() + SEARCH_COOLDOWN_S
        self.lost_since = None

        if not people:
            print("[360] Nobody found around Spot.")
            self.last_result = "nobody found"
            return True
        # Same person in several frames -> one entry (closest bearing wins).
        merged = []
        for b, hfrac, sc in sorted(people, key=lambda p: -p[1]):
            if all(abs((b - m[0] + 180) % 360 - 180) > 10 for m in merged):
                merged.append((b, hfrac, sc))
        txt = ", ".join(f"{b:+.0f} deg (height {h:.0%})" for b, h, _ in merged)
        print(f"[360] Found {len(merged)} person(s): {txt}")
        bearing, hfrac, _ = merged[0]           # tallest = closest
        if self.correct is not None:
            raw, bearing = bearing, self.correct(bearing)
            print(f"[360] Survey correction: {raw:+.0f} -> {bearing:+.0f} deg")
        self.last_result = f"closest at {bearing:+.0f} deg"
        if abs(bearing) < FRONT_IGNORE_DEG:
            print(f"[360] Closest person is already in front ({bearing:+.0f} deg); "
                  "not turning (they may be facing away from the PTZ).")
            return True
        if not self.allow_turn or self.adapter is None:
            print(f"[360] Would turn {bearing:+.0f} deg (turning off or --drive none).")
            return True
        self.turns += 1
        self.next_allowed = time.time() + abs(bearing) / 20.0 + SEARCH_COOLDOWN_S
        print(f"[360] Turning {bearing:+.0f} deg toward the closest person.")
        angle = math.radians(bearing)

        def turn(stop_event):
            self.adapter.turn_by(angle, stop_event=stop_event)

        self.motion.dispatch(turn)
        self.align_until = time.time() + abs(bearing) / 20.0 + 6.0
        return True

    def want_align(self):
        """True once, right after a 360 turn, when the person is back in the
        PTZ view: the caller then fine-turns to centre them."""
        if time.time() < self.align_until:
            self.align_until = 0.0
            return True
        return False

    def close(self):
        self.finder.close()
        if self.drawer is not None:
            self.drawer.close()

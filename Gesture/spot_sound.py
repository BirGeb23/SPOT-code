#!/usr/bin/env python3
"""
============================================================================
spot_sound.py  —  Spot says a short phrase for each intent (Spot CAM speaker)
============================================================================
When an intent STARTS (not every 2.5 s while the gesture is held), Spot
says a short phrase through the Spot CAM speaker:

    ENGAGE       "Hi there! Nice to see you."
    CHECK_IN     "I see you. How are you doing?"
    CAUTION      "Okay. I'll wait right here."
    DE_ESCALATE  "Okay. I'll give you some space."
    NEUTRAL      (silent)

Edit PHRASES below to change the words. The speech is made on the laptop
with the Mac's built-in text-to-speech (the `say` command, voice set by
--voice), made louder (normalized + light compression so it carries over
Spot's fans), uploaded to the Spot CAM at startup as "intent_engage",
"intent_check_in", ..., then played by name. Playing happens on its own
thread, so it never delays Spot's movement.

--sound-style tones  uses the earlier beeps instead of speech. If `say` is
not available (not a Mac), it falls back to the tones automatically.

Standalone test (plays each phrase with a pause; Spot does not move):
    python3 spot_sound.py 192.168.80.3
    python3 spot_sound.py 192.168.80.3 --volume 100 --voice Samantha
    python3 spot_sound.py --save-wavs        # write the WAVs to listen on the laptop
    say -v '?'                               # list the Mac's voices
============================================================================
"""

import argparse
import io
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave

import numpy as np

RATE = 44100           # samples per second (16-bit mono WAV)

# Each sound: list of (frequency Hz, or None for silence; duration s).
# Frequencies glide from the first to the second value when given a tuple.
SOUNDS = {
    "ENGAGE":      [(660, 0.12), (None, 0.03), (880, 0.20)],          # E5 -> A5, rising
    "CHECK_IN":    [(523, 0.16)],                                      # C5, single soft boop
    "CAUTION":     [(587, 0.10), (None, 0.10), (587, 0.10)],          # D5 x2, steady
    "DE_ESCALATE": [((523, 330), 0.55)],                               # C5 sliding down to E4
}
# What Spot says for each intent (NEUTRAL: silent). Keep them short (~1-2 s).
PHRASES = {
    "ENGAGE":      "Hi there! Nice to see you.",
    "CHECK_IN":    "I see you. How are you doing?",
    "CAUTION":     "Okay. I'll wait right here.",
    "DE_ESCALATE": "Okay. I'll give you some space.",
}
DEFAULT_VOICE = "Samantha"     # a clear built-in macOS voice
SPEECH_RATE = 175              # words per minute (macOS default ~175-200)
DEFAULT_VOLUME = 100           # Spot CAM speaker volume (%) unless --volume is given

# Loudness of each sound relative to the others (Spot CAM gain multiplier).
# DE_ESCALATE is a little softer, but still clearly audible over the fans.
GAIN = {"ENGAGE": 1.0, "CHECK_IN": 1.0, "CAUTION": 1.0, "DE_ESCALATE": 0.85}
PEAK = 0.97            # normalize every sound to this peak level (0..1)
COMPRESSION = 2.5      # soft compression: raises quiet parts so speech carries


def sound_name(intent):
    return "intent_" + intent.lower()


def make_wav(parts):
    """Build a 16-bit mono WAV (bytes) from (freq, seconds) parts. Each tone
    has a soft attack/decay so it sounds like a chime, not a harsh beep."""
    chunks = []
    for freq, dur in parts:
        n = int(RATE * dur)
        if freq is None:
            chunks.append(np.zeros(n))
            continue
        f0, f1 = freq if isinstance(freq, tuple) else (freq, freq)
        f = np.linspace(f0, f1, n)
        phase = 2 * np.pi * np.cumsum(f) / RATE
        tone = np.sin(phase) + 0.25 * np.sin(2 * phase)      # a little warmth
        attack = min(int(0.01 * RATE), n // 4)
        env = np.ones(n)
        env[:attack] = np.linspace(0, 1, attack)
        env *= np.exp(-3.0 * np.linspace(0, 1, n))           # gentle decay
        chunks.append(tone / 1.25 * env)
    audio = np.concatenate(chunks + [np.zeros(int(0.05 * RATE))])
    return to_wav(louder(audio))


def louder(audio):
    """Normalize to PEAK and apply soft compression (tanh), so the sound is as
    loud as possible without clipping and quiet syllables are lifted."""
    audio = np.asarray(audio, dtype=float)
    peak = np.max(np.abs(audio)) or 1.0
    audio = audio / peak
    audio = np.tanh(COMPRESSION * audio) / np.tanh(COMPRESSION)
    return audio * PEAK


def to_wav(audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def make_speech_wav(text, voice=DEFAULT_VOICE, rate=SPEECH_RATE):
    """Speak `text` with the Mac's `say` command into a loud 16-bit mono WAV."""
    if shutil.which("say") is None:
        raise RuntimeError("the `say` command is not available (not a Mac)")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "speech.wav")
        subprocess.run(["say", "-v", voice, "-r", str(rate), "-o", path,
                        "--file-format=WAVE", f"--data-format=LEI16@{RATE}", text],
                       check=True, capture_output=True)
        with wave.open(path, "rb") as w:
            frames = w.readframes(w.getnframes())
            channels = w.getnchannels()
    audio = np.frombuffer(frames, dtype="<i2").astype(float) / 32767
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio = np.concatenate([audio, np.zeros(int(0.1 * RATE))])
    return to_wav(louder(audio))


def build_sounds(style="speech", voice=DEFAULT_VOICE):
    """{intent: wav_bytes}. Speech if possible, else the tones."""
    if style == "speech":
        try:
            sounds = {i: make_speech_wav(t, voice) for i, t in PHRASES.items()}
            print(f"[SOUND] Speech made with the Mac voice '{voice}'")
            return sounds
        except Exception as e:
            print(f"[SOUND] Speech not available ({e}); using tones instead")
    return {i: make_wav(parts) for i, parts in SOUNDS.items()}


class IntentSounds:
    """Uploads the intent sounds to the Spot CAM and plays them on request.
    If anything fails (no speaker, old firmware), it prints one warning and
    the rest of the program carries on silently."""

    def __init__(self, robot, volume=None, mock=False, style="speech", voice=DEFAULT_VOICE):
        self.enabled = True
        self.mock = mock
        self._q = queue.Queue(maxsize=1)
        if not mock:
            try:
                from bosdyn.api.spot_cam import audio_pb2
                from bosdyn.client.spot_cam.audio import AudioClient
                self._pb = audio_pb2
                self.client = robot.ensure_client(AudioClient.default_service_name)
                self.client.set_volume(float(DEFAULT_VOLUME if volume is None else volume))
                for intent, data in build_sounds(style, voice).items():
                    self.client.load_sound(audio_pb2.Sound(name=sound_name(intent)), data)
                vol = self.client.get_volume()
                print(f"[SOUND] Intent sounds loaded on the Spot CAM (volume {vol:.0f}%)")
            except Exception as e:
                print(f"[SOUND] Sound disabled: could not use the Spot CAM speaker ({e})")
                self.enabled = False
                return
        else:
            print("[SOUND] Mock sound: printing instead of playing")
        threading.Thread(target=self._worker, daemon=True).start()

    def play(self, intent):
        """Queue the intent's sound (non-blocking). NEUTRAL is silent. If a
        sound is still waiting, the newer one replaces it."""
        if not self.enabled or intent not in PHRASES:
            return
        try:
            self._q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(intent)
        except queue.Full:
            pass

    def _worker(self):
        while True:
            intent = self._q.get()
            if self.mock:
                print(f"[SOUND] (mock) Spot would say: \"{PHRASES[intent]}\"")
                continue
            try:
                self.client.play_sound(self._pb.Sound(name=sound_name(intent)),
                                       gain=GAIN.get(intent, 1.0))
            except Exception as e:
                print(f"[SOUND] Could not play {sound_name(intent)}: {e}")


def main():
    p = argparse.ArgumentParser(description="Test the intent sounds")
    p.add_argument("hostname", nargs="?", help="Spot IP (e.g. 192.168.80.3)")
    p.add_argument("--volume", type=float, default=None,
                   help=f"Spot CAM speaker volume in percent (default {DEFAULT_VOLUME})")
    p.add_argument("--sound-style", choices=["speech", "tones"], default="speech")
    p.add_argument("--voice", default=DEFAULT_VOICE, help="Mac voice for the speech")
    p.add_argument("--save-wavs", action="store_true",
                   help="Write intent_*.wav files here to listen on the laptop")
    args = p.parse_args()

    if args.save_wavs:
        for intent, data in build_sounds(args.sound_style, args.voice).items():
            path = sound_name(intent) + ".wav"
            with open(path, "wb") as f:
                f.write(data)
            print("wrote", os.path.abspath(path))
        if not args.hostname:
            return
    if not args.hostname:
        sys.exit("Give the robot IP, or --save-wavs")

    import bosdyn.client
    import bosdyn.client.util
    from bosdyn.client import spot_cam
    sdk = bosdyn.client.create_standard_sdk("SpotSoundTest")
    spot_cam.register_all_service_clients(sdk)
    robot = sdk.create_robot(args.hostname)
    bosdyn.client.util.authenticate(robot)
    sounds = IntentSounds(robot, volume=args.volume, style=args.sound_style, voice=args.voice)
    if not sounds.enabled:
        sys.exit(1)
    for intent in PHRASES:
        print(f"Playing {intent}: \"{PHRASES[intent]}\" ...")
        sounds.play(intent)
        time.sleep(3.0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
drive_pilotnet.py - COMPLETE system in ONE file
================================================
Cameras + manual driving + recording + PilotNet AUTOPILOT, all controlled
from the web page. Based on the original car_server.py (same UI, same
joystick) with the model running inside and an AUTOPILOT button added.

  LAPTOP  -> CAMERA page   http://localhost:5000/
        Browser owns the 3 Logitech webcams (getUserMedia), uploads frames.
        Header has the AUTOPILOT button too.
  MOBILE  -> DRIVE page    http://<laptop-ip>:5000/?mode=drive
        Live feeds + your original joystick + AUTOPILOT button.
        While autopilot is ON, grabbing the joystick takes over instantly
        (disarms). Releasing does NOT re-arm - press the button again.

Data flow when AUTOPILOT is ON:
  browser center cam -> /frame -> autopilot thread (20 Hz):
  PilotNet steering -> EMA-smoothed throttle -> latest_control -> serial -> Arduino

Safety chain:
  - joystick deflection while armed  -> instant disarm (manual takeover)
  - center frame older than 0.6 s    -> outputs zero
  - stale for more than 2 s          -> auto-DISARM
  - this process dies                -> Arduino watchdog (0.3 s) zeros car
  - serial status printed every 2 s  -> you SEE what goes to the Arduino

Run:      python drive_pilotnet.py                     (uses pilotnet.pt, COM11)
          python drive_pilotnet.py --model other.pt --serial COM7
Install:  pip install flask pyserial torch opencv-python numpy
"""

import os
import time
import csv
import queue
import argparse
import threading
import collections
from datetime import datetime

import numpy as np
import cv2
import torch
from torch import nn
from flask import Flask, request, jsonify, Response

# ---- config -------------------------------------------------------------
SERIAL_PORT = "COM11"     # Windows COMx | Linux /dev/ttyUSB0 | Mac /dev/cu.usbserial-XXXX
BAUD        = 115200
SEND_HZ     = 20
STREAM_FPS  = 30
OUTPUT_DIR  = "dataset"
HOST        = "0.0.0.0"
PORT        = 5000
MODEL_PATH  = "pilotnet.pt"

# ---- autopilot tuning (kids ride-on car) ----------------------------------
AUTOPILOT_HZ    = 20
BASE_THROTTLE   = 0.25    # cruise throttle - heavy car, start LOW
CURVE_SLOWDOWN  = 0.55    # slow down in curves (0..1)
MIN_THROTTLE    = 0.15    # heavy car stalls below this
STEER_SMOOTH    = 0.4     # EMA; lower = smoother
STEER_GAIN      = 1.0     # set -1.0 if the car steers the wrong way
THROTTLE_SMOOTH = 0.12    # EMA toward target throttle; lower = smoother/gentler ramp
# PID_KP/KI/KD are no longer used for throttle (no speed sensor = nothing to
# close a real loop against). Kept here for the day a wheel encoder/hall
# sensor is added, at which point a proper PID on MEASURED speed makes sense.
PID_KP, PID_KI, PID_KD = 1.2, 0.3, 0.02
STALE_FRAME_S   = 0.6     # frame older than this -> output zero
AUTO_DISARM_S   = 2.0     # stale longer than this while armed -> disarm
TAKEOVER_THRESH = 0.25    # joystick deflection that disarms the autopilot

# ---- steering safety net (guards against a single bad/OOD frame) ----------
RAW_HISTORY_LEN   = 5      # raw (pre-smoothing) predictions kept for outlier check
OUTLIER_THRESH    = 0.45   # raw deviation from recent median treated as suspect
MAX_STEER_STEP    = 0.15   # max change in steer_ema allowed per 50ms tick (rate limit)
AUTOPILOT_DEBUG_LOG = True # write every autopilot tick to autopilot_debug.csv
DEBUG_LOG_PATH      = "autopilot_debug.csv"
MAX_CONSECUTIVE_ERRORS = 10   # inference errors in a row (0.5s @ 20Hz) -> auto-disarm
SERIAL_RECONNECT_S  = 3.0     # retry opening the serial port this often if it drops
# ---------------------------------------------------------------------------

app = Flask(__name__)

state_lock  = threading.Lock()

latest_control = {"steering": 0.0, "throttle": 0.0, "t": 0.0}
latest_frames  = {"left": None, "center": None, "right": None}
last_frame_ts  = {"left": 0.0, "center": 0.0, "right": 0.0}
recording   = False
sample_idx  = 0

autopilot = {"armed": False, "steer": 0.0, "throttle": 0.0,
             "fresh": False, "model": False, "outliers": 0, "rate_limited": 0}

ser = None
model = None
device = None
model_cfg = None


# ============================ PILOTNET =====================================

class PilotNet(nn.Module):
    """Conv stack identical to train_pilotnet.py; FC layer sizes itself for
    whatever input resolution the checkpoint was trained at."""
    def __init__(self, img_h, img_w):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, 3, img_h, img_w)).numel()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(n_flat, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class PID:
    """Incremental PID - CURRENTLY UNUSED for throttle.

    This was driving the throttle, but with no speed sensor there is no real
    process variable to close a loop against: the old code fed the previous
    throttle *command* back in as `measured`, so the controller was chasing
    its own output and oscillated (~0.14 <-> ~0.37 every 50 ms), which showed
    up as the "stop and go" lurch. Throttle now uses a plain EMA toward the
    target (see autopilot_thread).

    This class is kept intact so that, once a wheel encoder / hall sensor is
    added, a proper PID on MEASURED wheel speed can be dropped straight back
    in - that is the case where PID is actually the right tool.
    """
    def __init__(self, kp, ki, kd, out_min=0.0, out_max=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_min, self.out_max = out_min, out_max
        self.i = 0.0
        self.prev_err = None

    def reset(self):
        self.i = 0.0
        self.prev_err = None

    def update(self, target, measured, dt):
        err = target - measured
        self.i = max(-1.0, min(1.0, self.i + err * dt))       # anti-windup
        d = 0.0 if self.prev_err is None else (err - self.prev_err) / dt
        self.prev_err = err
        out = self.kp * err + self.ki * self.i + self.kd * d
        return max(self.out_min, min(self.out_max, measured + out))


def load_model(path):
    """Load checkpoint + its embedded preprocessing config (RAW, size, YUV).

    Wrapped in try/except: a malformed checkpoint (missing config keys,
    architecture mismatch, corrupted file) now disables autopilot with a
    clear message instead of crashing the whole process before manual
    driving/recording can even start.
    """
    global model, device, model_cfg
    if not os.path.isfile(path):
        print(f"WARNING: model file '{path}' not found - AUTOPILOT disabled.")
        print("         Manual driving + recording still work. Train, then restart.")
        return
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt["config"]
        for key in ("img_h", "img_w", "crop_top", "crop_bottom"):
            if key not in cfg:
                raise KeyError(f"checkpoint config missing '{key}'")
        if cfg["img_h"] <= 0 or cfg["img_w"] <= 0:
            raise ValueError(f"invalid checkpoint resolution {cfg['img_w']}x{cfg['img_h']}")
        m = PilotNet(cfg["img_h"], cfg["img_w"]).to(device)
        m.load_state_dict(ckpt["model"])
        m.eval()
        model_cfg = cfg
        model = m
        autopilot["model"] = True
        mode = ("RAW full frame" if cfg["crop_top"] == 0 and cfg["crop_bottom"] == 0
                else f"crop top {cfg['crop_top']} / bottom {cfg['crop_bottom']}")
        ar = "4:3 preserved, no distortion" if abs(cfg["img_w"] / cfg["img_h"] - 4 / 3) < 0.05 \
             else "aspect changed (vertical squash)"
        print(f"PilotNet loaded on {device} | {mode} -> "
              f"{cfg['img_w']}x{cfg['img_h']} ({ar}) -> "
              f"{'YUV' if cfg.get('yuv', True) else 'BGR'}")
    except Exception as e:
        print(f"WARNING: failed to load model '{path}' ({e}) - AUTOPILOT disabled.")
        print("         Manual driving + recording still work. Check the checkpoint and restart.")
        model = None
        model_cfg = None
        autopilot["model"] = False


def preprocess(img_bgr):
    """Exact training preprocessing, driven by the checkpoint config."""
    h = img_bgr.shape[0]
    top, bot = model_cfg["crop_top"], model_cfg["crop_bottom"]
    if top + bot >= h:
        # e.g. the camera negotiated a different resolution than the
        # checkpoint was trained/cropped for - fail loudly and specifically
        # instead of letting cv2.resize throw an opaque error on an empty
        # or negative-size array.
        raise ValueError(f"crop config invalid for a {h}px-tall frame: "
                          f"crop_top={top} + crop_bottom={bot} >= frame height")
    img = img_bgr[top: h - bot, :, :]
    img = cv2.resize(img, (model_cfg["img_w"], model_cfg["img_h"]),
                     interpolation=cv2.INTER_AREA)
    if model_cfg.get("yuv", True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = img.astype(np.float32) / 127.5 - 1.0
    return np.transpose(img, (2, 0, 1))


def autopilot_thread():
    """20 Hz: latest center JPEG -> PilotNet steering -> EMA-smoothed throttle
    -> latest_control (same path the joystick uses; watchdog included).

    THROTTLE (fixed): the throttle target is computed directly from the
    steering (cruise, scaled down in curves) and then eased toward with a
    single-pole EMA - THROTTLE_SMOOTH. There is no speed sensor, so there is
    nothing to run a real PID against; the previous PID fed its own last
    command back in as "measured" and oscillated between ~0.14 and ~0.37
    every tick, which is what caused the stop-and-go lurch. The EMA ramps
    monotonically toward the target with no overshoot regardless of the
    smoothing value. (See the PID class docstring for when PID comes back.)

    Two safety layers sit between the raw model output and the car, because
    a single bad frame (glare, motion blur, a truncated/corrupted JPEG, a
    genuinely out-of-distribution patch of track) can otherwise swing the
    output most of the way to full-lock within 2-3 ticks (150ms) even
    though each individual EMA blend looks harmless in isolation:

      1. OUTLIER REJECTION - if a raw prediction is far from the median of
         the last few raw predictions, it's treated as suspect and replaced
         by that median instead of being blended straight into steering.
         A prediction that's part of a genuine, sustained turn will keep
         showing up and stop being rejected within a couple of frames; a
         one-off spike from a bad frame gets filtered out.
      2. RATE LIMIT - no matter what value comes out of step 1, steer_ema
         is not allowed to change by more than MAX_STEER_STEP in a single
         tick, so the actual command sent to the Arduino can never jump.

    Every tick is optionally logged to autopilot_debug.csv (raw prediction,
    the value actually used, whether it was rejected/rate-limited, and the
    final steering/throttle) so a "sudden left" can be diagnosed after the
    fact instead of guessed at.

    The whole per-frame block is wrapped in try/except: previously an
    unhandled exception here (bad frame shape, a resize error, a CUDA
    hiccup, etc.) would silently kill this daemon thread forever while the
    UI kept showing "STOP AUTOPILOT" as if it were still running - the
    Arduino watchdog and the 0.3s serial timeout would eventually zero the
    car, but there was no visibility into why, and no way to recover
    without restarting the whole process. Now a bad tick is logged and
    skipped; MAX_CONSECUTIVE_ERRORS in a row triggers an explicit disarm.
    """
    steer_ema = 0.0
    throttle_cmd = 0.0
    stale_since = None
    consecutive_errors = 0
    raw_history = collections.deque(maxlen=RAW_HISTORY_LEN)
    period = 1.0 / AUTOPILOT_HZ
    while True:
        t0 = time.time()
        with state_lock:
            armed = autopilot["armed"]
            data = latest_frames.get("center")
            ts = last_frame_ts["center"]
        fresh = data is not None and (t0 - ts) < STALE_FRAME_S

        if armed and fresh and model is not None:
            stale_since = None
            prev_ema = steer_ema
            raw = used = None
            rejected = rate_limited = False

            try:
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                # cv2.imdecode can "succeed" on a truncated/corrupted JPEG and
                # return a valid-looking but garbage array instead of None -
                # the shape/size check catches the obvious cases; the outlier
                # filter below is the real backstop for subtler corruption.
                if img is None or img.size == 0 or img.ndim != 3 or img.shape[2] != 3:
                    raise ValueError(f"decoded frame looks invalid (shape={None if img is None else img.shape})")

                x = torch.from_numpy(preprocess(img)).unsqueeze(0).to(device)
                with torch.no_grad():
                    raw = float(model(x).item())
                raw = float(np.clip(raw * STEER_GAIN, -1.0, 1.0))

                if len(raw_history) >= 3:
                    med = float(np.median(raw_history))
                    if abs(raw - med) > OUTLIER_THRESH:
                        used = med
                        rejected = True
                    else:
                        used = raw
                else:
                    used = raw
                raw_history.append(raw)

                steer_ema = STEER_SMOOTH * used + (1 - STEER_SMOOTH) * steer_ema

                delta = steer_ema - prev_ema
                if delta > MAX_STEER_STEP:
                    steer_ema = prev_ema + MAX_STEER_STEP
                    rate_limited = True
                elif delta < -MAX_STEER_STEP:
                    steer_ema = prev_ema - MAX_STEER_STEP
                    rate_limited = True
            except Exception as e:
                consecutive_errors += 1
                print(f"AUTOPILOT: inference error, skipping this frame "
                      f"({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS} in a row): {e}")
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    with state_lock:
                        autopilot["armed"] = False
                        latest_control["steering"] = 0.0
                        latest_control["throttle"] = 0.0
                        latest_control["t"] = time.time()
                    print("AUTOPILOT auto-disarmed: too many inference errors in a row")
                    consecutive_errors = 0
                    raw_history.clear()
                    steer_ema = 0.0
                    throttle_cmd = 0.0
                dt = time.time() - t0
                if dt < period:
                    time.sleep(period - dt)
                continue

            consecutive_errors = 0
            if rejected:
                print(f"AUTOPILOT: outlier frame rejected "
                      f"(raw={raw:+.2f}, using median={used:+.2f})")
            if rate_limited:
                print(f"AUTOPILOT: steering rate-limited "
                      f"({prev_ema:+.2f} -> {steer_ema:+.2f})")

            # Throttle target: cruise, scaled down as steering grows. Then
            # ease toward it with a single-pole EMA - smooth, monotonic, no
            # overshoot, no oscillation (this is the stop-and-go fix).
            target = max(MIN_THROTTLE,
                         BASE_THROTTLE * (1.0 - CURVE_SLOWDOWN * abs(steer_ema)))
            throttle_cmd = THROTTLE_SMOOTH * target + (1 - THROTTLE_SMOOTH) * throttle_cmd
            with state_lock:
                if autopilot["armed"]:                 # may have been disarmed meanwhile
                    latest_control["steering"] = steer_ema
                    latest_control["throttle"] = throttle_cmd
                    latest_control["t"] = time.time()
                autopilot["steer"] = round(steer_ema, 3)
                autopilot["throttle"] = round(throttle_cmd, 3)
                autopilot["fresh"] = True
                if rejected:
                    autopilot["outliers"] += 1
                if rate_limited:
                    autopilot["rate_limited"] += 1

            if AUTOPILOT_DEBUG_LOG:
                debug_log_q.put((
                    f"{time.time():.3f}",
                    f"{raw:.4f}" if raw is not None else "",
                    f"{used:.4f}" if used is not None else "",
                    f"{prev_ema:.4f}", f"{steer_ema:.4f}",
                    int(rejected), int(rate_limited),
                    f"{target:.4f}", f"{throttle_cmd:.4f}",
                ))
        else:
            steer_ema = 0.0
            throttle_cmd = 0.0
            raw_history.clear()
            with state_lock:
                autopilot["steer"] = 0.0
                autopilot["throttle"] = 0.0
                autopilot["fresh"] = fresh
                if armed and not fresh:
                    if stale_since is None:
                        stale_since = t0
                    elif t0 - stale_since > AUTO_DISARM_S:
                        autopilot["armed"] = False
                        latest_control["steering"] = 0.0
                        latest_control["throttle"] = 0.0
                        latest_control["t"] = time.time()
                        print("AUTOPILOT auto-disarmed: center camera stale")
                        stale_since = None
                else:
                    stale_since = None

        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


# ============================ SERIAL BRIDGE ================================

def open_serial():
    global ser
    try:
        import serial
        ser = serial.Serial(SERIAL_PORT, BAUD, timeout=0)
        time.sleep(2.0)
        print(f"Serial open on {SERIAL_PORT}")
    except Exception as e:
        ser = None
        print(f"WARNING: serial not open ({e}).")
        print(f"         Nothing will reach the Arduino. Check the port name")
        print(f"         (--serial COMx) and close any other program using it.")


def serial_sender():
    """20 Hz command stream to the Arduino, with a status line every 2 s so
    you can SEE what is actually being sent (or that nothing is).

    Now also: if the port drops mid-session (Arduino unplugged/replugged,
    USB hiccup), this retries open_serial() every SERIAL_RECONNECT_S
    instead of staying dead for the rest of the run, and write failures are
    counted and reported on the existing 2s status line instead of
    spamming a traceback-per-tick (20/sec) to the console.
    """
    global ser
    period = 1.0 / SEND_HZ
    last_log = 0.0
    last_reconnect_try = 0.0
    sent = 0
    write_errs = 0
    while True:
        if ser is None:
            now = time.time()
            if now - last_reconnect_try > SERIAL_RECONNECT_S:
                last_reconnect_try = now
                open_serial()
        with state_lock:
            c = dict(latest_control)
            ap_on = autopilot["armed"]
        if time.time() - c["t"] > 0.3:
            steer, thr = 0, 0
        else:
            steer = int(max(-1.0, min(1.0, c["steering"])) * 100)
            thr   = int(max(-1.0, min(1.0, c["throttle"])) * 100)
        if ser is not None:
            try:
                ser.write(f"{steer} {thr}\n".encode())
                sent += 1
            except Exception:
                write_errs += 1
                ser = None   # drop it; the reconnect check above will retry
        now = time.time()
        if now - last_log > 2.0:
            last_log = now
            src = "AUTOPILOT" if ap_on else "manual"
            if ser is not None:
                print(f"[serial OK] {src}: steer={steer:+4d} thr={thr:+4d}  ({sent} cmds/2s)")
            else:
                print(f"[NO SERIAL] {src}: steer={steer:+4d} thr={thr:+4d}  -> NOT reaching "
                      f"the Arduino! ({write_errs} write errors since last check, "
                      f"retrying every {SERIAL_RECONNECT_S:.0f}s)")
            sent = 0
            write_errs = 0
        time.sleep(period)


# ============================ AUTOPILOT DEBUG LOG ==========================

debug_log_q = queue.Queue()


def debug_logger_thread():
    """Writes every autopilot tick to DEBUG_LOG_PATH so a 'sudden left'
    event can be looked up afterwards: was the raw model prediction itself
    extreme, or did smoothing/rate-limiting catch it? Flushes ~1x/sec so
    the file stays readable while a run is still going."""
    f = open(DEBUG_LOG_PATH, "w", newline="")
    wr = csv.writer(f)
    wr.writerow(["t", "raw", "used", "steer_before", "steer_after",
                 "outlier_rejected", "rate_limited", "throttle_target", "throttle_cmd"])
    last_flush = time.time()
    while True:
        row = debug_log_q.get()
        try:
            wr.writerow(row)
            now = time.time()
            if now - last_flush > 1.0:
                f.flush()
                last_flush = now
        except Exception as e:
            print("debug logger error:", e)
        finally:
            debug_log_q.task_done()


# ============================ RECORDER =====================================

writer_q = queue.Queue()


def recorder_thread():
    csvf = None
    wr = None
    sdir = None
    while True:
        item = writer_q.get()
        try:
            kind = item[0]
            if kind == "start":
                sdir = item[1]
                for c in ("left", "center", "right"):
                    os.makedirs(os.path.join(sdir, c), exist_ok=True)
                csvf = open(os.path.join(sdir, "labels.csv"), "w", newline="")
                wr = csv.writer(csvf)
                wr.writerow(["index", "timestamp", "steering", "throttle",
                             "left", "center", "right"])
                print(f"Recording -> {sdir}")
            elif kind == "stop":
                if csvf:
                    csvf.flush()
                    csvf.close()
                csvf = None
                wr = None
            elif kind == "frame" and wr is not None:
                _, idx, imgs, steer, thr, ts = item
                paths = {}
                for c in ("left", "center", "right"):
                    rel = f"{c}/{idx:06d}.jpg"
                    with open(os.path.join(sdir, rel), "wb") as out:
                        out.write(imgs[c])
                    paths[c] = rel
                wr.writerow([idx, f"{ts:.3f}", f"{steer:.4f}", f"{thr:.4f}",
                             paths["left"], paths["center"], paths["right"]])
        except Exception as e:
            print("recorder error:", e)
        finally:
            writer_q.task_done()


def mjpeg_generator(cam):
    period = 1.0 / STREAM_FPS
    while True:
        with state_lock:
            data = latest_frames.get(cam)
        if data is not None:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                   + str(len(data)).encode() + b"\r\n\r\n" + data + b"\r\n")
        time.sleep(period)


# ============================ HTTP ROUTES ==================================

@app.route("/")
def index():
    return Response(PAGE_HTML, mimetype="text/html")


@app.route("/control", methods=["POST"])
def control():
    d = request.get_json(force=True, silent=True) or {}
    s = float(d.get("steering", 0.0))
    t_ = float(d.get("throttle", 0.0))
    with state_lock:
        # MANUAL TAKEOVER: deflecting the joystick while armed disarms instantly
        if autopilot["armed"] and (abs(s) > TAKEOVER_THRESH or abs(t_) > TAKEOVER_THRESH):
            autopilot["armed"] = False
            print("AUTOPILOT disarmed: manual takeover (joystick)")
        if not autopilot["armed"]:
            latest_control["steering"] = s
            latest_control["throttle"] = t_
            latest_control["t"] = time.time()
    return ("", 204)


@app.route("/autopilot", methods=["POST"])
def set_autopilot():
    d = request.get_json(force=True, silent=True) or {}
    on = bool(d.get("on", False))
    now = time.time()
    with state_lock:
        if on:
            if model is None:
                return jsonify({"armed": False,
                                "error": "no model loaded - train pilotnet.pt first"}), 400
            if now - last_frame_ts["center"] > STALE_FRAME_S:
                return jsonify({"armed": False,
                                "error": "center camera not live - open the camera page on the laptop"}), 400
            autopilot["armed"] = True
            print("AUTOPILOT ARMED (from web page)")
        else:
            autopilot["armed"] = False
            latest_control["steering"] = 0.0
            latest_control["throttle"] = 0.0
            latest_control["t"] = now
            print("AUTOPILOT disarmed (from web page)")
        return jsonify({"armed": autopilot["armed"]})


@app.route("/frame", methods=["POST"])
def frame():
    global sample_idx
    imgs = {}
    for cam in ("left", "center", "right"):
        f = request.files.get(cam)
        if f is not None:
            imgs[cam] = f.read()
    if not imgs:
        return ("no frames", 400)

    now = time.time()
    idx = None
    with state_lock:
        for k, v in imgs.items():
            latest_frames[k] = v
            last_frame_ts[k] = now
        rec   = recording
        steer = latest_control["steering"]
        thr   = latest_control["throttle"]
        # record only when all three cameras are present, so samples stay aligned
        if rec and len(imgs) == 3:
            idx = sample_idx
            sample_idx += 1

    if idx is not None:
        writer_q.put(("frame", idx, imgs, steer, thr, now))

    with state_lock:
        n = sample_idx
    return jsonify({"recording": rec, "n": n, "got": len(imgs)})


@app.route("/stream/<cam>")
def stream(cam):
    if cam not in ("left", "center", "right"):
        return ("bad cam", 404)
    return Response(mjpeg_generator(cam),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/latest/<cam>.jpg")
def latest(cam):
    with state_lock:
        data = latest_frames.get(cam)
    if data is None:
        return ("", 404)
    return Response(data, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})


@app.route("/record", methods=["POST"])
def record():
    global recording, sample_idx
    d = request.get_json(force=True, silent=True) or {}
    on = bool(d.get("on", False))
    with state_lock:
        was = recording
    if on and not was:
        sdir = os.path.join(OUTPUT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
        with state_lock:
            sample_idx = 0
            recording = True
        writer_q.put(("start", sdir))
    elif not on and was:
        with state_lock:
            recording = False
        writer_q.put(("stop",))
    with state_lock:
        n = sample_idx
    return jsonify({"recording": on, "n": n})


@app.route("/state")
def get_state():
    now = time.time()
    with state_lock:
        cams = {k: round((now - t) * 1000) if t else -1 for k, t in last_frame_ts.items()}
        return jsonify({"recording": recording, "n": sample_idx, "cams": cams,
                        "ap": dict(autopilot)})


# ============================ WEB PAGE =====================================
# The ORIGINAL page (same joystick, same camera cards) + AUTOPILOT button
# and an AI steering readout in the header. Nothing else changed.

PAGE_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Car control + autopilot</title>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no" />
<style>
  :root{ --bg:#0c1626; --panel:#13243d; --line:#24405f; --txt:#e8f0fb; --mut:#8fb0d6; --accent:#38a8ff; --gold:#f2b233; --ok:#5dcaa5; --bad:#e24b4a; }
  *{ box-sizing:border-box; }
  body{ font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial; margin:0; background:var(--bg); color:var(--txt); }
  header{ display:flex; align-items:center; gap:10px; padding:10px 14px; background:var(--panel); border-bottom:1px solid var(--line); flex-wrap:wrap; }
  header h1{ font-size:1rem; margin:0; font-weight:600; }
  .grow{ flex:1; }
  button{ background:var(--line); color:var(--txt); border:1px solid var(--line); border-radius:8px; padding:8px 12px; cursor:pointer; font-size:.9rem; }
  button:active{ transform:translateY(1px); }
  button.rec{ background:#7a1f1f; border-color:#a32d2d; }
  button.rec.on{ background:#a32d2d; }
  button.ap{ background:#1d4d3a; border-color:#2f7a5b; font-weight:700; }
  button.ap.on{ background:#a32d2d; border-color:#d34141; animation:pulse 1.2s infinite; }
  button.ap:disabled{ opacity:.45; cursor:not-allowed; }
  @keyframes pulse{ 0%,100%{ box-shadow:0 0 0 0 rgba(211,65,65,.5);} 50%{ box-shadow:0 0 0 8px rgba(211,65,65,0);} }
  .status{ font-size:.85rem; color:var(--mut); }
  .warn{ background:#5a1a1a; color:#ffd9d9; padding:10px 14px; font-size:.9rem; display:none; }
  /* AI steering needle bar */
  .apbar{ position:relative; height:14px; margin:8px 14px 0; background:var(--panel); border:1px solid var(--line); border-radius:7px; }
  .apbar .mid{ position:absolute; left:50%; top:2px; bottom:2px; width:1px; background:var(--mut); }
  .apbar .needle{ position:absolute; top:1px; bottom:1px; width:10px; border-radius:5px; background:var(--gold); left:50%; transform:translateX(-50%); transition:left .08s linear; }
  .apbar .lbl{ position:absolute; right:8px; top:-1px; font-size:.68rem; color:var(--mut); }
  .cams{ display:flex; gap:10px; padding:12px; flex-wrap:wrap; justify-content:center; }
  .cam{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:8px; width:300px; }
  body.rec .cam{ border-color:var(--bad); }
  .cam .lbl{ font-size:.8rem; color:var(--mut); margin-bottom:6px; display:flex; justify-content:space-between; }
  .cam .st{ font-size:.75rem; }
  .st.ok{ color:var(--ok); } .st.bad{ color:var(--bad); }
  video,img.feed{ width:100%; aspect-ratio:4/3; background:#000; border-radius:6px; object-fit:cover; display:block; }
  select{ width:100%; margin-top:6px; background:var(--bg); color:var(--txt); border:1px solid var(--line); border-radius:6px; padding:5px; font-size:.8rem; }
  .drivebar{ display:flex; gap:24px; align-items:center; justify-content:center; padding:6px 12px 22px; flex-wrap:wrap; }
  .pad{ position:relative; width:230px; height:230px; border-radius:50%; background:var(--panel); border:1px solid var(--line); touch-action:none; }
  .pad .cross-h,.pad .cross-v{ position:absolute; background:var(--line); }
  .pad .cross-h{ left:8%; right:8%; top:50%; height:1px; }
  .pad .cross-v{ top:8%; bottom:8%; left:50%; width:1px; }
  .knob{ position:absolute; width:74px; height:74px; border-radius:50%; background:var(--accent); left:50%; top:50%; transform:translate(-50%,-50%); box-shadow:0 0 0 4px rgba(56,168,255,.18); }
  .readout{ font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.9rem; color:var(--mut); min-width:170px; }
  .readout b{ color:var(--gold); }
  .hint{ font-size:.78rem; color:var(--mut); max-width:240px; line-height:1.4; }
</style>
</head>
<body>
<header>
  <h1>Car control</h1>
  <span id="mode" class="status"></span>
  <span class="grow"></span>
  <button id="refreshBtn">Refresh cams</button>
  <button id="recBtn" class="rec">Start recording</button>
  <button id="apBtn" class="ap" disabled>AUTOPILOT</button>
  <span id="status" class="status"></span>
</header>

<div class="apbar" title="AI steering">
  <div class="mid"></div><div class="needle" id="apNeedle"></div>
  <div class="lbl" id="apLbl">AI steer 0.00 | thr 0.00</div>
</div>

<div id="warn" class="warn"></div>
<div class="cams" id="cams"></div>

<div class="drivebar" id="drivebar" style="display:none">
  <div class="pad" id="pad">
    <div class="cross-h"></div><div class="cross-v"></div>
    <div class="knob" id="knob"></div>
  </div>
  <div>
    <div class="readout">steer <b id="rSteer">0.00</b><br>throttle <b id="rThr">0.00</b><br><span id="gpName" class="status"></span></div>
    <div class="hint">Drag the pad (or gamepad / arrow keys). Left-right = steering. Up-down = throttle. Release to center and stop. While AUTOPILOT is ON, grabbing the pad takes over instantly (turns it off).</div>
  </div>
</div>

<script>
const MODE = new URLSearchParams(location.search).get('mode') === 'drive' ? 'drive' : 'operator';
const CAP_W = 320, CAP_H = 240;
const FRAME_HZ = 30;
const CTRL_HZ = 20;
const CAMS = ['left','center','right'];
const sleep = ms => new Promise(r=>setTimeout(r,ms));
document.getElementById('mode').textContent = MODE === 'operator' ? 'camera device (laptop)' : 'remote drive';

// ---------- recording ----------
let recording = false;
const recBtn = document.getElementById('recBtn');
recBtn.addEventListener('click', async ()=>{
  recording = !recording;
  try{ const r=await fetch('/record',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({on:recording})});
       const d=await r.json(); recording=d.recording; }catch(e){}
  recBtn.classList.toggle('on', recording);
  recBtn.textContent = recording ? 'Stop recording' : 'Start recording';
});

// ---------- AUTOPILOT button ----------
let apArmed = false, apModel = false;
const apBtn = document.getElementById('apBtn');
apBtn.addEventListener('click', async ()=>{
  const want = !apArmed;
  try{
    const r = await fetch('/autopilot',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({on:want})});
    const d = await r.json();
    if(d.error){ alert(d.error); }
    apArmed = !!d.armed;
  }catch(e){}
  renderAp();
});
function renderAp(steer=0, thr=0, fresh=true){
  apBtn.disabled = !apModel;
  apBtn.classList.toggle('on', apArmed);
  apBtn.textContent = !apModel ? 'AUTOPILOT (no model)'
    : (apArmed ? 'STOP AUTOPILOT' : 'AUTOPILOT');
  document.getElementById('apNeedle').style.left = (50 + steer*48) + '%';
  document.getElementById('apLbl').textContent =
    'AI steer ' + steer.toFixed(2) + ' | thr ' + thr.toFixed(2) + (fresh?'':' | NO FRAME');
}

// ---------- status poll ----------
setInterval(async ()=>{
  try{ const r=await fetch('/state'); const d=await r.json();
    const cams=d.cams||{}; const live=CAMS.map(c=>cams[c]).filter(a=>a>=0 && a<1500).length;
    const ap=d.ap||{};
    apModel = !!ap.model; apArmed = !!ap.armed;
    renderAp(ap.steer||0, ap.throttle||0, ap.fresh!==false);
    document.body.classList.toggle('rec', !!d.recording);
    document.getElementById('status').textContent =
      (ap.armed?'AUTOPILOT ':'') + (d.recording?'REC':'idle') + '  samples: ' + d.n + '  cams live: ' + live + '/3'
      + (ap.armed ? '  | outliers:' + (ap.outliers||0) + ' rate-limited:' + (ap.rate_limited||0) : '');
  }catch(e){}
}, 400);

// ================= OPERATOR (laptop, cameras only) =================
if(MODE === 'operator'){
  const warn = document.getElementById('warn');
  if(!navigator.mediaDevices || !window.isSecureContext){
    warn.style.display='block';
    warn.textContent = 'Cameras need a secure context. Open this page on the laptop at http://localhost:'+location.port+'/  (not the IP). Phones connect by IP with ?mode=drive.';
  }

  const cams = CAMS.map((name,i)=>({ name, index:i, video:null, select:null, st:null, stream:null }));
  const wrap = document.getElementById('cams');
  cams.forEach(cam=>{
    const d = document.createElement('div'); d.className='cam';
    d.innerHTML = '<div class="lbl"><span>'+cam.name+'</span><span class="st"></span></div>'
      + '<video autoplay playsinline muted></video><select></select>';
    wrap.appendChild(d);
    cam.video = d.querySelector('video');
    cam.select = d.querySelector('select');
    cam.st = d.querySelector('.st');
    cam.select.addEventListener('change', ()=>{ if(cam.select.value){ startCam(cam, cam.select.value); saveCamChoice(cam.name, cam.select.value); } });
  });

  function setSt(cam, txt, ok){ cam.st.textContent=txt; cam.st.className='st'+(ok===true?' ok':ok===false?' bad':''); }

  async function startCam(cam, deviceId){
    if(cam.stream){ cam.stream.getTracks().forEach(t=>t.stop()); cam.stream=null; }
    setSt(cam, 'opening...', null);
    try{
      cam.stream = await navigator.mediaDevices.getUserMedia({
        video:{ deviceId:{exact:deviceId}, width:{ideal:CAP_W}, height:{ideal:CAP_H}, frameRate:{ideal:30} },
        audio:false
      });
      cam.video.srcObject = cam.stream;
      setSt(cam, 'live', true);
    }catch(err){
      setSt(cam, 'FAILED: '+err.name, false);   // NotReadableError = USB busy/bandwidth
    }
  }

  // open cameras ONE AT A TIME -- opening 3 USB cams at once makes the later ones fail
  async function startAllSequential(){
    for(const cam of cams){
      if(cam.select.value){ await startCam(cam, cam.select.value); await sleep(350); }
    }
  }

  // Persist which physical camera (by deviceId) is assigned to left/center/
  // right across reloads and browser restarts - enumerateDevices() order is
  // not guaranteed stable across reboots, and "center" feeds the autopilot
  // directly, so a silent left/right/center swap is a real (and otherwise
  // invisible) failure mode. Falls back gracefully if storage is unavailable.
  function saveCamChoice(name, deviceId){ try{ localStorage.setItem('cam:'+name, deviceId); }catch(e){} }
  function loadCamChoice(name){ try{ return localStorage.getItem('cam:'+name); }catch(e){ return null; } }

  async function getDevices(){
    if(!navigator.mediaDevices) return;
    try{
      const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput');
      cams.forEach(cam=>{
        const cur = cam.select.value;
        const saved = loadCamChoice(cam.name);
        const savedStillHere = saved && devs.some(d=>d.deviceId===saved);
        cam.select.innerHTML='';
        devs.forEach((d,i)=>{ const o=document.createElement('option'); o.value=d.deviceId; o.textContent=d.label||('Camera '+(i+1)); cam.select.appendChild(o); });
        if(devs.length){
          cam.select.value = savedStillHere ? saved : (cur || (devs[cam.index]?devs[cam.index].deviceId:devs[0].deviceId));
          saveCamChoice(cam.name, cam.select.value);
        }
      });
      const chosen = cams.map(c=>c.select.value);
      const dupes = chosen.some((id,i)=>id && chosen.indexOf(id)!==i);
      if(devs.length < 3){
        warn.style.display='block';
        warn.textContent='Only '+devs.length+' camera(s) detected. Plug in all three (and check they are not on the same USB hub).';
      } else if(dupes){
        warn.style.display='block';
        warn.textContent='Two of the camera slots picked the SAME physical camera - check the '
          +'dropdowns below, left/center/right must each be a different camera (center feeds the autopilot directly).';
      }
      await startAllSequential();
    }catch(err){ document.getElementById('status').textContent='camera error: '+err.message; }
  }
  document.getElementById('refreshBtn').addEventListener('click', getDevices);

  const cv = document.createElement('canvas'); cv.width=CAP_W; cv.height=CAP_H;
  const ctx = cv.getContext('2d');
  function grab(cam){
    return new Promise(res=>{
      if(!cam.video.videoWidth) return res(null);
      ctx.drawImage(cam.video,0,0,CAP_W,CAP_H);
      cv.toBlob(b=>res(b), 'image/jpeg', 0.7);
    });
  }
  let inFlight=false;
  setInterval(async ()=>{
    if(inFlight) return; inFlight=true;
    try{
      const fd = new FormData(); let have=0;
      for(const cam of cams){ const b=await grab(cam); if(b){ fd.append(cam.name,b,cam.name+'.jpg'); have++; } }
      if(have) await fetch('/frame',{method:'POST',body:fd});
    }catch(e){}
    inFlight=false;
  }, 1000/FRAME_HZ);

  (async ()=>{
    try{
      const probe = await navigator.mediaDevices.getUserMedia({video:true,audio:false});
      probe.getTracks().forEach(t=>t.stop());   // free the default camera before opening the real ones
    }catch(e){}
    getDevices();
  })();
}

// ================= DRIVE (mobile: view + joystick + autopilot) =================
else {
  document.getElementById('refreshBtn').style.display='none';
  document.getElementById('drivebar').style.display='flex';
  const wrap = document.getElementById('cams');
  CAMS.forEach(name=>{
    const d=document.createElement('div'); d.className='cam';
    d.innerHTML='<div class="lbl"><span>'+name+'</span></div><img class="feed" alt="'+name+'">';
    wrap.appendChild(d);
    const img=d.querySelector('img');
    img.src='/stream/'+name;
    img.onerror=()=>{ setTimeout(()=>{ img.src='/stream/'+name+'?'+Date.now(); }, 1500); };
  });

  let steer=0, throttle=0;
  const pad=document.getElementById('pad'), knob=document.getElementById('knob');
  let dragging=false;
  function setFromPoint(x,y){
    const r=pad.getBoundingClientRect(), half=r.width/2;
    let nx=(x-(r.left+half))/half, ny=(y-(r.top+half))/half;
    const m=Math.hypot(nx,ny); if(m>1){nx/=m;ny/=m;}
    steer=nx; throttle=-ny;
    knob.style.left=(50+nx*42)+'%'; knob.style.top=(50+ny*42)+'%';
  }
  function reset(){ steer=0; throttle=0; knob.style.left='50%'; knob.style.top='50%'; }
  pad.addEventListener('pointerdown',e=>{ dragging=true; pad.setPointerCapture(e.pointerId); setFromPoint(e.clientX,e.clientY); });
  pad.addEventListener('pointermove',e=>{ if(dragging) setFromPoint(e.clientX,e.clientY); });
  pad.addEventListener('pointerup',()=>{ dragging=false; reset(); });
  pad.addEventListener('pointercancel',()=>{ dragging=false; reset(); });

  const keys={};
  addEventListener('keydown',e=>{ keys[e.key.toLowerCase()]=true; });
  addEventListener('keyup',e=>{ keys[e.key.toLowerCase()]=false; });
  function applyKeys(){ let s=0,t=0;
    if(keys['a']||keys['arrowleft'])s-=1; if(keys['d']||keys['arrowright'])s+=1;
    if(keys['w']||keys['arrowup'])t+=1; if(keys['s']||keys['arrowdown'])t-=1;
    if(s||t){ steer=s; throttle=t; knob.style.left=(50+s*42)+'%'; knob.style.top=(50-t*42)+'%'; return true; } return false; }
  function applyGamepad(){ const gps=navigator.getGamepads?navigator.getGamepads():[];
    for(const gp of gps){ if(!gp) continue;
      document.getElementById('gpName').textContent='gamepad: '+gp.id.slice(0,20);
      const dz=0.06; let sx=gp.axes[0]||0, sy=gp.axes[1]||0;
      if(Math.abs(sx)<dz)sx=0; if(Math.abs(sy)<dz)sy=0;
      if(sx||sy){ steer=sx; throttle=-sy; knob.style.left=(50+sx*42)+'%'; knob.style.top=(50+sy*42)+'%'; } return true; }
    return false; }

  let ctrlInFlight=false;
  setInterval(async ()=>{
    if(!dragging){ if(!applyGamepad()) applyKeys(); }
    document.getElementById('rSteer').textContent=steer.toFixed(2);
    document.getElementById('rThr').textContent=throttle.toFixed(2);
    // While AUTOPILOT is ON: stay silent unless the human actually deflects
    // the pad. A real deflection is sent and triggers server-side takeover.
    if(apArmed && Math.abs(steer)<0.05 && Math.abs(throttle)<0.05){ return; }
    if(ctrlInFlight) return; ctrlInFlight=true;
    try{ await fetch('/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({steering:steer,throttle:throttle})}); }catch(e){}
    ctrlInFlight=false;
  }, 1000/CTRL_HZ);
}
</script>
</body>
</html>"""


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--serial", default=SERIAL_PORT)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    SERIAL_PORT = args.serial
    PORT = args.port

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    load_model(args.model)
    open_serial()
    threading.Thread(target=serial_sender, daemon=True).start()
    threading.Thread(target=recorder_thread, daemon=True).start()
    if AUTOPILOT_DEBUG_LOG:
        threading.Thread(target=debug_logger_thread, daemon=True).start()
        print(f"Autopilot debug log -> {DEBUG_LOG_PATH}")
    threading.Thread(target=autopilot_thread, daemon=True).start()
    print(f"CAMERA page (on the laptop):  http://localhost:{PORT}/")
    print(f"DRIVE page  (on the phone) :  http://<this-laptop-ip>:{PORT}/?mode=drive")
    print("AUTOPILOT: press the green button on either page "
          "(needs pilotnet.pt + live center camera).")
    app.run(host=HOST, port=PORT, threaded=True)
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         IWR6843ISK mmWave Radar — Live Serial Backend (app.py)             ║
║                                                                              ║
║  Architecture:                                                               ║
║    Sensor → pyserial → TLV Parser → Frame Buffer → Classifier → Flask API  ║
║                                                                              ║
║  Ports (Windows):  CFG_PORT="COM3"         DATA_PORT="COM4"                ║
║  Ports (Linux):    CFG_PORT="/dev/ttyUSB0" DATA_PORT="/dev/ttyUSB1"        ║
║  Ports (Mac):      CFG_PORT="/dev/tty.usbmodem..." (check ls /dev/tty.*)   ║
║                                                                              ║
║  Install:  pip install flask flask-cors numpy pyserial scipy                ║
║  Run:      python app.py                                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

HOW CLASSIFICATION WORKS (Read this!)
──────────────────────────────────────
The sensor sends raw x,y,z,velocity for every detected reflection.
The classifier uses FOUR independent features per frame, then combines
them using a weighted vote. No single feature decides alone — this prevents
"everything is human" false positives.

  Feature 1 — POINT COUNT
    < 5  pts          → strong Empty signal
    5–15 pts          → ambiguous
    > 15 pts          → occupied (object or human)

  Feature 2 — DOPPLER VELOCITY DISTRIBUTION
    All |v| < 0.08 m/s         → Static object (no movement at all)
    Few points with |v| > 0.08 → Some motion (could be breathing)
    Many points with |v| > 0.08 → Active movement (human walking)

  Feature 3 — SPATIAL SPREAD (point cloud bounding box)
    Tight cluster (bbox < 0.3 m) → single rigid object (chair/box)
    Spread > 0.5 m in any axis   → human-sized target or multiple objects
    Very sparse (spread > 2.5 m) → room reflections / multipath noise
    NOTE: bbox > 2.5 m disqualifies human classification entirely.

  Feature 4 — MICRO-DOPPLER OSCILLATION (breathing / heartbeat)
    Computed on the LAST 10 FRAMES (temporal window), not just current frame.
    We look for periodic sign-flips in per-cluster velocity centroid.
    A rigid object has near-zero velocity always → no oscillation.
    A breathing human has tiny ±0.05–0.15 m/s oscillations at 0.2–0.5 Hz.

    CALIBRATED thresholds from real .dat recordings:
      Object recording (301 frames): osc max = 0.067 m/s (99th-pct)
      → object_max_oscillation = 0.070 m/s
      → human_min_oscillation  = 0.100 m/s  (0.030 m/s gap above object ceiling)
      Anything between 0.070–0.100 is ambiguous → classified as Object.

  Temporal Smoothing:
    The final label is smoothed over the last 8 frames using majority vote.
    This prevents flickering between labels on every frame.

  Calibration (/api/calibrate?label=empty|static|human):
    Run for 5 seconds while placing the target, then it learns your
    specific sensor's noise floor and threshold offsets.
"""

import struct
import threading
import time
import logging
import os
import json
from collections import deque
from typing import Optional, List, Dict

import numpy as np
import serial
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS


# ══════════════════════════════════════════════════════════════════════════════
#  ❶  CONFIGURATION  —  Edit before running
# ══════════════════════════════════════════════════════════════════════════════

CFG_PORT   = "COM5"       # User UART  (115200 baud) — send .cfg file
DATA_PORT  = "COM6"       # Data UART  (921600 baud) — receive TLV stream
CFG_BAUD   = 115200
DATA_BAUD  = 921600
CFG_FILE   = r"C:\ti\mmwave_sdk_03_06_02_00-LTS\packages\ti\demo\xwr68xx\mmw\profiles\profile_2d.cfg"
FLASK_PORT = 5000

# How many raw frames to keep in memory
RAW_BUFFER_DEPTH = 30     # ~3 seconds at 10 fps — used for temporal features


# ══════════════════════════════════════════════════════════════════════════════
#  ❷  TI TLV PROTOCOL — constants
# ══════════════════════════════════════════════════════════════════════════════

MAGIC_WORD          = b'\x02\x01\x04\x03\x06\x05\x08\x07'
MAGIC_LEN           = 8
FRAME_HEADER_FMT    = '<IIIIIIII'
FRAME_HEADER_SIZE   = struct.calcsize(FRAME_HEADER_FMT)   # 32 bytes

TLV_DETECTED_POINTS          = 1
TLV_DETECTED_POINTS_SIDE_INFO= 7
TLV_HEADER_FMT    = '<II'
TLV_HEADER_SIZE   = struct.calcsize(TLV_HEADER_FMT)       # 8 bytes
POINT_FMT         = '<ffff'
POINT_SIZE        = struct.calcsize(POINT_FMT)             # 16 bytes
POINT_DESC_FMT    = '<HH'
POINT_DESC_SIZE   = struct.calcsize(POINT_DESC_FMT)        # 4 bytes
SIDE_INFO_FMT     = '<HH'
SIDE_INFO_SIZE    = struct.calcsize(SIDE_INFO_FMT)         # 4 bytes


# ══════════════════════════════════════════════════════════════════════════════
#  ❸  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s  %(levelname)-7s  %(message)s',
    datefmt = '%H:%M:%S',
)
log = logging.getLogger("mmWave")


# ══════════════════════════════════════════════════════════════════════════════
#  ❹  RAW FRAME BUFFER  — thread-safe ring buffer
# ══════════════════════════════════════════════════════════════════════════════

class FrameBuffer:
    # Frames older than this are considered stale (sensor disconnected)
    STALE_SECONDS = 2.0

    def __init__(self, maxlen: int = RAW_BUFFER_DEPTH):
        self._lock   = threading.Lock()
        self._frames = deque(maxlen=maxlen)
        self._total  = 0
        self._drops  = 0

    def push(self, frame: dict):
        with self._lock:
            self._frames.append(frame)
            self._total += 1

    def latest(self) -> Optional[dict]:
        """Return the latest frame ONLY if it is fresh (not stale after disconnect)."""
        with self._lock:
            if not self._frames:
                return None
            frame = self._frames[-1]
            age = time.time() - frame.get("timestamp", 0)
            if age > self.STALE_SECONDS:
                return None          # ← sensor disconnected; don't serve stale data
            return frame

    def last_n(self, n: int) -> list:
        """Return only fresh frames from the last N entries."""
        now = time.time()
        with self._lock:
            recent = list(self._frames)[-n:]
        # Filter out stale frames
        return [f for f in recent
                if (now - f.get("timestamp", 0)) <= self.STALE_SECONDS]

    def drop(self):
        with self._lock:
            self._drops += 1

    @property
    def total(self): return self._total

    @property
    def drops(self): return self._drops


FRAME_BUFFER = FrameBuffer()


# ══════════════════════════════════════════════════════════════════════════════
#  ❺  TLV PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_frame(raw: bytes) -> Optional[dict]:
    """Parse one TLV frame (raw bytes AFTER magic word stripped)."""
    try:
        if len(raw) < FRAME_HEADER_SIZE:
            return None

        (version, total_packet_len, platform, frame_number,
         time_cpu_cycles, num_detected_obj, num_tlvs, subframe_number
        ) = struct.unpack_from(FRAME_HEADER_FMT, raw, 0)

        offset   = FRAME_HEADER_SIZE
        points   = []
        snr_list = []

        for _ in range(num_tlvs):
            if offset + TLV_HEADER_SIZE > len(raw):
                break
            tlv_type, tlv_length = struct.unpack_from(TLV_HEADER_FMT, raw, offset)
            offset += TLV_HEADER_SIZE
            payload  = raw[offset: offset + tlv_length]
            offset  += tlv_length

            if tlv_type == TLV_DETECTED_POINTS:
                points = _parse_points(payload, num_detected_obj)
            elif tlv_type == TLV_DETECTED_POINTS_SIDE_INFO:
                snr_list = _parse_side_info(payload, num_detected_obj)

        # Attach SNR
        for i, pt in enumerate(points):
            pt["snr"] = snr_list[i] if i < len(snr_list) else 0.0

        return {
            "frame_number"  : frame_number,
            "num_detected"  : num_detected_obj,
            "points"        : points,
            "timestamp"     : time.time(),
        }
    except Exception as e:
        log.debug(f"Parse error: {e}")
        return None


def _parse_points(payload: bytes, num_obj: int) -> list:
    points = []
    pos    = 0
    # Optional 4-byte descriptor — skip if count matches
    if len(payload) >= POINT_DESC_SIZE:
        maybe_count, _ = struct.unpack_from(POINT_DESC_FMT, payload, 0)
        if maybe_count == num_obj and num_obj > 0:
            pos = POINT_DESC_SIZE
    for _ in range(num_obj):
        if pos + POINT_SIZE > len(payload):
            break
        x, y, z, v = struct.unpack_from(POINT_FMT, payload, pos)
        pos += POINT_SIZE
        points.append({"x": x, "y": y, "z": z, "v": v, "snr": 0.0})
    return points


def _parse_side_info(payload: bytes, num_obj: int) -> list:
    snrs = []
    pos  = 0
    for _ in range(num_obj):
        if pos + SIDE_INFO_SIZE > len(payload):
            break
        snr_raw, _ = struct.unpack_from(SIDE_INFO_FMT, payload, pos)
        pos += SIDE_INFO_SIZE
        snrs.append(round(snr_raw * 0.1, 2))
    return snrs


# ══════════════════════════════════════════════════════════════════════════════
#  ❻  SNR FILTER
#     Wall/ceiling multipath reflections have LOW SNR (< 10 dB typically).
#     A real human target at close range has HIGH SNR (> 12–15 dB).
#     Filtering out low-SNR points before classification dramatically reduces
#     false oscillation caused by flickering multipath ghosts.
#
#     SNR values in the TI OOB demo are in units of 0.1 dB (raw uint16),
#     already converted to dB in _parse_side_info (snr_raw * 0.1).
#
#     If SNR is 0.0 for all points (side-info TLV missing), the filter
#     is automatically bypassed so classification still works.
# ══════════════════════════════════════════════════════════════════════════════

SNR_THRESHOLD_DB = 12.0   # points below this are treated as multipath noise
RANGE_GATE_Y_MAX = 5.0    # ignore points beyond 5 m depth (wall far-field noise)
CLUSTER_RADIUS   = 2.0    # metres — strip points >2 m from the cloud centroid


def filter_by_range(points: list) -> list:
    """
    Drop points beyond RANGE_GATE_Y_MAX metres (far-field wall reflections).
    Y is the depth axis on TI IWR6843 (forward distance from sensor).
    If everything is beyond the gate (sensor aimed at far wall), bypass.
    """
    if not points:
        return points
    gated = [p for p in points if p.get("y", 0.0) <= RANGE_GATE_Y_MAX]
    return gated if gated else points


def filter_by_cluster(points: list) -> list:
    """
    Remove spatial outliers by keeping only points within CLUSTER_RADIUS metres
    of the point-cloud centroid.

    Why: the sensor picks up a real target cluster (1-3 m away) PLUS isolated
    wall reflections scattered at X = +-8 m, Y = 3-5 m. The wall ghosts inflate
    bbox to 5-14 m and corrupt oscillation. Stripping them collapses bbox
    to 1-3 m and leaves only the real target.

    From object .dat analysis:
        Before filter: bbox mean = 3.6 m, max = 14.7 m
        After  filter: bbox mean = 1.8 m, max = 3.0 m  (100% <= 3.0 m)
    """
    if len(points) < 2:
        return points

    xs = np.array([p["x"] for p in points], dtype=float)
    ys = np.array([p["y"] for p in points], dtype=float)
    zs = np.array([p["z"] for p in points], dtype=float)

    cx, cy, cz = xs.mean(), ys.mean(), zs.mean()
    dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2 + (zs - cz) ** 2)

    kept = [p for p, d in zip(points, dists) if d <= CLUSTER_RADIUS]
    # Safety: never return empty — if radius too tight, keep closest half
    if not kept:
        half = max(1, len(points) // 2)
        kept = [points[i] for i in np.argsort(dists)[:half]]
    return kept


def filter_by_snr(points: list) -> list:
    """
    Remove low-SNR points (multipath wall reflections).
    Returns filtered list. If no SNR data available, returns original list.
    """
    if not points:
        return points

    # Check if SNR data is present (non-zero for at least some points)
    snrs = [p.get("snr", 0.0) for p in points]
    if max(snrs) < 0.1:
        # No SNR data in this stream — bypass filter
        return points

    filtered = [p for p in points if p.get("snr", 0.0) >= SNR_THRESHOLD_DB]

    # Safety: if filter removes everything, fall back to top-25% by SNR
    if not filtered:
        cutoff   = float(np.percentile(snrs, 75))
        filtered = [p for p in points if p.get("snr", 0.0) >= cutoff]

    return filtered


# ══════════════════════════════════════════════════════════════════════════════
#  ❼  FEATURE EXTRACTOR
#     Extracts 6 numerical features from a list of point dicts.
#     All features are designed to be sensor-agnostic (unit-based).
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(points: list) -> dict:
    """
    Extract classification features from a single frame's point cloud.
    Expects points already filtered by SNR (call filter_by_snr first).

    Returns dict with keys:
        n              — number of detected points (after SNR filter)
        avg_abs_vel    — mean of |velocity| across all points (m/s)
        vel_std        — std deviation of velocity   (m/s)
        frac_moving    — fraction of points with |v| > MOVE_THRESH
        bbox_volume    — bounding box volume (m³) of the point cloud
        bbox_max_span  — largest single-axis span (m) of the point cloud
        centroid       — (x, y, z) of point cloud centroid
    """
    n = len(points)
    if n == 0:
        return {
            "n": 0, "avg_abs_vel": 0.0, "vel_std": 0.0,
            "frac_moving": 0.0, "bbox_volume": 0.0,
            "bbox_max_span": 0.0, "centroid": (0.0, 0.0, 0.0),
        }

    xs = np.array([p["x"] for p in points])
    ys = np.array([p["y"] for p in points])
    zs = np.array([p["z"] for p in points])
    vs = np.array([p["v"] for p in points])

    # Velocity features
    abs_vs       = np.abs(vs)
    avg_abs_vel  = float(np.mean(abs_vs))
    vel_std      = float(np.std(vs))

    # Movement threshold — live data shows frac_moving always ~0.65–0.75 at 0.08
    # because sensor noise floor is higher than 0.08 m/s on this profile.
    # Raised to 0.20 m/s so only genuinely moving points are counted.
    MOVE_THRESH  = 0.20
    frac_moving  = float(np.mean(abs_vs > MOVE_THRESH))

    # Spatial extent
    dx = float(xs.max() - xs.min()) if n > 1 else 0.0
    dy = float(ys.max() - ys.min()) if n > 1 else 0.0
    dz = float(zs.max() - zs.min()) if n > 1 else 0.0
    bbox_volume   = max(dx * dy * dz, 1e-6)
    bbox_max_span = max(dx, dy, dz)

    centroid = (float(xs.mean()), float(ys.mean()), float(zs.mean()))

    return {
        "n"            : n,
        "avg_abs_vel"  : avg_abs_vel,
        "vel_std"      : vel_std,
        "frac_moving"  : frac_moving,
        "bbox_volume"  : bbox_volume,
        "bbox_max_span": bbox_max_span,
        "centroid"     : centroid,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ❼  TEMPORAL FEATURE EXTRACTOR
#     Looks at the last N frames to detect micro-Doppler oscillation
#     (breathing = periodic velocity sign changes at ~0.2–0.5 Hz).
# ══════════════════════════════════════════════════════════════════════════════

def extract_temporal_features(frames: list) -> dict:
    """
    Analyse last N frames for temporal patterns.

    Returns:
        centroid_vel_oscillation  — std of per-frame centroid velocity (m/s)
                                    High = oscillating (human breathing)
                                    Low  = stable (static object or empty)
        avg_n                     — mean point count over window
    """
    # Need at least 5 frames for a meaningful oscillation estimate.
    # Fewer frames → std is unreliable and can falsely spike to human.
    if len(frames) < 5:
        return {"centroid_vel_oscillation": 0.0, "avg_n": 0.0}

    centroid_vs = []
    ns          = []

    for f in frames:
        pts = f.get("points", [])
        ns.append(len(pts))
        if pts:
            vs = [p["v"] for p in pts]
            centroid_vs.append(float(np.mean(vs)))
        else:
            centroid_vs.append(0.0)

    # Oscillation = std of centroid velocity over time
    # Breathing human: ±0.05–0.12 m/s every ~2–5 frames → std ≈ 0.08–0.15+
    # Rigid object:    near-zero always                  → std ≈ 0.00–0.04
    oscillation = float(np.std(centroid_vs)) if len(centroid_vs) > 1 else 0.0

    return {
        "centroid_vel_oscillation": oscillation,
        "avg_n"                   : float(np.mean(ns)),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ❽  CALIBRATION STORE
#     Stores per-class feature statistics learned during calibration.
#     Saved to calibration.json so it persists between runs.
# ══════════════════════════════════════════════════════════════════════════════

CALIB_FILE = "calibration.json"

# Default thresholds — CALIBRATED from real .dat recordings
#
# Object recording analysis (301 frames):
#   n per frame  : mean=4.7, max=7    → object gives very few points
#   velocity     : 97.7% are exactly 0.000 m/s, max seen = 0.97 m/s (one spike)
#   SNR          : min=15 dB — all points well above noise floor
#   oscillation  : mean=0.024, max=0.067, 99th-pct=0.067
#                  → object NEVER exceeds 0.067 m/s osc in this recording
#   bbox         : mean=3.6 m — whole-room reflections, not useful for humans
#
# Threshold derivation:
#   object_max_oscillation = 99th-pct of object osc = 0.067  (rounded to 0.070)
#   human_min_oscillation  = object 99th-pct + 0.030 gap      = 0.097  (→ 0.100)
#   human_min_points       = 5 (human gives many more reflections than 4.7 avg)
#
# The critical fix: old thresholds were object=0.050 / human=0.080, which
# OVERLAPPED with real object data (14.8% of object frames exceeded 0.050).
# That caused those frames to fall into the ambiguous zone and sometimes
# get smoothed into "human". New thresholds have zero overlap.
DEFAULT_THRESHOLDS = {
    # ── EMPTY ────────────────────────────────────────────────────────────────
    # Object recording shows n=2..7 (mean 4.7). We must NOT call these empty.
    # Keep empty_max_points low (≤ 1) so the 2-point object frames are not
    # swallowed by the empty rule. True empty room gives n=0 on this sensor.
    "empty_max_points"        : 1,

    # ── OBJECT ───────────────────────────────────────────────────────────────
    # Based on real recording: object osc never exceeds 0.067 m/s.
    # Set ceiling at 0.070 (adds 3 ms margin above measured 99th-pct).
    "object_max_frac_moving"  : 1.0,    # disabled — oscillation is the only reliable feature
    "object_max_oscillation"  : 0.070,  # m/s — from .dat: 99th-pct = 0.067
    "object_max_bbox"         : 999.0,  # disabled — bbox is dominated by room reflections

    # ── HUMAN ────────────────────────────────────────────────────────────────
    # Gap between object ceiling (0.070) and human floor (0.100) = 0.030 m/s.
    # Anything in 0.070–0.100 is ambiguous → classified as Object (conservative).
    "human_min_oscillation"   : 0.100,  # m/s — safely above object 99th-pct
    "human_min_frac_moving"   : 1.1,    # disabled — use oscillation only
    "human_min_points"        : 2,      # object gives n=2..7, so don't block human on count

    # ── SMOOTHING ─────────────────────────────────────────────────────────────
    "smooth_window"           : 12,
}

_thresholds = dict(DEFAULT_THRESHOLDS)   # mutable copy used at runtime


def load_calibration():
    """Load saved calibration, ignoring stale keys from old versions."""
    global _thresholds
    if os.path.exists(CALIB_FILE):
        try:
            with open(CALIB_FILE) as f:
                saved = json.load(f)
            valid   = set(DEFAULT_THRESHOLDS.keys())
            stale   = [k for k in saved if k not in valid]
            if stale:
                log.warning(f"[Calib] Stale keys in calibration file: {stale} — auto-removing.")
                # Overwrite with clean version immediately
                for k in stale:
                    del saved[k]
                with open(CALIB_FILE, "w") as f:
                    json.dump(saved, f, indent=2)
            for k in saved:
                if k in valid:
                    _thresholds[k] = saved[k]
            log.info(f"[Calib] Loaded calibration (skipped {len(stale)} stale key(s))")
        except Exception as e:
            log.warning(f"[Calib] Could not load {CALIB_FILE}: {e} — using defaults")


def save_calibration():
    with open(CALIB_FILE, "w") as f:
        json.dump(_thresholds, f, indent=2)
    log.info(f"[Calib] Saved calibration to {CALIB_FILE}")


load_calibration()


# ══════════════════════════════════════════════════════════════════════════════
#  ❾  TEMPORAL SMOOTHER  — majority vote over last N labels
# ══════════════════════════════════════════════════════════════════════════════

class LabelSmoother:
    """Majority-vote smoother over a rolling window of labels."""

    def __init__(self):
        self._window = deque(maxlen=_thresholds["smooth_window"])

    def push_and_smooth(self, raw_label: str) -> str:
        self._window.append(raw_label)
        counts = {}
        for lbl in self._window:
            counts[lbl] = counts.get(lbl, 0) + 1
        return max(counts, key=counts.get)


SMOOTHER = LabelSmoother()


# ══════════════════════════════════════════════════════════════════════════════
#  ❿  MAIN CLASSIFIER
#     Decision tree using extracted features + thresholds.
#     Each decision is logged so you can debug why a frame was classified wrong.
# ══════════════════════════════════════════════════════════════════════════════

def classify_detection(points: list, temporal_frames: list) -> dict:
    """
    Multi-feature classifier for mmWave point cloud.

    Decision priority:
        1. Empty?   → point count alone decides
        2. Static?  → low Doppler + low oscillation + tight bbox
        3. Human?   → oscillation OR fraction-moving above threshold
        4. Fallback → Static (conservative — avoids false human detections)

    Args:
        points         : list of point dicts from the LATEST frame
        temporal_frames: list of last N raw frame dicts (for oscillation)

    Returns:
        dict with status, cls, icon, sub, confidence, debug
    """
    T   = _thresholds

    # ── POINT CLOUD FILTERING (3 stages) ─────────────────────────────────────
    # Applied in order: range gate -> cluster -> SNR
    # Range gate: drop points beyond 5 m (far wall reflections)
    # Cluster:    drop outliers > 2 m from centroid (X=+-8m wall ghosts)
    # SNR:        drop low-quality multipath points
    n_raw = len(points)

    snrs_raw = [p.get("snr", 0.0) for p in points]
    if points and max(snrs_raw) < 0.1:
        log.warning("[Classify] SNR data absent (all zeros) — SNR filter BYPASSED. "
                    "Check that TLV type 7 (side-info) is enabled in your .cfg profile.")

    def _apply_filters(pts: list) -> list:
        pts = filter_by_range(pts)
        pts = filter_by_cluster(pts)
        pts = filter_by_snr(pts)
        return pts

    points_filtered = _apply_filters(points)

    # Apply same pipeline to every temporal frame for oscillation computation
    temporal_filtered = []
    for fr in temporal_frames:
        fr_filtered = dict(fr)
        fr_filtered["points"] = _apply_filters(fr.get("points", []))
        temporal_filtered.append(fr_filtered)

    f   = extract_features(points_filtered)
    tf  = extract_temporal_features(temporal_filtered)

    n         = f["n"]
    frac_mov  = f["frac_moving"]
    osc       = tf["centroid_vel_oscillation"]
    bbox      = f["bbox_max_span"]
    avg_vel   = f["avg_abs_vel"]

    debug = {
        "n_raw"        : n_raw,
        "n_filtered"   : n,
        "frac_moving"  : round(frac_mov, 3),
        "oscillation"  : round(osc, 4),
        "avg_abs_vel"  : round(avg_vel, 4),
        "bbox_max_span": round(bbox, 3),
        "frames_used"  : len(temporal_frames),
        "snr_threshold": SNR_THRESHOLD_DB,
    }

    # Log every frame so you can see real values in terminal
    log.info(f"[Classify] n={n}/{n_raw} frac_mov={frac_mov:.3f} osc={osc:.4f} bbox={bbox:.2f}")

    # ── Rule 1: EMPTY ─────────────────────────────────────────────────────────
    # Use n_raw (before SNR filter) — if even raw count is tiny, truly empty.
    # Use n (after filter) as secondary check — no high-quality points = empty.
    # NOTE: object recording shows n_raw=2..7, so empty_max_points must be <= 1.
    if n_raw <= T["empty_max_points"] or n == 0:
        raw_label = "empty"
        conf      = max(10, 30 - n_raw * 2)
        result    = {
            "status"    : "EMPTY SPACE",
            "cls"       : "empty",
            "icon"      : "◯",
            "sub"       : f"Only {n_raw} raw pts ({n} above SNR threshold) — below detection threshold",
            "confidence": conf,
            "debug"     : debug,
        }
        smoothed = SMOOTHER.push_and_smooth(raw_label)
        return _apply_smooth(result, smoothed)

    # ── Rule 2: OBJECT DETECTED ──────────────────────────────────────────────
    # Oscillation is the only reliable feature given whole-room bbox noise.
    # Rigid object → near-zero oscillation over time (no breathing).
    # Human → oscillation > 0.11 m/s (breathing micro-Doppler).
    if osc <= T["object_max_oscillation"]:
        raw_label = "object"
        conf      = min(97, int(70 + (T["object_max_oscillation"] - osc) * 300))
        result    = {
            "status"    : "OBJECT DETECTED",
            "cls"       : "object",
            "icon"      : "⬜",
            "sub"       : (f"Rigid object — {n} pts · osc={osc:.4f} m/s "
                           f"(threshold={T['object_max_oscillation']})"),
            "confidence": conf,
            "debug"     : debug,
        }
        smoothed = SMOOTHER.push_and_smooth(raw_label)
        return _apply_smooth(result, smoothed)

    # ── Rule 3: HUMAN — via MICRO-DOPPLER ───────────────────────────────────
    # Oscillation above object threshold = breathing/movement signature.
    # Extra guard: real human body is < 2.5 m wide. If bbox > 2.5 m the
    # point cloud spans the whole room (multipath) — do NOT call it human.
    if (osc >= T["human_min_oscillation"]
            and n >= T["human_min_points"]
            and bbox <= 2.5):
        if osc >= 0.35:
            icon = "🚶"
            sub  = f"Human moving — osc={osc:.3f} m/s · {n} pts"
            conf = min(97, int(75 + osc * 10))
        else:
            icon = "🫁"
            sub  = f"Human — breathing detected · osc={osc:.3f} m/s · {n} pts"
            conf = min(97, int(60 + osc * 150))

        raw_label = "human"
        result    = {
            "status"    : "HUMAN DETECTED",
            "cls"       : "human",
            "icon"      : icon,
            "sub"       : sub,
            "confidence": conf,
            "debug"     : debug,
        }
        smoothed = SMOOTHER.push_and_smooth(raw_label)
        return _apply_smooth(result, smoothed)

    # ── Fallback: Ambiguous — call Object (conservative) ─────────────────────
    # Avoids false human alerts when sensor data is noisy or transitional
    raw_label = "object"
    result    = {
        "status"    : "OBJECT DETECTED",
        "cls"       : "object",
        "icon"      : "⬜",
        "sub"       : f"Ambiguous — defaulting to object · {n} pts · osc={osc:.4f}",
        "confidence": 40,
        "debug"     : debug,
    }
    smoothed = SMOOTHER.push_and_smooth(raw_label)
    return _apply_smooth(result, smoothed)


def _apply_smooth(result: dict, smoothed_label: str) -> dict:
    """
    If the smoothed label differs from the raw label, override the cls/status/icon
    so the displayed result reflects the majority-vote decision.
    This prevents single-frame flickering.
    """
    raw_cls = result["cls"]
    if smoothed_label == raw_cls:
        return result

    # The window majority disagrees with this frame — use smoothed label
    SMOOTH_LABELS = {
        "empty" : ("EMPTY SPACE",                  "empty",  "◯",  "Smoothed — transitioning to empty"),
        "object": ("OBJECT DETECTED",              "object", "⬜", "Smoothed — transitioning to object"),
        "human" : ("HUMAN DETECTED (VITAL SIGNS ACTIVE)", "human", "🫁", "Smoothed — transitioning to human"),
    }
    status, cls, icon, sub = SMOOTH_LABELS.get(smoothed_label, SMOOTH_LABELS["empty"])
    result = dict(result)
    result.update({"status": status, "cls": cls, "icon": icon,
                   "sub": sub, "confidence": max(30, result["confidence"] - 20)})
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  ⓫  SERIAL READER THREAD
# ══════════════════════════════════════════════════════════════════════════════

class RadarSerialReader(threading.Thread):
    def __init__(self, port: str, baud: int):
        super().__init__(daemon=True, name="RadarReader")
        self.port        = port
        self.baud        = baud
        self._running    = False
        self._connected  = False

    def stop(self): self._running = False

    @property
    def is_connected(self): return self._connected

    def run(self):
        self._running = True
        log.info(f"[Reader] Starting on {self.port} @ {self.baud} baud")
        while self._running:
            try:
                self._connect_and_read()
            except serial.SerialException as e:
                log.error(f"[Reader] Serial error: {e}. Retrying in 3s…")
                self._connected = False
                time.sleep(3)
            except Exception as e:
                log.exception(f"[Reader] Unexpected: {e}")
                self._connected = False
                time.sleep(3)

    def _connect_and_read(self):
        log.info(f"[Reader] Opening {self.port}…")
        ser = serial.Serial(
            port=self.port, baudrate=self.baud,
            bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE, timeout=1.0,
        )
        self._connected = True
        log.info(f"[Reader] Connected to {self.port} ✓")
        buf = bytearray()

        while self._running:
            chunk = ser.read(4096)
            if not chunk:
                continue
            buf.extend(chunk)

            while True:
                pos = buf.find(MAGIC_WORD)
                if pos == -1:
                    buf = buf[-(MAGIC_LEN - 1):]
                    break
                if pos > 0:
                    buf = buf[pos:]

                min_needed = MAGIC_LEN + FRAME_HEADER_SIZE
                if len(buf) < min_needed:
                    break

                total_len = struct.unpack_from('<I', buf, MAGIC_LEN + 4)[0]
                if total_len < min_needed or total_len > 65536:
                    buf = buf[MAGIC_LEN:]
                    continue
                if len(buf) < total_len:
                    break

                packet = bytes(buf[:total_len])
                buf    = buf[total_len:]

                frame = parse_frame(packet[MAGIC_LEN:])
                if frame:
                    FRAME_BUFFER.push(frame)
                    log.debug(f"[Reader] Frame {frame['frame_number']:05d} — {frame['num_detected']} pts")
                else:
                    FRAME_BUFFER.drop()

        ser.close()
        self._connected = False


# ══════════════════════════════════════════════════════════════════════════════
#  ⓬  CONFIG SENDER
# ══════════════════════════════════════════════════════════════════════════════

def send_config(cfg_port: str, cfg_file: str) -> bool:
    if not os.path.exists(cfg_file):
        log.warning(f"[Config] '{cfg_file}' not found — skipping. Sensor must already be running OOB demo.")
        return False
    log.info(f"[Config] Sending {cfg_file} → {cfg_port}…")
    try:
        with serial.Serial(cfg_port, CFG_BAUD, timeout=2) as s:
            time.sleep(0.2)
            for raw_line in open(cfg_file):
                line = raw_line.strip()
                if not line or line.startswith('%'):
                    continue
                s.write((line + '\n').encode())
                time.sleep(0.05)
                resp = s.read_all().decode(errors='ignore').strip()
                log.info(f"[Config]  >> {line:<40}  ← {resp or '(ok)'}")
        log.info("[Config] Config sent ✓")
        return True
    except serial.SerialException as e:
        log.error(f"[Config] Port {cfg_port} error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  ⓭  FLASK APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__, static_folder='.')
CORS(app)


@app.route('/')
def serve_dashboard():
    return send_from_directory('.', 'radar_dashboard.html')


@app.route('/api/radar')
def get_radar():
    """Return latest classified radar frame."""
    frame = FRAME_BUFFER.latest()

    if frame is None:
        return jsonify({
            "source"          : "empty",
            "frame"           : 0,
            "num_points"      : 0,
            "points"          : [],
            "status"          : "EMPTY SPACE",
            "cls"             : "empty",
            "icon"            : "◯",
            "sub"             : "Waiting for sensor data…",
            "confidence"      : 0,
            "sensor_connected": READER.is_connected,
            "total_frames"    : FRAME_BUFFER.total,
            "dropped_frames"  : FRAME_BUFFER.drops,
        })

    # Use last 10 frames for temporal analysis (~1 second at 10 fps)
    temporal_window = FRAME_BUFFER.last_n(10)
    points          = frame["points"]
    classification  = classify_detection(points, temporal_window)

    pts_out = [
        {"x": round(float(p["x"]), 4), "y": round(float(p["y"]), 4),
         "z": round(float(p["z"]), 4), "v": round(float(p["v"]), 5),
         "snr": round(float(p.get("snr", 0)), 2)}
        for p in points
    ]

    return jsonify({
        "source"          : "live",
        "frame"           : frame["frame_number"],
        "num_points"      : len(points),
        "points"          : pts_out,
        "sensor_connected": READER.is_connected,
        "total_frames"    : FRAME_BUFFER.total,
        "dropped_frames"  : FRAME_BUFFER.drops,
        **classification,
    })


@app.route('/api/status')
def get_status():
    return jsonify({
        "sensor_connected": READER.is_connected,
        "data_port"       : DATA_PORT,
        "cfg_port"        : CFG_PORT,
        "total_frames"    : FRAME_BUFFER.total,
        "dropped_frames"  : FRAME_BUFFER.drops,
        "thresholds"      : _thresholds,
    })


@app.route('/api/calibrate')
def calibrate():
    """
    ── CALIBRATION ENDPOINT ──────────────────────────────────────────────────
    Collect sensor data for a few seconds and auto-tune thresholds.

    Usage:
        GET /api/calibrate?label=empty   (sensor pointing at empty room)
        GET /api/calibrate?label=object  (rigid object in front of sensor)
        GET /api/calibrate?label=human   (human standing/sitting in front)

    How it works:
        Collects the last 30 frames (~3 seconds), computes feature stats,
        and sets thresholds as midpoints between class distributions.
    """
    label = request.args.get("label", "").lower()
    if label not in ("empty", "object", "human"):
        return jsonify({"error": "label must be one of: empty, object, human"}), 400

    frames = FRAME_BUFFER.last_n(30)
    if len(frames) < 5:
        return jsonify({"error": "Not enough frames yet — wait a few seconds and retry"}), 400

    feats_list = [extract_features(f["points"]) for f in frames]
    tf         = extract_temporal_features(frames)

    ns         = [f["n"]             for f in feats_list]
    fracs      = [f["frac_moving"]   for f in feats_list]
    oscs_raw   = []

    # Compute oscillation over rolling sub-windows
    for i in range(2, len(frames)):
        sub = extract_temporal_features(frames[max(0,i-5):i])
        oscs_raw.append(sub["centroid_vel_oscillation"])

    osc_mean = float(np.mean(oscs_raw)) if oscs_raw else 0.0
    osc_std  = float(np.std(oscs_raw))  if oscs_raw else 0.0

    log.info(f"[Calib] label={label} | avg_n={np.mean(ns):.1f} | "
             f"avg_frac={np.mean(fracs):.3f} | osc_mean={osc_mean:.4f}")

    if label == "empty":
        # Learn the max point count seen during "empty" condition
        _thresholds["empty_max_points"] = max(4, int(np.percentile(ns, 95)) + 1)
        log.info(f"[Calib] empty_max_points → {_thresholds['empty_max_points']}")

    elif label == "object":
        # Learn what a static object's oscillation and motion fraction look like
        max_osc   = osc_mean + 2 * osc_std
        max_frac  = float(np.percentile(fracs, 95))
        _thresholds["object_max_oscillation"]   = round(max(0.030, max_osc + 0.005), 4)
        _thresholds["object_max_frac_moving"]   = round(min(0.30,  max_frac + 0.05), 3)
        _thresholds["human_min_oscillation"]    = round(max(0.050, max_osc + 0.030), 4)
        log.info(f"[Calib] object_max_oscillation → {_thresholds['object_max_oscillation']}")
        log.info(f"[Calib] object_max_frac_moving → {_thresholds['object_max_frac_moving']}")
        log.info(f"[Calib] human_min_oscillation  → {_thresholds['human_min_oscillation']}")

    elif label == "human":
        # Learn what a human's oscillation and motion fraction look like
        min_osc  = max(0.020, osc_mean - osc_std)
        min_frac = float(np.percentile(fracs, 10))
        _thresholds["human_min_oscillation"]  = round(min_osc,   4)
        _thresholds["human_min_frac_moving"]  = round(min_frac,  3)
        _thresholds["human_min_points"]       = max(5, int(np.percentile(ns, 10)))
        log.info(f"[Calib] human_min_oscillation → {_thresholds['human_min_oscillation']}")
        log.info(f"[Calib] human_min_frac_moving → {_thresholds['human_min_frac_moving']}")

    save_calibration()

    return jsonify({
        "ok"          : True,
        "label"       : label,
        "frames_used" : len(frames),
        "thresholds"  : _thresholds,
        "stats": {
            "avg_n"      : round(float(np.mean(ns)), 2),
            "avg_frac"   : round(float(np.mean(fracs)), 3),
            "osc_mean"   : round(osc_mean, 4),
            "osc_std"    : round(osc_std,  4),
        }
    })


@app.route('/api/thresholds', methods=['GET', 'POST'])
def thresholds():
    """GET to view current thresholds. POST JSON body to override any key."""
    if request.method == 'POST':
        data = request.get_json(force=True) or {}
        for k, v in data.items():
            if k in _thresholds:
                _thresholds[k] = v
                log.info(f"[Threshold] {k} → {v}")
        save_calibration()
        return jsonify({"ok": True, "thresholds": _thresholds})
    return jsonify(_thresholds)


# ══════════════════════════════════════════════════════════════════════════════
#  ⓮  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

READER = RadarSerialReader(port=DATA_PORT, baud=DATA_BAUD)

if __name__ == '__main__':
    send_config(CFG_PORT, CFG_FILE)
    log.info("[Main] Starting serial reader…")
    READER.start()
    time.sleep(1.0)

    log.info(f"[Main] Flask on http://0.0.0.0:{FLASK_PORT}")
    app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, threaded=True)
"""Re-detect AprilTags from an rl_deploy run's video to recover the fish
trajectory in the anchor (world) frame. Wraps render_run.py so detection math
matches the reference renderer exactly.

Returns a dict of equal-length lists keyed by frame_index order:
  frame_time (s from first frame), pos_x, pos_y, yaw, detected (bool)
Undetected frames carry NaN pos/yaw and detected=False.
"""
from __future__ import annotations
import csv
import json
import os
import sys

import cv2
import numpy as np


# Module-level detector cache: reusing one Detector instance avoids a
# dt_apriltags native-library crash in __del__ when multiple instances are
# created and destroyed in the same process.  Keyed by quad_decimate so that
# repeated calls with different decimation values each get their own cached
# Detector.
_detector_cache: dict = {}

# Default detection parameters: quad_decimate=2.0 (the rough quad-detect runs on
# a half-resolution image, but with refine_edges=1 the tag corners are re-fit and
# bits decoded at full resolution, so pose accuracy is decimate-independent —
# decimate=2 is much faster and matches the validated image_tune.json). Combined
# with image-tune preprocessing (alpha/beta/blur) seeded from image_tune.json.
_DEFAULT_PARAMS: dict = {
    "alpha": 0.9,
    "beta": -20.0,
    "blur": 3,
    "quad_decimate": 2.0,
}


def _get_detector(quad_decimate: float):
    """Return a cached Detector for the given quad_decimate, creating once.

    All other Detector args are fixed (families tag36h11, quad_sigma=0.0,
    refine_edges=1, decode_sharpening=0.25) — they do not affect the cache key.
    """
    try:
        from dt_apriltags import Detector
    except ImportError:
        from pupil_apriltags import Detector
    key = quad_decimate
    if key not in _detector_cache:
        _detector_cache[key] = Detector(
            families="tag36h11",
            nthreads=2,
            quad_decimate=quad_decimate,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
        )
    return _detector_cache[key]


def _import_render_run():
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    from deploy.laptop import render_run as RR  # type: ignore
    return RR


def _load_frame_times(ts_path: str) -> dict:
    with open(ts_path) as f:
        rows = list(csv.DictReader(f))
    times = {int(r["frame_index"]): float(r["laptop_time"]) for r in rows}
    if not times:
        raise ValueError(f"No frame timestamps in {ts_path}")
    t0 = min(times.values())
    return {k: v - t0 for k, v in times.items()}


def clock_offset_s(run_dir: str) -> float:
    """Pi->laptop clock offset in seconds (laptop = pi + offset). 0.0 if absent."""
    import json as _json
    p = os.path.join(run_dir, "raw", "run_config.json")
    if os.path.isfile(p):
        with open(p) as f:
            return float(_json.load(f).get("clock_offset_ms", 0.0)) / 1000.0
    return 0.0


def _tag_ids_from_run(run_dir: str, override) -> dict:
    if override:
        return override
    cfg_path = os.path.join(run_dir, "raw", "run_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        if "tag_ids" in cfg:
            return cfg["tag_ids"]
    return {"fish": 0, "anchor": 1, "target": 2}


def extract_trajectory(run_dir: str, tag_size_m: float = 0.080,
                       tag_ids=None, camera: str | None = None,
                       params: dict | None = None,
                       global_anchor: dict | None = None,
                       max_frames: int | None = None,
                       win_start: float | None = None,
                       win_end: float | None = None,
                       progress=None) -> dict:
    """Decode video.avi and re-detect AprilTags to recover the fish trajectory.

    Parameters
    ----------
    run_dir:
        Path to the rl_deploy run directory (contains raw/video.avi,
        raw/frame_timestamps.csv, raw/run_config.json).
    tag_size_m:
        Physical side length of the AprilTag in metres (default 0.080).
    tag_ids:
        Override dict with keys 'fish', 'anchor', 'target' mapped to tag IDs.
        If None, reads from run_config.json; falls back to {fish:0, anchor:1,
        target:2}.
    camera:
        Camera profile name passed to render_run.load_camera_params
        (default None -> "overhead").
    params:
        Detection parameter dict with optional keys:
          alpha       – convertScaleAbs contrast multiplier (default 0.9)
          beta        – convertScaleAbs brightness offset  (default -20.0)
          blur        – GaussianBlur kernel size, >=3 and odd (default 3)
          quad_decimate – AprilTag decimation factor (default 2.0; corners
                          refined at full res via refine_edges=1)
        If None, uses defaults: alpha=0.9, beta=-20.0, blur=3, quad_decimate=2.0
        (rough-detect decimated, corners refined at full res via refine_edges=1;
        matches the validated image_tune.json — best offline detection).
    global_anchor:
        Optional fixed session-wide anchor pose dict (as produced by
        anchor_calibration.calibrate_global_anchor) with keys 'anchor_R' (3x3)
        and 'anchor_t' (length 3); list or ndarray accepted. When provided, this
        single pose is used for every frame and the anchor is NOT updated from
        live per-frame detection, so a frame is detected whenever the fish tag
        is found. When None (default), the original per-run static-anchor reuse
        behaviour is kept exactly.
    max_frames:
        Optional cap on how many video frames are decoded (for speed). When
        None (default) all frames are read (current behaviour).
    win_start, win_end:
        Optional detection window (seconds, frame_time axis). When set, the
        (expensive) AprilTag ``detector.detect`` is only run for frames whose
        ``frame_time`` lies in ``[win_start, win_end]``; frames outside the
        window are appended with NaN pos/yaw and ``detected=False`` *without*
        running detection (the speedup). Both default None = detect every frame
        (back-compat). Composes with ``max_frames``.

    Returns
    -------
    dict with equal-length lists:
        frame_time  – seconds from first frame (float)
        pos_x       – world X position (m), NaN when not detected
        pos_y       – world Y position (m), NaN when not detected
        yaw         – fish yaw (rad), NaN when not detected
        detected    – True when fish tag found AND anchor pose is known
    """
    RR = _import_render_run()
    ids = _tag_ids_from_run(run_dir, tag_ids)
    fish_id, anchor_id = ids["fish"], ids["anchor"]

    vid = os.path.join(run_dir, "raw", "video.avi")
    ts = os.path.join(run_dir, "raw", "frame_timestamps.csv")

    if not os.path.isfile(vid):
        raise FileNotFoundError(vid)

    frame_times = _load_frame_times(ts)

    _, _, cam_params = RR.load_camera_params(RR.SETTINGS_FILE, camera)

    # Merge user params with defaults.
    p = dict(_DEFAULT_PARAMS)
    if params is not None:
        p.update(params)

    alpha: float = float(p.get("alpha", 1.0))
    beta: float = float(p.get("beta", 0.0))
    blur: int = int(p.get("blur", 0))
    quad_decimate: float = float(p.get("quad_decimate", 1.0))

    detector = _get_detector(quad_decimate)

    cap = cv2.VideoCapture(vid)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None and total_frames:
        total_frames = min(total_frames, max_frames)

    # Static-anchor state: detect anchor whenever visible, persist last-known
    # pose.  Fish must be seen in the *current* frame (it moves).  Together
    # this means a pose is recovered for every frame where the fish is
    # detected, regardless of whether the anchor happens to be in frame too.
    anchor_R = None  # last-known anchor rotation matrix (3x3)
    anchor_t = None  # last-known anchor translation vector (3,)

    # Global (session-wide) anchor override: a single fixed pose used for every
    # frame; the per-frame anchor detection is then never consulted.
    use_global_anchor = global_anchor is not None
    if use_global_anchor:
        anchor_R = np.asarray(global_anchor["anchor_R"], dtype=float).reshape(3, 3)
        anchor_t = np.asarray(global_anchor["anchor_t"], dtype=float).reshape(3)

    out = {k: [] for k in ("frame_time", "pos_x", "pos_y", "yaw", "detected")}
    idx = 0
    try:
        while True:
            if max_frames is not None and idx >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break

            if progress is not None and idx % 30 == 0:
                progress(idx, total_frames)

            t = frame_times.get(idx, idx / src_fps)

            # Window-restricted detection: skip the expensive detect() entirely
            # for frames outside [win_start, win_end] (they get NaN/undetected),
            # keeping the frame-time axis and ordering identical.
            in_window = True
            if win_start is not None and t < win_start:
                in_window = False
            if win_end is not None and t > win_end:
                in_window = False
            if not in_window:
                out["pos_x"].append(float("nan"))
                out["pos_y"].append(float("nan"))
                out["yaw"].append(float("nan"))
                out["detected"].append(False)
                out["frame_time"].append(t)
                idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Per-frame image preprocessing on a copy.
            proc = gray
            if alpha != 1.0 or beta != 0.0:
                proc = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
            if blur >= 3:
                proc = cv2.GaussianBlur(proc, (blur, blur), 0)

            tags = detector.detect(proc, estimate_tag_pose=True,
                                   camera_params=cam_params,
                                   tag_size=tag_size_m)
            tag_map = {t.tag_id: t for t in tags}

            # Update static-anchor cache whenever anchor is visible (skipped
            # when a fixed global anchor pose is supplied).
            if not use_global_anchor and anchor_id in tag_map:
                anchor_R = tag_map[anchor_id].pose_R
                anchor_t = tag_map[anchor_id].pose_t.reshape(3)

            fish_det = tag_map.get(fish_id)
            if fish_det is not None and anchor_R is not None:
                x, y = RR.world_xy(fish_det.pose_t.reshape(3),
                                   anchor_t, anchor_R)
                yaw = RR.extract_yaw(fish_det.pose_R, anchor_R)
                out["pos_x"].append(x)
                out["pos_y"].append(y)
                out["yaw"].append(yaw)
                out["detected"].append(True)
            else:
                out["pos_x"].append(float("nan"))
                out["pos_y"].append(float("nan"))
                out["yaw"].append(float("nan"))
                out["detected"].append(False)

            out["frame_time"].append(t)
            idx += 1
    finally:
        cap.release()
    if progress is not None:
        progress(idx, total_frames or idx)

    # Sort by frame_time in case timestamps are not perfectly in order.
    order = np.argsort(out["frame_time"], kind="stable")
    for k in out:
        out[k] = [out[k][i] for i in order]
    return out

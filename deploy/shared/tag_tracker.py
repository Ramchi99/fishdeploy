"""per-tag ROI tracking state and detection helpers for AprilTag vision."""

import cv2
import numpy as np

MISS_THRESH      = 3
PAD_FACTOR       = 1.5
ANCHOR_INTERVAL  = 30   # full-frame anchor re-detect every N ROI frames
TARGET_INTERVAL  = 10   # sparse-mode target re-detect every N ROI frames


class TagTracker:
    """Tracks one AprilTag's last known position and pose across frames."""

    def __init__(self):
        self.cx_full    = None   # tag centre in full-frame pixels
        self.cy_full    = None
        self.bbox_w     = None   # tag bounding-box size in full-frame pixels
        self.bbox_h     = None
        self.miss_count = 0
        self.pose_R     = None   # last valid camera-frame rotation matrix (3,3)
        self.pose_t     = None   # last valid camera-frame translation (3,)

    @property
    def initialised(self) -> bool:
        return self.cx_full is not None

    @property
    def use_fullframe(self) -> bool:
        """True when this tag should trigger a full-frame fallback scan."""
        return (not self.initialised) or (self.miss_count >= MISS_THRESH)

    def get_roi(self, frame_w: int, frame_h: int):
        """Return (x0, y0, x1, y1) padded ROI, clamped to frame bounds."""
        hw = (self.bbox_w / 2) * PAD_FACTOR
        hh = (self.bbox_h / 2) * PAD_FACTOR
        x0 = max(0,        int(self.cx_full - hw))
        y0 = max(0,        int(self.cy_full - hh))
        x1 = min(frame_w,  int(self.cx_full + hw))
        y1 = min(frame_h,  int(self.cy_full + hh))
        return x0, y0, x1, y1

    def update(self, tag, crop_offset=(0, 0)):
        # pose_R/t are stored as-is: shifting the principal point doesn't affect 3D geometry
        ox, oy   = crop_offset
        corners  = tag.corners            # (4, 2) in crop space
        self.cx_full    = float(corners[:, 0].mean()) + ox
        self.cy_full    = float(corners[:, 1].mean()) + oy
        self.bbox_w     = float(corners[:, 0].max() - corners[:, 0].min())
        self.bbox_h     = float(corners[:, 1].max() - corners[:, 1].min())
        self.miss_count = 0
        if tag.pose_R is not None:
            self.pose_R = tag.pose_R.copy()
            self.pose_t = tag.pose_t.flatten().copy()

    def on_miss(self):
        self.miss_count += 1


def _roi_detect(detector, gray_full, tag_id, tracker, cam_params, tag_size, frame_w, frame_h,
                alpha=1.0, beta=0.0, blur=0):
    """Detect tag_id in a padded ROI crop; update tracker; return detection or None.

    alpha/beta/blur: preprocess only the crop (cheaper than full-frame preprocessing).
    """
    x0, y0, x1, y1 = tracker.get_roi(frame_w, frame_h)
    crop = gray_full[y0:y1, x0:x1]
    if alpha != 1.0 or beta != 0.0:
        crop = cv2.convertScaleAbs(crop, alpha=alpha, beta=beta)
    if blur >= 3:
        crop = cv2.GaussianBlur(crop, (blur, blur), 0)
    fx, fy, cx, cy = cam_params
    cam_crop = [fx, fy, cx - x0, cy - y0]  # shift principal point into crop space

    detections = detector.detect(
        crop,
        estimate_tag_pose=True,
        camera_params=cam_crop,
        tag_size=tag_size,
    )
    match = next((t for t in detections if t.tag_id == tag_id), None)
    if match:
        tracker.update(match, crop_offset=(x0, y0))
    else:
        tracker.on_miss()
    return match


def _fullframe_detect(detector, gray_full, cam_params, tag_size, trackers: dict):
    """Detect all tag IDs in the full frame; update all trackers; return {tag_id: detection_or_None}."""
    detections = detector.detect(
        gray_full,
        estimate_tag_pose=True,
        camera_params=cam_params,
        tag_size=tag_size,
    )
    det_map = {t.tag_id: t for t in detections}
    result  = {}
    for tag_id, trk in trackers.items():
        found = det_map.get(tag_id)
        if found:
            trk.update(found)
        else:
            trk.on_miss()
        result[tag_id] = found
    return result

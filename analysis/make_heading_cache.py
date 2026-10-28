"""Precompute the initial heading of every reach run from its video, so the figures do not need it.

Writes data/02_deploy_20260520/analysis/cache_heading/initial_heading.json (the thirty AprilTag-derived
headings that paper Fig. 8 reads). Needs the raw videos, which are not in git: fetch the run folders from
the OneDrive archive into data/02_deploy_20260520/<run>/raw/video.avi first.

    python analysis/make_heading_cache.py [OUT.json]
"""
import csv, glob, json, math, os, sys

import numpy as np, cv2
from dt_apriltags import Detector

import sys as _sys; _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import DEPLOY, ANALYSIS

OUT = sys.argv[1] if len(sys.argv) > 1 else str(ANALYSIS / "cache_heading" / "initial_heading.json")
CAM = [1055.0202, 1054.2105, 945.8357, 489.0351]
FISH, ANCHOR = 0, 1
_DET = Detector(families="tag36h11", nthreads=4, quad_decimate=1.0, refine_edges=1)


def initial_heading(run, n_want=12):
    vids = glob.glob(str(DEPLOY / f"*{run}*" / "raw" / "video.avi"))
    if not vids:
        return None
    cap = cv2.VideoCapture(vids[0]); yaws = []; fi = 0
    while len(yaws) < n_want and fi < 400:
        ret, frame = cap.read()
        if not ret:
            break
        fi += 1
        tm = {t.tag_id: t for t in _DET.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
              estimate_tag_pose=True, camera_params=CAM, tag_size=0.08)}
        if FISH in tm and ANCHOR in tm:
            fwd = (tm[ANCHOR].pose_R.T @ tm[FISH].pose_R)[:, 1]
            yaws.append(math.atan2(fwd[1], fwd[0]))
    cap.release()
    if not yaws:
        return None
    return float(np.arctan2(np.mean(np.sin(yaws)), np.mean(np.cos(yaws))))


runs = [r["run"] for r in csv.DictReader(open(ANALYSIS / "reachability" / "run_geometry.csv"))]
out = {}
for r in runs:
    h = initial_heading(r)
    out[r] = h
    print(f"  {r:<28s} {'None' if h is None else round(h, 6)}")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"), indent=1, sort_keys=True)
print(f"\nwrote {len(out)} headings ({sum(v is not None for v in out.values())} resolved) to {OUT}")

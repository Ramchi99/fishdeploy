#!/usr/bin/env python3
"""
Phase D — RUN_CATALOG.md: one table covering every run in the 2026-05-20 session
with its name, outcome, video, frame-drop status, detection quality and (for
static/disturbance runs) the fish-frame target angle/radius. Reads the caches
written by build_analysis.py.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))   # repository root
from paths import DEPLOY  # noqa: E402
sys.path.insert(0, _HERE)
import build_analysis as B

MISS = B.MISS_FLAG_THRESH

GROUP_TITLE = {
    "static": "Static-target runs",
    "disturbance": "Disturbance runs (perturbed)",
    "special": "Special",
    "moving": "Moving-target runs",
    "figure_eight": "Figure-eight runs",
}
OUTCOME_TAG = {"success": "✅ success", "fail": "❌ fail", "semi": "🟡 semi", "review": "🔍 review"}


def cache_stats(name):
    p = os.path.join(B.CACHE, name + ".npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    if bool(d["corrupt"]):
        return dict(corrupt=True)
    total = int(d["total"]); n_ts = int(d["n_ts"])
    drop = max(n_ts - total, 0)
    return dict(corrupt=False, total=total, n_ts=n_ts, drop=drop,
                drop_pct=(100 * drop / n_ts if n_ts else 0.0),
                miss=float(d["miss_frac"]) * 100,
                tl=str(d["tl_method"]))


def main():
    geo = {g["name"]: g for g in (B.reach_geometry(r) for r in B.MANIFEST if r["reach"])
           if g is not None}

    lines = []
    lines.append("# RL-deploy session catalog — 2026-05-20\n")
    lines.append("Hardware RL closed-loop swimming runs (CPG 24-D policy). Every run "
                 "below has a clean **real-time** trace video in `videos/`. Static- and "
                 "disturbance-target runs also feed the fish-frame reachability plot in "
                 "`reachability/`.\n")
    lines.append("**Raw data is read-only** — nothing here was written into the "
                 "`runs/.../raw/` folders.\n")
    lines.append("Legend: drops = recorder dropped frames (timing recovered, see "
                 "the troubleshooting table in `deploy/README.md`); miss% = frames where the fish tag was "
                 "not detected (gaps linearly interpolated); ⚠️ marks miss% > "
                 f"{int(MISS*100)}%.\n")

    # group ordering preserves the session story
    order = ["static", "disturbance", "special", "moving", "figure_eight"]
    flagged_miss, flagged_drop, corrupt = [], [], []

    by_group = {k: [r for r in B.MANIFEST if r["group"] == k] for k in order}
    for grp in order:
        runs = by_group[grp]
        if not runs:
            continue
        lines.append(f"\n## {GROUP_TITLE[grp]}\n")
        if grp in ("static", "disturbance"):
            lines.append("| run | outcome | target r (m) | target θ (°) | min dist (m) | drops | miss% | notes |")
            lines.append("|---|---|---|---|---|---|---|---|")
        else:
            lines.append("| run | outcome | frames | drops | miss% | notes |")
            lines.append("|---|---|---|---|---|---|")
        for r in runs:
            name = r["given"]
            st = cache_stats(name)
            note = r["note"]
            if st is None:
                row_note = (note + " — *(no cache)*").strip(" —")
                if grp in ("static", "disturbance"):
                    lines.append(f"| {name} | {OUTCOME_TAG[r['outcome']]} | – | – | – | ? | ? | {row_note} |")
                else:
                    lines.append(f"| {name} | {OUTCOME_TAG[r['outcome']]} | ? | ? | ? | {row_note} |")
                continue
            if st.get("corrupt"):
                corrupt.append(name)
                cn = (note).strip()
                if grp in ("static", "disturbance"):
                    lines.append(f"| {name} | {OUTCOME_TAG[r['outcome']]} | – | – | – | – | – | **CORRUPT video (0 frames)** — {cn} |")
                else:
                    lines.append(f"| {name} | {OUTCOME_TAG[r['outcome']]} | 0 | – | – | **CORRUPT video (0 frames)** — {cn} |")
                continue
            drop_s = f"{st['drop']} ({st['drop_pct']:.0f}%)" if st["drop"] else "—"
            if st["drop"]:
                flagged_drop.append(name)
            miss_s = f"{st['miss']:.0f}%"
            if st["miss"] > MISS * 100:
                miss_s = f"⚠️{st['miss']:.0f}%"
                flagged_miss.append(name)
            if grp in ("static", "disturbance"):
                g = geo.get(name)
                if g:
                    lines.append(f"| {name} | {OUTCOME_TAG[r['outcome']]} | {g['radius']:.2f} | "
                                 f"{g['angle']:+.0f} | {g['min_dist']:.3f} | {drop_s} | {miss_s} | {note} |")
                else:
                    lines.append(f"| {name} | {OUTCOME_TAG[r['outcome']]} | – | – | – | {drop_s} | {miss_s} | {note} *(geometry unavailable)* |")
            else:
                lines.append(f"| {name} | {OUTCOME_TAG[r['outcome']]} | {st['total']} | {drop_s} | {miss_s} | {note} |")

    # ---- assumptions / things to know ----
    lines.append("\n## Assumptions & things to know\n")
    lines.append("- **Real-time playback:** videos are variable-frame-rate; each frame is "
                 "held for its true inter-frame time. Runs with recorder frame-drops were "
                 "time-corrected (two-phase reconstruction) so the tail does not speed up. "
                 "Affected: " + (", ".join(flagged_drop) if flagged_drop else "none") + ".")
    lines.append(f"- **Detection gaps (miss% > {int(MISS*100)}%):** the fish tag is "
                 "intermittently lost to motion blur when the fish swims fast (lowering the "
                 "AprilTag decimation did **not** help). Those frames are linearly "
                 "interpolated for both the trace and the reachability path. Flagged runs: "
                 + (", ".join(sorted(set(flagged_miss))) if flagged_miss else "none") + ".")
    if corrupt:
        lines.append("- **Corrupt recording:** " + ", ".join(corrupt) +
                     " has 0 frames in video.avi (the old single-VideoWriter pipeline "
                     "produced an empty file). No video or geometry possible; listed for "
                     "completeness.")
    lines.append("- **Fish frame:** right-handed top-down — origin at the fish's initial "
                 "position, +x = its initial swim direction, +y = its left. Target "
                 "angle/radius are in this frame (0° ahead, +90° left, −90° right, ±180° "
                 "behind). Verified against the two 'target behind' runs (finetune_run_6/7 "
                 "→ ±175°/+158°). Initial pose = position median + circular-mean yaw over "
                 "the first 8 detected frames.")
    lines.append("- **Outlier rejection:** the AprilTag pose occasionally flips "
                 "(ambiguity / anchor jitter), producing world-position jumps up to ~2 m. "
                 "The fish world-track is velocity-gated (>~1 m/s steps dropped & "
                 "re-interpolated) then lightly smoothed before the fish-frame transform. "
                 "Most affected: finetune_run_4 (40 pts), finetune_run_6 (19), "
                 "finetune_run_7 (3), finetune_run_3 (1).")
    lines.append("- **Trajectory truncation:** successful runs are drawn up to the point "
                 "of closest approach to the target; failed runs are drawn to the end of "
                 "the run.")
    lines.append("- **Reachability scope:** only static + disturbance runs (target fixed). "
                 "Moving-target and figure-eight runs are video-only. `polaris` is excluded "
                 "and flagged for your review.")
    lines.append("- **Reached ratios are descriptive only** — targets were placed by hand, "
                 "not sampled uniformly, so the ratios are not an unbiased reachability rate.")

    out = str(DEPLOY / "RUN_CATALOG.md")
    open(out, "w").write("\n".join(lines) + "\n")
    print("saved", out)
    print(f"flagged miss>{int(MISS*100)}%: {sorted(set(flagged_miss))}")
    print(f"drop runs: {flagged_drop}")
    print(f"corrupt: {corrupt}")


if __name__ == "__main__":
    main()

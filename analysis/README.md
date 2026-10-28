# analysis — RL-deploy session analysis

Self-contained analysis of the **2026-05-20** hardware RL swimming session
(CPG 24-D policy). **Raw data is strictly read-only** — every output here is
generated from `../runs/<run>/raw/{video.avi, frame_timestamps.csv}` without
modifying them.

## Contents

| path | what |
|---|---|
| `RUN_CATALOG.md` | master table: every run, outcome, video, drops, detection quality, target angle/radius |
| `videos/<run>.mp4` | clean **real-time** (VFR) trace video — viridis time-trace + fish→target arrow, no clutter |
| `reachability/reachability_trajectories.png` | fish-frame reachability: all static + disturbance runs overlaid |
| `reachability/reachability_polar.png` | target angle vs radius, coloured by outcome |
| `reachability/run_geometry.csv` | per-run radius / angle / min-distance / detection miss |
| `reachability/reachability_summary.md` | reached ratios + angle/radius coverage + caveats |
| `cache/<run>.npz` | per-run detection cache (pixel + world positions + corrected timeline) |

## Colour key (reachability)

- 🟩 **green** — static target, success
- 🟥 **red** — static target, fail
- 🟪 **indigo** — disturbance run, success
- 🟪 **magenta** — disturbance run, fail

## Rebuild

```bash
# from fishdeploy/ repo root
python analysis/build_analysis.py   # detect + render videos (resumable)
python analysis/build_reach.py      # reachability plots + tables
python analysis/build_catalog.py    # RUN_CATALOG.md
python analysis/make_heading_cache.py   # initial heading per reach run (needs the videos from the archive)
```

`build_analysis.py` is idempotent: it skips runs whose cache/video already
exist (use `--force` to rebuild, `--only <name>` for one run).

## Method notes

- **Fish frame:** right-handed top-down — origin at the fish's initial position,
  +x = initial swim direction, +y = fish's left. Matches `render_run.py`'s
  default plotting convention.
- **Frame-drops:** some long recordings dropped frames (old recorder bottleneck,
  see the last row of the troubleshooting table in `../deploy/README.md`). Timing is reconstructed so playback stays
  real-time. Counts are in `RUN_CATALOG.md`.
- **Detection gaps:** the fish tag is sometimes lost to motion blur; gaps are
  linearly interpolated and any run >10% missed is flagged in the catalog.

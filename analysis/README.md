# analysis: from the deployment videos to the paper's inputs

The closed-loop runs of 2026-05-20 were filmed by the laptop camera. These scripts turn each run's video
and log into the tracked trajectories, the reach table, the run catalogue and the strobe composites that
the paper's figures read. Their outputs are committed under `data/02_deploy_20260520/analysis/`, so
fishsim rebuilds every figure from a clone of this repository; rerunning the scripts needs the videos from
the archive next to the runs' `run.csv`.

| Script | Reads | Writes | Read by |
|---|---|---|---|
| `build_analysis.py` | each run's `video.avi`, `frame_timestamps.csv`, `run_config.json`; the run list with outcomes is at its top | `analysis/cache/<run>.npz` (fish, anchor and target per frame, corrected timeline), `analysis/videos/<run>.mp4` (real-time trace video) | everything below; paper Fig. 7 and Fig. 8 |
| `redetect_trajectories.py` | the videos, the caches | `analysis/cache_redetect/<run>_redetect.npz` (metric world tracks), `coverage_summary.md` | paper Fig. 7 and Fig. 8, `plot_nav_collage.py` |
| `make_heading_cache.py` | the videos, `run_geometry.csv` | `analysis/cache_heading/initial_heading.json` | paper Fig. 8 |
| `build_reach.py` | the caches | `analysis/reachability/run_geometry.csv` (radius, bearing, outcome, closest approach per run), `reachability_summary.md` (13/22 static, 7/8 disturbance, 20/30), two plots | paper Fig. 8, the paper's reach numbers |
| `build_catalog.py` | the caches, the run configs | `RUN_CATALOG.md` (every run with outcome, target geometry, frame drops, detection quality) | the appendix run table, fishsim `paper/docs/DEPLOY_RUNS.md` |
| `gen_strobes.py` (with `build_strobe_plots.py`) | the videos, the caches | the six `analysis/report_strobes/*_traceover_strobe.png` | paper Fig. 1 and Fig. 7 |
| `plot_nav_collage.py` | the caches | the six-panel navigation collage (not in the paper) | fishsim's Fig. 7 imports its `panel()` to draw the five static-target panels |
| `canonical/` | the videos, `run.csv`, the curation manifest `canonical/layer2_parts.yml` | `canonical/<num>_<name>__p<part>.csv`, the 18 curated 1 kHz segments | fishsim `sysid/closed_loop/` (the direct-action fluid refit and the identified motor) |

```bash
python analysis/build_analysis.py              # resumable; --force rebuilds, --only <name> for one run
python analysis/redetect_trajectories.py
python analysis/make_heading_cache.py
python analysis/build_reach.py
python analysis/build_catalog.py
python analysis/gen_strobes.py
python analysis/canonical/convert_segments.py  # --num <n> [--part <p>] for one segment
```

Outcomes are hand-labelled in the run list at the top of `build_analysis.py` and flow from there into
every table; `finetune_run_4` is a fail (closest approach 246 mm with half its frames dropped), which is
what the paper's 13/22 and 20/30 count.

Conventions: the fish frame is right-handed and top-down, origin at the fish's initial position, +x its
initial heading, +y its left; bearings are 0° ahead, +90° left, ±180° behind. Long recordings dropped
frames (the recorder sustained about 20 fps at 1080p); the timeline is reconstructed from the frame
timestamps and the catalogue lists the counts. Where the fish tag is lost to motion blur the gap is
interpolated linearly; runs with more than 10 % missed frames are flagged.

Segment and detection file names under `canonical/` are the manifest's run names with everything but
letters, digits and underscores replaced (`canonical.curation.slug`); the same rule is in fishsim's
`sysid/closed_loop/canonical.py`.

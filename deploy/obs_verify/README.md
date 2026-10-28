# SOFI Observation Verifier

Walks through 10 physical tests to confirm every observation dimension
matches what the MuJoCo training environment expects.

## Quick start

**Step 1 — (optional) start the Pi sensor server** for IMU + motor tests:
```bash
# on the Pi
python3 deploy/obs_verify/pi_sensor_server.py
```

**Step 2 — run the verifier** on the laptop:
```bash
# vision-only (tests 1–5, no Pi needed)
python deploy/obs_verify/obs_verifier.py --device 1

# full suite (tests 1–10, Pi server must be running)
python deploy/obs_verify/obs_verifier.py --device 1 --pi-ip raspberrypi.local
```

> Stop `vision_streamer.py` first — both tools open the camera.

## Controls (during each test)

| Key | Action |
|-----|--------|
| `ENTER` | Mark PASS, advance to next test |
| `f` | Mark FAIL, advance |
| `s` | Skip (no judgment) |
| `r` | Reset frame buffer (fresh assembler) |
| `q` | Quit |

## Test sequence

| # | What you do | What should happen |
|---|-------------|-------------------|
| 1 | Place all 3 tags in view | fps > 20, tags {0,1,2} visible |
| 2 | Tape-measure fish→target distance | obs[6] ≈ measured ± 10 cm |
| 3 | Point fish nose **directly at target** | obs[7] (vec_x) **< 0**, obs[8] ≈ 0 |
| 4 | Target 90° to fish's **LEFT** | obs[8] (vec_y) **< 0** |
| 5 | Target 90° to fish's **RIGHT** | obs[8] (vec_y) **> 0** |
| 6 | Spin fish **CCW** from above (nose-left) | obs[5] (wz) **> 0** |
| 7 | Spin fish **CW** from above (nose-right) | obs[5] (wz) **< 0** |
| 8 | Roll fish onto its **right side** | \|obs[3]\| (wx) > 0.3 |
| 9 | Tilt fish **nose up** | \|obs[4]\| (wy) > 0.3 |
| 10 | Motor at centre (tail straight) | obs[0] > 0.9, \|obs[1]\| < 0.15 |

## Why obs[7] is negative when pointing at the target

The MuJoCo fish model swims toward its body **−X** axis (nose = −X).
Training computes the target vector with `R(−headRollZ)`, so a target
directly ahead → `vec_x = −dist` (negative).

Deploy `fish_yaw` tracks the nose direction and differs from `headRollZ`
by **π**, which negates both `vec_x` and `vec_y`.  The fix is in
`obs_assembler.py` — all three assembler classes negate both components.

## Bearing display (live stats line)

The bearing shown on screen uses the **human-intuitive** convention
(0° = nose toward target, +90° = target to your left) and is **not**
sent to the policy.  The policy receives the correctly signed `vec_x/vec_y`.

## ROI tag tracking and the `--no-roi` flag

By default the verifier (and `vision_streamer.py`) use **ROI-crop detection**: instead
of scanning the full camera frame for tags every frame, each tag is detected in a small
padded crop around its last known position.  The anchor tag is held for 29 out of every
30 frames and only re-verified periodically.  A full-frame fallback fires automatically
whenever a fish or target tag has been missed 3 frames in a row.

`quad_decimate` stays at **4.0** in both modes.

### Expected improvement

In steady-state tracking (all tags visible and stable), each ROI crop is roughly
1.5× the tag bounding box — typically **5–15 % of the full frame area**.  Running
`detector.detect()` on a fraction of the pixels at the same decimation gives a
**3–5× FPS increase** in practice.  The verifier's FPS readout is an accurate proxy
for what `vision_streamer.py` will achieve on the same machine and camera resolution,
because both tools use identical detection compute (minus UDP sends and video recording,
which are async/non-blocking).

### Potential difficulties

| Situation | What happens |
|-----------|--------------|
| Tag partially occluded | Miss count increments; full-frame fallback fires at miss count ≥ 3 |
| Camera bumped / fish moves fast | ROI may lag one to two frames before re-centring |
| Tag very small (fish far away) | ROI crop + decimate=4 can push effective resolution below reliable threshold — try decimate=2 if misses increase |
| Cold start (no tags seen yet) | Full-frame scan fires on every frame until all tags are initialised — same cost as `--no-roi` |

The live stats line shows `[ROI]` or `[FULL]` next to FPS so you can see exactly which
path is active and how often fallbacks occur.

### Comparing ROI vs full-frame

```bash
# New default — ROI tracking:
python deploy/obs_verify/obs_verifier.py --device 1

# Old behaviour — full-frame scan every frame (pre-ROI baseline):
python deploy/obs_verify/obs_verifier.py --device 1 --no-roi
```

Run both back-to-back and compare the FPS number in the live stats line.

## `deploy/shared/tag_tracker.py` — why it exists

`tag_tracker.py` is a **new shared module** (it did not exist before this feature).
It was created to avoid duplicating the ROI tracking logic between two tools that need
to behave identically:

- `vision_streamer.py` — the live deployment tool that streams poses to the Pi
- `obs_verifier.py` — the offline verification tool that measures achievable FPS

Both tools import `TagTracker`, `_roi_detect`, and `_fullframe_detect` from the shared
module.  The key reason to centralise this code is the FPS measurement guarantee: if the
verifier used a different detection path from the streamer, its FPS reading would be
meaningless.  With a single shared implementation, any change to detection logic
automatically applies to both tools at once.

`TagTracker` holds the per-tag state between frames (last-known centre pixel, bounding
box size, miss count, and the last valid rotation/translation from the pose solver).
The two helper functions implement the two detection paths and update all trackers
in-place, so callers never touch the tracker internals directly.

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `obs_verifier.py` | Laptop | Main test tool |
| `pi_sensor_server.py` | Pi | Streams IMU + motor at 50 Hz over UDP port 5007 |
| `../shared/tag_tracker.py` | Laptop | ROI tracking state + detection helpers (shared with `vision_streamer.py`) |

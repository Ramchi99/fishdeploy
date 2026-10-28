# Frame-drop analysis — deploy recordings (2026-05-20 session)

## TL;DR

- The async MJPEG recorder's `cv2.VideoWriter` sustains only **~20 fps at 1080p**
  on this 8-core machine, but the camera captures at **~45–50 fps**. The encoder
  is permanently ~2.5× too slow, so on long runs its queue fills and frames are
  dropped — roughly **every other frame** once dropping starts.
- **8 of 49 runs** dropped frames (plus 1 fully corrupt). All drop runs are
  **long / late in the session**; every run shorter than ~33 s is clean.
- **Timing is now fixed** (recorder writes a timestamp only per written frame →
  exact VFR; `render_trace.py` recovers old runs). **Frame *loss* is not yet
  fixed** — see "How to never drop a frame".

## Which runs are clean vs dropped

**Dropped (8) — detail:**

| run | drop | drop% | onset | phase-1 fps | phase-2 fps | phase-2 cadence |
|---|---|---|---|---|---|---|
| 162457_hope_didnt_work_5 | 264 | 13.6% | 20.4 s | 52.8 | 23.0 | keep 1 in 1.4 |
| 165858_full_success_3 | 614 | 24.5% | 23.3 s | 51.8 | 20.1 | keep 1 in 1.9 |
| 170419_insane_run_2 | 141 | 7.3% | 28.8 s | 50.2 | 24.2 | keep 1 in 1.4 |
| 192737_figure_eight_5 | 399 | 18.3% | 36.9 s | 45.3 | 7.4 | keep 1 in 4.4 |
| 193023_figure_eight_6 | 729 | 20.6% | 38.0 s | 50.3 | 24.3 | keep 1 in 1.8 |
| 194717_finetune_run_4 | 2251 | 49.6% | 11.5 s | 53.4 | 19.9 | keep 1 in 2.3 |
| 195142_finetune_run_6 | 115 | 6.1% | 22.4 s | 52.8 | 31.7 | keep 1 in 1.2 |
| 200038_finetune_run_9 | 2579 | 47.7% | 14.5 s | 52.3 | 21.8 | keep 1 in 2.2 |

**Corrupt (1):** `175136_amazing_run_4` — 0 AVI frames, 2233 timestamps. The
VideoWriter produced an empty file (total encode failure / write never flushed).

**Clean (40, zero drops):** one_last_time_3/4, hope_didnt_work_1/2/3/4/6,
polaris, polaris_2, crazy_run_1–5, full_success_1/2, insane_run_1/3–8,
amazing_run_1/2/3/5/6/7/8, figure_eight_2/3/4, finetune_run_1/2/3/4(194207)/5/7/8/10.

## Why frames drop

`AsyncVideoRecorder` (vision_streamer.py) pushes every captured frame into a
1024-deep queue; a background thread pulls frames and calls
`cv2.VideoWriter.write()` (MJPEG/AVI). Measured throughput on this machine:

| encoder path | 1080p | 720p |
|---|---|---|
| `cv2.VideoWriter` MJPG (what we use) | **20 fps** | 35 fps |
| `cv2.imencode('.jpg')` 1 thread | 71 fps | — |
| `cv2.imencode('.jpg')` 4 threads | **206 fps** | — |

Capture is ~45–50 fps. Since the writer drains at ~20 fps, the queue backlog
grows at ~25–30 fps; once it hits 1024 the producer starts dropping. From then
on the writer can still only emit ~20 fps, so it keeps ~20 of every ~45 captured
→ **drops roughly every other frame**.

This also explains the per-run pattern:
- **Onset varies (11–38 s)** — when the queue fills depends on the exact capture
  rate and concurrent system load that run, so it is **not a fixed cadence**.
- **Phase-2 written fps clusters at ~20–24 fps** — that is the encoder ceiling.
  (`figure_eight_5` at 7.4 fps is the outlier; that recording was under heavier
  load, so the writer ran even slower.)
- **Keep-ratio ~0.5–0.7** for most (≈ every other frame); worst case 0.23.

So the cadence is *semi*-consistent: once saturated you keep ~the encoder's
20 fps and discard the rest — but the exact ratio and onset drift with load.

## How to NEVER drop a frame — implementation plan (NOT yet implemented)

### Why the obvious knobs don't work
- **Bigger queue** — no. The writer (20 fps) is permanently slower than capture
  (45–50 fps), so the backlog grows without bound and any finite queue fills. A
  bigger buffer only delays the onset; it cannot prevent drops on a long run.
- **Lower JPEG quality / 720p** — helps a bit (720p VideoWriter = 35 fps) but
  still below capture; not a guarantee on its own.

### The fix: raise encode throughput above capture rate
The bottleneck is `cv2.VideoWriter`'s internal single-threaded mjpeg encoder,
**not** JPEG encoding. Per-frame `cv2.imencode('.jpg')` is 71 fps single-thread
and 206 fps on 4 threads — comfortably above 50 fps capture at full 1080p.

### Concrete changes (in `vision_streamer.py` `AsyncVideoRecorder`)
1. **Queue carries `(index, frame, capture_time)`**; `index` assigned
   monotonically at push time so output order is preserved regardless of which
   worker encodes it.
2. **Worker pool of 2–3 threads**; each loops: `cv2.imencode('.jpg', frame, [JPEG_QUALITY,92])`
   → write bytes to `frames/{index:06d}.jpg`, and append `(index, capture_time)`
   to a shared timestamp list under a lock (or per-worker lists merged at stop).
3. **Back-pressure instead of silent drop** — use a blocking `queue.put()` with a
   modest `maxsize` (e.g. 256) rather than `put_nowait`. If the pool ever can't
   keep up, the capture loop briefly waits (and `cv2.VideoCapture` buffers in the
   driver) rather than discarding a frame. With imencode this should never
   actually block, but it converts "silent drop" into "never drop".
4. **At stop**: join workers, sort timestamps by index, write
   `frame_timestamps.csv`, then `ffmpeg` concat-assemble `frames/*.jpg` →
   `video.avi`/`.mp4` (same VFR backbone the render scripts use), and optionally
   delete the frames dir.
5. **Keep** the already-applied rule: one timestamp per *written* frame.
6. **Live monitoring**: print encode fps + queue depth in the 1 Hz status line so
   any future backlog is visible immediately.

### Sizing / headroom
- 8-core box: a 2–3 thread imencode pool gives ~130–180 fps — ~3× margin over
  capture. Reserve cores for AprilTag detection (already `nthreads=4`); 2 encode
  threads is the safe default.
- Weaker machines: record 720p, drop JPEG quality to ~85, or use a GPU encoder
  (NVENC H.264) which does 1080p at hundreds of fps.

### Verification when built
Re-run a 2+ minute recording and confirm `n_avi == n_ts` (zero drops) and that
the status line never reports a growing queue. The corrupt-file case
(`amazing_run_4`) also disappears, since assembly happens from on-disk JPEGs
rather than a single VideoWriter that can fail silently.

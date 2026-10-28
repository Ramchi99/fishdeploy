#!/usr/bin/env python3
"""
tune_vision.py — Find optimal alpha/beta/blur/quad_decimate for AprilTag detection.

Usage (Windows, from repo root):
    python deploy\\laptop\\tune_vision.py --device 1

Phase 1 — capture: live preview while you wave the fish tag (tag 0) around the
          scene so all lighting/position conditions are represented.
          Press Q to stop early (minimum ~10 frames needed).
Phase 2 — sweep: replay all captured frames through every
          (quad_decimate, blur, alpha, beta) combination and count how often
          each tag is detected.
Phase 3 — result: matplotlib heatmap of product detection score, best params
          printed and saved to deploy/configs/image_tune.json.

Sweep time estimate (Windows, 1920×1080, ~180 captured frames):
  default decimates=[2,4,8]:  ~6 min
  with decimate=1 added:      ~20 min  (quad search at full res is ~15× slower)
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from dt_apriltags import Detector
except ImportError:
    from pupil_apriltags import Detector

_REPO_ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_TUNE_FILE     = os.path.join(_REPO_ROOT, "deploy", "configs", "image_tune.json")
_SETTINGS_FILE = os.path.join(_REPO_ROOT, "deploy", "configs", "camera_settings.json")

# Search grid
ALPHAS   = [round(a, 1) for a in np.arange(0.3, 2.1, 0.2)]  # 0.3 … 1.9  (9 values)
BETAS    = list(range(-80, 60, 20))                            # -80 … 40   (7 values)
BLURS    = [0, 3]                                              # off / light smooth
DECIMATES = [2, 4, 8]   # quad_decimate values — higher = faster but needs larger tags

CAPTURE_S    = 30   # capture window [s]; press Q to stop early
SAMPLE_EVERY = 5    # keep 1 frame in N to limit sweep time


def _load_cam_params(settings_file, camera_name=None):
    try:
        with open(settings_file) as f:
            d = json.load(f)
        cams = d.get("cameras", {})
        name = camera_name or d.get("default_camera", next(iter(cams)))
        cam = cams[name]
        return list(cam["camera_params"])  # [fx, fy, cx, cy]
    except Exception:
        return [1055.0, 1054.2, 960.0, 540.0]


def _preprocess(gray, alpha, beta, blur):
    img = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    if blur >= 3:
        img = cv2.GaussianBlur(img, (blur, blur), 0)
    return img


def main():
    parser = argparse.ArgumentParser(description="Tune vision streamer alpha/beta/blur/decimate.")
    parser.add_argument("--device",    type=int,   default=1,        help="Camera device index")
    parser.add_argument("--camera",    default=None,                  help="Camera profile from camera_settings.json")
    parser.add_argument("--capture_s", type=float, default=CAPTURE_S, help="Capture duration [s]")
    parser.add_argument("--tag_size",  type=float, default=0.080,    help="AprilTag size [m] (unused in scoring)")
    parser.add_argument("--robustness_sigma", type=float, default=1.5,
                        help="Gaussian smoothing radius [grid steps] for robust selection (default 1.5)")
    parser.add_argument("--decimate_preference", type=float, default=0.05,
                        help="Score bonus per doubling of quad_decimate (default 0.05). "
                             "Higher values more aggressively prefer faster detection. "
                             "0.05 means decimate=8 wins over decimate=2 only if its "
                             "product score is within 15%% of decimate=2.")
    parser.add_argument("--decimates", type=int, nargs="+", default=None,
                        help="quad_decimate values to sweep (default: 2 4 8). "
                             "Add 1 for highest accuracy but expect ~20 min sweep.")
    args = parser.parse_args()

    decimates = sorted(set(args.decimates)) if args.decimates else DECIMATES

    cam_params = _load_cam_params(_SETTINGS_FILE, args.camera)

    # ── Camera open ────────────────────────────────────────────────────────────
    if sys.platform != "linux":
        cap = cv2.VideoCapture(args.device, cv2.CAP_MSMF)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
    else:
        cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
        cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera: {actual_w}x{actual_h}")
    print(f"Sweep grid: {len(decimates)} decimates × {len(BLURS)} blur × "
          f"{len(ALPHAS)} alpha × {len(BETAS)} beta = "
          f"{len(decimates)*len(BLURS)*len(ALPHAS)*len(BETAS)} combos")
    if 1 in decimates:
        print("  NOTE: decimate=1 included — quad search runs at full resolution, expect long sweep.")

    # ── One Detector per quad_decimate value ───────────────────────────────────
    detectors = {
        d: Detector(
            families="tag36h11",
            nthreads=6,
            quad_decimate=float(d),
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
        )
        for d in decimates
    }

    # ── Phase 1: capture ───────────────────────────────────────────────────────
    print(f"\n=== CAPTURE ({args.capture_s:.0f}s) ===")
    print("Wave the FISH TAG (tag 0) all around the scene.")
    print("Keep anchor (tag 1) and target (tag 2) in view when possible.")
    print("Press Q to stop early.\n")

    frames = []
    frame_idx = 0
    t_end = time.time() + args.capture_s

    while time.time() < t_end:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_idx += 1

        remaining = max(0, t_end - time.time())
        disp = frame.copy()
        cv2.putText(disp, f"Capturing: {remaining:.0f}s left  ({len(frames)} frames kept)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(disp, "Wave the FISH TAG around!  Q = stop early",
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        cv2.imshow("tune_vision — capture", disp)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if frame_idx % SAMPLE_EVERY == 0:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    cap.release()
    cv2.destroyAllWindows()

    if len(frames) < 5:
        print(f"Only {len(frames)} frames — too few. Re-run and let it capture longer.")
        return
    print(f"Captured {len(frames)} frames for sweep.")

    # ── Phase 2: sweep ─────────────────────────────────────────────────────────
    n_dec  = len(decimates)
    n_bl   = len(BLURS)
    n_al   = len(ALPHAS)
    n_bt   = len(BETAS)
    n_fr   = len(frames)
    total  = n_dec * n_bl * n_al * n_bt * n_fr

    print(f"\n=== SWEEP ===")
    print(f"{n_dec * n_bl * n_al * n_bt} combos × {n_fr} frames = {total} detections")

    # results[decimation, blur, alpha, beta, tag_id]
    results = np.zeros((n_dec, n_bl, n_al, n_bt, 3), dtype=np.int32)

    done = 0
    t_sweep = time.time()
    for di, dec in enumerate(decimates):
        det = detectors[dec]
        for bi, blur in enumerate(BLURS):
            for ai, alpha in enumerate(ALPHAS):
                for bti, beta in enumerate(BETAS):
                    for gray in frames:
                        img  = _preprocess(gray, alpha, beta, blur)
                        tags = det.detect(img, estimate_tag_pose=False,
                                          camera_params=None, tag_size=None)
                        for tag in tags:
                            if 0 <= tag.tag_id <= 2:
                                results[di, bi, ai, bti, tag.tag_id] += 1
                    done += n_fr
                pct = 100 * done / total
                eta = (time.time() - t_sweep) / done * (total - done) if done else 0
                print(f"  {pct:5.1f}%  ETA {eta:4.0f}s  "
                      f"(dec={dec} blur={blur} alpha={alpha})", end="\r")

    print(f"\nSweep done in {time.time() - t_sweep:.1f}s")

    # ── Phase 3: score & select ────────────────────────────────────────────────
    rate  = results / n_fr  # detection rate, shape: (dec, blur, alpha, beta, 3)

    # Product score: zero if any tag is never seen; maximised when all three are
    # reliably detected simultaneously.  An additive score would sacrifice the
    # anchor or target tag to gain fish detection.
    score = rate[..., 0] * rate[..., 1] * rate[..., 2]

    # Gaussian smooth over (alpha, beta) axes for each (decimate, blur) pair.
    # mode='constant', cval=0.0: off-grid territory is treated as 0 detection,
    # so edge combos are penalised relative to well-surrounded interior combos.
    sigma = args.robustness_sigma
    robust_score = np.zeros_like(score)
    for di in range(n_dec):
        for bi in range(n_bl):
            robust_score[di, bi] = gaussian_filter(
                score[di, bi], sigma=sigma, mode='constant', cval=0.0
            )

    # Decimation preference: multiply each decimate level's robust score by a
    # small bonus so that equal-quality combos prefer the higher (faster) decimation.
    # pref=0.05/octave → decimate=8 gets 1.15× vs decimate=2's 1.05×, so it wins
    # only if its product score is within ~10% of decimate=2's score.
    pref = args.decimate_preference
    weighted = robust_score.copy()
    for di, d in enumerate(decimates):
        weighted[di] *= (1.0 + pref * math.log2(d))

    flat_idx = np.argmax(weighted)
    di, bi, ai, bti = np.unravel_index(flat_idx, weighted.shape)
    best_alpha    = ALPHAS[ai]
    best_beta     = int(BETAS[bti])
    best_blur     = BLURS[bi]
    best_decimate = decimates[di]

    # Raw peak (before preference weighting) for comparison
    raw_flat = np.argmax(score)
    rdi, rbi, rai, rbti = np.unravel_index(raw_flat, score.shape)

    print(f"\nRobust best (sigma={sigma:.1f} steps, pref={pref:.2f}/octave):")
    print(f"  alpha={best_alpha}  beta={best_beta}  blur={best_blur}  "
          f"quad_decimate={best_decimate}")
    t0, t1, t2 = rate[di, bi, ai, bti, 0], rate[di, bi, ai, bti, 1], rate[di, bi, ai, bti, 2]
    print(f"  tag0={t0:.1%}  tag1={t1:.1%}  tag2={t2:.1%}  "
          f"product={score[di,bi,ai,bti]:.3f}")

    if (rdi, rbi, rai, rbti) != (di, bi, ai, bti):
        rs = score[rdi, rbi, rai, rbti]
        print(f"  (raw peak: dec={decimates[rdi]}  alpha={ALPHAS[rai]}  "
              f"beta={int(BETAS[rbti])}  blur={BLURS[rbi]}  product={rs:.3f})")

    # Per-decimation summary: best product score achievable at each decimation level
    print(f"\n  Decimation summary (best product score across all alpha/beta/blur):")
    for di2, d2 in enumerate(decimates):
        best_prod = score[di2].max()
        best_flat2 = np.argmax(score[di2])
        bi2, ai2, bti2 = np.unravel_index(best_flat2, score[di2].shape)
        marker = "  ← chosen" if d2 == best_decimate else ""
        print(f"    decimate={d2:2d}:  best product={best_prod:.3f}  "
              f"(a={ALPHAS[ai2]}  b={int(BETAS[bti2])}  blur={BLURS[bi2]}){marker}")

    # ── Edge warning ───────────────────────────────────────────────────────────
    on_alpha_edge = (ai == 0 or ai == len(ALPHAS) - 1)
    on_beta_edge  = (bti == 0 or bti == len(BETAS) - 1)
    if on_alpha_edge or on_beta_edge:
        edge_axes = []
        if on_alpha_edge:
            edge_axes.append(f"alpha={best_alpha} ({'min' if ai == 0 else 'max'} of grid)")
        if on_beta_edge:
            edge_axes.append(f"beta={best_beta} ({'min' if bti == 0 else 'max'} of grid)")
        print(f"\n  *** WARNING: best combo is on the grid EDGE: {', '.join(edge_axes)}")
        print(f"  *** There may be better (or worse) values beyond the grid.")
        print(f"  *** Expand ALPHAS / BETAS in the script and re-tune.")

    # ── Save ───────────────────────────────────────────────────────────────────
    with open(_TUNE_FILE, "w") as f:
        json.dump({
            "alpha":         best_alpha,
            "beta":          float(best_beta),
            "blur":          best_blur,
            "quad_decimate": float(best_decimate),
        }, f, indent=2)
    print(f"\nSaved → {_TUNE_FILE}")

    # ── Heatmap ────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        n_rows = n_dec
        n_cols = n_bl
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(6 * n_cols, 4.5 * n_rows),
            squeeze=False,
        )

        B_arr = np.array(BETAS)
        A_arr = np.array(ALPHAS)
        BB, AA = np.meshgrid(B_arr, A_arr)

        for di2, dec2 in enumerate(decimates):
            for bi2, blur2 in enumerate(BLURS):
                ax   = axes[di2][bi2]
                data = score[di2, bi2]           # product score for this (dec, blur)
                im   = ax.imshow(
                    data, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="viridis",
                    extent=[BETAS[0] - 10, BETAS[-1] + 10,
                            ALPHAS[0] - 0.1, ALPHAS[-1] + 0.1],
                )
                ax.set_xlabel("beta")
                ax.set_ylabel("alpha")
                ax.set_title(f"decimate={dec2}  blur={blur2}  |  product score")
                ax.set_xticks(BETAS)
                ax.set_yticks(ALPHAS)
                plt.colorbar(im, ax=ax, label="product score")

                # Robustness contours
                rs = robust_score[di2, bi2]
                rs_max = rs.max()
                if rs_max > 0:
                    levels = [rs_max * t for t in (0.70, 0.85, 0.95)]
                    ax.contour(BB, AA, rs, levels=levels,
                               colors="white", alpha=0.5, linewidths=0.8)

                # Mark chosen winner
                if di2 == di and bi2 == bi:
                    edge_flag = "  ⚠ EDGE" if (on_alpha_edge or on_beta_edge) else ""
                    ax.plot(best_beta, best_alpha, "r*", markersize=16,
                            label=f"chosen ({best_alpha},{best_beta}){edge_flag}")
                    ax.legend(fontsize=9)

                # Mark raw peak if different from chosen winner
                if (rdi, rbi, rai, rbti) != (di, bi, ai, bti) \
                        and di2 == rdi and bi2 == rbi:
                    ax.plot(int(BETAS[rbti]), ALPHAS[rai], "w+",
                            markersize=14, markeredgewidth=2,
                            label=f"raw peak ({ALPHAS[rai]},{int(BETAS[rbti])})")
                    ax.legend(fontsize=9)

        plt.suptitle(
            f"Chosen: dec={best_decimate}  α={best_alpha}  β={best_beta}  "
            f"blur={best_blur}  |  "
            f"tag0={t0:.0%}  tag1={t1:.0%}  tag2={t2:.0%}  "
            f"(σ={sigma:.1f} steps, pref={pref:.2f}/oct)",
            fontsize=11,
        )
        plt.tight_layout()
        out_png = os.path.join(_REPO_ROOT, "deploy", "configs", "tune_vision_result.png")
        plt.savefig(out_png, dpi=100)
        print(f"Heatmap → {out_png}")
        plt.show()
    except ImportError:
        print("matplotlib not available — skipping heatmap (result already saved).")


if __name__ == "__main__":
    main()

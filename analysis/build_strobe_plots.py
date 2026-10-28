#!/usr/bin/env python3
import os
import sys
import math
import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
import sys as _sys; _sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..")))   # repository root
from paths import DEPLOY, ANALYSIS  # noqa: E402
_LAPTOP = os.path.abspath(os.path.join(_HERE, "..", "deploy", "laptop"))
sys.path.insert(0, _LAPTOP)
from render_trace import viridis_bgr, OUT_W
from matplotlib import colormaps as _mpl_colormaps
_PLASMA = _mpl_colormaps["plasma"]
def plasma_bgr(frac):
    # yellow -> red -> violet -> blue (Liam's palette); returns BGR for cv2
    r, g, b, _ = _PLASMA(float(np.clip(frac, 0.0, 1.0)))
    return (int(b * 255), int(g * 255), int(r * 255))

RUNS_DIR = str(DEPLOY)
CACHE_DIR = str(ANALYSIS / "cache")
OUT_DIR = str(ANALYSIS / "strobe_plots")
os.makedirs(OUT_DIR, exist_ok=True)

TRACE_ALPHA = 0.9   # trace opacity (higher = pops more; full opacity ~1.0)
OUT_SUFFIX = ""     # appended to output filename (set non-empty to avoid overwriting)
TRACE_OUTLINE = True  # dark outline under the coloured trace (set False for a borderless line)
FISH_ON_TOP = False # if True: draw the trace first, segment the fish, composite only the fish on top
TRACE_FADE = False   # if True: v2 composite, but dim the on-top trace where it crosses a fish crop


def draw_trace_faded(img, trail, alpha, scale, fade_mask, fade=0.55):
    """Draw the trace on top (exactly like the v2 trace), but reduce its opacity where fade_mask~1
    (the fish-crop regions), so the fish shows through where the line would otherwise overwrite it."""
    if len(trail) < 2:
        return img
    overlay = img.copy()
    pts = [(int(x), int(y)) for (x, y, _f) in trail]
    thick1 = max(1, int(4 * scale))
    thick2 = max(1, int(2 * scale))
    for i in range(1, len(pts)):
        cv2.line(overlay, pts[i - 1], pts[i], (20, 20, 20), thick1, cv2.LINE_AA)
    for i in range(1, len(pts)):
        cv2.line(overlay, pts[i - 1], pts[i], plasma_bgr(1.0 - trail[i][2]), thick2, cv2.LINE_AA)
    a = np.full(img.shape[:2], float(alpha), np.float32) * (1.0 - fade * np.clip(fade_mask, 0, 1))
    a = a[..., None]
    return (overlay.astype(np.float32) * a + img.astype(np.float32) * (1.0 - a)).astype(np.uint8)


def segment_fish(frame, cx, cy, r, scale):
    """Full fish mask (body + tag + thin tail): pixels differing from the local water median,
    strongly closed so the thin tail connects to the body, then the component containing the fish
    centre is kept. A wide patch ensures a long tail is not clipped."""
    H, W = frame.shape[:2]
    rs = int(r * 1.5)                                     # wide patch so the tail fits
    x1, x2 = max(0, cx - rs), min(W, cx + rs)
    y1, y2 = max(0, cy - rs), min(H, cy + rs)
    patch = frame[y1:y2, x1:x2].astype(np.uint8)
    if patch.size == 0:
        return np.zeros((H, W), np.float32)
    gray = cv2.medianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), 5).astype(np.float32)
    water = float(np.median(gray))
    m = (np.abs(gray - water) > 22).astype(np.uint8)     # catches faint tail without much water
    kc = max(3, int(13 * scale)) | 1                     # close bridges the tail to the body
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((kc, kc), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    pcx, pcy = int((x2 - x1) / 2), int((y2 - y1) / 2)
    n, lbl, stats, cent = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:                                            # largest blob whose centroid is near the fish centre
        best, bestarea = 0, -1
        for i in range(1, n):
            d = np.hypot(cent[i][0] - pcx, cent[i][1] - pcy)
            if d < 1.1 * r and stats[i, cv2.CC_STAT_AREA] > bestarea:
                bestarea, best = stats[i, cv2.CC_STAT_AREA], i
        m = (lbl == best).astype(np.uint8) if best > 0 else m
    # bound the mask to a circle around the fish so it can't spill into the far wake
    bound = np.zeros_like(m)
    cv2.circle(bound, (pcx, pcy), int(r * 1.05), 1, -1)
    m = m & bound
    m = cv2.dilate(m, np.ones((max(3, int(5 * scale)) | 1,) * 2, np.uint8))
    full = np.zeros((H, W), np.float32)
    full[y1:y2, x1:x2] = m.astype(np.float32)
    fb = max(3, int(7 * scale)) | 1
    return cv2.GaussianBlur(full, (fb, fb), 0)

CONFIG = {
    "hope_didnt_work_2": {"base_run": "hope_didnt_work_2", "min_dist_px": 220, "stop_at_min": True},
    "insane_run_8": {"base_run": "insane_run_8", "min_dist_px": 300, "stop_at_min": True},
    "insane_run_8_middle_bg": {"base_run": "insane_run_8", "min_dist_px": 300, "stop_at_min": True, "bg_frame_idx": 502},
    "finetune_run_6": {"base_run": "finetune_run_6", "min_dist_px": 250, "stop_at_min": True},
    "figure_eight_6": {"base_run": "figure_eight_6", "min_dist_px": 280, "stop_at_min": False},
    "figure_eight_6_with_9": {"base_run": "figure_eight_6", "min_dist_px": 280, "stop_at_min": False, "keep_9th": True},
    "figure_eight_6_clean": {"base_run": "figure_eight_6", "min_dist_px": 280, "stop_at_min": False, "keep_9th": True, "clean_stick": True},
    
    # HIGHRES VARIANTS
    "hope_didnt_work_2_highres": {"base_run": "hope_didnt_work_2", "min_dist_px": 220, "stop_at_min": True, "highres": True},
    "insane_run_8_highres": {"base_run": "insane_run_8", "min_dist_px": 300, "stop_at_min": True, "highres": True},
    "insane_run_8_middle_bg_highres": {"base_run": "insane_run_8", "min_dist_px": 300, "stop_at_min": True, "bg_frame_idx": 502, "highres": True},
    "finetune_run_6_highres": {"base_run": "finetune_run_6", "min_dist_px": 250, "stop_at_min": True, "highres": True},
    "figure_eight_6_clean_highres": {"base_run": "figure_eight_6", "min_dist_px": 280, "stop_at_min": False, "keep_9th": True, "clean_stick": True, "highres": True},

    # Report §5.2 gallery additions (2026-07-09): (c) backward-velocity success, (e) OOD failure.
    "insane_run_2_highres": {"base_run": "insane_run_2", "min_dist_px": 240, "stop_at_min": True, "highres": True},
    "finetune_run_8_highres": {"base_run": "finetune_run_8", "min_dist_px": 250, "stop_at_min": False, "highres": True},
    "finetune_run_7_highres": {"base_run": "finetune_run_7", "min_dist_px": 250, "stop_at_min": True, "highres": True, "match_bg": True},
}

def compute_median_background(cap, total, out_w, out_h):
    print("Computing median background...")
    idx = np.linspace(0, total - 1, min(total, 60)).astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.resize(frame, (out_w, out_h)))
    if not frames:
        return None
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)

def draw_transparent_trace(img, trail, alpha=0.5, scale=1.0):
    overlay = img.copy()
    if len(trail) < 2:
        return img
    pts = [(int(x), int(y)) for (x, y, _f) in trail]
    thick1 = max(1, int(4 * scale))   # thinner dark outline
    thick2 = max(1, int(2 * scale))   # thinner coloured line
    if TRACE_OUTLINE:
        for i in range(1, len(pts)):
            cv2.line(overlay, pts[i - 1], pts[i], (20, 20, 20), thick1, cv2.LINE_AA)
    for i in range(1, len(pts)):
        # plasma reversed: yellow(start) -> red -> violet -> blue(end), Liam's palette
        cv2.line(overlay, pts[i - 1], pts[i], plasma_bgr(1.0 - trail[i][2]), thick2, cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def main():
    for run_name, cfg in CONFIG.items():
        print(f"\n=== Processing {run_name} ===")
        base_run_name = cfg.get("base_run", run_name)
        cache_path = os.path.join(CACHE_DIR, f"{base_run_name}.npz")
        if not os.path.exists(cache_path):
            print(f"Cache not found: {cache_path}")
            continue
        
        d = np.load(cache_path, allow_pickle=True)
        if bool(d["corrupt"]):
            print("Run marked corrupt.")
            continue
        
        folders = [f for f in os.listdir(RUNS_DIR) if base_run_name in f]
        if not folders:
            print(f"Raw run folder not found for {base_run_name}")
            continue
        run_folder = folders[0]
        vid_path = os.path.join(RUNS_DIR, run_folder, "raw", "video.avi")
        if not os.path.exists(vid_path):
            vid_path = os.path.join(RUNS_DIR, run_folder, "raw", "video.mp4")
            if not os.path.exists(vid_path):
                print(f"Video not found in {run_folder}")
                continue
        
        fish_px = d["fish_px"]
        tgt_px = d["tgt_px"]
        frame_time = d["frame_time"]
        total = int(d["total"])
        
        cap = cv2.VideoCapture(vid_path)
        Wv = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        Hv = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if Wv == 0:
            print("Invalid video dimensions")
            continue
        
        if cfg.get("highres"):
            current_out_w = Wv
            out_h = Hv
            scale = Wv / OUT_W
        else:
            current_out_w = OUT_W
            out_h = int(round(OUT_W * Hv / Wv)); out_h -= out_h % 2
            scale = 1.0
            
        r_fish = int(120 * scale)
        blur_k = int(31 * scale) | 1
        
        min_idx = total - 1
        if cfg["stop_at_min"]:
            fw = d["fish_world"]
            tw = d["tgt_world"]
            valid = ~np.isnan(fw[:, 0]) & ~np.isnan(tw[:, 0])
            if valid.sum() > 0:
                tgt_median = np.median(tw[valid], axis=0)
                dist = np.hypot(fw[:, 0] - tgt_median[0], fw[:, 1] - tgt_median[1])
                valid_indices = np.where(valid)[0]
                min_valid_idx = np.argmin(dist[valid])
                min_idx = valid_indices[min_valid_idx]
                print(f"Truncating trace at index {min_idx}/{total} (min dist = {dist[valid][min_valid_idx]:.3f}m)")

        if "bg_frame_idx" in cfg:
            target_f = total // 2 if cfg["bg_frame_idx"] == -1 else cfg["bg_frame_idx"]
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_f)
            ret, frame = cap.read()
            if ret:
                bg = cv2.resize(frame, (current_out_w, out_h)).astype(np.float32)
                print(f"Using single frame {target_f} as background.")
            else:
                bg = compute_median_background(cap, total, current_out_w, out_h)
        else:
            bg = compute_median_background(cap, total, current_out_w, out_h)
            
        if bg is None:
            continue
            
        comp = bg.copy()
        
        last_cx, last_cy = -999, -999
        
        trail = []
        t0 = frame_time[0]
        t_span = max(frame_time[min_idx] - t0, 1e-6)
        
        rendered_count = 0
        render_actions = []
        
        for fi in range(total):
            t = frame_time[fi]
            
            if fi <= min_idx:
                fpx = fish_px[fi]
                if not np.isnan(fpx[0]):
                    frac = (t - t0) / t_span
                    trail.append((fpx[0] * scale, fpx[1] * scale, frac))
            
            if fi <= min_idx:
                fpx = fish_px[fi]
                if not np.isnan(fpx[0]):
                    cx, cy = int(fpx[0] * scale), int(fpx[1] * scale)
                    dist = math.hypot(cx - last_cx, cy - last_cy)
                    
                    if last_cx == -999 or dist > (cfg["min_dist_px"] * scale):
                        rendered_count += 1
                        
                        # Fix for figure 8: handle 9th and 13th fish
                        if base_run_name == "figure_eight_6":
                            skip_list = [13] if cfg.get("keep_9th") else [9, 13]
                            if rendered_count in skip_list:
                                last_cx, last_cy = cx, cy
                                continue
                            
                        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                        ret, frame = cap.read()
                        if not ret:
                            continue
                        frame_resized = cv2.resize(frame, (current_out_w, out_h)).astype(np.float32)
                        
                        # Fix for finetune_run_6: sun came out, match brightness of crop to bg
                        if base_run_name == "finetune_run_6" and rendered_count in [2, 3]:
                            x1, x2 = max(0, cx - r_fish), min(current_out_w, cx + r_fish)
                            y1, y2 = max(0, cy - r_fish), min(out_h, cy + r_fish)
                            bg_patch = bg[y1:y2, x1:x2]
                            fg_patch = frame_resized[y1:y2, x1:x2]
                            diff = np.mean(fg_patch) - np.mean(bg_patch)
                            if diff > 0:
                                frame_resized = np.clip(frame_resized - diff, 0, 255)

                        # General per-snapshot colour/brightness match: shift each crop so the water
                        # ring around the fish matches the median background (fixes bright/dark haloes
                        # when lighting drifts over the run). Median is robust to the dark fish in-patch.
                        if cfg.get("match_bg"):
                            x1, x2 = max(0, cx - r_fish), min(current_out_w, cx + r_fish)
                            y1, y2 = max(0, cy - r_fish), min(out_h, cy + r_fish)
                            bgp = bg[y1:y2, x1:x2]
                            fgp = frame_resized[y1:y2, x1:x2]
                            for ch in range(3):
                                d = float(np.median(fgp[:, :, ch]) - np.median(bgp[:, :, ch]))
                                frame_resized[:, :, ch] = np.clip(frame_resized[:, :, ch] - d, 0, 255)
                                
                        if FISH_ON_TOP:
                            # tight fish mask so the underlying trace shows through the water
                            mask = segment_fish(frame_resized, cx, cy, r_fish, scale)
                            mask = mask[..., None]
                            render_actions.append((frame_resized, mask))
                            last_cx, last_cy = cx, cy
                            continue

                        mask = np.zeros((out_h, current_out_w), dtype=np.float32)
                        cv2.circle(mask, (cx, cy), r_fish, 1.0, -1)

                        # Fix for insane_run_8: stick overlaps first fish.
                        if base_run_name == "insane_run_8" and rendered_count == 4:
                            cv2.rectangle(mask, (int(cx - 180 * scale), int(cy - 20 * scale)), (int(cx + 180 * scale), out_h), 1.0, -1)
                            
                        # Remove stick for figure 8 clean run
                        if cfg.get("clean_stick") and rendered_count in [5, 9]:
                            tpx = tgt_px[fi]
                            if not np.isnan(tpx[0]):
                                tx, ty = int(tpx[0] * scale), int(tpx[1] * scale)
                                Y, X = np.ogrid[:out_h, :current_out_w]
                                dist_from_tgt = np.hypot(X - tx, Y - ty)
                                is_bright = (frame_resized[:,:,0] > 160) & (frame_resized[:,:,1] > 160) & (frame_resized[:,:,2] > 160)
                                stick_mask = is_bright & (dist_from_tgt > 45 * scale)
                                mask[stick_mask] = 0.0
                            
                        mask = cv2.GaussianBlur(mask, (blur_k, blur_k), 0)
                        mask = mask[..., None]
                        
                        render_actions.append((frame_resized, mask))
                        last_cx, last_cy = cx, cy
                        
        if base_run_name == "figure_eight_6" and len(render_actions) > 0:
            render_actions.pop()
            
        # Fish-on-top mode: draw the trace on the background FIRST, so the fish crops sit over it.
        if FISH_ON_TOP:
            comp = draw_transparent_trace(np.clip(comp, 0, 255).astype(np.uint8),
                                          trail, alpha=TRACE_ALPHA, scale=scale).astype(np.float32)

        # Reverse chronological composite so earlier fishes sit ON TOP of later huge crops (stick)
        for frame_resized, mask in reversed(render_actions):
            comp = frame_resized * mask + comp * (1.0 - mask)

        comp_uint = np.clip(comp, 0, 255).astype(np.uint8)
        if FISH_ON_TOP:
            pass                                          # trace already drawn under the fish
        elif TRACE_FADE:
            fade_mask = np.zeros((out_h, current_out_w), np.float32)
            for _, mk in render_actions:
                fade_mask = np.maximum(fade_mask, mk[..., 0])
            comp_uint = draw_trace_faded(comp_uint, trail, TRACE_ALPHA, scale, fade_mask)
        else:
            comp_uint = draw_transparent_trace(comp_uint, trail, alpha=TRACE_ALPHA, scale=scale)

        out_path = os.path.join(OUT_DIR, f"{run_name}{OUT_SUFFIX}_strobe.png")
        cv2.imwrite(out_path, comp_uint, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        print(f"Saved {out_path}")
        
    print("All done!")

if __name__ == "__main__":
    main()

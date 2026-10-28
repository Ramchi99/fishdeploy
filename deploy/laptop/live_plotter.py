"""
Live diagnostic plots for SOFI vision streamer.

Runs in a separate process so it never blocks the camera loop.
Shows 4 panels:
  - Fish→Target vector (anchor frame)
  - Relative velocity vector
  - Distance to target over time
  - Approach speed over time (smoothed)

Usage: imported by vision_streamer.py, not run directly.
"""
import collections
import math
import multiprocessing as mp

import numpy as np


# ── plotter process ───────────────────────────────────────────────────────────

def _plotter_process(queue: mp.Queue, window_s: float = 7.0, flip_y: bool = True) -> None:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    times    = collections.deque()
    rel_x_b  = collections.deque()
    rel_y_b  = collections.deque()
    vel_x_b  = collections.deque()
    vel_y_b  = collections.deque()
    dist_b   = collections.deque()
    spd_b    = collections.deque()

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("SOFI — Live Vision Diagnostics", fontsize=13)
    ax_vec, ax_vel, ax_dist, ax_spd = axes.flat

    # ── vector plot ───────────────────────────────────────────────────────────
    ax_vec.set_title("Fish → Target  (anchor frame)")
    ax_vec.set_aspect("equal")
    ax_vec.axhline(0, color="#555", lw=0.5)
    ax_vec.axvline(0, color="#555", lw=0.5)
    ax_vec.set_xlabel("X (m)")
    ax_vec.set_ylabel("Y (m)")
    ax_vec.set_xlim(-1, 1)
    ax_vec.set_ylim(-1, 1)
    ax_vec.plot(0, 0, "gs", ms=10, label="fish (origin)", zorder=5)
    q_vec = ax_vec.quiver(
        0, 0, 0, 0, angles="xy", scale_units="xy", scale=1,
        color="cyan", width=0.015, zorder=4,
    )
    dot_target = ax_vec.plot([], [], "ro", ms=8, label="target", zorder=5)[0]
    ax_vec.legend(fontsize=8, loc="upper right")

    # ── velocity vector plot ──────────────────────────────────────────────────
    ax_vel.set_title("Relative velocity  (fish frame)")
    ax_vel.set_aspect("equal")
    ax_vel.axhline(0, color="#555", lw=0.5)
    ax_vel.axvline(0, color="#555", lw=0.5)
    ax_vel.set_xlabel("Vx (m/s)")
    ax_vel.set_ylabel("Vy (m/s)")
    ax_vel.set_xlim(-1, 1)
    ax_vel.set_ylim(-1, 1)
    ax_vel.plot(0, 0, "k+", ms=10, zorder=5)
    q_vel = ax_vel.quiver(
        0, 0, 0, 0, angles="xy", scale_units="xy", scale=1,
        color="orange", width=0.015, zorder=4,
    )

    # ── distance plot ─────────────────────────────────────────────────────────
    ax_dist.set_title("Distance to target")
    ax_dist.set_xlabel("Time (s)")
    ax_dist.set_ylabel("m")
    (line_dist,) = ax_dist.plot([], [], "g-", lw=1.5)
    ax_dist.set_xlim(0, window_s)
    ax_dist.set_ylim(0, 3)

    # ── speed plot ────────────────────────────────────────────────────────────
    ax_spd.set_title("Approach speed  (smoothed)")
    ax_spd.set_xlabel("Time (s)")
    ax_spd.set_ylabel("m/s")
    (line_spd,) = ax_spd.plot([], [], color="#cc44ff", lw=1.5)
    ax_spd.set_xlim(0, window_s)
    ax_spd.set_ylim(0, 0.5)

    plt.tight_layout()

    session_start = [None]  # fixed to the first timestamp ever received

    def animate(_):
        # drain queue
        while True:
            try:
                item = queue.get_nowait()
            except Exception:
                break
            if item is None:
                plt.close("all")
                return
            t, rx, ry, vx, vy, dist, spd = item
            times.append(t)
            rel_x_b.append(rx)
            rel_y_b.append(ry)
            vel_x_b.append(vx)
            vel_y_b.append(vy)
            dist_b.append(dist)
            spd_b.append(spd)

        if not times:
            return

        if session_start[0] is None:
            session_start[0] = times[0]

        # trim all deques together to the last window_s seconds
        cutoff_abs = times[-1] - window_s
        while len(times) > 1 and times[0] < cutoff_abs:
            times.popleft();   rel_x_b.popleft(); rel_y_b.popleft()
            vel_x_b.popleft(); vel_y_b.popleft()
            dist_b.popleft();  spd_b.popleft()

        t_cur = times[-1] - session_start[0]
        t_rel = [t - session_start[0] for t in times]
        rx, ry = rel_x_b[-1], rel_y_b[-1]
        vx, vy = vel_x_b[-1], vel_y_b[-1]

        if flip_y:
            ry = -ry
            vy = -vy

        # vector — fixed ±1 m scale
        q_vec.set_UVC(rx, ry)
        dot_target.set_data([rx], [ry])

        # velocity — fixed ±1 m/s scale
        q_vel.set_UVC(vx, vy)

        # distance
        line_dist.set_data(t_rel, list(dist_b))
        ax_dist.set_xlim(max(0, t_cur - window_s), max(t_cur, 0.01))
        ax_dist.set_ylim(0, max(max(dist_b) * 1.2, 0.1))

        # speed — already EMA-smoothed via LivePlotter.push(), no convolution needed
        spd_arr = np.array(list(spd_b))
        line_spd.set_data(t_rel, spd_arr)
        ax_spd.set_xlim(max(0, t_cur - window_s), max(t_cur, 0.01))
        ax_spd.set_ylim(0, max(float(spd_arr.max()) * 1.2, 0.05))

        fig.canvas.draw_idle()

    _ani = animation.FuncAnimation(fig, animate, interval=100, cache_frame_data=False)
    plt.show()


# ── public class ──────────────────────────────────────────────────────────────

class LivePlotter:
    """Push tag detections from the camera loop; plots update in a background process."""

    def __init__(self, window_s: float = 7.0, smooth_alpha: float = 0.25, flip_y: bool = True):
        self._q: mp.Queue = mp.Queue(maxsize=600)
        self._proc = mp.Process(
            target=_plotter_process, args=(self._q, window_s, flip_y), daemon=True
        )
        self._last_rel = None        # (rx, ry) at previous frame
        self._last_t: float = None
        self._sv = [0.0, 0.0]        # EMA-smoothed relative velocity
        self._alpha = smooth_alpha

    def start(self) -> None:
        self._proc.start()

    def push(self, t: float, fish_x: float, fish_y: float,
             target_x: float, target_y: float) -> None:
        """Call every frame when fish, anchor, and target are all detected."""
        rx = target_x - fish_x
        ry = target_y - fish_y
        dist = math.sqrt(rx ** 2 + ry ** 2)

        vx = vy = 0.0
        if self._last_rel is not None and self._last_t is not None:
            dt = t - self._last_t
            if 0 < dt < 0.5:
                raw_vx = (rx - self._last_rel[0]) / dt
                raw_vy = (ry - self._last_rel[1]) / dt
                self._sv[0] = self._alpha * raw_vx + (1 - self._alpha) * self._sv[0]
                self._sv[1] = self._alpha * raw_vy + (1 - self._alpha) * self._sv[1]
                vx, vy = self._sv

        spd = math.sqrt(vx ** 2 + vy ** 2)
        self._last_rel = (rx, ry)
        self._last_t = t

        try:
            self._q.put_nowait((t, rx, ry, vx, vy, dist, spd))
        except Exception:
            pass  # drop frame if queue full

    def stop(self) -> None:
        try:
            self._q.put_nowait(None)
        except Exception:
            pass
        self._proc.join(timeout=2)

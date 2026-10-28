import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import numpy as np


@dataclass
class VisionState:
    fish_x: float
    fish_y: float
    fish_yaw: float
    target_x: float
    target_y: float
    flags: int
    pi_arrival_time: float
    frame_seq: int
    laptop_timestamp: float = 0.0


class ObsAssemblerCPG24D:
    """Assembles the 24-dimensional observation for env_sofi.py (BaseFishEnv).

    Obs layout (matches training env exactly):
      [0]     cos(motor_pos)
      [1]     sin(motor_pos)
      [2]     motor_vel [rad/s]
      [3]     imu_wz [rad/s]  (yaw rate only — not wx/wy)
      [4-5]   cos/sin(cpg_phase)
      [6-9]   frame_buffer[0]: dist, vecX, vecY, age_s  (oldest)
      [10-13] frame_buffer[1]: dist, vecX, vecY, age_s
      [14-17] frame_buffer[2]: dist, vecX, vecY, age_s  (most recent)
      [18-19] action_history[0]: omega, theta_center    (oldest)
      [20-21] action_history[1]: omega, theta_center
      [22-23] action_history[2]: omega, theta_center    (most recent)

    Frame buffer updates once per new vision frame (detected via frame_seq).
    Action history updates every build() call — caller passes the action that
    was just sent (prev_omega, prev_theta_center).
    """

    _FRAME_BUFFER_LEN   = 3
    _ACTION_HISTORY_LEN = 3
    _AGE_SENTINEL_S     = 0.200  # padding age at episode start, matches training

    def __init__(self, config: dict):
        self._stale_warn_ms = config["vision_stale_warn_ms"]
        self._stale_stop_ms = config["vision_stale_stop_ms"]
        self._com_offset    = np.array(config["tag0_to_com_offset"], dtype=np.float64)
        self._clock_offset_s: float = 0.0

        self._frame_buffer:   List[Tuple[float, float, float, float]] = []
        self._action_history: List[Tuple[float, float]] = []
        self._last_frame_seq: Optional[int] = None

    def set_clock_offset(self, offset_s: float) -> None:
        """Update Pi-vs-laptop clock offset (seconds, positive = laptop ahead)."""
        self._clock_offset_s = offset_s

    def _frame_age_ms(self, vision: VisionState) -> float:
        return (time.time() - vision.laptop_timestamp + self._clock_offset_s) * 1000.0

    def vision_status(self, vision: VisionState) -> str:
        """Returns 'nominal', 'stale', or 'lost'."""
        age_ms = self._frame_age_ms(vision)
        if age_ms >= self._stale_stop_ms:
            return "lost"
        if age_ms >= self._stale_warn_ms:
            return "stale"
        return "nominal"

    def build(
        self,
        motor: dict,
        imu: dict,
        vision: VisionState,
        cpg_phase: float,
        prev_omega: float,
        prev_theta_center: float = 0.0,
    ) -> np.ndarray:
        pos_rad   = motor["pos_rad"]
        vel_rad_s = motor["vel_rad_s"]

        # World → body-frame target vector (with CoM offset)
        yaw = vision.fish_yaw
        fish_x_com = (vision.fish_x
                      + np.cos(yaw) * self._com_offset[0]
                      - np.sin(yaw) * self._com_offset[1])
        fish_y_com = (vision.fish_y
                      + np.sin(yaw) * self._com_offset[0]
                      + np.cos(yaw) * self._com_offset[1])
        dx    = vision.target_x - fish_x_com
        dy    = vision.target_y - fish_y_com
        vec_x = float( np.cos(yaw) * dx + np.sin(yaw) * dy)
        vec_y = float(-np.sin(yaw) * dx + np.cos(yaw) * dy)
        dist  = float(np.sqrt(dx * dx + dy * dy))

        # Frame buffer: push on each new frame_seq, storing laptop capture time
        if vision.frame_seq != self._last_frame_seq:
            self._last_frame_seq = vision.frame_seq
            self._frame_buffer.append((dist, vec_x, vec_y, vision.laptop_timestamp))
            if len(self._frame_buffer) > self._FRAME_BUFFER_LEN:
                self._frame_buffer.pop(0)

        # Action history: push every step
        self._action_history.append((float(prev_omega), float(prev_theta_center)))
        if len(self._action_history) > self._ACTION_HISTORY_LEN:
            self._action_history.pop(0)

        # 12-dim frame buffer obs — oldest-first, sentinels pad the front
        now = time.time()
        buf = list(self._frame_buffer)
        # Sentinel age: insert as a past laptop_timestamp so age computes to _AGE_SENTINEL_S
        sentinel_t = now + self._clock_offset_s - self._AGE_SENTINEL_S
        while len(buf) < self._FRAME_BUFFER_LEN:
            buf.insert(0, (0.0, 0.0, 0.0, sentinel_t))
        buf_obs: List[float] = []
        for d, vx, vy, t in buf:
            age = float(np.clip(now - t + self._clock_offset_s, 0.0, self._AGE_SENTINEL_S))
            buf_obs.extend([d, vx, vy, age])

        # 6-dim action history obs — oldest-first, zeros pad the front
        hist = list(self._action_history)
        while len(hist) < self._ACTION_HISTORY_LEN:
            hist.insert(0, (0.0, 0.0))
        act_obs: List[float] = []
        for omega, theta in hist:
            act_obs.extend([omega, theta])

        return np.array([
            np.cos(pos_rad),    # 0
            np.sin(pos_rad),    # 1
            vel_rad_s,          # 2
            imu["wz"],          # 3 — yaw rate only
            np.cos(cpg_phase),  # 4
            np.sin(cpg_phase),  # 5
            *buf_obs,           # 6-17
            *act_obs,           # 18-23
        ], dtype=np.float32)

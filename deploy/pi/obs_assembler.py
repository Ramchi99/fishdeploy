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


class ObsAssemblerDA40D:
    """40-dim observation for the Direct-Action (DA-MotorID / sofi_da_rl_policy) policy.

    MUST match rl/environments/env_sofi_da.py `_get_obs`
    (FRAME_BUFFER_LEN=5, TRACK_ACTUAL_MOTOR=True). Layout:
      [0]    cos(motor_pos)            encoder
      [1]    sin(motor_pos)
      [2]    motor_vel  [rad/s]
      [3]    imu_wz     [rad/s]        single axis (yaw rate)
      [4:24] 5 vision frames x [dist, vecX, vecY, age]  oldest->newest
             vecX/vecY = body-frame target vector; age = clip(now - frame_time, 0, 0.2)
      [24:40] 8 x [setpoint, actual]   oldest->newest, interleaved
             setpoint = policy action in [-1,1]; actual = motor_pos / ctrlrange (clipped)

    !!! VALIDATE WITH deploy/obs_verify/obs_verifier.py BEFORE ANY WATER RUN. !!!
    The frame-buffer age handling and motor-history pairing are timing-sensitive.
    """
    _AGE_SENTINEL_S = 0.200   # MUST equal env_sofi_da._AGE_SENTINEL_S
    _VEL_ZERO_AFTER_MS = 500.0

    def __init__(self, config: dict):
        self._stale_warn_ms = config["vision_stale_warn_ms"]
        self._stale_stop_ms = config["vision_stale_stop_ms"]
        self._com_offset = np.array(config["tag0_to_com_offset"], dtype=np.float64)
        self._clock_offset_s: float = 0.0
        self._fb_len = int(config.get("frame_buffer_len", 5))
        self._mh_len = int(config.get("motor_obs_history_len", 8))
        self._ctrlrange = float(config.get("motor_limit_rad", 1.0))
        self._frame_buffer: deque = deque(maxlen=self._fb_len)  # (dist, vx, vy, frame_time_pi)
        self._last_frame_seq: Optional[int] = None
        self._setpoint_hist: deque = deque([0.0] * self._mh_len, maxlen=self._mh_len)
        self._actual_hist:   deque = deque([0.0] * self._mh_len, maxlen=self._mh_len)
        self._pending_setpoint: float = 0.0

    def set_clock_offset(self, offset_s: float) -> None:
        self._clock_offset_s = offset_s

    def _frame_age_ms(self, vision: VisionState) -> float:
        return (time.time() - vision.laptop_timestamp + self._clock_offset_s) * 1000.0

    def vision_status(self, vision: VisionState) -> str:
        age = self._frame_age_ms(vision)
        if age >= self._stale_stop_ms:
            return "lost"
        if age >= self._stale_warn_ms:
            return "stale"
        return "ok"

    def set_pending_action(self, setpoint_norm: float) -> None:
        """Call AFTER policy.predict each step: the action just applied (in [-1,1]).
        It is paired with next step's measured motor position in the motor history."""
        self._pending_setpoint = float(np.clip(setpoint_norm, -1.0, 1.0))

    def build(self, motor: dict, imu: dict, vision: VisionState) -> np.ndarray:
        # Pair the last applied setpoint with the resulting (current) actual motor pos.
        actual = float(np.clip(motor["pos_rad"] / self._ctrlrange, -1.0, 1.0))
        self._setpoint_hist.append(self._pending_setpoint)
        self._actual_hist.append(actual)

        yaw = vision.fish_yaw
        fish_x = vision.fish_x + np.cos(yaw) * self._com_offset[0] - np.sin(yaw) * self._com_offset[1]
        fish_y = vision.fish_y + np.sin(yaw) * self._com_offset[0] + np.cos(yaw) * self._com_offset[1]
        dx = vision.target_x - fish_x
        dy = vision.target_y - fish_y
        vec_x =  np.cos(yaw) * dx + np.sin(yaw) * dy   # body-frame (matches RL rot)
        vec_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
        dist = float(np.hypot(vec_x, vec_y))
        self.last_dist = dist   # live target distance (for the deploy-loop done/success check)

        now = time.time()
        frame_time_pi = vision.laptop_timestamp - self._clock_offset_s
        if vision.frame_seq != self._last_frame_seq:
            self._frame_buffer.append((dist, float(vec_x), float(vec_y), frame_time_pi))
            self._last_frame_seq = vision.frame_seq

        buf = list(self._frame_buffer)
        fb: List[float] = []
        for _ in range(self._fb_len - len(buf)):        # pad oldest with sentinel
            fb.extend([0.0, 0.0, 0.0, self._AGE_SENTINEL_S])
        for (d, vx, vy, t) in buf:
            fb.extend([d, vx, vy, float(np.clip(now - t, 0.0, self._AGE_SENTINEL_S))])

        mh = [v for pair in zip(self._setpoint_hist, self._actual_hist) for v in pair]

        pos = motor["pos_rad"]
        return np.concatenate([
            [np.cos(pos), np.sin(pos), motor["vel_rad_s"], imu["wz"]],
            fb, mh,
        ]).astype(np.float32)


class ObsAssemblerCPG14D:
    """Assembles the 14-dimensional observation vector for env_sofi_phase1_1_hwobsNoise.

    Obs layout:
      [0]  cos(motor_pos)
      [1]  sin(motor_pos)
      [2]  motor_vel  [rad/s]
      [3]  imu_wx     [rad/s]
      [4]  imu_wy     [rad/s]
      [5]  imu_wz     [rad/s]
      [6]  dist_target  [m]
      [7]  vec_target_x (body-frame forward)
      [8]  vec_target_y (body-frame lateral)
      [9]  vel_vec_target_x (EWM smoothed, α=0.5)
      [10] vel_vec_target_y (EWM smoothed, α=0.5)
      [11] cos(cpg_phase)
      [12] sin(cpg_phase)
      [13] prev_omega
    """

    _EWM_ALPHA = 0.25
    _VEL_ZERO_AFTER_MS = 200.0

    def __init__(self, config: dict):
        self._stale_warn_ms = config["vision_stale_warn_ms"]
        self._stale_stop_ms = config["vision_stale_stop_ms"]
        self._com_offset = np.array(config["tag0_to_com_offset"], dtype=np.float64)
        self._clock_offset_s: float = 0.0

        # Rolling buffer for numerical differentiation of vec_target
        self._prev_vec: Optional[np.ndarray] = None
        self._prev_vec_time: Optional[float] = None
        self._vel_vec_ewm = np.zeros(2, dtype=np.float64)

    def set_clock_offset(self, offset_s: float) -> None:
        """Update Pi-vs-laptop clock offset (seconds, positive = laptop ahead)."""
        self._clock_offset_s = offset_s

    def _frame_age_ms(self, vision: VisionState) -> float:
        """True age from frame capture to now, corrected for clock skew."""
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
    ) -> np.ndarray:
        pos_rad = motor["pos_rad"]
        vel_rad_s = motor["vel_rad_s"]

        # World → body-frame target vector
        yaw = vision.fish_yaw
        # Apply tag→CoM offset in world frame
        fish_x_com = vision.fish_x + np.cos(yaw) * self._com_offset[0] - np.sin(yaw) * self._com_offset[1]
        fish_y_com = vision.fish_y + np.sin(yaw) * self._com_offset[0] + np.cos(yaw) * self._com_offset[1]
        dx = vision.target_x - fish_x_com
        dy = vision.target_y - fish_y_com
        vec_x =  np.cos(yaw) * dx + np.sin(yaw) * dy
        vec_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
        dist = np.sqrt(dx * dx + dy * dy)

        # Velocity of target vector (numerical diff + EWM)
        now = time.time()
        age_ms = self._frame_age_ms(vision)
        if age_ms >= self._VEL_ZERO_AFTER_MS:
            self._vel_vec_ewm[:] = 0.0
        elif self._prev_vec is not None and self._prev_vec_time is not None:
            dt = now - self._prev_vec_time
            if dt > 0:
                raw_vel = (np.array([vec_x, vec_y]) - self._prev_vec) / dt
                self._vel_vec_ewm = (self._EWM_ALPHA * raw_vel
                                     + (1.0 - self._EWM_ALPHA) * self._vel_vec_ewm)
        self._prev_vec = np.array([vec_x, vec_y])
        self._prev_vec_time = now

        obs = np.array([
            np.cos(pos_rad),          # 0
            np.sin(pos_rad),          # 1
            vel_rad_s,                # 2
            imu["wx"],                # 3
            imu["wy"],                # 4
            imu["wz"],                # 5
            dist,                     # 6
            vec_x,                    # 7
            vec_y,                    # 8
            self._vel_vec_ewm[0],     # 9
            self._vel_vec_ewm[1],     # 10
            np.cos(cpg_phase),        # 11
            np.sin(cpg_phase),        # 12
            prev_omega,               # 13
        ], dtype=np.float32)

        return obs


class ObsAssemblerCPG12D:
    """Assembles the 12-dimensional observation (14D minus vel_vec).

    Obs layout:
      [0]  cos(motor_pos)
      [1]  sin(motor_pos)
      [2]  motor_vel  [rad/s]
      [3]  imu_wx     [rad/s]
      [4]  imu_wy     [rad/s]
      [5]  imu_wz     [rad/s]
      [6]  dist_target  [m]
      [7]  vec_target_x (body-frame forward)
      [8]  vec_target_y (body-frame lateral)
      [9]  cos(cpg_phase)
      [10] sin(cpg_phase)
      [11] prev_omega
    """

    def __init__(self, config: dict):
        self._stale_warn_ms = config["vision_stale_warn_ms"]
        self._stale_stop_ms = config["vision_stale_stop_ms"]
        self._com_offset    = np.array(config["tag0_to_com_offset"], dtype=np.float64)
        self._clock_offset_s: float = 0.0

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
    ) -> np.ndarray:
        pos_rad   = motor["pos_rad"]
        vel_rad_s = motor["vel_rad_s"]

        yaw = vision.fish_yaw
        fish_x_com = (vision.fish_x
                      + np.cos(yaw) * self._com_offset[0]
                      - np.sin(yaw) * self._com_offset[1])
        fish_y_com = (vision.fish_y
                      + np.sin(yaw) * self._com_offset[0]
                      + np.cos(yaw) * self._com_offset[1])
        dx    = vision.target_x - fish_x_com
        dy    = vision.target_y - fish_y_com
        vec_x =  np.cos(yaw) * dx + np.sin(yaw) * dy
        vec_y = -np.sin(yaw) * dx + np.cos(yaw) * dy
        dist  = float(np.sqrt(dx * dx + dy * dy))

        return np.array([
            np.cos(pos_rad),   # 0
            np.sin(pos_rad),   # 1
            vel_rad_s,         # 2
            imu["wx"],         # 3
            imu["wy"],         # 4
            imu["wz"],         # 5
            dist,              # 6
            vec_x,             # 7
            vec_y,             # 8
            np.cos(cpg_phase), # 9
            np.sin(cpg_phase), # 10
            prev_omega,        # 11
        ], dtype=np.float32)


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

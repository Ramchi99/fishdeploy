# SOURCE OF TRUTH: fishCREATE/RL/cpg_oscillator.py
# If that file is updated, manually copy changes here.
import numpy as np


class CPGOscillator:
    """First-order-filtered CPG producing a bounded motor-position setpoint.

    State: phase phi and six filtered parameters [k1c, k1s, k2c, k2s, omega, theta_center].
    Each step(desired) advances the filter, advances phase by omega*dt, and
    returns the clipped motor setpoint.

    All quantities are in applied units (radians, rad/s) — action scaling is
    the caller's job.
    """

    def __init__(self, dt_phys: float, tau: float, motor_limit: float):
        self.dt_phys = float(dt_phys)
        self.tau = float(tau)
        self.motor_limit = float(motor_limit)
        self._applied = np.zeros(6, dtype=np.float64)
        self._phase = 0.0

    def reset(self):
        self._applied[:] = 0.0
        self._phase = 0.0

    def step(self, desired: np.ndarray) -> float:
        alpha = self.dt_phys / self.tau
        self._applied += alpha * (np.asarray(desired, dtype=np.float64) - self._applied)

        k1c, k1s, k2c, k2s, omega, theta_c = self._applied
        self._phase = (self._phase + omega * self.dt_phys) % (2 * np.pi)

        theta_m = (theta_c
                   + k1c * np.cos(self._phase)     + k1s * np.sin(self._phase)
                   + k2c * np.cos(2 * self._phase) + k2s * np.sin(2 * self._phase))
        return float(np.clip(theta_m, -self.motor_limit, self.motor_limit))

    @property
    def applied_params(self) -> np.ndarray:
        return self._applied.copy()

    @property
    def phase(self) -> float:
        return self._phase

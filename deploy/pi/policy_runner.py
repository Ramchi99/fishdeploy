import math
import numpy as np

# Action scaling constants (from env_sofi_phase1_1_hwobsNoise.py)
K1_MAX = 0.30
K2_MAX = 0.15
OMEGA_MAX = 12 * math.pi  # rad/s ceiling (≈6 Hz)
THETA_CENTER_MAX = 0.15

_SCALES = np.array([K1_MAX, K1_MAX, K2_MAX, K2_MAX, OMEGA_MAX, THETA_CENTER_MAX])


def scale_action(raw_action: np.ndarray) -> np.ndarray:
    """Scale SAC output [-1,1]^6 → applied CPG units (rad, rad/s)."""
    a = np.asarray(raw_action, dtype=np.float64)
    scaled = a * _SCALES
    scaled[4] = 0.5 * (a[4] + 1.0) * OMEGA_MAX  # omega half-rectified to [0, OMEGA_MAX]
    return scaled


class PolicyRunner:
    """Wraps SB3 SAC model for inference.

    Usage:
        runner = PolicyRunner(model_path)
        action_raw, action_scaled = runner.predict(obs_14d)
    """

    def __init__(self, model_path: str):
        from stable_baselines3 import SAC
        # SB3 appends .zip automatically — strip it first to avoid .zip.zip
        if model_path.endswith(".zip"):
            model_path = model_path[:-4]
        self._model = SAC.load(model_path, device="cpu")
        print(f"[PolicyRunner] Loaded: {model_path}")

    def predict(self, obs: np.ndarray):
        """Returns (raw_action [−1,1]^6, scaled_action in CPG units)."""
        # MUST use deterministic=False. SAC for behind-targets learns a bimodal
        # distribution (turn left OR turn right). The deterministic mean ≈ 0 → no
        # turn → fails on lateral/behind targets. Stochastic sampling picks a
        # direction and commits to it, which is correct real-robot behaviour.
        raw, _ = self._model.predict(obs, deterministic=False)
        return raw, scale_action(raw)

    def predict_raw(self, obs: np.ndarray):
        """Direct-action policies: return raw action in [-1,1] with NO CPG scaling
        (scale_action is CPG-only, assumes a 6-dim action). Stochastic, as required."""
        raw, _ = self._model.predict(obs, deterministic=False)
        return raw

    def load_actor_weights(self, state_dict: dict):
        """Hot-reload actor weights between episodes (Mode B fine-tuning)."""
        self._model.policy.actor.load_state_dict(state_dict)

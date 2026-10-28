import math


class ZeroTrajectory:
    name = "zero"

    def step(self, t: float) -> float:
        return 0.0

    def cpg_phase(self, t: float) -> float:
        return 0.0


class SineTrajectory:
    name = "sine"

    def __init__(self, freq: float, amp: float, offset: float = 0.0):
        self.freq = freq
        self.amp = amp
        self.offset = offset

    def step(self, t: float) -> float:
        return self.offset + self.amp * math.sin(2.0 * math.pi * self.freq * t)

    def cpg_phase(self, t: float) -> float:
        return (2.0 * math.pi * self.freq * t) % (2.0 * math.pi)


class SineGlideTrajectory:
    name = "sine_glide"

    def __init__(self, freq: float, amp: float, offset: float = 0.0, sine_duration: float = 10.0):
        self.freq = freq
        self.amp = amp
        self.offset = offset
        self.sine_duration = sine_duration

    def step(self, t: float) -> float:
        if t < self.sine_duration:
            return self.offset + self.amp * math.sin(2.0 * math.pi * self.freq * t)
        return 0.0

    def cpg_phase(self, t: float) -> float:
        t_eff = min(t, self.sine_duration)
        return (2.0 * math.pi * self.freq * t_eff) % (2.0 * math.pi)


class SineHoldTrajectory:
    """Sine wave for sine_duration seconds, then hold at hold_offset.

    Useful for bias-glide experiments: swim at full amplitude for a few
    seconds, then freeze the tail at a fixed deflection (0.1, 0.2, 0.3 rad…)
    to measure passive drift / yaw response.
    """
    name = "sine_hold"

    def __init__(self, freq: float, amp: float, offset: float = 0.0,
                 sine_duration: float = 10.0, hold_offset: float = 0.0):
        self.freq = freq
        self.amp = amp
        self.offset = offset
        self.sine_duration = sine_duration
        self.hold_offset = hold_offset

    def step(self, t: float) -> float:
        if t < self.sine_duration:
            return self.offset + self.amp * math.sin(2.0 * math.pi * self.freq * t)
        return self.hold_offset

    def cpg_phase(self, t: float) -> float:
        t_eff = min(t, self.sine_duration)
        return (2.0 * math.pi * self.freq * t_eff) % (2.0 * math.pi)


def make_trajectory(name: str, **kwargs):
    """Factory: name in {'zero', 'sine', 'sine_glide', 'sine_hold'}."""
    if name == "zero":
        return ZeroTrajectory()
    if name == "sine":
        return SineTrajectory(
            freq=float(kwargs["freq"]),
            amp=float(kwargs["amp"]),
            offset=float(kwargs.get("offset", 0.0)),
        )
    if name == "sine_glide":
        return SineGlideTrajectory(
            freq=float(kwargs["freq"]),
            amp=float(kwargs["amp"]),
            offset=float(kwargs.get("offset", 0.0)),
            sine_duration=float(kwargs.get("sine_duration", 10.0)),
        )
    if name == "sine_hold":
        return SineHoldTrajectory(
            freq=float(kwargs["freq"]),
            amp=float(kwargs["amp"]),
            offset=float(kwargs.get("offset", 0.0)),
            sine_duration=float(kwargs.get("sine_duration", 10.0)),
            hold_offset=float(kwargs.get("hold_offset", 0.0)),
        )
    raise ValueError(f"Unknown trajectory: {name!r}. Choose: zero, sine, sine_glide, sine_hold")

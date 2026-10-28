# Sofi Field Test — Cross-Run Evaluation

**Session:** 2026-04-01  |  **Runs:** 1–12  |  **Motor:** Dynamixel XL430 @ ~11 V  |  **Pulley radius:** 17.5 mm (analysis) / 12.5 mm (field software)

> **Amplitude note:** The field software (`unified_test.py`) computed `amp_ticks` with the default `--radius 12.5 mm`. All amplitudes in this report are converted from encoder ticks using **r = 17.5 mm**, which is the physically measured pulley radius used in the analysis pipeline. A "5 mm nominal" run therefore corresponds to ≈ 6.98 mm of actual tendon travel at 17.5 mm. Commanded and actual amplitudes are always compared in the same unit (mm at 17.5 mm), so `actual < commanded` is physically guaranteed.

---

## 1  Run Overview

| Run | Type | Freq (Hz) | Nominal amp (mm) | Cmd amp @17.5mm | Offset (mm) | Duration (s) |
|-----|------|-----------|------------------|-----------------|-------------|--------------|
| R1 | straight | 1.0 | 5 | 6.98 | +0.0 | 21.3 |
| R2 | straight | 2.0 | 5 | 6.98 | +0.0 | 14.8 |
| R3 | straight | 1.0 | 9 | 12.59 | +0.0 | 14.4 |
| R4 | straight | 2.0 | 9 | 12.59 | +0.0 | 9.8 |
| R5 | straight | 1.0 | 7 | 9.80 | +0.0 | 20.1 |
| R6 | straight | 2.0 | 7 | 9.80 | +0.0 | 10.8 |
| R7 | straight | 3.0 | 7 | 9.80 | +0.0 | 8.7 |
| R8 | curve | 2.0 | 6 | 8.38 | +3.0 | 9.9 |
| R9 | curve | 2.0 | 6 | 8.38 | -3.0 | 12.2 |
| R10 | straight | 4.0 | 7 | 9.80 | +0.0 | 8.5 |
| R11 | straight | 2.0 | 12 | 16.78 | +0.0 | 9.9 |
| R12 | straight | 2.0 | 12 | 16.78 | +0.0 | 9.8 |

---

## 2  Trajectories

![Trajectories](fig1_trajectories.png)

![Trajectories overlaid](fig1b_trajectories_overlay.png)

All 12 runs at equal spatial scale, using the same coordinate convention as `04_plot_field_test.py`: X axis = swim direction (decreases as fish swims), Y axis = lateral. Blue = straight, orange = curve. Start (green circle) and end (red ×) markers shown.

Straight runs cover ~3–3.5 m in the swim direction. Curve runs (R8, R9) cover only 1.5–1.9 m along the swim axis because the fish curves, spending more lateral displacement. Run 2 (2 Hz/5 mm) shows notable lateral drift despite being a straight run.

---

## 3  Steady-State Speed

**Method:** Linear regression on the swim-axis position over the **50–80% window** of each run. This avoids the acceleration transient (0–50%) and the Savitzky-Golay boundary effect at the very end (>80%). Since this uses raw position data — not a differentiated speed estimate — it has no SG boundary noise.

> **On the speed-still-increasing question:** If the speed profile is still rising between 50–70% of normalised time, the fish has not yet reached steady state within the run. In that case the regression slope gives an *average effective speed* rather than a true steady-state. This is most likely for short runs (< 10 s). The speed profile figure (Fig. 2a) shows the 50–80% window shaded — inspect whether individual runs have plateaued within it.

<p><img src="fig2a_speed_profiles.png" width="49%"/> <img src="fig2b_steady_state_speed.png" width="49%"/></p>

<p><img src="fig3a_speed_vs_freq.png" width="49%"/> <img src="fig3b_speed_vs_amplitude.png" width="49%"/></p>

Fig. 3a shows steady-state speed vs. commanded frequency; each point is coloured by actual amplitude (blue = low, red = high). Fig. 3b plots speed against *actual* amplitude — this is the physically meaningful relationship because the fish cannot know what was commanded, only what the tail actually does.

### Straight-run summary

| Run | Freq (Hz) | Cmd amp (mm) | Act amp (mm) | Tracking | SS speed (m/s) | Accel (m/s²) |
|-----|-----------|-------------|-------------|----------|----------------|--------------|
| R1 | 1.0 | 6.99 | 6.24 | 0.893 | **0.178** | 0.015 |
| R2 | 2.0 | 6.99 | 6.11 | 0.873 | **0.257** | 0.049 |
| R3 | 1.0 | 12.60 | 11.24 | 0.892 | **0.276** | 0.059 |
| R4 | 2.0 | 12.60 | 8.86 | 0.703 | **0.417** | 0.100 |
| R5 | 1.0 | 9.81 | 8.83 | 0.900 | **0.218** | 0.027 |
| R6 | 2.0 | 9.81 | 8.08 | 0.823 | **0.359** | 0.067 |
| R7 | 3.0 | 9.81 | 5.77 | 0.588 | **0.422** | 0.096 |
| R10 | 4.0 | 9.90 | 4.09 | 0.414 | **0.479** | 0.157 |
| R11 | 2.0 | 16.79 | 9.10 | 0.542 | **0.427** | 0.096 |
| R12 | 2.0 | 16.79 | 8.96 | 0.534 | **0.396** | 0.098 |

**Trends:**
- Higher frequency → higher speed at the same amplitude.
- Higher actual amplitude → higher speed at the same frequency.
- R10 (4 Hz/7 mm nominal) achieves the highest measured speed despite the lowest   actual amplitude among all 4 Hz runs — high-frequency small flaps still produce   meaningful thrust even when the motor is severely velocity-limited.
- R11 vs R12 (same 2 Hz/12 mm nominal, P=800 vs P=1600): the higher P gain barely   changes actual amplitude but slightly reduces steady-state speed, suggesting   increased oscillation/overshoot rather than better tracking.

---

## 4  Motor Amplitude Tracking

Both commanded and actual amplitudes are computed from encoder **ticks** using **r = 17.5 mm**, so they are on the same physical scale. `actual < commanded` is physically guaranteed for position-controlled servos.

<p><img src="fig4a_motor_amplitude.png" width="49%"/> <img src="fig4b_tracking_ratio.png" width="49%"/></p>

### Tracking table

| Run | Nominal (mm) | Cmd @17.5mm | Act @17.5mm | Tracking |
|-----|-------------|-------------|------------|----------|
| R1 | 5 | 6.99 | 6.24 | 0.893 |
| R2 | 5 | 6.99 | 6.11 | 0.873 |
| R3 | 9 | 12.60 | 11.24 | 0.892 |
| R4 | 9 | 12.60 | 8.86 | 0.703 |
| R5 | 7 | 9.81 | 8.83 | 0.900 |
| R6 | 7 | 9.81 | 8.08 | 0.823 |
| R7 | 7 | 9.81 | 5.77 | 0.588 |
| R8 | 6 | 8.39 | 7.11 | 0.847 |
| R9 | 6 | 8.39 | 7.07 | 0.843 |
| R10 | 7 | 9.90 | 4.09 | 0.414 |
| R11 | 12 | 16.79 | 9.10 | 0.542 |
| R12 | 12 | 16.79 | 8.96 | 0.534 |

Tracking degrades sharply above ~2 Hz or for large amplitudes. This is explained by the velocity budget below — the commanded sine requires faster motor shaft rotation than the battery-limited motor can deliver. Adjusting the P gain (R12 vs R11) has minimal effect on amplitude tracking because the bottleneck is velocity, not position gain.

---

## 5  Velocity Budget — Why Tracking Degrades

The Dynamixel XL430 has a theoretical no-load speed of **60.9 RPM (266 units) at 12 V**. With the field battery at ~11 V, the practical limit observed in the data is **50.4 RPM (220 units)** — the `Vel` register consistently caps at 220 units across runs 11 and 12 (the highest-demand runs).

The **required peak velocity** to track a sinusoidal position command is:

> `v_peak = 2π · f · A_rad` (rad/s) = `60 · f · A_rad / 0.229` (Dxl units)

where `A_rad` is the commanded amplitude in radians.

![Velocity budget](fig5_velocity_budget.png)

### Velocity budget table

| Run | Freq | Cmd amp | Required vel (units) | Required (% of limit) | Measured peak (units) | Saturating? |
|-----|------|---------|---------------------|----------------------|----------------------|-------------|
| R1 | 1.0 Hz | 7.0 mm | 105 | 48% | 105 | no |
| R2 | 2.0 Hz | 7.0 mm | 209 | 95% | 189 | no |
| R3 | 1.0 Hz | 12.6 mm | 189 | 86% | 192 | no |
| R4 | 2.0 Hz | 12.6 mm | 377 | 172% | 218 | **YES** |
| R5 | 1.0 Hz | 9.8 mm | 147 | 67% | 150 | no |
| R6 | 2.0 Hz | 9.8 mm | 294 | 133% | 212 | **YES** |
| R7 | 3.0 Hz | 9.8 mm | 441 | 200% | 201 | **YES** |
| R8 | 2.0 Hz | 8.4 mm | 251 | 114% | 213 | **YES** |
| R9 | 2.0 Hz | 8.4 mm | 251 | 114% | 206 | **YES** |
| R10 | 4.0 Hz | 9.9 mm | 593 | 269% | 187 | **YES** |
| R11 | 2.0 Hz | 16.8 mm | 503 | 229% | 220 | **YES** |
| R12 | 2.0 Hz | 16.8 mm | 503 | 229% | 220 | **YES** |

Runs where required velocity exceeds the practical limit (~220 units) show a **clipped, non-sinusoidal position profile** — the motor moves at near-constant maximum velocity instead of following the smooth sine, making the position profile triangular rather than sinusoidal. This reduces effective amplitude and introduces phase lag. The effect is most severe for R10 (4 Hz/7 mm, required = 587 units = 267% of limit — motor achieves only 51% of commanded amplitude).

---

## 6  Path Straightness

The raw trajectory mixes two physically distinct phenomena: slow **heading drift** (hydrodynamic asymmetry, misalignment) and high-frequency **lateral oscillations** from the tail flap. The old single-metric approach (RMS from a best-fit line) confounds both. These plots use a decomposition instead:

1. **Smooth path** — the trajectory low-pass filtered at $f_{\rm swim}/2$    (4th-order zero-phase Butterworth). This retains slow heading drift and removes    the per-cycle oscillation.
2. **Path drift (fig. 6a)** — RMS perpendicular distance of the *smooth path* from    the best-fit line through the original trajectory. Pure heading quality.
3. **Oscillation magnitude (fig. 6b)** — RMS of (actual trajectory − smooth path).    Pure per-cycle lateral displacement of the fish body.

<p><img src="fig6a_path_drift.png" width="49%"/> <img src="fig6b_osc_magnitude.png" width="49%"/></p>

| Run | Freq (Hz) | Cmd amp (mm) | Act amp (mm) | Path drift (mm) | Osc. magnitude (mm) |
|-----|-----------|-------------|-------------|----------------|---------------------|
| R10 | 4.0 | 9.90 | 4.09 | 17.1 | 4.39 |
| R11 | 2.0 | 16.79 | 9.10 | 17.7 | 4.99 |
| R1 | 1.0 | 6.99 | 6.24 | 7.6 | 8.27 |
| R2 | 2.0 | 6.99 | 6.11 | 39.4 | 8.20 |
| R3 | 1.0 | 12.60 | 11.24 | 11.5 | 17.63 |
| R4 | 2.0 | 12.60 | 8.86 | 33.2 | 6.68 |
| R5 | 1.0 | 9.81 | 8.83 | 25.3 | 12.12 |
| R6 | 2.0 | 9.81 | 8.08 | 22.5 | 7.08 |
| R7 | 3.0 | 9.81 | 5.77 | 60.1 | 8.87 |

**Path drift** reflects heading stability — how consistently the fish tracks a straight course ignoring the flapping. **Oscillation magnitude** reflects the physical lateral body displacement per flap cycle, which scales with actual tail amplitude and frequency.

---

## 7  Curve Runs: R8 vs R9

Both curve runs use a **2 Hz / 6 mm** oscillation with a **3 mm tendon offset** in opposite directions. The offset shifts the equilibrium position, creating asymmetric forward/backward strokes and net lateral thrust.

![Curve comparison](fig7_curves.png)

### Asymmetry analysis (all values @ 17.5mm radius)

| Metric | R8 (+3mm offset) | R9 (−3mm offset) |
|--------|-----------------|-----------------|
| Commanded offset (bias) | +3.98 mm | -3.99 mm |
| **Actual offset (bias)** | **+3.45 mm** | **-3.40 mm** |
| Commanded +stroke amp | 8.37 mm | 8.28 mm |
| **Actual +stroke amp** | **7.18 mm** | **7.00 mm** |
| Commanded −stroke amp | 8.36 mm | 8.44 mm |
| **Actual −stroke amp** | **7.13 mm** | **7.18 mm** |
| SS speed (m/s) | 0.245 | 0.148 |
| Fitted turning radius (m) | 1.98 | 2.23 |

**Key observations:**
- R9 (−3mm) turns significantly tighter than R8 (+3mm): radius 2.23 m vs 1.98 m.
- The actual motor bias differs between runs: the motor has an easier time reaching   the offset in one direction than the other (mechanical/preload asymmetry in the   tendon drive). This means a +3mm commanded offset ≠ −3mm commanded offset in terms   of actual achieved bias and resulting curvature.
- The asymmetry in actual +/- stroke amplitudes is visible: R8 acts   0.05 mm differently between sides;   R9 acts 0.18 mm differently.
- Both curve runs are slower than the symmetric 2 Hz/6 mm equivalent (the offset   effectively reduces mean driving amplitude).

---

## 8  Voltage-Driven Tracking Loss

![Saturation analysis](fig8_saturation_analysis.png)

**X axis:** saturation depth = required peak velocity / measured peak velocity. Values > 1 mean the commanded sine demands faster shaft rotation than the motor can deliver at the current battery voltage. **Y axis:** amplitude tracking ratio. The dashed line is the ideal triangular-wave saturation model: tracking = min(1, 1/x). Colour encodes mean battery voltage per run.

**Key observations:**
- All runs lie *above* the ideal 1/x curve — the motor achieves better tracking   than a pure triangular-wave prediction. This is expected: a velocity-clipped   sine is not perfectly triangular; the motor still partially follows the   curvature near the turning points.
- Unsaturated runs (depth < 1) cluster near tracking ≈ 0.90 rather than 1.0,   revealing a ~10% baseline droop from the P-only controller's steady-state   position error (no I term to eliminate offset under load).
- Battery voltage (colour) correlates with saturation depth: runs later in the   session (lower voltage, darker points) sit further right — the lower motor   speed ceiling raises the saturation depth for the same commanded sine.

---

_Generated by `field_test/evaluation/evaluate_all_runs.py`_

# obs_verify: does the hardware observation match the simulator?

Ten physical tests, one per observation convention, run before a water session or after any change to
the sensors, the tags or `pi/obs_assembler.py`. The verifier builds the observation with the same
`ObsAssemblerCPG24D` and the same tag tracker as the deployment, so what it shows is what the policy sees.

```bash
python3 deploy/obs_verify/pi_sensor_server.py                    # Pi, for tests 6 to 10 (IMU + motor over UDP 5007)
python deploy/obs_verify/obs_verifier.py --device 1               # laptop, tests 1 to 5 (vision only)
python deploy/obs_verify/obs_verifier.py --device 1 --pi-ip raspberrypi.local      # all ten
```

Stop `vision_streamer.py` first; both open the camera. Keys: ENTER pass, `f` fail, `s` skip, `r` reset
the frame buffer, `q` quit.

| # | What you do | What the observation must show |
|---|---|---|
| 1 | all three tags in view | fps > 20, tags 0 (fish), 1 (anchor), 2 (target) seen |
| 2 | tape-measure fish to target | obs[14] = distance ± 10 cm |
| 3 | nose pointing at the target | obs[15] (vec_x) < 0, obs[16] (vec_y) ≈ 0 |
| 4 | target 90° to the fish's left | obs[16] < 0 |
| 5 | target 90° to the fish's right | obs[16] > 0 |
| 6 | spin the fish counter-clockwise from above | obs[3] (wz) > 0 |
| 7 | spin it clockwise | obs[3] < 0 |
| 8 | roll right, tilt nose up | wx, wy spike on the Pi server's terminal (not in the observation) |
| 9 | fish at rest | obs[3] ≈ 0 |
| 10 | tail straight | obs[0] > 0.9, obs[1] ≈ 0 |

obs[6..17] are three vision frames (dist, vec_x, vec_y, age), oldest first; the tests read the most
recent one. obs[18..23] are the three past actions.

**Why vec_x is negative with the target straight ahead:** the MuJoCo fish swims along its body −X axis,
and training computes the target vector with `R(−headRollZ)`. The deployment's `fish_yaw` follows the
nose and differs from `headRollZ` by π, which the assembler compensates by negating both components. The
bearing printed in the status line (0° ahead, +90° left) is for the human and is not sent to the policy.

Detection runs with the same region-of-interest tracking as the streamer (`shared/tag_tracker.py`), so the
fps shown here is the fps the streamer will reach on the same machine; `--no-roi` gives the full-frame
baseline.

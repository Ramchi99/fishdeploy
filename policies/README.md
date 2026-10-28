# Policies

| File | md5 | What |
|---|---|---|
| `sofi_cpg_rl_policy.zip` | `7bbe08b8e0999fb4324b073d324d8c78` | The deployed policy. Stable-Baselines3 SAC, 24-D observation, 6 CPG parameters as action. Trained in [fishsim](https://github.com/srl-ethz/fishsim) (W&B run `e06g760j`, curriculum stage 8 "lateral", targets up to ±120°). Every one of the 43 paper runs in `deploy/runs/` records this file's hash in its `run_config.json`. |

Run it with stochastic inference (`deterministic=False`), see `deploy/README.md`. The direct-action policy
and the training code are in fishsim; the archive (see `DATA.md`) has both checkpoints.

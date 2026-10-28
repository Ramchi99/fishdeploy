import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from deploy.pi.deploy_main import compute_reward, make_run_dir

def test_reward_distance_penalty():
    # reward = -dist + action_penalty + success_bonus
    # obs[6] = dist_target = 1.0, action = zeros → reward = -1.0
    import numpy as np
    obs = np.zeros(14, dtype=np.float32)
    obs[6] = 1.0
    action = np.zeros(6, dtype=np.float32)
    r = compute_reward(obs, action)
    assert abs(r - (-1.0)) < 1e-5

def test_reward_success_bonus():
    import numpy as np
    obs = np.zeros(14, dtype=np.float32)
    obs[6] = 0.03  # < 0.05 threshold → success
    action = np.zeros(6, dtype=np.float32)
    r = compute_reward(obs, action)
    assert r > 900  # dominated by 1000 success bonus

def test_make_run_dir_format():
    d = make_run_dir("deploy/data", "pool_test_1")
    import re
    assert re.match(r"deploy/data/\d{8}_\d{6}_pool_test_1", d)

def test_make_run_dir_with_ts():
    """make_run_dir accepts an explicit timestamp and produces matching name."""
    import time
    from deploy.pi.deploy_main import make_run_dir
    ts = 1746000000.0  # 2025-04-30 20:00:00 UTC
    d = make_run_dir("deploy/data", "myrun", ts=ts)
    import re
    assert re.match(r"deploy/data/\d{8}_\d{6}_myrun", d)

def test_open_loop_make_run_dir_with_ts():
    """open_loop_main re-uses make_run_dir from deploy_main — verify after both tasks done."""
    from deploy.pi.deploy_main import make_run_dir
    import re
    ts = 1746000000.0
    d = make_run_dir("deploy/data", "sine_1", ts=ts)
    assert re.match(r"deploy/data/\d{8}_\d{6}_sine_1", d)

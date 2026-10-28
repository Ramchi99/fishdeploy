import sys, os, tempfile, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import pandas as pd
from deploy.pi.data_logger import DataLogger

def _make_logger(tmpdir, record=False, finetune=False):
    config = {
        "env_type": "cpg_14d",
        "model_path": "policy/test.zip",
    }
    return DataLogger(
        run_dir=tmpdir,
        config=config,
        record=record,
        finetune=finetune,
        model_hash="abc123",
        git_hash="def456",
        pi_ip="10.55.163.213",
        clock_offset_ms=2.5,
    )

def test_run_csv_created(tmp_path):
    logger = _make_logger(str(tmp_path))
    obs = np.zeros(14, dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)
    logger.log_step(
        pi_time=1.0, laptop_time=0.99,
        obs=obs, action=action, motor_setpoint=0.1,
        reward=-0.5, done=False,
        vision_age_ms=15.0, vision_flags=0x03,
        frame_seq=10, cpg_phase=1.57,
    )
    logger.close()
    csv_path = os.path.join(str(tmp_path), "run.csv")
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert "pi_time" in df.columns
    assert "obs_00" in df.columns
    assert "act_00" in df.columns

def test_run_config_json_created(tmp_path):
    logger = _make_logger(str(tmp_path))
    logger.close()
    cfg_path = os.path.join(str(tmp_path), "run_config.json")
    assert os.path.exists(cfg_path)
    with open(cfg_path) as f:
        cfg = json.load(f)
    assert cfg["model_hash"] == "abc123"
    assert cfg["clock_offset_ms"] == 2.5

def test_replay_buffer_written_when_finetune(tmp_path):
    logger = _make_logger(str(tmp_path), finetune=True)
    obs = np.zeros(14, dtype=np.float32)
    action = np.zeros(6, dtype=np.float32)
    logger.log_step(
        pi_time=1.0, laptop_time=0.99,
        obs=obs, action=action, motor_setpoint=0.1,
        reward=-0.5, done=False,
        vision_age_ms=10.0, vision_flags=0x03,
        frame_seq=1, cpg_phase=0.0,
    )
    logger.close()
    rb_path = os.path.join(str(tmp_path), "replay_buffer.csv")
    assert os.path.exists(rb_path)
    df = pd.read_csv(rb_path)
    assert "s_00" in df.columns
    assert "a_00" in df.columns
    assert "reward" in df.columns

def test_no_replay_buffer_without_flag(tmp_path):
    logger = _make_logger(str(tmp_path), finetune=False)
    logger.log_step(1.0, 0.99, np.zeros(14), np.zeros(6), 0.0, -0.5, False, 10.0, 0x03, 1, 0.0)
    logger.close()
    assert not os.path.exists(os.path.join(str(tmp_path), "replay_buffer.csv"))

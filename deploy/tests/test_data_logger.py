import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import numpy as np
import pandas as pd
from deploy.pi.data_logger import DataLogger


def _make_logger(tmpdir, record=False):
    return DataLogger(run_dir=tmpdir, config={"env_type": "cpg_24d", "model_path": "policies/test.zip"},
                      record=record, model_hash="abc123", git_hash="def456", pi_ip="10.55.163.213",
                      clock_offset_ms=2.5)


def test_run_csv_has_the_24d_columns(tmp_path):
    logger = _make_logger(str(tmp_path))
    logger.log_step(pi_time=1.0, laptop_time=0.99, obs=np.zeros(24, dtype=np.float32),
                    action=np.zeros(6, dtype=np.float32), motor_setpoint=0.1, reward=-0.5, done=False,
                    vision_age_ms=15.0, vision_flags=0x03, frame_seq=10, cpg_phase=1.57)
    logger.close()
    df = pd.read_csv(os.path.join(str(tmp_path), "run.csv"))
    assert len(df) == 1
    assert [c for c in df.columns if c.startswith("obs_")] == [f"obs_{i:02d}" for i in range(24)]
    assert [c for c in df.columns if c.startswith("act_")] == [f"act_{i:02d}" for i in range(6)]
    assert list(df.columns[:2]) == ["pi_time", "laptop_time"]
    assert df.columns[-1] == "motor_load"


def test_run_config_json_records_the_provenance(tmp_path):
    logger = _make_logger(str(tmp_path), record=True)
    logger.close()
    with open(os.path.join(str(tmp_path), "run_config.json")) as f:
        cfg = json.load(f)
    assert cfg["env_type"] == "cpg_24d"
    assert cfg["model_hash"] == "abc123"
    assert cfg["clock_offset_ms"] == 2.5
    assert cfg["record"] is True
    assert "finetune" not in cfg

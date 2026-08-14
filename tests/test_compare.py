"""Milestone-5 gates: the comparison stack's trial runner and persistence."""
import numpy as np

import mfoptmm as mf
from mfoptmm import compare


def test_run_trial_and_roundtrip(tmp_path):
    spec = mf.bbg_example(variant="a1", K_blocks=4, profile="replication")
    cfg = mf.TrainConfig(r=16, n=8, epochs=60, lr0=0.3, lr1=3e-3, seed=0)
    res = compare.run_trial(spec, cfg, arm="reg", fresh_n=256)
    for v in (res.J_train, res.J_fresh, res.se_fresh, res.arb_center,
              res.arb_avg, res.wall_s):
        assert np.isfinite(v)
    # dominance of the exact arbiter (learned <= v*, both fine-grid)
    assert res.arb_avg <= res.refs["vstar_avg"] + 1e-6
    assert res.refs["jref_avg"] < res.refs["vstar_avg"]
    compare.save_trial(res, tmp_path / "t.npz")
    back = compare.load_trials(tmp_path)[0]
    assert back.cfg == res.cfg
    assert abs(back.arb_avg - res.arb_avg) < 1e-9
    assert "reg" in compare.gap_table([back])

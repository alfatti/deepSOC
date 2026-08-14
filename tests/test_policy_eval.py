"""Policy-evaluation gates.

The linear evaluator shares the stencil, gate and source with the HJB, so two
exact statements hold and are asserted: plugging the HJB's own per-step quote
maps reproduces v to machine precision (envelope identity through the full
backward pass), and any schedule prices <= v pointwise (monotone comparison,
valid because the evaluator's realised diagonal stays positive).

The frozen-block gate prices the K = 36 decision-grid restriction the BCR
trainer lives in: a zero-order hold of the optimal maps at block starts.
Measured: 1.77e-4 of v* at the reference state — the restriction is ~2 bp.
The myopic baseline prices the whole inventory-management channel: measured
51.4% of v* at the centre, i.e. the D-gap the learner can close is ~half the
achievable value.
"""
import numpy as np
import mfoptmm as mf
from mfoptmm.hjb import make_ops, at_state
from mfoptmm.policy_eval import policy_value, repeat_block_start, myopic_policy

NU0 = 0.0225


def test_envelope_identity_through_backward_pass(body):
    # Measured |u0 - v0|max = 2.9e-11 on a 1.63e5 scale (1.8e-16 relative).
    spec, sol = body
    pv = policy_value(spec, sol.quotes)
    scale = float(np.abs(np.asarray(sol.v0)).max())
    assert float(np.abs(np.asarray(pv.u0) - np.asarray(sol.v0)).max()) \
        <= 1e-8 * scale
    assert float(pv.min_diag.min()) > 0.5


def test_frozen_block_restriction_price(body):
    spec, sol = body
    qf = repeat_block_start(sol.quotes, spec.time_grid.substeps_per_block)
    u = policy_value(spec, qf)
    ops = make_ops(spec)
    v_c = float(at_state(sol.v0, ops, NU0, 0.0))
    gap_rel = (v_c - float(at_state(u.u0, ops, NU0, 0.0))) / v_c
    # measured 1.77e-4; dominance measured max(u - v) = -28.9 EUR
    assert 0.0 <= gap_rel < 5e-3, f"frozen-block gap {gap_rel:.2e}"
    assert float((np.asarray(u.u0) - np.asarray(sol.v0)).max()) < 1e-6


def test_myopic_baseline(body):
    spec, sol = body
    u = policy_value(spec, myopic_policy(spec))
    ops = make_ops(spec)
    ratio = float(at_state(u.u0, ops, NU0, 0.0)) / float(at_state(sol.v0, ops, NU0, 0.0))
    # measured 0.514: the inventory-management channel is worth ~half of v*
    assert 0.35 < ratio < 0.70, f"myopic/optimal ratio {ratio:.3f}"
    assert float((np.asarray(u.u0) - np.asarray(sol.v0)).max()) < 1e-6
    assert float(u.min_diag.min()) > 0.5


def test_terminal_map_is_myopic(body):
    # v(T) = 0 => p = 0 => the last map equals the myopic quotes exactly.
    spec, sol = body
    myo = np.asarray(mf.myopic_quotes(spec.book))
    err = np.abs(np.asarray(sol.quotes[-1]) - myo[None, None, :, None]).max()
    assert err <= 1e-12


def test_schedule_builders(body):
    import pytest
    spec, sol = body
    with pytest.raises(ValueError):
        repeat_block_start(sol.quotes, 7)          # 180 % 7 != 0
    q = myopic_policy(spec)
    assert q.shape == (spec.time_grid.M, spec.nu_grid.n, spec.vpi_grid.n,
                       spec.book.strike.shape[0], 2)

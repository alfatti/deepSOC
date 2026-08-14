"""Gates E4 (realised monotonicity) and E2 (Remark-6 collapse), structural
orderings, and the dt first-order convergence contract.

All numeric thresholds are pinned from measured values with headroom; the
measured values are recorded in the assertions' comments and in the README.
"""
import numpy as np
import jax.numpy as jnp
import mfoptmm as mf
from mfoptmm.hjb import make_ops, at_state

NU0 = 0.0225


def test_gate_e4_replication(body):
    # Worst-case bound rejects M = 180 (config.stability_rates); the realised
    # diagonal licenses it. Measured min 0.7162, max dt*sum(Lambda*) 0.3356.
    _, sol = body
    assert float(sol.min_diag.min()) > 0.5
    assert float(sol.lam_dt_max.max()) < 0.5


def test_gate_e4_worst_case():
    spec = mf.bbg_example(variant="body", profile="worst_case")
    sol = mf.hjb_solve(spec)          # check=True asserts > 0 at runtime
    assert float(sol.min_diag.min()) > 0.7   # measured 0.7973 at M = 252


def test_e2_nu_collapse(collapsed):
    # The scheme preserves nu-constants identically: measured variation 0.0.
    _, sol = collapsed
    v = np.asarray(sol.v0)
    assert np.abs(v - v[:1]).max() <= 1e-9 * np.abs(v).max()
    q = np.asarray(sol.quotes[0])
    assert np.abs(q - q[:1]).max() <= 1e-9


def test_e2_bid_ask_symmetry(collapsed):
    # With carry = 0 the problem is exactly symmetric under Vpi -> -Vpi with
    # bid <-> ask. Measured: v-flip 0.0 abs, quote mirror 1.3e-16.
    spec, sol = collapsed
    v = np.asarray(sol.v0)
    assert np.abs(v - v[:, ::-1]).max() <= 1e-9 * np.abs(v).max()
    g = np.asarray(make_ops(spec).gate)
    q = np.where(g[None], np.asarray(sol.quotes[0]), np.nan)
    mirror = np.abs(q[:, :, :, 0] - q[:, ::-1, :, 1])
    assert np.nanmax(mirror) <= 1e-9


def test_e2_independent_1d_reference(collapsed):
    """Remark 6's 1-D equation, solved by an independent numpy recursion
    (np.interp for the jump shifts, its own gate and loop), must reproduce
    any nu-row of the 2-D solve."""
    spec, sol = collapsed
    book, r, tg = spec.book, spec.risk, spec.time_grid
    V = np.linspace(-r.Vbar, r.Vbar, spec.vpi_grid.n)
    zV, z = np.asarray(book.zV), np.asarray(book.z)
    tgt = V[None, None, :] + np.array([1.0, -1.0])[None, :, None] * zV[:, None, None]
    gate = (np.abs(tgt) <= r.Vbar * (1 + 1e-12))              # (N, 2, n_V)
    gate_t = np.transpose(gate, (2, 1, 0))                     # (n_V, 2, N)
    pen = float(r.gamma) * spec.market.xi ** 2 / 8.0 * r.c_rho * V ** 2
    tgt_c = np.clip(tgt, -r.Vbar, r.Vbar)
    w = np.zeros(V.size)
    for _ in range(tg.M):
        v_post = np.stack([[np.interp(tgt_c[i, s], V, w) for s in (0, 1)]
                           for i in range(zV.size)])           # (N, 2, n_V)
        p = (w[None, None, :] - v_post) / z[:, None, None]
        Hv = np.asarray(mf.H(book, jnp.asarray(np.transpose(p, (2, 1, 0)))))
        jump = (gate_t * z[None, None, :] * Hv).sum(axis=(1, 2))
        w = w + tg.dt * (jump - pen)
    v_row = np.asarray(sol.v0)[0]
    assert np.abs(w - v_row).max() <= 1e-6 * np.abs(w).max()


def test_body_asymmetry_is_carry(body):
    # With a^P != a^Q the carry Vpi (a^P - a^Q)/(2 sqrt(nu)) is odd in Vpi and
    # breaks the symmetry by O(carry * T) ~ 1.4e3 EUR at the vega limit.
    # Measured max asymmetry 549 EUR; long vega carries positively.
    spec, sol = body
    v = np.asarray(sol.v0)
    asym = np.abs(v - v[:, ::-1]).max()
    assert 10.0 < asym < 2.5e3
    ops = make_ops(spec)
    assert (float(at_state(sol.v0, ops, NU0, +5e6))
            > float(at_state(sol.v0, ops, NU0, -5e6)))


def test_dt_refinement_first_order():
    # Explicit Euler: error ratio ~ 2 under dt halving. Measured 2.03.
    from mfoptmm.config import TimeGrid
    base = mf.bbg_example(variant="body", profile="replication")
    c = {}
    for M in (180, 360, 720):
        spec = base._replace(time_grid=TimeGrid(T=0.0012, M=M, K_blocks=36))
        sol = mf.hjb_solve(spec)
        c[M] = float(at_state(sol.v0, make_ops(spec), NU0, 0.0))
    ratio = (c[180] - c[360]) / (c[360] - c[720])
    assert 1.5 < ratio < 2.7, f"first-order contract broken: ratio {ratio:.3f}"


def test_value_ordering_a1_above_body(body, a1):
    # A.1's penalty is 0.75x the body's with identical dynamics and revenue,
    # so v_a1 >= v_body pointwise (comparison principle). Measured min gap
    # +5.85e3 EUR.
    _, sb = body
    _, sa = a1
    assert float((sa.v0 - sb.v0).min()) > 0.0

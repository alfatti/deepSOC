"""Structural identities of the A.1 problem definition."""
import numpy as np
import jax
import jax.numpy as jnp

from mfoptmm import (
    bbg_example, a_P, a_Q, carry, penalty_reduced, penalty_hedge, eta_star,
    post_trade_ok,
)

SPEC = bbg_example(variant="a1")
M, R = SPEC.market, SPEC.risk
VPI = jnp.linspace(-1e7, 1e7, 41)


def test_c_rho_value():
    assert np.isclose(R.c_rho, 1.0 - M.rho ** 2)
    assert np.isclose(R.c_rho, 0.75)


def test_hedge_penalty_minimum_recovers_reduced_penalty():
    """min_eta penalty_hedge = penalty_reduced with c_rho = 1 - rho^2 —
    the algebra behind the A.1 reduction, verified numerically."""
    at_star = penalty_hedge(M, R, VPI, eta_star(M, VPI))
    reduced = penalty_reduced(M, R, VPI)
    rel = np.abs(np.asarray(at_star - reduced)) / np.maximum(np.asarray(reduced), 1e-30)
    assert rel[np.asarray(VPI) != 0].max() < 1e-12


def test_eta_star_is_stationary_point():
    g = jax.vmap(jax.grad(lambda e, v: penalty_hedge(M, R, v, e)), (0, 0))
    grads = np.asarray(g(eta_star(M, VPI), VPI))
    scale = float(R.gamma) * np.maximum(np.abs(np.asarray(VPI)), 1.0)
    assert (np.abs(grads) / scale).max() < 1e-12


def test_eta_star_slope():
    # eta* = -(rho xi / 2) Vpi; with rho = -0.5, xi = 0.2 the slope is +0.05
    slope = float(eta_star(M, 1.0) - eta_star(M, 0.0))
    assert np.isclose(slope, -0.5 * M.rho * M.xi) and np.isclose(slope, 0.05)


def test_drift_and_carry_values_at_nu0():
    gap = float(a_P(M, M.nu0) - a_Q(M, M.nu0))
    assert np.isclose(gap, 0.035)  # 2*(0.04-0.0225) - 3*0 at nu0 = thetaQ
    c = float(carry(M, M.nu0, 1.0))
    assert np.isclose(c, 0.035 / (2 * np.sqrt(0.0225)))  # = 0.11666.. per yr


def test_penalty_scale_sanity():
    # (gamma xi^2 / 8) * 0.75 * Vbar^2 * T: the max running penalty over the
    # horizon, in EUR — fixes the magnitude the objective lives at.
    val = float(penalty_reduced(M, R, R.Vbar) * SPEC.time_grid.T)
    assert np.isclose(val, 1e-3 * 0.04 / 8 * 0.75 * 1e14 * 0.0012)
    assert 4.0e5 < val < 5.0e5


def test_gate():
    ok = np.asarray(post_trade_ok(R, jnp.asarray([-1.1e7, -1e7, 0.0, 1e7, 1.1e7])))
    assert (ok == np.array([False, True, True, True, False])).all()

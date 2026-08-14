"""Milestone-4 gates: the BCR trainer's machinery, cheaply and decisively.

All runs are seed-fixed, so measured values are deterministic on a given
JAX/CPU stack; thresholds carry generous margins against version drift.
Measured (r=32, n=32, K=4, 300-400 epochs, straight-through fills):

* eta-aware arbiter identity: |u(eta = eta*) - u(reduced)| = 2.9e-11.
* soft-mode gradient vs central FD: 8.9e-7 relative.
* last-block eta head recovers the closed-form hedge line eta* = +0.05 Vpi:
  slope +0.0440, corr +0.999 after 400 epochs.
* full backward induction captures 32.5% of (v* - J_ref) at smoke scale,
  respects dominance (max(u_learned - v*) = -5.7e4), and the generalisation
  gap is alive: J_train - J_fresh = +8,418 EUR at n = 32.

Two training-dynamics findings recorded in the README: the loss must be
normalised (objective_scale, the myopic revenue bound) or per-neuron drift
r * grad explodes at any sane lr; and the canonical-MFLD noise sqrt(2 lr)/beta
replaces the Overlearning repo's sigma = sqrt(0.1) * beta, which is ~20x
hotter than the beta = 100 stationary measure warrants.
"""
import numpy as np
import jax
import jax.numpy as jnp
import pytest

import mfoptmm as mf
from mfoptmm.hjb import make_ops, at_state
from mfoptmm.model import eta_star
from mfoptmm.policy_eval import policy_value, myopic_policy

NU0 = 0.0225
KEY = jax.random.PRNGKey(0)


@pytest.fixture(scope="module")
def a1_k4():
    spec = mf.bbg_example(variant="a1", K_blocks=4, profile="replication")
    return spec, mf.hjb_solve(spec, store_quotes=True)


def test_eta_identity_in_arbiter(a1_k4):
    spec, sol = a1_k4
    ops = make_ops(spec)
    eta_grid = jnp.broadcast_to(
        eta_star(spec.market, ops.V)[None, None, :],
        (spec.time_grid.M, ops.nu.size, ops.V.size))
    u_red = policy_value(spec, sol.quotes)
    u_eta = policy_value(spec, sol.quotes, eta=eta_grid)
    scale = float(jnp.abs(u_red.u0).max())
    assert float(jnp.abs(u_eta.u0 - u_red.u0).max()) <= 1e-8 * scale


def test_gradient_matches_fd_in_soft_mode(a1_k4):
    spec, _ = a1_k4
    cm = mf.control_maps(spec)
    dk = jax.random.fold_in(KEY, 1)
    stack = mf.init_stack(jax.random.fold_in(KEY, 2), 4, 16)
    th = jax.tree_util.tree_map(lambda f: f[3], stack)
    starts = mf.rollin_states(spec, dk, 8, 3 * spec.time_grid.substeps_per_block)
    f = lambda t: mf.rollout_value(t, 3, stack, cm, spec, starts, dk, 0.01)
    g = float(jax.tree_util.tree_leaves(jax.grad(f)(th))[0][0, 0])
    flat, td = jax.tree_util.tree_flatten(th)

    def perturb(s):
        l = [x.copy() for x in flat]
        l[0] = l[0].at[0, 0].add(s)
        return jax.tree_util.tree_unflatten(td, l)

    eps = 1e-4
    fd = float((f(perturb(eps)) - f(perturb(-eps))) / (2 * eps))
    assert abs(g - fd) / max(abs(fd), 1e-12) < 1e-4
    # straight-through mode: gradients finite and nonzero
    g_st = jax.grad(lambda t: mf.rollout_value(t, 3, stack, cm, spec, starts,
                                               dk, 0.0))(th)
    leaves = jax.tree_util.tree_leaves(g_st)
    assert all(bool(jnp.isfinite(x).all()) for x in leaves)
    assert sum(float(jnp.abs(x).sum()) for x in leaves) > 0.0


def test_last_block_recovers_eta_line(a1_k4):
    # eta* = -(rho xi / 2) Vpi = +0.05 Vpi (rho = -0.5): the eta head's
    # optimum is pointwise and closed-form. Measured slope +0.0440, corr 0.999.
    spec, _ = a1_k4
    cm = mf.control_maps(spec)
    cfg = mf.TrainConfig(r=32, n=32, epochs=400, lr0=0.3, lr1=3e-3, seed=0)
    stack, hist = mf.train_block(
        3, mf.init_stack(jax.random.fold_in(KEY, 2), 4, cfg.r), spec, cm,
        cfg, jax.random.fold_in(KEY, 1), jax.random.fold_in(KEY, 103))
    assert float(hist[-1]) > float(hist[0])          # 6,926 -> 16,787
    net3 = jax.tree_util.tree_map(lambda f: f[3], stack)
    Vs = jnp.linspace(-0.5e7, 0.5e7, 21)
    _, eta = mf.controls(net3, cm, jnp.full(21, NU0), Vs)
    slope = float(np.polyfit(np.asarray(Vs), np.asarray(eta), 1)[0])
    corr = float(np.corrcoef(np.asarray(Vs), np.asarray(eta))[0, 1])
    assert 0.03 < slope < 0.07, f"eta slope {slope:.4f} vs target +0.05"
    assert corr > 0.98


def test_smoke_backward_induction(a1_k4):
    spec, sol = a1_k4
    ops = make_ops(spec)
    cfg = mf.TrainConfig(r=32, n=32, epochs=300, lr0=0.3, lr1=3e-3, seed=0)
    pol = mf.train(spec, cfg)
    assert bool(jnp.isfinite(pol.history).all())
    qs, es = mf.materialise_schedule(pol.stack, pol.cm, spec)
    uL = policy_value(spec, qs, eta=es)
    u_ref = policy_value(spec, myopic_policy(spec),
                         eta=jnp.zeros((spec.time_grid.M, ops.nu.size,
                                        ops.V.size)))
    jl = float(at_state(uL.u0, ops, NU0, 0.0))
    jr = float(at_state(u_ref.u0, ops, NU0, 0.0))
    va1 = float(at_state(sol.v0, ops, NU0, 0.0))
    captured = (jl - jr) / (va1 - jr)                # measured 0.325
    assert captured > 0.20, f"smoke captured only {captured:.2%}"
    assert float((uL.u0 - sol.v0).max()) < 1e-6      # dominance
    # generalisation gap exists and points the right way (measured +8,418)
    dk = jax.random.fold_in(jax.random.PRNGKey(cfg.seed), 1)
    jt = float(mf.full_objective(pol.stack, pol.cm, spec, dk, cfg.n))
    jf = float(mf.full_objective(pol.stack, pol.cm, spec,
                                 jax.random.PRNGKey(777), 512))
    assert jt - jf > 1e3


def test_erm_determinism(a1_k4):
    spec, _ = a1_k4
    cfg = mf.TrainConfig(r=8, n=8, epochs=3, seed=0)
    cm = mf.control_maps(spec)
    args = (3, mf.init_stack(jax.random.fold_in(KEY, 2), 4, cfg.r), spec, cm,
            cfg, jax.random.fold_in(KEY, 1), jax.random.fold_in(KEY, 103))
    s1, h1 = mf.train_block(*args)
    s2, h2 = mf.train_block(*args)
    for x, y in zip(jax.tree_util.tree_leaves(s1),
                    jax.tree_util.tree_leaves(s2)):
        assert float(jnp.abs(x - y).max()) == 0.0
    assert float(jnp.abs(h1 - h2).max()) == 0.0

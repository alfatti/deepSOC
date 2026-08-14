"""Linear policy evaluation: the exact arbiter for any feedback quote policy.

For a given quote schedule delta_bar (M, n_nu, n_V, N, 2) -- a map per fine
step, evaluated at the state each substep, so still Markov feedback -- solve
the *linear* backward equation on the identical stencil as the HJB:

  u^m = u^{m+1} + dt * [ L_nu u + carry - penalty
        + sum_{i,s} gate Lambda^i(delta_bar) ( z^i delta_bar + u_shift - u ) ],
  u^M = 0.

Because the stencil, interpolation, gate and source are shared with hjb.py,
two exact identities hold and are gate-tested:

* plugging the HJB's own per-step optimal quotes reproduces v to machine
  precision (envelope identity z H(p) = Lambda(d*) (z d* + v_shift - v));
* any schedule prices <= v pointwise (monotone comparison), provided the
  evaluator's own realised diagonal stays >= 0 (returned for asserting).

Schedules are materialised as full (M, ...) arrays -- 69 MB at the BBG grid,
fine on CPU. The H200 lane will switch to index-based schedules to avoid the
repeat; keep the seam in `repeat_block_start`.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ProblemSpec
from .hjb import Ops, make_ops, _l_nu, _v_shift
from .model import intensity, myopic_quotes, penalty_hedge, penalty_reduced


class PolicyValue(NamedTuple):
    u0: jnp.ndarray          # (n_nu, n_V)
    min_diag: jnp.ndarray    # (M,) forward-time realised diagonal of the
                             # *evaluator* (Lambda at the given quotes)


@jax.jit
def _eval_scan(market, ops: Ops, quotes_rev: jnp.ndarray):
    dt = ops.dt

    def step(u, q):
        lam = intensity(ops.book2, q)
        u_shift = _v_shift(ops, u)
        jump = jnp.sum(
            jnp.where(ops.gate[None],
                      lam * (ops.z * q + u_shift - u[:, :, None, None]), 0.0),
            axis=(2, 3))
        u_new = u + dt * (_l_nu(market, ops, u) + ops.source + jump)
        lam_diag = jnp.sum(
            jnp.where(ops.gate[None], (1.0 - ops.w_self) * lam, 0.0),
            axis=(2, 3))
        diag = 1.0 - dt * (ops.diff_diag[:, None] + ops.drift_diag[:, None]
                           + lam_diag)
        return u_new, jnp.min(diag)

    u_T = jnp.zeros((ops.nu.size, ops.V.size))
    u0, mind = jax.lax.scan(step, u_T, quotes_rev)
    return u0, mind


@jax.jit
def _eval_scan_eta(market, ops: Ops, spec_risk, quotes_rev, eta_rev,
                   pen_red_row):
    dt = ops.dt

    def step(u, x):
        q, eta = x
        lam = intensity(ops.book2, q)
        u_shift = _v_shift(ops, u)
        jump = jnp.sum(
            jnp.where(ops.gate[None],
                      lam * (ops.z * q + u_shift - u[:, :, None, None]), 0.0),
            axis=(2, 3))
        src = ops.source + pen_red_row[None, :] \
            - penalty_hedge(market, spec_risk, ops.V[None, :], eta)
        u_new = u + dt * (_l_nu(market, ops, u) + src + jump)
        lam_diag = jnp.sum(
            jnp.where(ops.gate[None], (1.0 - ops.w_self) * lam, 0.0),
            axis=(2, 3))
        diag = 1.0 - dt * (ops.diff_diag[:, None] + ops.drift_diag[:, None]
                           + lam_diag)
        return u_new, jnp.min(diag)

    u_T = jnp.zeros((ops.nu.size, ops.V.size))
    return jax.lax.scan(step, u_T, (quotes_rev, eta_rev))


def policy_value(spec: ProblemSpec, quotes, eta=None) -> PolicyValue:
    """quotes: (M, n_nu, n_V, N, 2), forward-time indexed (quotes[m] applies
    on [t_m, t_{m+1})).

    eta: optional hedge schedule (M, n_nu, n_V). When given, the running
    penalty becomes the hedge-channel quadratic penalty_hedge(Vpi, eta) —
    the exact arbiter for learned (delta, eta) controls. eta = eta*(Vpi)
    reproduces the reduced-penalty value identically (A.1 identity,
    gate-tested)."""
    ops = make_ops(spec)
    if quotes.shape[0] != spec.time_grid.M:
        raise ValueError(f"schedule has {quotes.shape[0]} steps, "
                         f"grid has M={spec.time_grid.M}")
    if eta is None:
        u0, mind = _eval_scan(spec.market, ops, jnp.flip(quotes, axis=0))
    else:
        pen_red = penalty_reduced(spec.market, spec.risk, ops.V)
        u0, mind = _eval_scan_eta(spec.market, ops, spec.risk,
                                  jnp.flip(quotes, axis=0),
                                  jnp.flip(eta, axis=0), pen_red)
    return PolicyValue(u0=u0, min_diag=jnp.flip(mind, 0))


# ----------------------------------------------------------------------------
# Schedule builders
# ----------------------------------------------------------------------------

def repeat_block_start(quotes, substeps: int):
    """Zero-order-hold restriction: freeze, for every step of block k, the map
    from the block's first (forward-time) step, m = k * substeps. This is the
    policy class the BCR trainer lives in (one map per decision epoch,
    re-read at the state every substep)."""
    M = quotes.shape[0]
    if M % substeps != 0:
        raise ValueError(f"M={M} not a multiple of substeps={substeps}")
    starts = quotes[::substeps]
    return jnp.repeat(starts, substeps, axis=0)


def constant_policy(spec: ProblemSpec, delta_per_option) -> jnp.ndarray:
    """Broadcast a per-(option, side) quote array (N, 2) -- or (N,), applied
    to both sides -- to a full state-independent schedule."""
    d = jnp.asarray(delta_per_option)
    if d.ndim == 1:
        d = jnp.stack([d, d], axis=-1)
    tg, nug, vg = spec.time_grid, spec.nu_grid, spec.vpi_grid
    return jnp.broadcast_to(d[None, None, None],
                            (tg.M, nug.n, vg.n) + d.shape)


def myopic_policy(spec: ProblemSpec) -> jnp.ndarray:
    """The reference policy: state-independent revenue-maximising quotes."""
    return constant_policy(spec, myopic_quotes(spec.book))

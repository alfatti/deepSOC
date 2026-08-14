"""Lockstep CRN Monte-Carlo simulator for quote-schedule policies.

Forward Euler MDP on the HJB's fine time grid. Common random numbers come
from counter-based keys (fold_in per step; nothing materialised), so any two
policies simulated with the same base key see identical Brownian increments
and fill uniforms — the hjbpinn lockstep pattern, GPU-ready.

Step at t_m, state (nu, Vpi), quote map delta_m from the schedule (bilinear
in (nu, Vpi) at the state, edge-clamped):

    reward_m  = dt [ sum_{i,s} gate z^i delta Lambda(delta) + carry - pen ]   (compensated)
                or   sum_{i,s} fill z^i delta + dt [ carry - pen ]            (realised)
    fill^{i,s} = gate * 1{ U < Lambda(delta) dt }                              (hard)
               = gate * sigmoid( (Lambda(delta) dt - U) / tau )                (soft, same U)
    Vpi_{m+1} = Vpi + sum_{i,s} fill * s * z^i V^i
    nu_{m+1}  = nu + a^P(nu) dt + xi sqrt(nu) dW.

The compensated estimator books the objective's own compensator form (its
revenue term is z delta Lambda dt in (2)), so fills only drive the *state*
and the estimator variance is an order of magnitude below realised booking;
both have the same mean.

Semantics deliberately mirror the policy-evaluation PDE except for O(dt)
discretisation effects: multi-fire steps (independent per-channel Bernoullis
can fire together; simultaneous fills may overshoot Vbar, so the reachable
set is [-Vbar - max zV, Vbar + max zV]), Euler nu vs the generator, and
off-grid policy interpolation. Gate E5 measures the total residual against
the exact evaluator under known policies and pins it with a dt-refinement
contract (`refine_schedule` holds the maps fixed while halving dt).

Per-channel Bernoulli fidelity: lam_max * dt = 0.05 on the M = 180 grid
(asserted), so single-channel thinning error is O((lam dt)^2) ~ 2.5e-3 per
channel-step; the aggregate is what E5 sees.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .config import ProblemSpec
from .hjb import Ops, make_ops
from .model import a_P, carry, intensity, penalty_reduced


class SimResult(NamedTuple):
    objective: jnp.ndarray   # (n_paths,) total objective per path
    Vpi_T: jnp.ndarray       # (n_paths,) terminal portfolio vega
    fills: jnp.ndarray       # (n_paths,) total fills (fractional under soft)


def _lookup(quotes_m, ops: Ops, nu, Vpi):
    """Bilinear, edge-clamped lookup of a quote map (n_nu, n_V, N, 2) at
    batched states nu, Vpi (B,) -> (B, N, 2)."""
    xn = jnp.clip((nu - ops.nu[0]) / ops.dnu, 0.0, ops.nu.size - 1.0)
    xv = jnp.clip((Vpi - ops.V[0]) / ops.dV, 0.0, ops.V.size - 1.0)
    i0 = jnp.clip(jnp.floor(xn).astype(jnp.int32), 0, ops.nu.size - 2)
    j0 = jnp.clip(jnp.floor(xv).astype(jnp.int32), 0, ops.V.size - 2)
    fn = (xn - i0)[:, None, None]
    fv = (xv - j0)[:, None, None]
    q00 = quotes_m[i0, j0]
    q01 = quotes_m[i0, j0 + 1]
    q10 = quotes_m[i0 + 1, j0]
    q11 = quotes_m[i0 + 1, j0 + 1]
    return ((1 - fn) * ((1 - fv) * q00 + fv * q01)
            + fn * ((1 - fv) * q10 + fv * q11))


def simulate_schedule(spec: ProblemSpec, quotes, key, n_paths: int,
                      Vpi0, nu0: float | None = None, tau: float = 0.0,
                      realized: bool = False) -> SimResult:
    """Simulate a quote schedule (M, n_nu, n_V, N, 2) under CRN.

    Vpi0: scalar or (n_paths,) initial vega. tau = 0.0 -> hard fills;
    tau > 0 -> sigmoid-relaxed fills sharing the same uniforms (as tau -> 0
    the soft path converges to the hard path realisation-by-realisation).
    """
    ops = make_ops(spec)
    m_, book, r, tg = spec.market, spec.book, spec.risk, spec.time_grid
    if quotes.shape[0] != tg.M:
        raise ValueError(f"schedule has {quotes.shape[0]} steps, M={tg.M}")
    lam_dt = float(jnp.max(book.lam)) * tg.dt
    if lam_dt >= 1.0:
        raise ValueError(f"lam_max * dt = {lam_dt:.3f} >= 1: refine the grid")
    dt = tg.dt
    sides = jnp.asarray([1.0, -1.0])
    szV = book.zV[:, None] * sides[None, :]          # (N, 2) signed vega print
    z_r = book.z[None, :, None]                       # (1, N, 1)
    nu_init = jnp.full((n_paths,), m_.nu0 if nu0 is None else nu0)
    Vpi_init = jnp.broadcast_to(jnp.asarray(Vpi0, dtype=jnp.float64),
                                (n_paths,)).astype(jnp.float64)

    def step(state, xs):
        nu, Vpi, obj, nf = state
        q_m, m = xs
        k = jax.random.fold_in(key, m)
        kn, ku = jax.random.split(k)
        dW = jnp.sqrt(dt) * jax.random.normal(kn, (n_paths,))
        U = jax.random.uniform(ku, (n_paths, book.zV.size, 2))
        delta = _lookup(q_m, ops, nu, Vpi)
        gate = (jnp.abs(Vpi[:, None, None] + szV[None])
                <= r.Vbar * (1.0 + 1e-12)).astype(jnp.float64)
        lam = intensity(ops.book2, delta)             # (B, N, 2)
        if tau == 0.0:
            fill = gate * (U < lam * dt)
        else:
            fill = gate * jax.nn.sigmoid((lam * dt - U) / tau)
        if realized:
            revenue = (fill * z_r * delta).sum(axis=(1, 2))
        else:
            revenue = dt * (gate * z_r * delta * lam).sum(axis=(1, 2))
        obj = obj + revenue + dt * (carry(m_, nu, Vpi)
                                    - penalty_reduced(m_, r, Vpi))
        Vpi = Vpi + (fill * szV[None]).sum(axis=(1, 2))
        nu = nu + a_P(m_, nu) * dt + m_.xi * jnp.sqrt(jnp.maximum(nu, 1e-12)) * dW
        return (nu, Vpi, obj, nf + fill.sum(axis=(1, 2))), None

    def run(q):
        init = (nu_init, Vpi_init, jnp.zeros(n_paths), jnp.zeros(n_paths))
        (nu, Vpi, obj, nf), _ = jax.lax.scan(
            step, init, (q, jnp.arange(tg.M)))
        return obj, Vpi, nf

    obj, Vpi_T, nf = jax.jit(run)(quotes)
    return SimResult(objective=obj, Vpi_T=Vpi_T, fills=nf)


def refine_schedule(quotes, factor: int):
    """Zero-order hold of the same maps on a factor-x finer time grid: the
    priced policy is unchanged, only the simulator's dt shrinks — the E5
    refinement contract isolates pure simulator discretisation."""
    return jnp.repeat(quotes, factor, axis=0)


def refined_spec(spec: ProblemSpec, factor: int) -> ProblemSpec:
    from .config import TimeGrid
    tg = spec.time_grid
    return spec._replace(time_grid=TimeGrid(T=tg.T, M=tg.M * factor,
                                            K_blocks=tg.K_blocks))


def draw_vpi0(key, n_paths: int, Vbar: float, width: float = 0.5):
    """Randomised exploration starts (Remark 8): Vpi0 ~ U[-w Vbar, w Vbar]."""
    return jax.random.uniform(key, (n_paths,), minval=-width * Vbar,
                              maxval=width * Vbar)

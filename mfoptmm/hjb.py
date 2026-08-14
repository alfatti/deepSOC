"""Exact solver for the reduced HJB on (t, nu, Vpi), with quote extraction.

Scheme: monotone explicit Euler backward from v(T) = 0, linear interpolation
in Vpi for the RFQ jumps (BBG Section 4.2's scheme), one code path for body
and A.1 via Risk.c_rho:

  v^m = v^{m+1} + dt * [ a^P d_nu v (upwind)  +  0.5 nu xi^2 d2_nu v (Neumann)
                         + carry(nu, Vpi) - penalty(Vpi)
                         + sum_{i,s} gate^{i,s} z^i H^i(p^{i,s}) ],
  p^{i,s} = ( v - v(., Vpi + s zV^i) ) / z^i,     s = +1 bid, -1 ask,
  gate^{i,s} = 1{ |Vpi + s zV^i| <= Vbar }.

Side axis convention everywhere in this repo: index 0 = bid (s = +1, MM buys,
vega jumps up), index 1 = ask (s = -1).

Monotonicity (Gate E4). The update's diagonal coefficient at each state is

  D = 1 - dt * ( diff_diag(nu) + drift_diag(nu)
                 + sum_{i,s} gate (1 - w_self^{i,s}) Lambda^i(delta*^i(p)) )

where w_self is the interpolation weight the jump target places back on the
origin node -- substantial for jumps smaller than one Vpi cell, so the
realised diagonal is *better* than the naive 1 - dt (rates + sum Lambda*).
The solver returns min D per step; this is the empirical licence for BBG's
M = 180 grid, which the worst-case bound (config.stability_rates) rejects.
Gate E4 is asserted at runtime: hjb_solve(check=True) syncs the per-step
minima to host after the scan and raises on a violation (the scan itself
stays jit-pure); check=False opts out for hot GPU lanes that assert later.
"""
from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from .config import ProblemSpec
from .model import H, H_p, a_P, carry, delta_star, penalty_reduced


class Ops(NamedTuple):
    """Precomputed grid geometry and stencils. Static per ProblemSpec."""

    nu: jnp.ndarray          # (n_nu,)
    V: jnp.ndarray           # (n_V,)
    dnu: float
    dV: float
    dt: float
    book2: object            # Book with lam, b reshaped (N, 1) for (..., N, 2)
    z: jnp.ndarray           # (1, 1, N, 1)
    idx0: jnp.ndarray        # (n_V, N, 2) int, lower interp node of jump target
    frac: jnp.ndarray        # (n_V, N, 2) interp weight on idx0 + 1 (post-clip)
    w_self: jnp.ndarray      # (N, 2) weight the jump target puts on the origin
    gate: jnp.ndarray        # (n_V, N, 2) bool, post-trade |Vpi'| <= Vbar
    diff_diag: jnp.ndarray   # (n_nu,) diagonal rate of the diffusion stencil
    drift_diag: jnp.ndarray  # (n_nu,) diagonal rate of the upwind stencil
    source: jnp.ndarray      # (n_nu, n_V) carry - penalty


def make_ops(spec: ProblemSpec) -> Ops:
    m, book, r = spec.market, spec.book, spec.risk
    nug, vg, tg = spec.nu_grid, spec.vpi_grid, spec.time_grid
    nu = jnp.linspace(nug.nu_min, nug.nu_max, nug.n)
    V = jnp.linspace(-r.Vbar, r.Vbar, vg.n)
    dnu, dV, dt = nug.dnu, V[1] - V[0], tg.dt   # dV trace-safe

    book2 = book._replace(lam=book.lam[:, None], b=book.b[:, None])
    z = book.z[None, None, :, None]

    sides = jnp.asarray([1.0, -1.0])                       # (2,)
    shift = sides[None, :] * book.zV[:, None] / dV         # (N, 2) in cells
    w_self = jnp.maximum(0.0, 1.0 - jnp.abs(shift))        # (N, 2)
    k = jnp.arange(vg.n)[:, None, None]                    # (n_V, 1, 1)
    pos = k + shift[None]                                  # (n_V, N, 2) in cells
    idx0 = jnp.clip(jnp.floor(pos), 0, vg.n - 2).astype(jnp.int32)
    # weight from the *clipped* index: exact when a target lands on the top
    # node (pos = n_V - 1 -> idx0 = n_V - 2, frac = 1), where the pre-clip
    # fractional part would wrongly read the second-to-last node. Off-domain
    # positions saturate to an edge read; the gate masks those terms anyway.
    frac = jnp.clip(pos - idx0, 0.0, 1.0)                  # (n_V, N, 2)
    Vpost = V[:, None, None] + sides[None, None, :] * book.zV[None, :, None]
    gate = jnp.abs(Vpost) <= r.Vbar * (1.0 + 1e-12)        # (n_V, N, 2)

    cc = 0.5 * nu * m.xi ** 2 / dnu ** 2                   # (n_nu,)
    diff_diag = 2.0 * cc
    diff_diag = diff_diag.at[0].set(cc[0]).at[-1].set(cc[-1])
    a = a_P(m, nu)
    drift_diag = jnp.abs(a) / dnu
    drift_diag = drift_diag.at[-1].set(jnp.where(a[-1] > 0, 0.0, drift_diag[-1]))
    drift_diag = drift_diag.at[0].set(jnp.where(a[0] < 0, 0.0, drift_diag[0]))

    source = carry(m, nu[:, None], V[None, :]) \
        - penalty_reduced(m, r, V)[None, :]
    return Ops(nu=nu, V=V, dnu=dnu, dV=dV, dt=dt, book2=book2, z=z,
               idx0=idx0, frac=frac, w_self=w_self, gate=gate,
               diff_diag=diff_diag, drift_diag=drift_diag, source=source)


# ----------------------------------------------------------------------------
# Shared stencil pieces
# ----------------------------------------------------------------------------

def _l_nu(m, ops: Ops, v):
    """Upwind drift + central diffusion in nu, homogeneous Neumann via edge
    clamping (ghost node = edge node). `m` is the Market NamedTuple (pytree);
    ProblemSpec itself contains plain dataclasses and must stay out of jit."""
    vp = jnp.concatenate([v[:1], v, v[-1:]], axis=0)
    d2 = (vp[2:] - 2.0 * v + vp[:-2]) / ops.dnu ** 2
    diff = 0.5 * ops.nu[:, None] * m.xi ** 2 * d2
    a = a_P(m, ops.nu)[:, None]
    fwd = (vp[2:] - v) / ops.dnu
    bwd = (v - vp[:-2]) / ops.dnu
    drift = jnp.maximum(a, 0.0) * fwd + jnp.minimum(a, 0.0) * bwd
    return diff + drift


def _v_shift(ops: Ops, v):
    """v at the post-trade vega, linearly interpolated: (n_nu, n_V, N, 2)."""
    lo = v[:, ops.idx0]                       # (n_nu, n_V, N, 2)
    hi = v[:, ops.idx0 + 1]
    return (1.0 - ops.frac) * lo + ops.frac * hi


def _p(ops: Ops, v):
    """Per-unit reservation spread p^{i,s} = (v - v_shift)/z^i."""
    return (v[:, :, None, None] - _v_shift(ops, v)) / ops.z


# ----------------------------------------------------------------------------
# HJB solve
# ----------------------------------------------------------------------------

class HJBSolution(NamedTuple):
    v0: jnp.ndarray            # (n_nu, n_V) value at t = 0
    quotes: jnp.ndarray | None  # (M, n_nu, n_V, N, 2), forward-time indexed:
                                # quotes[m] is the map used on [t_m, t_{m+1})
    min_diag: jnp.ndarray      # (M,) forward-time min realised diagonal
    lam_dt_max: jnp.ndarray    # (M,) max_state dt * sum_{i,s} gate Lambda*


@functools.partial(jax.jit, static_argnums=(2, 3))
def _hjb_scan(market, ops: Ops, M: int, store_quotes: bool):
    dt = ops.dt

    def step(v, _):
        p = _p(ops, v)
        lam_star = -H_p(ops.book2, p)              # = Lambda(delta*(p)) >= 0
        ham = jnp.sum(jnp.where(ops.gate[None], ops.z * H(ops.book2, p), 0.0),
                      axis=(2, 3))
        v_new = v + dt * (_l_nu(market, ops, v) + ops.source + ham)
        ham_diag = jnp.sum(
            jnp.where(ops.gate[None], (1.0 - ops.w_self) * lam_star, 0.0),
            axis=(2, 3))
        diag = 1.0 - dt * (ops.diff_diag[:, None] + ops.drift_diag[:, None]
                           + ham_diag)
        lam_sum = jnp.sum(jnp.where(ops.gate[None], lam_star, 0.0),
                          axis=(2, 3))
        out = (jnp.min(diag), jnp.max(dt * lam_sum))
        if store_quotes:
            out = out + (delta_star(ops.book2, p),)
        return v_new, out

    v_T = jnp.zeros((ops.nu.size, ops.V.size))
    v0, ys = jax.lax.scan(step, v_T, None, length=M)
    return v0, ys


def hjb_solve(spec: ProblemSpec, store_quotes: bool = False,
              check: bool = True) -> HJBSolution:
    ops = make_ops(spec)
    v0, ys = _hjb_scan(spec.market, ops, spec.time_grid.M, store_quotes)
    if store_quotes:
        mind, lamdt, quotes = ys
        quotes = jnp.flip(quotes, axis=0)   # scan runs m = M-1 -> 0
    else:
        mind, lamdt = ys
        quotes = None
    if check:
        worst = float(jnp.min(mind))        # Gate E4, runtime
        if worst < 0.0:
            raise RuntimeError(
                f"Gate E4: realised scheme diagonal reached {worst:.4f} < 0 "
                f"(M={spec.time_grid.M}); rerun with profile='worst_case' or "
                f"a larger force_M.")
    return HJBSolution(v0=v0, quotes=quotes,
                       min_diag=jnp.flip(mind, 0),
                       lam_dt_max=jnp.flip(lamdt, 0))


# ----------------------------------------------------------------------------
# Interpolation helpers (evaluation at off-grid nu / Vpi)
# ----------------------------------------------------------------------------

def interp_axis(field, grid, x, axis: int):
    """Linear interpolation of `field` along `axis` at scalar x."""
    grid = np.asarray(grid)
    q = (x - grid[0]) / (grid[1] - grid[0])
    k = int(np.clip(np.floor(q), 0, grid.size - 2))
    w = q - k
    lo = jnp.take(field, k, axis=axis)
    hi = jnp.take(field, k + 1, axis=axis)
    return (1.0 - w) * lo + w * hi


def at_state(field, ops: Ops, nu0: float, V0: float):
    """field (n_nu, n_V, ...) evaluated at (nu0, V0) by bilinear interpolation."""
    return interp_axis(interp_axis(field, ops.nu, nu0, 0), ops.V, V0, 0)

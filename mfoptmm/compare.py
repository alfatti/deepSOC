"""Levels 1-4 comparison stack (strategy doc section 5).

Level 0 (identities: Gate E5, tau study) lives in the test suite. This module
provides the trial runner, references, metrics and figure builders shared by
the notebooks and tests/run_h200.py:

* Level 1 — generalisation gap |J_train - J_fresh| per (n, r, trial, arm);
  boxplots vs n with the n^{-1} reference line.
* Level 2 — optimality gap v* - J(learned) via (a) fresh hard-fill MC and
  (b) the exact policy-evaluation PDE at n_V = 161 (the milestone-3 finding:
  the 40-node arbiter's policy-dependent spatial error would swamp learned
  gaps). The reference control's gap locates the starting line.
* Level 3 — policy space: learned vs exact quote curves and the eta line;
  per-option errors both raw-sup and visitation-weighted L2 (weights = the
  Vpi occupation measure of the learned policy under fresh noise — sup over
  never-visited states is exactly what the BCR bound does not control).
* Level 4 — overlearning exhibit: unregularised small-n inventory paths on
  the training environments vs fresh ones.

Values reported at the reference state (nu0, Vpi = 0) and averaged over the
training start distribution Vpi0 ~ U[-Vbar/2, Vbar/2] (exact on the 161-node
grid: +-Vbar/2 are grid nodes).
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import ProblemSpec, VpiGrid, bbg_example
from .hjb import at_state, hjb_solve, interp_axis, make_ops
from .nets import controls, materialise_schedule
from .policy_eval import myopic_policy, policy_value
from .train import (TrainConfig, TrainedPolicy, _env_consts, _step_noise,
                    _step_state, _V0_TAG, full_objective, train)

NU0 = 0.0225
N_FINE = 161


# ----------------------------------------------------------------------------
# References (computed once per spec family)
# ----------------------------------------------------------------------------

def fine_spec(spec: ProblemSpec) -> ProblemSpec:
    return spec._replace(vpi_grid=VpiGrid(n=N_FINE))


def _start_avg(u0, ops):
    """Average over the training start law Vpi0 ~ U[-Vbar/2, Vbar/2]."""
    row = np.asarray(interp_axis(u0, ops.nu, NU0, 0))
    V = np.asarray(ops.V)
    m = np.abs(V) <= 0.5 * V[-1] * (1 + 1e-12)
    return float(np.trapezoid(row[m], V[m]) / (V[m][-1] - V[m][0]))


@lru_cache(maxsize=4)
def _references(variant: str, K_blocks: int):
    """v* and the reference-control value on the fine grid. Cached."""
    spec = bbg_example(variant=variant, K_blocks=K_blocks,
                       profile="replication")
    sf = fine_spec(spec)
    sol = hjb_solve(sf, store_quotes=False)
    ops = make_ops(sf)
    eta0 = jnp.zeros((sf.time_grid.M, ops.nu.size, ops.V.size))
    u_ref = policy_value(sf, myopic_policy(sf), eta=eta0)
    return {
        "vstar_center": float(at_state(sol.v0, ops, NU0, 0.0)),
        "vstar_avg": _start_avg(sol.v0, ops),
        "jref_center": float(at_state(u_ref.u0, ops, NU0, 0.0)),
        "jref_avg": _start_avg(u_ref.u0, ops),
    }


def references(spec: ProblemSpec) -> dict:
    variant = "a1" if abs(spec.risk.c_rho - 0.75) < 1e-9 else "body"
    return _references(variant, spec.time_grid.K_blocks)


# ----------------------------------------------------------------------------
# Trial runner (Levels 1-2)
# ----------------------------------------------------------------------------

class TrialResult(NamedTuple):
    cfg: dict
    J_train: float
    J_fresh: float
    se_fresh: float
    arb_center: float       # exact PDE value of the learned policy, n_V = 161
    arb_avg: float
    refs: dict
    history: np.ndarray     # (K, n_hist) downsampled per-block ERM curves
    wall_s: float


def _fresh_objective_se(stack, cm, spec, key, n):
    """Fresh-sample hard-fill objective: mean and per-path SE."""
    r = spec.risk
    Vpi0 = jax.random.uniform(jax.random.fold_in(key, _V0_TAG), (n,),
                              minval=-0.5 * r.Vbar, maxval=0.5 * r.Vbar)
    nu0 = jnp.full((n,), spec.market.nu0)
    ops, szV = _env_consts(spec)
    s = spec.time_grid.substeps_per_block
    dt = spec.time_grid.dt

    def step(state, m):
        nu, Vpi, obj = state
        net = jax.tree_util.tree_map(lambda f: f[m // s], stack)
        delta, eta = controls(net, cm, nu, Vpi)
        dW, U = _step_noise(key, m, n, spec.book.zV.size)
        reward, nu, Vpi = _step_state(spec, ops, szV, nu, Vpi, delta, eta,
                                      U, jnp.sqrt(dt) * dW, 0.0)
        return (nu, Vpi, obj + reward), None

    (nu, Vpi, obj), _ = jax.lax.scan(
        step, (nu0, Vpi0, jnp.zeros(n)), jnp.arange(spec.time_grid.M))
    return float(obj.mean()), float(obj.std() / np.sqrt(n))


def run_trial(spec: ProblemSpec, cfg: TrainConfig, arm: str = "reg",
              fresh_n: int = 4096, hist_points: int = 200) -> TrialResult:
    t0 = time.time()
    pol = train(spec, cfg)
    data_key = jax.random.fold_in(jax.random.PRNGKey(cfg.seed), 1)
    J_train = float(full_objective(pol.stack, pol.cm, spec, data_key, cfg.n))
    J_fresh, se = _fresh_objective_se(
        pol.stack, pol.cm, spec,
        jax.random.fold_in(jax.random.PRNGKey(cfg.seed), 999), fresh_n)
    sf = fine_spec(spec)
    qs, es = materialise_schedule(pol.stack, pol.cm, sf)
    u = policy_value(sf, qs, eta=es)
    opsF = make_ops(sf)
    hist = np.asarray(pol.history)
    stride = max(1, hist.shape[1] // hist_points)
    return TrialResult(
        cfg={"n": cfg.n, "r": cfg.r, "epochs": cfg.epochs, "seed": cfg.seed,
             "arm": arm, "K": spec.time_grid.K_blocks, "tau": cfg.tau,
             "beta": cfg.beta, "regularised": cfg.regularised},
        J_train=J_train, J_fresh=J_fresh, se_fresh=se,
        arb_center=float(at_state(u.u0, opsF, NU0, 0.0)),
        arb_avg=_start_avg(u.u0, opsF),
        refs=references(spec),
        history=hist[:, ::stride], wall_s=time.time() - t0)


def save_trial(res: TrialResult, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, cfg=json.dumps(res.cfg), refs=json.dumps(res.refs),
             history=res.history,
             scalars=np.array([res.J_train, res.J_fresh, res.se_fresh,
                               res.arb_center, res.arb_avg, res.wall_s]))


def load_trials(directory) -> list[TrialResult]:
    out = []
    for p in sorted(Path(directory).glob("*.npz")):
        z = np.load(p, allow_pickle=False)
        s = z["scalars"]
        out.append(TrialResult(cfg=json.loads(str(z["cfg"])),
                               J_train=float(s[0]), J_fresh=float(s[1]),
                               se_fresh=float(s[2]), arb_center=float(s[3]),
                               arb_avg=float(s[4]),
                               refs=json.loads(str(z["refs"])),
                               history=z["history"], wall_s=float(s[5])))
    return out


# ----------------------------------------------------------------------------
# Level 1-2 figures and tables
# ----------------------------------------------------------------------------

def gap_table(results: list[TrialResult]) -> str:
    lines = ["arm        n     r     trials  |gen gap|   opt gap (avg)  "
             "captured"]
    keys = sorted({(r.cfg["arm"], r.cfg["n"], r.cfg["r"]) for r in results})
    for arm, n, rr in keys:
        sel = [r for r in results if (r.cfg["arm"], r.cfg["n"],
                                      r.cfg["r"]) == (arm, n, rr)]
        gg = np.array([abs(r.J_train - r.J_fresh) for r in sel])
        og = np.array([r.refs["vstar_avg"] - r.arb_avg for r in sel])
        cap = np.array([(r.arb_avg - r.refs["jref_avg"])
                        / (r.refs["vstar_avg"] - r.refs["jref_avg"])
                        for r in sel])
        lines.append(f"{arm:<10} {n:<5} {rr:<5} {len(sel):<7} "
                     f"{gg.mean():>9.0f}   {og.mean():>12.0f}   "
                     f"{cap.mean():>7.1%}")
    return "\n".join(lines)


def plot_gap_vs_n(results, out, level: str = "gen"):
    """Level 1 (level='gen') or Level 2 (level='opt') boxplots vs n, per arm,
    log-log, with a C/n guide anchored at the largest-n regularised median."""
    arms = sorted({r.cfg["arm"] for r in results})
    ns = sorted({r.cfg["n"] for r in results})
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    width = 0.8 / max(len(arms), 1)
    med_anchor = None
    for a_i, arm in enumerate(arms):
        data, pos = [], []
        for n in ns:
            vals = [abs(r.J_train - r.J_fresh) if level == "gen"
                    else r.refs["vstar_avg"] - r.arb_avg
                    for r in results
                    if r.cfg["arm"] == arm and r.cfg["n"] == n]
            if vals:
                data.append(vals)
                pos.append(n * (1 + 0.25 * (a_i - (len(arms) - 1) / 2)))
                if arm.startswith("reg") and n == ns[-1]:
                    med_anchor = (n, float(np.median(vals)))
        if data:
            bp = ax.boxplot(data, positions=pos,
                            widths=[p * width for p in pos],
                            manage_ticks=False, patch_artist=True)
            for box in bp["boxes"]:
                box.set_facecolor(f"C{a_i}")
                box.set_alpha(0.45)
        ax.plot([], [], color=f"C{a_i}", label=arm)
    if med_anchor and level == "gen":
        nn = np.array([min(ns) * 0.8, max(ns) * 1.25])
        ax.plot(nn, med_anchor[1] * med_anchor[0] / nn, "k--", lw=1,
                label=r"$\propto n^{-1}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(ns); ax.set_xticklabels(ns)
    ax.set_xlabel("n (ERM sample count)")
    ax.set_ylabel("|J_train - J_fresh|  [EUR]" if level == "gen"
                  else "v* - J(learned)  [EUR, start-averaged]")
    ax.set_title("Level 1: generalisation gap" if level == "gen"
                 else "Level 2: optimality gap")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ----------------------------------------------------------------------------
# Level 3: policy space
# ----------------------------------------------------------------------------

def sample_vpi_paths(pol: TrainedPolicy, spec: ProblemSpec, key, n: int):
    """Vpi trajectories (M+1, n) under the trained stack, hard fills."""
    r = spec.risk
    Vpi0 = jax.random.uniform(jax.random.fold_in(key, _V0_TAG), (n,),
                              minval=-0.5 * r.Vbar, maxval=0.5 * r.Vbar)
    nu0 = jnp.full((n,), spec.market.nu0)
    ops, szV = _env_consts(spec)
    s = spec.time_grid.substeps_per_block
    dt = spec.time_grid.dt

    def step(state, m):
        nu, Vpi = state
        net = jax.tree_util.tree_map(lambda f: f[m // s], pol.stack)
        delta, eta = controls(net, pol.cm, nu, Vpi)
        dW, U = _step_noise(key, m, n, spec.book.zV.size)
        _, nu2, Vpi2 = _step_state(spec, ops, szV, nu, Vpi, delta, eta,
                                   U, jnp.sqrt(dt) * dW, 0.0)
        return (nu2, Vpi2), Vpi2

    (_, _), Vs = jax.lax.scan(step, (nu0, Vpi0),
                              jnp.arange(spec.time_grid.M))
    return jnp.concatenate([Vpi0[None], Vs], axis=0)


def policy_errors(pol: TrainedPolicy, spec: ProblemSpec, sol_quotes,
                  key=None, n_paths: int = 512) -> dict:
    """Per-option quote errors at the mid-horizon block, nu0 row, in spread
    units 1/b^i: raw sup over the gated grid, and visitation-weighted L2
    (weights = Vpi occupation of the learned policy under fresh noise)."""
    ops = make_ops(spec)
    K = spec.time_grid.K_blocks
    s = spec.time_grid.substeps_per_block
    m_mid = (K // 2) * s + s // 2
    q_ex = np.asarray(interp_axis(sol_quotes[m_mid], ops.nu, NU0, 0))
    V = np.asarray(ops.V)
    net = jax.tree_util.tree_map(lambda f: f[K // 2], pol.stack)
    d_l, _ = controls(net, pol.cm, jnp.full(V.size, NU0), jnp.asarray(V))
    q_l = np.asarray(d_l)
    gate = np.asarray(ops.gate)
    key = jax.random.PRNGKey(2024) if key is None else key
    paths = np.asarray(sample_vpi_paths(pol, spec, key, n_paths)).ravel()
    hist, edges = np.histogram(paths, bins=np.linspace(-V[-1], V[-1], 41))
    centers = 0.5 * (edges[:-1] + edges[1:])
    w_nodes = np.interp(V, centers, hist / max(hist.sum(), 1))
    w_nodes = w_nodes / w_nodes.sum()
    inv_b = np.asarray(1.0 / spec.book.b)
    err = np.where(gate, q_l - q_ex, np.nan) / inv_b[None, :, None]
    return {
        "sup": np.nanmax(np.abs(err), axis=(0, 2)),
        "l2_visit": np.sqrt(np.nansum(w_nodes[:, None, None] * err ** 2,
                                      axis=(0, 2)) / 2.0),
    }


def plot_policy_surfaces(pol: TrainedPolicy, spec: ProblemSpec, sol_quotes,
                         out, options=((12.0, 1.0), (10.0, 1.0), (8.0, 1.0))):
    ops = make_ops(spec)
    Kk = np.asarray(spec.book.strike); Tm = np.asarray(spec.book.mat)
    K = spec.time_grid.K_blocks; s = spec.time_grid.substeps_per_block
    m_mid = (K // 2) * s + s // 2
    q_ex = np.asarray(interp_axis(sol_quotes[m_mid], ops.nu, NU0, 0))
    V = np.asarray(ops.V)
    net = jax.tree_util.tree_map(lambda f: f[K // 2], pol.stack)
    d_l, eta_l = controls(net, pol.cm, jnp.full(V.size, NU0), jnp.asarray(V))
    d_l = np.asarray(d_l); gate = np.asarray(ops.gate)
    fig, axes = plt.subplots(1, len(options) + 1,
                             figsize=(3.1 * (len(options) + 1), 3.2))
    for ax, (kv, tv) in zip(axes, options):
        i = int(np.where((Kk == kv) & (Tm == tv))[0][0])
        g = gate[:, i, 0]
        ax.plot(V[g] / 1e7, q_ex[g, i, 0], "k-", lw=1.6, label="exact bid")
        ax.plot(V[g] / 1e7, d_l[g, i, 0], "C1--", lw=1.4, label="learned")
        ax.set_title(f"K={kv:.0f}, T={tv:g}", fontsize=9)
        ax.set_xlabel(r"$\mathcal{V}^\pi / 10^7$")
        if ax is axes[0]:
            ax.set_ylabel("bid quote [EUR]")
            ax.legend(fontsize=7)
    ax = axes[-1]
    ax.plot(V / 1e7, 0.05 * V / 1e5, "k-", lw=1.6,
            label=r"$\eta^*=+0.05\,\mathcal{V}^\pi$")
    ax.plot(V / 1e7, np.asarray(eta_l) / 1e5, "C1--", lw=1.4, label="learned")
    ax.set_title("hedge head", fontsize=9)
    ax.set_xlabel(r"$\mathcal{V}^\pi / 10^7$")
    ax.set_ylabel(r"$\eta$ [1e5]")
    ax.legend(fontsize=7)
    fig.suptitle("Level 3: mid-horizon block policy at $\\nu_0$", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Level 4: overlearning exhibit
# ----------------------------------------------------------------------------

def plot_overlearning_paths(pol: TrainedPolicy, spec: ProblemSpec, seed: int,
                            out, n_show: int = 12):
    data_key = jax.random.fold_in(jax.random.PRNGKey(seed), 1)
    fresh_key = jax.random.fold_in(jax.random.PRNGKey(seed), 999)
    t = np.linspace(0, spec.time_grid.T, spec.time_grid.M + 1)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    for ax, key, title in ((axes[0], data_key, "training environments"),
                           (axes[1], fresh_key, "fresh environments")):
        Vs = np.asarray(sample_vpi_paths(pol, spec, key, n_show))
        ax.plot(t, Vs / 1e7, lw=0.8, alpha=0.8)
        ax.axhline(1.0, color="r", ls=":", lw=0.8)
        ax.axhline(-1.0, color="r", ls=":", lw=0.8)
        ax.set_title(f"Level 4: {title}", fontsize=9)
        ax.set_xlabel("t [yr]")
    axes[0].set_ylabel(r"$\mathcal{V}^\pi / \bar{\mathcal{V}}$")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)

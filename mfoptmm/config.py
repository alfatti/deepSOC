"""Grids and time-step derivation. Single-source experiment configuration.

The step size Delta t is set by rates, not by the horizon: the explicit
monotone scheme's diagonal multiplier is
    1 - dt * ( nu_max xi^2 / dnu^2  +  max|a^P| / dnu  +  sum_{i,j} Lambda^{i,j} ),
and Lambda <= lam gives the worst-case (control-independent) bound. T only
determines the *count* M = ceil(T / dt). Profiles:

* worst-case-safe: dt from the bound above (with a safety factor < 1 if
  desired), M rounded up to a multiple of K_blocks;
* replication: force M = 180 to match BBG's published grid; the solver then
  relies on realised Lambda(delta*) << lam, which Gate E4 asserts at runtime.

The decision grid K_blocks (number of distinct per-epoch control networks) is
a separate, coarser grid: policies are piecewise-constant *maps* in t, re-read
the state every substep, and remain Markov feedback (so the policy-evaluation
PDE applies to them directly).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .model import Book, Market, Risk
from .instruments import paper_book


@dataclass(frozen=True)
class NuGrid:
    nu_min: float = 0.0144
    nu_max: float = 0.0324
    n: int = 30

    @property
    def dnu(self) -> float:
        return (self.nu_max - self.nu_min) / (self.n - 1)


@dataclass(frozen=True)
class VpiGrid:
    n: int = 40  # over [-Vbar, Vbar]; spacing depends on Risk.Vbar


@dataclass(frozen=True)
class TimeGrid:
    T: float
    M: int          # fine substeps (environment / PDE time steps)
    K_blocks: int   # decision epochs (distinct control networks)

    def __post_init__(self):
        if self.M % self.K_blocks != 0:
            raise ValueError(
                f"M={self.M} must be a multiple of K_blocks={self.K_blocks}")

    @property
    def dt(self) -> float:
        return self.T / self.M

    @property
    def substeps_per_block(self) -> int:
        return self.M // self.K_blocks

    @property
    def block_duration(self) -> float:
        return self.T / self.K_blocks


def stability_rates(market: Market, book: Book, nug: NuGrid) -> dict:
    """Worst-case per-year rates entering the explicit scheme's diagonal."""
    lam = np.asarray(book.lam)
    intensity_rate = float(2.0 * lam.sum())  # both sides, Lambda <= lam
    diffusion_rate = float(nug.nu_max * market.xi ** 2 / nug.dnu ** 2)
    aP_ends = [abs(market.kappaP * (market.thetaP - nug.nu_min)),
               abs(market.kappaP * (market.thetaP - nug.nu_max))]
    drift_rate = float(max(aP_ends) / nug.dnu)
    total = intensity_rate + diffusion_rate + drift_rate
    return dict(intensity=intensity_rate, diffusion=diffusion_rate,
                drift=drift_rate, total=total, dt_bound=1.0 / total)


def derive_time_grid(T: float, K_blocks: int, market: Market, book: Book,
                     nug: NuGrid, safety: float = 1.0,
                     force_M: int | None = None) -> TimeGrid:
    """Derive M from the worst-case stability bound (or force it), rounded up
    to a multiple of K_blocks."""
    if force_M is None:
        rates = stability_rates(market, book, nug)
        M = math.ceil(T / (safety * rates["dt_bound"]))
    else:
        M = force_M
    M = K_blocks * math.ceil(M / K_blocks)
    return TimeGrid(T=T, M=M, K_blocks=K_blocks)


class ProblemSpec(NamedTuple):
    market: Market
    book: Book
    risk: Risk
    nu_grid: NuGrid
    vpi_grid: VpiGrid
    time_grid: TimeGrid


BBG_T = 0.0012           # yr (0.3 day)
BBG_GAMMA = 1e-3
BBG_VBAR = 1e7


def bbg_example(variant: str = "a1", K_blocks: int = 36,
                profile: str = "replication", safety: float = 1.0,
                book: Book | None = None,
                force_M: int | None = None) -> ProblemSpec:
    """The BBG Section 4 example, in the body or A.1 variant.

    variant: 'a1' (penalty scaled by 1 - rho^2) or 'body'.
    profile: 'replication' (M = 180, BBG's published grid) or 'worst_case'
             (M from the stability bound; ~252 at K_blocks = 36, safety 1).
    force_M: overrides the profile's M (rounded up to a K_blocks multiple);
             used by the dt-refinement gate.
    """
    market = Market(S0=10.0, nu0=0.0225,
                    kappaP=2.0, thetaP=0.04,
                    kappaQ=3.0, thetaQ=0.0225,
                    xi=0.2, rho=-0.5, mu=0.0)
    if book is None:
        book = paper_book(market)
    if variant == "a1":
        c_rho = 1.0 - market.rho ** 2
    elif variant == "body":
        c_rho = 1.0
    else:
        raise ValueError(f"unknown variant {variant!r}")
    risk = Risk(gamma=BBG_GAMMA, Vbar=BBG_VBAR, c_rho=c_rho)
    nug = NuGrid()
    if force_M is not None:
        tg = derive_time_grid(BBG_T, K_blocks, market, book, nug,
                              force_M=force_M)
    elif profile == "replication":
        tg = derive_time_grid(BBG_T, K_blocks, market, book, nug, force_M=180)
    elif profile == "worst_case":
        tg = derive_time_grid(BBG_T, K_blocks, market, book, nug, safety=safety)
    else:
        raise ValueError(f"unknown profile {profile!r}")
    return ProblemSpec(market=market, book=book, risk=risk,
                       nu_grid=nug, vpi_grid=VpiGrid(), time_grid=tg)

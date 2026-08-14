"""Instrument data for the BBG Section 4 example, and its verification pricer.

Provenance
----------
PAPER_TABLE carries the (price, vega, IV) triples printed verbatim in the
figure legends of BBG Figures 4-13 ("Algorithmic market making for options").
These are the paper's own numbers, computed there by 1e5 Monte-Carlo paths per
option, so they carry MC noise plus two-decimal rounding (four decimals for
IV). The default book (`paper_book`) uses them, so that our lambda^i, z^i and
zV^i match the published example exactly.

Gate P1 (`verify_paper_table`) recomputes every entry with an independent
semi-analytic Heston pricer (characteristic function, trap-free Albrecher
formulation, Gauss-Legendre quadrature; vega by central finite difference in
nu0) and reports per-entry discrepancies. Tolerances in the test reflect the
*paper's* noise sources, not the pricer's.

The pricer lives in NumPy on purpose: it is a verification tool and a table
generator, never in a hot loop.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

import jax.numpy as jnp

from .model import Book, Market

# (strike, maturity) -> (price O0, vega V = dO/d sqrt(nu), implied vol)
# Source: BBG figure legends (Figs 4-13). K-major, maturity-minor order.
PAPER_TABLE = {
    (8.0, 1.0): (2.06, 0.41, 0.1622),
    (8.0, 1.5): (2.11, 0.46, 0.1583),
    (8.0, 2.0): (2.16, 0.48, 0.1563),
    (8.0, 3.0): (2.27, 0.46, 0.1544),
    (9.0, 1.0): (1.21, 0.91, 0.1538),
    (9.0, 1.5): (1.31, 0.84, 0.1518),
    (9.0, 2.0): (1.41, 0.76, 0.1515),
    (9.0, 3.0): (1.57, 0.65, 0.1508),
    (10.0, 1.0): (0.58, 1.25, 0.1460),
    (10.0, 1.5): (0.71, 1.05, 0.1459),
    (10.0, 2.0): (0.83, 0.92, 0.1468),
    (10.0, 3.0): (1.02, 0.76, 0.1476),
    (11.0, 1.0): (0.21, 1.05, 0.1392),
    (11.0, 1.5): (0.33, 0.96, 0.1409),
    (11.0, 2.0): (0.44, 0.88, 0.1427),
    (11.0, 3.0): (0.62, 0.75, 0.1448),
    (12.0, 1.0): (0.06, 0.54, 0.1344),
    (12.0, 1.5): (0.13, 0.65, 0.1367),
    (12.0, 2.0): (0.21, 0.67, 0.1395),
    (12.0, 3.0): (0.36, 0.65, 0.1423),
}

ALPHA = 0.7
BETA_LAM = 150.0
TRADE_NOTIONAL = 5.0e5  # EUR per transaction; z^i = TRADE_NOTIONAL / O^i_0


def rfq_rate(S0: float, strike) -> np.ndarray:
    """lambda^i = 252 * 30 / (1 + 0.7 |S0 - K^i|), per year."""
    return 252.0 * 30.0 / (1.0 + 0.7 * np.abs(S0 - np.asarray(strike)))


# ----------------------------------------------------------------------------
# Heston semi-analytic pricer (verification tool)
# ----------------------------------------------------------------------------

def _heston_cf(u, tau, x0, nu0, kappa, theta, xi, rho):
    """E^Q[exp(i u ln S_tau)] with r = q = 0. Accepts complex u.

    Trap-free formulation (Albrecher et al.): with s = i u,
      beta = kappa - rho xi s,  d = sqrt(beta^2 + xi^2 (s - s^2)),
      g = (beta - d) / (beta + d),
      A = kappa theta / xi^2 [ (beta - d) tau - 2 log((1 - g e^{-d tau})/(1 - g)) ],
      B = (beta - d) / xi^2 * (1 - e^{-d tau}) / (1 - g e^{-d tau}).
    Martingale check phi(-i) = S0 holds exactly (d = beta, g = 0).
    """
    s = 1j * np.asarray(u, dtype=complex)
    beta = kappa - rho * xi * s
    d = np.sqrt(beta * beta + xi * xi * (s - s * s))
    g = (beta - d) / (beta + d)
    edt = np.exp(-d * tau)
    A = kappa * theta / xi ** 2 * ((beta - d) * tau
                                   - 2.0 * np.log((1.0 - g * edt) / (1.0 - g)))
    B = (beta - d) / xi ** 2 * (1.0 - edt) / (1.0 - g * edt)
    return np.exp(s * x0 + A + B * nu0)


def _gl_nodes(n_gl: int, u_max: float):
    x, w = np.polynomial.legendre.leggauss(n_gl)
    u = 0.5 * u_max * (x + 1.0)
    w = 0.5 * u_max * w
    return u, w


def heston_call(S0, K, tau, nu0, kappa, theta, xi, rho,
                n_gl: int = 800, u_max: float = 200.0) -> float:
    """European call, r = q = 0, via the P1/P2 representation."""
    x0 = np.log(S0)
    k = np.log(K)
    u, w = _gl_nodes(n_gl, u_max)
    phi_u = _heston_cf(u, tau, x0, nu0, kappa, theta, xi, rho)
    phi_ui = _heston_cf(u - 1j, tau, x0, nu0, kappa, theta, xi, rho)
    e = np.exp(-1j * u * k)
    P1 = 0.5 + np.sum(w * np.real(e * phi_ui / (1j * u * S0))) / np.pi
    P2 = 0.5 + np.sum(w * np.real(e * phi_u / (1j * u))) / np.pi
    return float(S0 * P1 - K * P2)


def heston_price_and_vega(S0, K, tau, nu0, kappa, theta, xi, rho,
                          h: float = 1e-5, **kw):
    """(price, V) with V = dO/d sqrt(nu) at nu0 = 2 sqrt(nu0) * dO/d nu0,
    central finite difference in nu0."""
    c0 = heston_call(S0, K, tau, nu0, kappa, theta, xi, rho, **kw)
    cp = heston_call(S0, K, tau, nu0 + h, kappa, theta, xi, rho, **kw)
    cm = heston_call(S0, K, tau, nu0 - h, kappa, theta, xi, rho, **kw)
    dO_dnu = (cp - cm) / (2.0 * h)
    return c0, 2.0 * np.sqrt(nu0) * dO_dnu


def bs_call(S0, K, sigma, tau) -> float:
    """Black-Scholes call, r = q = 0."""
    if sigma <= 0.0 or tau <= 0.0:
        return float(max(S0 - K, 0.0))
    sq = sigma * np.sqrt(tau)
    d1 = (np.log(S0 / K)) / sq + 0.5 * sq
    d2 = d1 - sq
    return float(S0 * norm.cdf(d1) - K * norm.cdf(d2))


def implied_vol(price, S0, K, tau, lo: float = 1e-4, hi: float = 2.0) -> float:
    return float(brentq(lambda s: bs_call(S0, K, s, tau) - price, lo, hi,
                        xtol=1e-12))


# ----------------------------------------------------------------------------
# Book builders
# ----------------------------------------------------------------------------

def _assemble_book(S0: float, entries) -> Book:
    """entries: iterable of (strike, mat, O0, vega), K-major/mat-minor sorted."""
    entries = sorted(entries, key=lambda e: (e[0], e[1]))
    strike = np.array([e[0] for e in entries])
    mat = np.array([e[1] for e in entries])
    O0 = np.array([e[2] for e in entries])
    vega = np.array([e[3] for e in entries])
    lam = rfq_rate(S0, strike)
    z = TRADE_NOTIONAL / O0
    return Book(
        strike=jnp.asarray(strike), mat=jnp.asarray(mat),
        O0=jnp.asarray(O0), vega=jnp.asarray(vega),
        lam=jnp.asarray(lam), alpha=ALPHA, beta_lam=BETA_LAM,
        b=jnp.asarray(BETA_LAM / vega), z=jnp.asarray(z),
        zV=jnp.asarray(z * vega),
    )


def paper_book(market: Market) -> Book:
    """The published example book: (O0, vega) verbatim from the paper's legends."""
    entries = [(K, T, p, v) for (K, T), (p, v, _iv) in PAPER_TABLE.items()]
    return _assemble_book(market.S0, entries)


def pricer_book(market: Market, **kw) -> Book:
    """Same book with (O0, vega) recomputed by the semi-analytic pricer
    (pricing measure Q: kappaQ, thetaQ)."""
    entries = []
    for (K, T) in sorted(PAPER_TABLE):
        p, v = heston_price_and_vega(market.S0, K, T, market.nu0,
                                     market.kappaQ, market.thetaQ,
                                     market.xi, market.rho, **kw)
        entries.append((K, T, p, v))
    return _assemble_book(market.S0, entries)


def verify_paper_table(market: Market, **kw):
    """Gate P1: per-entry comparison of PAPER_TABLE against the pricer.

    Returns a list of dict rows with keys strike, mat, price_paper, price_mine,
    dprice, vega_paper, vega_mine, dvega, iv_paper, iv_mine, div.
    """
    rows = []
    for (K, T), (p_paper, v_paper, iv_paper) in sorted(PAPER_TABLE.items()):
        p, v = heston_price_and_vega(market.S0, K, T, market.nu0,
                                     market.kappaQ, market.thetaQ,
                                     market.xi, market.rho, **kw)
        iv = implied_vol(p, market.S0, K, T)
        rows.append(dict(strike=K, mat=T,
                         price_paper=p_paper, price_mine=p, dprice=p - p_paper,
                         vega_paper=v_paper, vega_mine=v, dvega=v - v_paper,
                         iv_paper=iv_paper, iv_mine=iv, div=iv - iv_paper))
    return rows

"""Single source of truth for the BBG A.1 problem definition.

Every consumer in this repository (HJB solver, policy evaluator, simulator,
trainer, comparison stack) imports these primitives; none re-implements them.
Cross-consumer tests then catch discretisation errors instead of re-deriving
the mathematics (the B(t)-bug principle).

Conventions
-----------
* Units: EUR, years. Quotes delta are mid-to-bid / ask-to-mid in EUR per
  contract. Portfolio vega Vpi in EUR * yr^{1/2}.
* Book arrays have trailing shape (N,). Functions broadcast `p` / `delta`
  against that trailing axis; callers align shapes.
* The A.1 (partial-delta-hedge) problem is the body problem with the running
  risk-penalty coefficient scaled by c_rho = 1 - rho^2; the nu-diffusion keeps
  the physical xi. `Risk.c_rho` carries the variant.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


class Market(NamedTuple):
    """One-factor stochastic volatility market (BBG Section 2.1)."""

    S0: float
    nu0: float
    kappaP: float
    thetaP: float
    kappaQ: float
    thetaQ: float
    xi: float
    rho: float
    mu: float = 0.0  # A.1 study runs with mu = 0 (see strategy doc, Decision 7)


class Book(NamedTuple):
    """Option book under the constant-vega approximation. Arrays: shape (N,)."""

    strike: jnp.ndarray
    mat: jnp.ndarray            # option maturities T^i (yr); horizon T << min mat
    O0: jnp.ndarray             # option prices at t=0 (EUR)
    vega: jnp.ndarray           # V^i = d O^i / d sqrt(nu) at (0, S0, nu0)
    lam: jnp.ndarray            # RFQ intensity scale lambda^i (1/yr)
    alpha: float                # logistic intercept
    beta_lam: float             # logistic slope scale (yr^{1/2})
    b: jnp.ndarray              # beta_lam / vega^i  (1/EUR)
    z: jnp.ndarray              # Dirac trade size z^i (contracts)
    zV: jnp.ndarray             # vega notional per trade z^i * V^i (EUR yr^{1/2})


class Risk(NamedTuple):
    gamma: float
    Vbar: float
    c_rho: float                # 1.0 (body) or 1 - rho^2 (A.1)
    delta_inf: float = -jnp.inf


# ----------------------------------------------------------------------------
# Drifts, carry, penalties, hedge channel
# ----------------------------------------------------------------------------

def a_P(m: Market, nu):
    return m.kappaP * (m.thetaP - nu)


def a_Q(m: Market, nu):
    return m.kappaQ * (m.thetaQ - nu)


def carry(m: Market, nu, Vpi):
    """Vega carry term Vpi * (a^P - a^Q) / (2 sqrt(nu))."""
    return Vpi * (a_P(m, nu) - a_Q(m, nu)) / (2.0 * jnp.sqrt(nu))


def penalty_reduced(m: Market, r: Risk, Vpi):
    """Reduced running risk penalty (gamma xi^2 / 8) c_rho Vpi^2."""
    return r.gamma * m.xi ** 2 / 8.0 * r.c_rho * Vpi ** 2


def penalty_hedge(m: Market, r: Risk, Vpi, eta):
    """Pre-reduction A.1 running penalty in the eta = sqrt(nu) S (Delta^pi + q^S)
    coordinate: (gamma/2) [ xi^2/4 Vpi^2 + eta^2 + rho xi Vpi eta ].

    Pointwise quadratic in eta; minimised at eta_star, where it equals
    penalty_reduced with c_rho = 1 - rho^2 (identity-tested).
    """
    return 0.5 * r.gamma * (m.xi ** 2 / 4.0 * Vpi ** 2 + eta ** 2
                            + m.rho * m.xi * Vpi * eta)


def eta_star(m: Market, Vpi):
    """Closed-form optimal partial hedge, eta* = -(rho xi / 2) Vpi."""
    return -0.5 * m.rho * m.xi * Vpi


# ----------------------------------------------------------------------------
# Intensities
# ----------------------------------------------------------------------------

def intensity(book: Book, delta):
    """Fill intensity Lambda^i(delta) = lam^i * sigmoid(-(alpha + b^i delta))."""
    return book.lam * jax.nn.sigmoid(-(book.alpha + book.b * delta))


def intensity_inv(book: Book, y):
    """Inverse of Lambda^i on (0, lam^i)."""
    return (jnp.log(book.lam / y - 1.0) - book.alpha) / book.b


# ----------------------------------------------------------------------------
# Wright omega: omega(l) solves w + log w = l, i.e. omega(l) = W(e^l)
# ----------------------------------------------------------------------------
# Log-domain evaluation of the Lambert W function on an exponential argument.
# Needed because the Hamiltonian argument exp(-(alpha + b p + 1)) over- or
# underflows for |b p| large, while omega(l) is perfectly tame.
#
# Newton on f(w) = w + log w - l:  w <- w - f * w / (w + 1). Initial guesses:
#   l > 2   : w0 = l - log l            (asymptotic branch)
#   l <= 2  : w0 = exp(min(l, 2))       (converges over the whole mid range)
#   l < -40 : omega(l) = exp(l) to double precision; returned directly.
# 20 iterations is far beyond quadratic-convergence needs; the loop is cheap.

_OMEGA_EXACT_BELOW = -40.0
_OMEGA_ITERS = 20


@jax.custom_jvp
def wright_omega(l):
    l = jnp.asarray(l)
    lc = jnp.maximum(l, _OMEGA_EXACT_BELOW)  # keep log(w) finite inside Newton
    big = lc > 2.0
    w0 = jnp.where(big,
                   lc - jnp.log(jnp.maximum(lc, 2.0)),
                   jnp.exp(jnp.minimum(lc, 2.0)))

    def body(w, _):
        f = w + jnp.log(w) - lc
        w = w - f * w / (w + 1.0)
        w = jnp.maximum(w, 1e-300)
        return w, None

    w, _ = lax.scan(body, w0, None, length=_OMEGA_ITERS)
    return jnp.where(l < _OMEGA_EXACT_BELOW, jnp.exp(l), w)


@wright_omega.defjvp
def _wright_omega_jvp(primals, tangents):
    (l,), (ld,) = primals, tangents
    w = wright_omega(l)
    # From w + log w = l:  dw/dl = w / (1 + w). Exact; no differentiating the loop.
    return w, w / (1.0 + w) * ld


# ----------------------------------------------------------------------------
# Logistic Hamiltonian in semi-closed form
# ----------------------------------------------------------------------------
# H^i(p) = sup_delta Lambda^i(delta) (delta - p) with logistic Lambda. With
# l(p) = -(alpha + b p + 1) and w = omega(l):
#   delta*(p) = p + (1 + w) / b
#   H(p)      = (lam / b) * w
#   H'(p)     = -lam w / (1 + w)  = -Lambda(delta*(p))   (envelope identity)
#   H''(p)    =  lam b w / (1 + w)^3 > 0                 (convex, decreasing)
# All identity-tested against brute-force sup and against each other.

def _ell(book: Book, p):
    return -(book.alpha + book.b * p + 1.0)


def H(book: Book, p):
    return book.lam / book.b * wright_omega(_ell(book, p))


def H_p(book: Book, p):
    w = wright_omega(_ell(book, p))
    return -book.lam * w / (1.0 + w)


def H_pp(book: Book, p):
    w = wright_omega(_ell(book, p))
    return book.lam * book.b * w / (1.0 + w) ** 3


def delta_star(book: Book, p, delta_inf: float = -jnp.inf):
    """Unconstrained maximiser of Lambda(delta)(delta - p), clamped at delta_inf."""
    w = wright_omega(_ell(book, p))
    return jnp.maximum(p + (1.0 + w) / book.b, delta_inf)


def myopic_quotes(book: Book):
    """Reference quotes: argmax_delta delta * Lambda^i(delta) = delta*(p=0)."""
    return delta_star(book, jnp.zeros_like(book.b))


# ----------------------------------------------------------------------------
# Risk-limit gate
# ----------------------------------------------------------------------------

def post_trade_ok(r: Risk, Vpi_post):
    """Hard vega-limit gate: trade admissible iff |Vpi_post| <= Vbar."""
    return jnp.abs(Vpi_post) <= r.Vbar

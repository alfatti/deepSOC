"""Gate E3: the semi-closed logistic Hamiltonian block.

Every closed form is checked against an independent brute-force optimisation
of Lambda(delta)(delta - p), on the actual 20-option book, and the algebraic
identities tying H, H', delta*, Lambda and Lambda^{-1} together are verified
to near machine precision.
"""
import numpy as np
import jax.numpy as jnp
from scipy.optimize import minimize_scalar
from scipy.special import expit

from mfoptmm import (
    bbg_example, intensity, intensity_inv, H, H_p, H_pp, delta_star,
    myopic_quotes,
)

SPEC = bbg_example()
BOOK = SPEC.book
P_GRID = np.linspace(-0.5, 0.5, 41)  # spans and exceeds realistic quote scales


def _lambda_np(i, delta):
    lam = float(BOOK.lam[i]); b = float(BOOK.b[i]); a = float(BOOK.alpha)
    return lam * expit(-(a + b * delta))


def _brute(i, p):
    """Grid scan + bounded refinement of sup_delta Lambda(delta)(delta - p)."""
    lo, hi = p - 1.0, p + 2.0
    grid = np.linspace(lo, hi, 30001)
    vals = _lambda_np(i, grid) * (grid - p)
    k = int(np.argmax(vals))
    a = grid[max(k - 2, 0)]
    b = grid[min(k + 2, grid.size - 1)]
    res = minimize_scalar(lambda d: -_lambda_np(i, d) * (d - p),
                          bounds=(a, b), method="bounded",
                          options={"xatol": 1e-13})
    return -res.fun, res.x


def test_hamiltonian_vs_brute_force():
    worst_H, worst_d = 0.0, 0.0
    for i in range(int(BOOK.lam.shape[0])):
        p = jnp.asarray(P_GRID)
        Hc = np.asarray(H(BOOK._replace(lam=BOOK.lam[i], b=BOOK.b[i]), p))
        dc = np.asarray(delta_star(BOOK._replace(lam=BOOK.lam[i], b=BOOK.b[i]), p))
        for j, pj in enumerate(P_GRID):
            Hb, db = _brute(i, pj)
            worst_H = max(worst_H, abs(Hc[j] - Hb) / max(1.0, abs(Hb)))
            worst_d = max(worst_d, abs(dc[j] - db))
    assert worst_H < 1e-10, f"H rel err {worst_H:.3e}"
    assert worst_d < 1e-7, f"delta* abs err {worst_d:.3e}"


def test_envelope_identity():
    """H'(p) = -Lambda(delta*(p)), algebraically exact."""
    p = jnp.linspace(-5.0, 5.0, 201)[:, None]
    lhs = np.asarray(H_p(BOOK, p))
    rhs = -np.asarray(intensity(BOOK, delta_star(BOOK, p)))
    rel = np.abs(lhs - rhs) / np.maximum(np.abs(rhs), 1e-300)
    assert rel.max() < 1e-12, f"envelope rel err {rel.max():.3e}"


def test_quote_extraction_identity():
    """Lambda^{-1}(-H'(p)) = delta*(p): the HJB quote-extraction path agrees
    with the direct closed form."""
    p = jnp.linspace(-0.6, 0.6, 121)[:, None]
    d1 = np.asarray(intensity_inv(BOOK, -H_p(BOOK, p)))
    d2 = np.asarray(delta_star(BOOK, p))
    assert np.abs(d1 - d2).max() < 1e-10


def test_H_convex_decreasing():
    # In the deep right tail (b p >> 1) Lambda(delta*) underflows — XLA's
    # flush-to-zero makes H' = -0.0 and H'' = 0.0 exactly, the correct limit.
    # So: non-strict signs globally, strict signs on the realistic quote band.
    p = jnp.linspace(-2.0, 2.0, 401)[:, None]
    hp = np.asarray(H_p(BOOK, p))
    hpp = np.asarray(H_pp(BOOK, p))
    assert (hp <= 0.0).all() and (hpp >= 0.0).all()
    band = jnp.linspace(-0.6, 0.6, 121)[:, None]
    assert (np.asarray(H_p(BOOK, band)) < 0.0).all()
    assert (np.asarray(H_pp(BOOK, band)) > 0.0).all()
    # analytic H'' against a finite difference of H'
    h = 1e-6
    fd = np.asarray((H_p(BOOK, p + h) - H_p(BOOK, p - h))) / (2 * h)
    rel = np.abs(hpp - fd) / np.maximum(np.abs(fd), 1e-300)
    assert rel.max() < 1e-5, f"H'' rel err {rel.max():.3e}"


def test_myopic_quotes_maximise_revenue():
    d = np.asarray(myopic_quotes(BOOK))
    for i in range(d.size):
        grid = np.linspace(-0.5, 1.5, 40001)
        rev = _lambda_np(i, grid) * grid
        k = int(np.argmax(rev))
        assert abs(d[i] - grid[k]) < 1e-4
        # first-order condition at the closed form
        eps = 1e-7
        r0 = _lambda_np(i, d[i] - eps) * (d[i] - eps)
        r1 = _lambda_np(i, d[i] + eps) * (d[i] + eps)
        rc = _lambda_np(i, d[i]) * d[i]
        assert rc >= r0 - 1e-12 and rc >= r1 - 1e-12

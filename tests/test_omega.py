"""wright_omega gate: oracle comparison against scipy.special.wrightomega,
extreme-argument behaviour, and derivative consistency."""
import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import wrightomega as scipy_omega

from mfoptmm import wright_omega


def test_omega_vs_scipy_wide_range():
    l = np.concatenate([
        np.linspace(-600.0, 600.0, 4001),
        np.linspace(-5.0, 5.0, 2001),
        np.random.default_rng(0).uniform(-50.0, 50.0, 1000),
    ])
    mine = np.asarray(wright_omega(jnp.asarray(l)))
    ref = scipy_omega(l).real
    # relative where representable, absolute-vs-exp(l) deep in the underflow tail
    mask = ref > 1e-280
    rel = np.abs(mine[mask] - ref[mask]) / ref[mask]
    assert rel.max() < 1e-12, f"max rel err {rel.max():.3e}"
    tail = ~mask
    if tail.any():
        assert np.allclose(mine[tail], np.exp(l[tail]), rtol=1e-12, atol=0.0)


def test_omega_defining_identity():
    l = jnp.linspace(-35.0, 600.0, 2001)  # log(w) finite region
    w = wright_omega(l)
    resid = np.asarray(w + jnp.log(w) - l)
    assert np.abs(resid).max() < 1e-11, f"max |w + log w - l| = {np.abs(resid).max():.3e}"


def test_omega_extremes_no_nan():
    l = jnp.asarray([-1e4, -745.0, -41.0, -40.0, 0.0, 2.0, 700.0, 1e4])
    w = np.asarray(wright_omega(l))
    assert np.isfinite(w).all()
    assert (w >= 0.0).all()
    # deep-negative branch equals exp(l) exactly (incl. underflow to 0)
    assert w[0] == 0.0 and np.isclose(w[2], np.exp(-41.0), rtol=1e-12)
    # asymptotic sanity at the top end: w ~ l - log l
    assert abs(w[-1] - (1e4 - np.log(1e4))) / 1e4 < 1e-3


def test_omega_tangent_matches_primal_fd():
    l = jnp.linspace(-20.0, 20.0, 81)
    g = jax.vmap(jax.grad(wright_omega))(l)
    h = 1e-6
    fd = (wright_omega(l + h) - wright_omega(l - h)) / (2 * h)
    rel = np.abs(np.asarray(g - fd)) / np.maximum(np.abs(np.asarray(fd)), 1e-300)
    assert rel.max() < 1e-7, f"max rel err {rel.max():.3e}"

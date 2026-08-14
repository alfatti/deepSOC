"""Time-grid derivation: rate arithmetic and profile behaviour."""
import numpy as np
import pytest

from mfoptmm import bbg_example, stability_rates, derive_time_grid, TimeGrid
from mfoptmm.config import NuGrid


def test_stability_rates_values():
    spec = bbg_example()
    r = stability_rates(spec.market, spec.book, spec.nu_grid)
    lam_sum = float(np.asarray(spec.book.lam).sum())
    assert np.isclose(lam_sum, 91016.4706, atol=1e-3)
    assert np.isclose(r["intensity"], 2 * lam_sum)
    dnu = spec.nu_grid.dnu
    assert np.isclose(r["diffusion"], 0.0324 * 0.04 / dnu ** 2)
    assert np.isclose(r["drift"], 2.0 * (0.04 - 0.0144) / dnu)
    assert np.isclose(r["total"],
                      r["intensity"] + r["diffusion"] + r["drift"])
    # the binding constraint is the intensity sum, by a wide margin
    assert r["intensity"] > 50 * r["diffusion"]


def test_worst_case_grid():
    spec = bbg_example(profile="worst_case")
    tg = spec.time_grid
    r = stability_rates(spec.market, spec.book, spec.nu_grid)
    assert tg.M % tg.K_blocks == 0
    assert tg.dt <= r["dt_bound"] * (1 + 1e-12)
    # T * total_rate ~ 222.6 -> M = 223 -> rounded up to 252 = 36 * 7
    assert tg.M == 252 and tg.substeps_per_block == 7


def test_replication_grid():
    spec = bbg_example(profile="replication")
    tg = spec.time_grid
    assert tg.M == 180 and tg.K_blocks == 36 and tg.substeps_per_block == 5
    assert np.isclose(tg.dt, 0.0012 / 180)
    # BBG's grid nominally violates the worst-case bound: the solver must
    # assert the *realised* diagonal at runtime (Gate E4, milestone 2)
    r = stability_rates(spec.market, spec.book, spec.nu_grid)
    assert tg.dt > r["dt_bound"]


def test_m_multiple_of_k_enforced():
    with pytest.raises(ValueError):
        TimeGrid(T=1.0, M=181, K_blocks=36)


def test_variants():
    a1 = bbg_example(variant="a1")
    body = bbg_example(variant="body")
    assert np.isclose(a1.risk.c_rho, 0.75)
    assert np.isclose(body.risk.c_rho, 1.0)


def test_book_magnitudes():
    """Trade vega prints span ~1e5 (deep ITM K=8) to ~4.5e6 (K=12 short)."""
    spec = bbg_example()
    zV = np.asarray(spec.book.zV)
    assert 0.9e5 < zV.min() < 1.1e5
    assert 4.0e6 < zV.max() < 5.0e6
    assert zV.max() < spec.risk.Vbar  # every single trade is admissible at Vpi=0


def test_custom_nu_grid():
    g = NuGrid(nu_min=0.01, nu_max=0.04, n=31)
    assert np.isclose(g.dnu, 0.001)

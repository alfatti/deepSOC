"""Gate P1: the paper's (price, vega, IV) table against an independent
semi-analytic Heston pricer, plus internal pricer gates (martingale identity,
Black-Scholes limit, quadrature convergence).

Tolerances reflect the paper's noise sources: 1e5-path Monte Carlo per option
plus two-decimal rounding (four for IV). The pricer itself is verified to far
higher accuracy by the internal gates.
"""
import numpy as np

from mfoptmm import bbg_example, bs_call, heston_call, verify_paper_table
from mfoptmm.instruments import _heston_cf, PAPER_TABLE

SPEC = bbg_example()
MKT = SPEC.market

PRICE_TOL = 0.02
VEGA_TOL = 0.03
# IV: the paper's IVs come from their *unrounded* MC prices. For deep-ITM K=8
# the BS IV-vega is only ~1.4, so their ~1e-2 (2 sigma) MC price noise maps to
# up to ~7e-3 of IV; observed max discrepancy is 2.7e-3 (all K=8 entries), with
# the corresponding *prices* agreeing to <1e-2 — i.e. their printed price and
# IV disagree with each other at exactly the MC-noise level.
IV_TOL = 0.004


def test_martingale_identity():
    phi = _heston_cf(np.array([-1j]), 2.0, np.log(MKT.S0), MKT.nu0,
                     MKT.kappaQ, MKT.thetaQ, MKT.xi, MKT.rho)[0]
    assert abs(phi - MKT.S0) < 1e-10 * MKT.S0


def test_bs_limit_small_xi():
    """xi -> 0 with nu0 = thetaQ pins the integrated variance at nu0, so the
    Heston price must approach Black-Scholes at sigma = sqrt(nu0) = 0.15.

    Run at rho = 0: the leading smile correction is O(rho * xi) — first order
    in xi — so any rho != 0 leaves a ~2e-4 skew residual at xi = 1e-3 that is
    genuine Heston, not pricer error. With rho = 0 the residual is O(xi^2) and
    the observed gap is ~2e-7 (quadrature floor)."""
    for (K, T) in sorted(PAPER_TABLE):
        h = heston_call(MKT.S0, K, T, MKT.nu0, MKT.kappaQ, MKT.thetaQ,
                        1e-3, 0.0)
        b = bs_call(MKT.S0, K, 0.15, T)
        assert abs(h - b) < 1e-6, f"(K,T)=({K},{T}): heston {h:.8f} vs bs {b:.8f}"


def test_quadrature_convergence():
    for (K, T) in [(8.0, 1.0), (10.0, 1.0), (12.0, 3.0)]:
        p1 = heston_call(MKT.S0, K, T, MKT.nu0, MKT.kappaQ, MKT.thetaQ,
                         MKT.xi, MKT.rho, n_gl=800, u_max=200.0)
        p2 = heston_call(MKT.S0, K, T, MKT.nu0, MKT.kappaQ, MKT.thetaQ,
                         MKT.xi, MKT.rho, n_gl=1600, u_max=400.0)
        assert abs(p1 - p2) < 1e-9, f"(K,T)=({K},{T}): {p1!r} vs {p2!r}"


def _fmt(rows):
    hdr = (f"{'K':>4} {'T':>4} | {'P_paper':>8} {'P_mine':>8} {'dP':>8} | "
           f"{'V_paper':>8} {'V_mine':>8} {'dV':>8} | "
           f"{'IV_paper':>8} {'IV_mine':>8} {'dIV':>8}")
    lines = [hdr]
    for r in rows:
        lines.append(
            f"{r['strike']:>4.0f} {r['mat']:>4.1f} | "
            f"{r['price_paper']:>8.3f} {r['price_mine']:>8.4f} {r['dprice']:>8.4f} | "
            f"{r['vega_paper']:>8.3f} {r['vega_mine']:>8.4f} {r['dvega']:>8.4f} | "
            f"{r['iv_paper']:>8.4f} {r['iv_mine']:>8.4f} {r['div']:>8.4f}")
    return "\n".join(lines)


def test_gate_p1_paper_table():
    rows = verify_paper_table(MKT)
    table = _fmt(rows)
    dp = max(abs(r["dprice"]) for r in rows)
    dv = max(abs(r["dvega"]) for r in rows)
    di = max(abs(r["div"]) for r in rows)
    msg = (f"\nGate P1 report (max |dP|={dp:.4f}, |dV|={dv:.4f}, |dIV|={di:.4f})\n"
           f"{table}")
    assert dp <= PRICE_TOL and dv <= VEGA_TOL and di <= IV_TOL, msg

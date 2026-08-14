"""Gate E1: replication of BBG's published example (body variant, their grid).

What this gate pins, and what it does not
-----------------------------------------
The equation, parameters and scheme are verbatim-verified against the paper
(objective (2), the reduced HJB with H^{i,j} and the post-trade gate, the
lambda/alpha/beta/z/Vbar/T/gamma values). This solve is internally exact
(envelope-identity evaluator at 1.8e-16 relative; Remark-6 collapse at 0.0)
and refinement-converged (dt ladder ratio 2.03; 0.03% over M 180 -> 1440;
0.5% over n_V 40 -> 80).

Against the paper's figures (tick-level, curves not machine-readable here):

* Figure 3 (raw (8,1) quotes, ticks [-0.0025, 0.0125]): matched tick-exactly
  (measured [-0.00261, +0.01160]).
* Figures 9-10 (IV ratios, K=8/9, ticks [0.94, 1.02]): matched.
* Figures 4-8 tick ranges imply skews 15-50% deeper at the wide end than both
  this solve *and the paper's own Figure 3* (their Fig 3 and Fig 4 mutually
  contradict for the same options), and Figure 2's value scale (<= 1.2e5) is
  below this converged solve's 1.63e5. A gamma = 2e-3 vintage probe
  reproduces Figs 5-6 tick-exactly but breaks Figs 3/9-10; no single
  parameter set reconciles the printed figure family. See README ("E1
  finding"); same class of paper-internal inconsistency as Gate P1's
  price-vs-IV mismatch. Pixel extraction from the PDF figures would close it.

The automated gate therefore asserts: the two tick-exact matches, tick
containment for Figures 4-8 (this solve must never exceed the printed
envelopes), quote structure (monotonicity, stationarity, nu-flatness), and
this solve's own value-scale contract.
"""
import numpy as np
import jax.numpy as jnp
import mfoptmm as mf
from mfoptmm.hjb import make_ops, at_state, interp_axis

NU0 = 0.0225
TICKS = {8: (-0.0075, 0.0125), 9: (-0.01, 0.03), 10: (-0.02, 0.08),
         11: (-0.05, 0.15), 12: (-0.05, 0.25)}


def _bid_at_nu0(spec, sol):
    """Gate-masked t=0 bid maps at nu0: (n_V, N) with NaN where inadmissible."""
    ops = make_ops(spec)
    q0 = np.asarray(interp_axis(sol.quotes[0], ops.nu, NU0, 0))
    g = np.asarray(ops.gate)
    return np.where(g[:, :, 0], q0[:, :, 0], np.nan)


def test_fig3_raw_quote_band(body):
    # (8,1) raw bid range vs Figure 3 ticks [-0.0025, 0.0125].
    # Measured [-0.00261, +0.01160].
    spec, sol = body
    K, Tm = np.asarray(spec.book.strike), np.asarray(spec.book.mat)
    bid = _bid_at_nu0(spec, sol)
    i = int(np.where((K == 8) & (Tm == 1.0))[0][0])
    lo, hi = np.nanmin(bid[:, i]), np.nanmax(bid[:, i])
    assert -0.004 < lo < -0.0015
    assert 0.010 < hi < 0.0135


def test_fig4_8_tick_containment_and_span(body):
    spec, sol = body
    K = np.asarray(spec.book.strike)
    O0 = np.asarray(spec.book.O0)
    bid = _bid_at_nu0(spec, sol)
    for k, (lo_t, hi_t) in TICKS.items():
        r = bid[:, K == k] / O0[None, K == k]
        lo, hi = np.nanmin(r), np.nanmax(r)
        assert lo_t <= lo and hi <= hi_t, f"K={k}: [{lo:.4f},{hi:.4f}]"
        assert hi >= 0.4 * hi_t, f"K={k}: max {hi:.4f} too shallow vs tick {hi_t}"


def test_fig9_10_iv_ratio_bands(body):
    # Measured K=8: [0.9483, 1.0110]; K=9: [0.9424, 1.0132]; ticks [0.94, 1.02].
    spec, sol = body
    K, Tm = np.asarray(spec.book.strike), np.asarray(spec.book.mat)
    O0 = np.asarray(spec.book.O0)
    bid = _bid_at_nu0(spec, sol)
    for k in (8.0, 9.0):
        lo, hi = np.inf, -np.inf
        for i in np.where(K == k)[0]:
            iv0 = mf.implied_vol(O0[i], 10.0, K[i], Tm[i])
            for b in bid[:, i]:
                if np.isfinite(b):
                    r = mf.implied_vol(O0[i] - b, 10.0, K[i], Tm[i]) / iv0
                    lo, hi = min(lo, r), max(hi, r)
        assert 0.935 < lo < 0.958, f"K={k}: IV min {lo:.4f}"
        assert 1.004 < hi < 1.020, f"K={k}: IV max {hi:.4f}"


def test_quote_monotonicity(body):
    # Paper: "optimal mid-to-bid quotes are increasing with the vega of the
    # portfolio"; the ask mirrors.
    spec, sol = body
    bid = _bid_at_nu0(spec, sol)
    ops = make_ops(spec)
    q0 = np.asarray(interp_axis(sol.quotes[0], ops.nu, NU0, 0))
    ask = np.where(np.asarray(ops.gate)[:, :, 1], q0[:, :, 1], np.nan)
    for i in range(bid.shape[1]):
        db = np.diff(bid[~np.isnan(bid[:, i]), i])
        da = np.diff(ask[~np.isnan(ask[:, i]), i])
        assert db.min() > -1e-9 and db.sum() > 0.0
        assert da.max() < 1e-9 and da.sum() < 0.0


def test_stationarity_at_t0(body):
    # Figure 3's claim: quotes converged to stationary values by t = 0.
    # Measured max |q(0) - q(0.25 T)| = 1.75e-5 EUR on a ~8e-3 EUR scale.
    _, sol = body
    drift = np.abs(np.asarray(sol.quotes[0])
                   - np.asarray(sol.quotes[45])).max()
    assert drift < 1e-3


def test_nu_dependence_low(body):
    # Paper: "the dependence in nu is low". Measured 3.5e-6 EUR max spread.
    spec, sol = body
    g = np.asarray(make_ops(spec).gate)
    q = np.where(g[None], np.asarray(sol.quotes[0]), np.nan)
    finite = np.isfinite(q)
    hi = np.where(finite, q, -np.inf).max(axis=0)
    lo = np.where(finite, q, np.inf).min(axis=0)
    spread = float(np.max(np.where(finite.any(axis=0), hi - lo, 0.0)))
    assert spread < 1e-3


def test_value_scale_contract(body):
    # Own converged scale (see module docstring re: Figure 2): centre
    # 163,021 EUR (M-ladder converged to 162,972; n_V=80 gives 163,858),
    # minimum 53,053 EUR.
    spec, sol = body
    c = float(at_state(sol.v0, make_ops(spec), NU0, 0.0))
    assert 1.5e5 < c < 1.75e5
    assert float(sol.v0.min()) > 4e4

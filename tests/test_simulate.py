"""Gate E5: simulator <-> PDE-stack consistency, plus the Level-0 tau study.

The consistency triangle this file closes (measured values in comments):

* HJB == policy evaluator: machine-exact (test_policy_eval, 1.8e-16 rel).
* Evaluator == simulator: at MC precision once spatially matched. The MC
  mean of the (interpolated ZOH) optimal schedule sits *inside the spatial
  refinement bracket* [v_pde(n_V=40), v_pde(n_V=161)] at every start, and is
  flat under simulator dt-refinement (the residual is PDE spatial error, not
  simulator error). On the myopic policy — wild, limit-hugging paths — the
  40-node evaluator underprices by +3.1%, and the 161-node evaluator recovers
  it to +0.15% (< 1 sigma): the discrepancy was reference-side spatial error.

Consequence recorded for milestone 5: the optimality-gap arbiter must run
policy_eval at n_V >= 161; the 40-node grid's policy-dependent 0.3-3% spatial
error would swamp learned-policy gaps of similar size.

Tau study (fixed CRN, optimal schedule): the sigmoid-uniform relaxation's
bias is non-monotone in tau (state under-dispersion pushes value up, small-
lambda*dt channels over-fire and push it down; they cross near tau ~ 5e-3)
with a slowly decaying O(tau) tail: +9.8k / +6.6k / +2.9k EUR at tau = 1e-3 /
5e-4 / 2e-4. Design consequence for milestone 4 (reverses the strategy doc's
default): straight-through gradients primary, soft fills as the ablation.
"""
import numpy as np
import jax
import mfoptmm as mf
from mfoptmm.hjb import make_ops, at_state
from mfoptmm.policy_eval import policy_value, myopic_policy
from mfoptmm.simulate import (draw_vpi0, refine_schedule, refined_spec,
                              simulate_schedule)

NU0 = 0.0225
KEY = jax.random.PRNGKey(0)


def _starts(ops):
    V = np.asarray(ops.V)
    return [float(V[19]), float(V[12]), float(V[27])]   # on-grid Vpi0


def test_gate_e5_bracket(body, fine):
    # Measured (n=20k): sim 163617/147236/147476 (+-175..195) vs
    # v40 163014/146837/147033 and v161 164086/147941/148136.
    spec, sol = body
    specF, solF = fine
    ops, opsF = make_ops(spec), make_ops(specF)
    n = 20000
    for v0s in _starts(ops):
        res = simulate_schedule(spec, sol.quotes, KEY, n, Vpi0=v0s)
        m = float(res.objective.mean())
        se = float(res.objective.std()) / np.sqrt(n)
        lo = float(at_state(sol.v0, ops, NU0, v0s))
        hi = float(at_state(solF.v0, opsF, NU0, v0s))
        assert lo - 3 * se <= m <= hi + 3 * se, \
            f"Vpi0={v0s:.2e}: sim {m:.0f} outside [{lo:.0f}, {hi:.0f}] +- 3se"
        assert abs(m - lo) <= 0.006 * lo and abs(m - hi) <= 0.006 * hi


def test_e5_simulator_dt_flatness(body):
    # The bias vs a fixed reference is flat under simulator dt-halving with
    # the same ZOH maps (measured +603 -> +659 at n=20k, within noise): the
    # simulator's own time error is sub-SE.
    spec, sol = body
    ops = make_ops(spec)
    v0s = _starts(ops)[0]
    n = 10000
    m1 = float(simulate_schedule(spec, sol.quotes, KEY, n,
                                 Vpi0=v0s).objective.mean())
    m2 = float(simulate_schedule(refined_spec(spec, 2),
                                 refine_schedule(sol.quotes, 2), KEY, n,
                                 Vpi0=v0s).objective.mean())
    se = float(simulate_schedule(spec, sol.quotes, KEY, n,
                                 Vpi0=v0s).objective.std()) / np.sqrt(n)
    assert abs(m2 - m1) <= 3 * np.sqrt(2) * se


def test_realised_equals_compensated(body):
    # Same conditional mean by compensation; CRN-paired difference is
    # zero-mean (measured +365 +- 412 at n=20k). Compensated booking halves
    # the SE (174 vs 338), which is why it is the primary estimator.
    spec, sol = body
    ops = make_ops(spec)
    v0s = _starts(ops)[0]
    n = 10000
    rc = simulate_schedule(spec, sol.quotes, KEY, n, Vpi0=v0s)
    rr = simulate_schedule(spec, sol.quotes, KEY, n, Vpi0=v0s, realized=True)
    d = np.asarray(rr.objective) - np.asarray(rc.objective)
    assert abs(d.mean()) <= 4 * d.std() / np.sqrt(n)
    assert rr.objective.std() > rc.objective.std()


def test_myopic_matched_resolution(body, fine):
    # 40-node evaluator underprices the limit-hugging myopic policy by +3.1%;
    # the 161-node evaluator recovers it: measured sim - u161 = +128 +- 733.
    spec, _ = body
    specF, _ = fine
    ops, opsF = make_ops(spec), make_ops(specF)
    v0s = _starts(ops)[0]
    n = 10000
    r = simulate_schedule(spec, myopic_policy(spec), KEY, n, Vpi0=v0s)
    m = float(r.objective.mean())
    se = float(r.objective.std()) / np.sqrt(n)
    u40 = float(at_state(policy_value(spec, myopic_policy(spec)).u0,
                         ops, NU0, v0s))
    u161 = float(at_state(policy_value(specF, myopic_policy(specF)).u0,
                          opsF, NU0, v0s))
    assert u161 - u40 > 0.02 * u40          # the 40-node underpricing, exact:
    #                                         measured +2471 EUR (+2.95%)
    assert abs(m - u161) <= max(3 * se, 0.01 * u161)


def test_tau_ladder(body):
    # Measured biases +9.8k / +6.6k / +2.9k at tau = 1e-3 / 5e-4 / 2e-4:
    # decaying ~O(tau) tail. See module docstring for the milestone-4
    # consequence (straight-through primary).
    spec, sol = body
    ops = make_ops(spec)
    v0s = _starts(ops)[0]
    n = 10000
    hard = float(simulate_schedule(spec, sol.quotes, KEY, n,
                                   Vpi0=v0s).objective.mean())
    b = [abs(float(simulate_schedule(spec, sol.quotes, KEY, n, Vpi0=v0s,
                                     tau=t).objective.mean()) - hard)
         for t in (1e-3, 5e-4, 2e-4)]
    assert b[0] > b[1] > b[2], f"tau ladder not decreasing: {b}"
    ref = float(at_state(sol.v0, ops, NU0, v0s))
    assert b[2] < 0.025 * ref


def test_crn_determinism(body):
    spec, sol = body
    a = simulate_schedule(spec, sol.quotes, KEY, 2000, Vpi0=0.0)
    b = simulate_schedule(spec, sol.quotes, KEY, 2000, Vpi0=0.0)
    assert float(np.abs(np.asarray(a.objective)
                        - np.asarray(b.objective)).max()) == 0.0
    c = simulate_schedule(spec, sol.quotes, jax.random.PRNGKey(1), 2000,
                          Vpi0=0.0)
    assert float(np.abs(np.asarray(a.objective)
                        - np.asarray(c.objective)).max()) > 0.0


def test_draw_vpi0(body):
    spec, _ = body
    v = np.asarray(draw_vpi0(KEY, 5000, spec.risk.Vbar))
    assert v.shape == (5000,)
    assert np.abs(v).max() <= 0.5 * spec.risk.Vbar
    assert np.abs(v).max() > 0.4 * spec.risk.Vbar

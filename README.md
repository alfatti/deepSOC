# mfoptmm

Mean-field learned quoting vs. exact HJB on the Baldacci–Bergault–Guéant (BBG)
options market-making problem, Appendix A.1 variant (partial delta hedge).
Instantiates the Baros–Cohen–Reisinger (BCR) mean-field ERM framework on a
control problem whose exact solution we can compute, so that the
generalisation gap (BCR's object) and the **optimality gap** (novel here) can
both be measured. Full research design: `docs/strategy.md`.

## Status

| Milestone | Contents | Gates | Status |
|---|---|---|---|
| **1** | `model.py`, `instruments.py`, `config.py` | E3, P1, ω-oracle, identities | **done — 27/27 green** |
| **2** | `hjb.py` exact 2-D monotone solver + quote extraction; `policy_eval.py` | E1 (figure replication), E2 (ν-collapse), E4 (runtime realised diagonal), block-frozen pricing | **done — 47/47 green** |
| **3** | `simulate.py` (lockstep CRN, soft/hard fills, compensated/realised booking) | E5 (simulator ↔ PDE consistency triangle), τ study | **done — 54/54 green** |
| **4** | `nets.py`, `train.py` (BCR Algorithm 1; straight-through primary per the τ study; η head; η-aware arbiter) | trainer gates (identity, gradients, η-line, smoke induction, ERM determinism) | **done — 59/59 green** |
| 5 | `compare.py`, notebooks, `tests/run_h200.py` sweep driver | Levels 0–4 comparison stack | planned |

## Milestone-1 gate numbers (observed, CPU x64)

| Gate | Check | Observed | Threshold |
|---|---|---|---|
| ω-oracle | `wright_omega` vs `scipy.special.wrightomega`, ℓ ∈ [−600, 600] | 2.9e-15 rel | 1e-12 |
| E3 | H vs brute-force sup over δ, 20 options × 41 p | 4.4e-16 rel | 1e-10 |
| E3 | δ* vs brute force | 2.3e-9 abs (Brent xatol-limited) | 1e-7 |
| E3 | envelope H′(p) = −Λ(δ*(p)) | 1.1e-13 rel | 1e-12 |
| E3 | Λ⁻¹(−H′) = δ* (quote-extraction path) | < 1e-10 | 1e-10 |
| P1 | paper prices vs semi-analytic Heston | max \|dP\| = 0.0093 | 0.02 |
| P1 | paper vegas | max \|dV\| = 0.0192 | 0.03 |
| P1 | paper IVs | max \|dIV\| = 0.0027 | 0.004 |
| — | Heston → BS limit (ρ=0, ξ=1e-3) | 2.0e-7 | 1e-6 |

P1 tolerances reflect the **paper's** noise (1e5-path MC + 2-decimal rounding),
not the pricer's; the BS-limit and martingale gates pin the pricer itself to
quadrature precision. The IV outliers are all deep-ITM K=8, where BS IV-vega
≈ 1.4 amplifies the paper's MC price noise into ~30 bp of IV — their printed
price and IV are mutually inconsistent at exactly that level. The BS-limit
test runs at ρ=0 because the leading smile correction is O(ρξ), first order
in ξ; at ρ=−0.5 a ~2e-4 residual is genuine Heston, not pricer error.

## Milestone-2 gate numbers (observed, body variant, BBG replication grid)

| Gate | Check | Observed | Threshold |
|---|---|---|---|
| E4 | realised scheme diagonal, M=180 (worst-case bound rejects this grid) | min 0.716 | > 0 (runtime) |
| E4 | realised diagonal, worst-case profile M=252 | min 0.797 | > 0.7 |
| E4 | dt-refinement first-order contract (M 180/360/720) | ratio 2.03 | ∈ [1.5, 2.7] |
| E2 | Remark-6 ν-collapse (a^P = a^Q) | 0.0 (exact) | ≤ 1e-9 rel |
| E2 | vs independent numpy 1-D recursion (`np.interp`, own loop) | ~1e-9 rel | ≤ 1e-6 rel |
| E2 | bid/ask mirror symmetry under carry = 0 | 1.3e-16 | ≤ 1e-9 |
| — | envelope identity through the full backward pass (evaluator ≡ HJB) | 1.8e-16 rel | ≤ 1e-8 rel |
| — | frozen-block (K=36 ZOH) restriction price | 1.77e-4 of v* | < 5e-3 |
| — | myopic baseline / v* at (ν₀, 0) | 0.514 | ∈ [0.35, 0.70] |
| — | terminal quote map ≡ myopic (v(T)=0 ⇒ p=0) | 1.7e-18 | ≤ 1e-12 |
| E1 | Fig 3 raw (8,1) quote range vs ticks [−0.0025, 0.0125] | [−0.0026, 0.0116] | tick band |
| E1 | Figs 9–10 IV-ratio ranges vs ticks [0.94, 1.02] | [0.948, 1.011] / [0.942, 1.013] | tick bands |
| E1 | Figs 4–8 tick containment + span; monotonicity; stationarity (1.75e-5 €); ν-flatness (3.5e-6 €) | all pass | — |
| — | v_A.1 ≥ v_body pointwise (penalty 0.75×, comparison principle) | min gap +5.9e3 € | > 0 |

Headline numbers for the study: v*(0, ν₀, 0) = **163,021 €** (converged: M-ladder
→ 162,972; n_V 40→80 moves +0.5%); the K=36 decision-grid restriction costs
**~2 bp of v***; the myopic baseline captures **51.4%** — the inventory-management
channel the learner must discover is worth about half the achievable value.

### E1 finding: the paper's printed figures are mutually inconsistent

The equation, parameters and scheme are verbatim-verified against the paper
(objective (2), reduced HJB, Remark 6, all Section-4.1 values), and this solve
is internally exact and refinement-converged. Against the figures: **Figure 3**
(raw (8,1) quotes) and **Figures 9–10** (K=8/9 IV ratios) match tick-exactly.
**Figures 4–8** imply wide-end skews 15–50% deeper than both this solve *and
the paper's own Figure 3* (their Figs 3 and 4 contradict each other for the
same four options), and **Figure 2**'s value scale (≤ 1.2e5) sits below the
converged 1.63e5. A γ = 2e-3 vintage probe reproduces Figs 5–6 tick-exactly
(K=9: [−0.0077, 0.0293] vs ticks (−0.01, 0.03); K=10: [−0.0213, 0.0781] vs
(−0.02, 0.08)) but breaks Figs 3/9–10; no single parameter set reconciles the
whole family — same class of paper-internal inconsistency as Gate P1's
price-vs-IV mismatch, presumably mixed figure vintages. Pixel extraction from
the PDF figures (not mounted in this container) would close it fully.

### Design note for milestone 3 (simulator)

Measured max per-step total dt·ΣΛ* = 0.336 on the M=180 grid — the *sum* over
40 (option, side) channels. Per channel, λ_max·dt = 0.05, so independent
per-channel Bernoulli thinning at the PDE dt is faithful to O((λdt)²) ≈ 2.5e-3
per channel-step; Gate E5 (simulator ↔ PDE value identity under the optimal
policy) quantifies the aggregate. Multiple fills per step are expected and
handled channel-independently.

## Milestone-3 gate numbers (Gate E5, body variant, n = 20k paths)

| Check | Observed | Gate |
|---|---|---|
| E5 bracket: sim mean ∈ [v_pde(n_V=40), v_pde(n_V=161)] at 3 starts | 163,617 ∈ [163,014, 164,086] (±175); likewise at ±3.85e6 | ± 3·SE slack |
| E5 vs either reference | ≤ 0.48% | ≤ 0.6% |
| Simulator dt-flatness (same ZOH maps, dt/2, dt/4) | +603 → +659 → +663 € (flat, sub-SE) | ≤ 3√2·SE |
| Realised ≡ compensated booking (CRN-paired) | +365 ± 412 € | ≤ 4·SE |
| Compensated vs realised SE | 174 vs 338 € | comp. primary |
| Myopic at matched resolution (n_V=161 evaluator) | +128 ± 733 € (+0.15%) | ≤ max(3·SE, 1%) |
| Myopic at n_V=40 evaluator (spatial underpricing, exact) | +2,471 € (+2.95%) | documented |
| τ ladder bias (1e-3 / 5e-4 / 2e-4) | +9.8k / +6.6k / +2.9k € | decreasing; final < 2.5% |
| CRN determinism | bitwise | exact |

**The consistency triangle is closed**: HJB ≡ evaluator at 1.8e-16 relative;
evaluator ≡ simulator at MC precision on two very different policies once
spatially matched. The simulator's own time error is sub-SE (dt-flatness);
residuals against coarse references are PDE spatial error, cleanly bracketed.

**Consequence for milestone 5**: the optimality-gap arbiter runs `policy_eval`
at n_V ≥ 161. The 40-node grid's policy-dependent spatial error (+0.3% on the
optimal policy, +3% on myopic) would swamp learned-policy gaps of that size.

**Consequence for milestone 4 (reverses the strategy-doc default)**: the
sigmoid-uniform soft-fill relaxation carries a non-monotone, slowly decaying
O(τ) bias (channels with λΔt ~ 3e-4 over-fire by ~τ·log2 each; fractional
fills under-disperse the state; the effects cross near τ ≈ 5e-3 and the tail
is still +2.9k € at τ = 2e-4). Straight-through gradients become the primary
training path; soft fills are the ablation. The simulator exposes both via
`tau` on shared CRN.

Simulator design: counter-based keys (fold_in per step; no materialised noise
arrays — GPU-ready), quotes as scanned `xs`, bilinear edge-clamped policy
lookup, per-channel Bernoulli thinning (λ_max·dt = 0.05), reachable set
extends to ±(𝒱̄ + max z𝒱) via multi-fire steps as anticipated in the design
doc.

## Milestone-4 gate numbers (A.1 variant, K = 4 smoke, seed-pinned)

| Gate | Check | Observed | Threshold |
|---|---|---|---|
| — | η-aware arbiter identity: `policy_value(eta = η*)` ≡ reduced | 2.9e-11 abs | ≤ 1e-8 rel |
| — | soft-mode rollout gradient vs central FD | 8.9e-7 rel | < 1e-4 |
| — | straight-through gradients | finite, nonzero | — |
| — | last-block η head vs closed form η* = +0.05 𝒱π | slope +0.0440, corr +0.999 | slope ∈ [0.03, 0.07], corr > 0.98 |
| — | smoke backward induction: captured share of (v* − J_ref) | 32.5% | > 20% |
| — | dominance: max(u_learned − v*) | −5.7e4 € | < 1e-6 |
| — | generalisation gap J_train − J_fresh at n = 32 | +8,418 € | > 1e3 |
| — | ERM determinism (same seeds → bitwise-equal params) | exact | exact |

Trainer structure: BCR Algorithm 1 — backward induction over blocks with
frozen trained downstream, reference-control roll-ins (myopic quotes, η = 0,
hard fills) from randomised 𝒱π₀ ~ U[−𝒱̄/2, 𝒱̄/2], compensated reward with the
hedge-channel penalty at the learned η, per-neuron noisy GD with the BCR
regulariser Γ(θ) = ‖θ‖^p + 1e-9·e^‖θ‖, geometric lr decay. Nets are per-block
mean-field one-hidden-layer (1/r output scaling); outputs are offsets from
the reference in natural units (quotes in 1/b spread units, η in
(|ρ|ξ/2)𝒱̄ units), so a zero net *is* the reference policy. ERM sample sets
are fixed by counter-based keys and never materialised; fresh-sample
evaluation uses a disjoint key.

Two training-dynamics findings:

1. **The loss must be normalised** by `objective_scale` (the myopic revenue
   supersolution bound, T·2Σz^iH^i(0) ≈ 2.05e5 €) — with the raw EUR
   objective, the per-neuron drift r·∇ explodes at any sane lr and training
   NaNs within epochs.
2. **Canonical MFLD noise** √(2·lr)/β replaces the Overlearning repo's
   σ = √0.1·β, which is ~20× hotter than the β = 100 stationary measure
   warrants; σ stays an explicit knob for the H200 sweep to replicate either.

The η head recovering the closed-form hedge line (slope +0.044 vs +0.05,
corr 0.999, from 400 epochs on the last block alone) is the decisive
machinery check: a known-linear optimal control is discovered from rollouts.
The 32.5% capture at r=32/n=32/300-epoch smoke scale leaves the n-sweep
science (gap vs n^{-1}, β ∝ n^{1/4} arm, unregularised arm, K = 36, r ∈
{256, 1024}, ~20k epochs) to `tests/run_h200.py` in milestone 5.

## Milestone 5: the comparison stack

`mfoptmm/compare.py` implements Levels 1–4 (Level 0 = the identity gates in
the test suite):

* **Level 1** — generalisation gap |J_train − J_fresh| per (arm, n, r, trial);
  log-log boxplots vs n with the n⁻¹ guide.
* **Level 2** — optimality gap v* − J(learned) via fresh hard-fill MC and the
  exact evaluator at n_V = 161 (per the milestone-3 finding), start-averaged
  over 𝒱π₀ ~ U[−𝒱̄/2, 𝒱̄/2] (±𝒱̄/2 are exact 161-grid nodes); the reference
  control's gap locates the starting line.
* **Level 3** — learned vs exact quote curves and the η line at the
  mid-horizon block; per-option errors both raw-sup and
  **visitation-weighted L²** (weights = the learned policy's 𝒱π occupation
  under fresh noise). Smoke measurement: sup 1.9–2.5 vs visit-L² 0.9–1.0
  spread units — the policy is good where its paths go; the sup lives on
  never-visited states the BCR bound does not control.
* **Level 4** — the overlearning exhibit: unregularised small-n inventory
  paths on the training environments vs fresh ones.

`tests/run_h200.py` is the staged sweep driver: per-trial `.npz` checkpoints
(skip-if-present, so it survives preemption; validated by resuming a
timed-out CPU smoke run), figures and the gap table rebuilt from disk after
every run. Stages: `smoke` (CPU, minutes: K=4, r=32, 300 epochs, n ∈ {8,32},
reg/unreg — results shipped in `results/smoke/`), `main` (H200: K=36, r=256,
20k epochs, n ∈ {8,16,32,64,128} × {reg, unreg, beta_n14 = 100(n/32)^{1/4}}
× 20 trials), `wide` (r=1024 subset). Per-trial cost at smoke scale ≈ 50 s
CPU warm.

`notebooks/01_exact_hjb.ipynb` (the exact stack: figure analogues, value
surface, baselines) and `notebooks/02_learning_vs_exact.ipynb` (Levels 1–4;
re-entrant: loads `results/main` when present, else `results/smoke`, else
runs the smoke stage inline; scales up automatically on GPU). Both validated
by cell-by-cell execution on CPU; outputs land in `docs/figures/`.

Smoke-scale caveat, stated once: at 2 trials × 300 epochs the Level-1/2
tables are machinery validation, not science — the n⁻¹ law, the temperature
arm and the unregularised contrast belong to the H200 main stage.

## Layout

```
mfoptmm/
  model.py        # single source of truth: params, Λ, Lambert-W Hamiltonian
                  # (Wright ω, custom_jvp), penalties, hedge channel, gate
  instruments.py  # paper (price, vega, IV) table + semi-analytic Heston
                  # verification pricer (NumPy; never in a hot loop)
  config.py       # grids; rate-driven Δt derivation; bbg_example() factory
  simulate.py     # lockstep CRN forward simulator (schedule policies,
                  # hard/soft fills, compensated/realised booking)
  nets.py         # per-block mean-field control nets (quote offsets + η head)
  compare.py      # Levels 1–4 metrics, trial runner, figure builders
  train.py        # BCR Algorithm 1 (backward induction, reference roll-ins,
                  # straight-through fills, per-neuron noisy GD + Γ)
tests/            # the gates; run `pytest tests/ -q` (60 tests, ~4.5 min CPU)
docs/strategy.md  # the full research design
```

Every consumer (solver, simulator, trainer, evaluators) imports `model.py`
primitives; none re-implements them, so cross-consumer identity tests catch
discretisation bugs at the source.

## Design invariants worth knowing

- **Δt is rate-set, not T-set.** The explicit scheme's diagonal is
  1 − Δt·(Σ2λⁱ + ν_max ξ²/Δν² + max|a^P|/Δν); the intensity sum
  (182,033 /yr) binds, giving Δt_bound ≈ 5.39e-6 yr. `config.derive_time_grid`
  parametrises (safety, K_blocks, T) and derives (M, K). Profiles:
  `worst_case` (M=252 at BBG's T) and `replication` (M=180, BBG's grid, which
  nominally violates the worst-case bound — the milestone-2 solver must assert
  the *realised* diagonal ≥ 0 at runtime, Gate E4).
- **A.1 = body problem with c_ρ = 1−ρ² = 0.75** on the running penalty; the
  ν-diffusion keeps physical ξ. Hedge channel η* = −(ρξ/2)𝒱π (slope +0.05);
  `penalty_hedge` at η* equals `penalty_reduced` (identity-tested).
- **Deep-tail behaviour is exact-by-underflow**: for bp ≫ 1, ω underflows
  (XLA flush-to-zero) so H = 0, H′ = −0.0 — the correct limit, and why sign
  tests are strict only on the realistic quote band.
- **x64 is enabled at `import mfoptmm`.** Import it before creating JAX
  arrays anywhere else in the process.

## Install

```bash
pip install -e .            # CPU
pip install -U "jax[cuda12]"   # on the H200 box, after the editable install
pytest tests/ -q            # 27 tests, ~25 s CPU
```

Nothing in milestone 1 is GPU-bound; GPU lanes (whole-loop `lax.scan`
kernels, auto-CUDA smoke/full profiles, staged `tests/run_h200.py` driver
with checkpointing) arrive with the solver and trainer milestones.

## Provenance

`PAPER_TABLE` carries the (price, vega, IV) triples verbatim from BBG figure
legends (Figs 4–13). `paper_book()` (default) uses them so λⁱ, zⁱ, z𝒱ⁱ match
the published example exactly; `pricer_book()` regenerates the book from the
semi-analytic pricer under Q (κ=3, θ=0.0225). Trade vega prints span ~1e5
(deep-ITM K=8) to ~4.5e6 (K=12 short) against 𝒱̄ = 1e7.

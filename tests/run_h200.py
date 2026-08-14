#!/usr/bin/env python
"""Staged sweep driver for the H200 (resumable; CPU-runnable at smoke scale).

Usage:
    python tests/run_h200.py --stage smoke              # CPU sanity, minutes
    python tests/run_h200.py --stage main  --trials 20  # H200 headline sweep
    python tests/run_h200.py --stage wide  --trials 8   # r = 1024 subset
    python tests/run_h200.py --figures results/main     # rebuild figures only

Stages (A.1 variant throughout; straight-through fills):

  smoke : K = 4,  r = 32,  epochs = 300,  n in {8, 32},           arms {reg, unreg}
  main  : K = 36, r = 256, epochs = 20000, n in {8,16,32,64,128}, arms {reg, unreg, beta_n14}
  wide  : K = 36, r = 1024, epochs = 20000, n in {16, 64},        arms {reg}

beta_n14: beta = 100 * (n / 32)^{1/4} — BCR's rate-matched temperature arm.

Each (arm, n, r, trial) checkpoints to <out>/<tag>.npz and is skipped when
present, so the driver survives preemption; rerun to resume. Figures and the
gap table are rebuilt from whatever is on disk at the end of every run.
"""
import argparse
import sys
import time
from pathlib import Path

import jax

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mfoptmm as mf                                   # noqa: E402
from mfoptmm import compare                            # noqa: E402


def stage_configs(stage: str, trials: int):
    if stage == "smoke":
        K, r, epochs, ns, arms = 4, 32, 300, (8, 32), ("reg", "unreg")
    elif stage == "main":
        K, r, epochs = 36, 256, 20000
        ns, arms = (8, 16, 32, 64, 128), ("reg", "unreg", "beta_n14")
    elif stage == "wide":
        K, r, epochs, ns, arms = 36, 1024, 20000, (16, 64), ("reg",)
    else:
        raise SystemExit(f"unknown stage {stage!r}")
    for arm in arms:
        for n in ns:
            for trial in range(trials):
                beta = 100.0 * (n / 32.0) ** 0.25 if arm == "beta_n14" \
                    else 100.0
                cfg = mf.TrainConfig(
                    r=r, n=n, epochs=epochs, lr0=0.3, lr1=1e-3, beta=beta,
                    regularised=(arm != "unreg"), tau=0.0, seed=1000 * trial + n)
                yield K, arm, trial, cfg


def build_figures(out: Path):
    results = compare.load_trials(out)
    if not results:
        print(f"no results in {out}")
        return
    print(f"\n{len(results)} trials loaded from {out}\n")
    print(compare.gap_table(results))
    compare.plot_gap_vs_n(results, out / "level1_gen_gap.png", level="gen")
    compare.plot_gap_vs_n(results, out / "level2_opt_gap.png", level="opt")
    print(f"figures -> {out}/level1_gen_gap.png, level2_opt_gap.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="smoke",
                    choices=["smoke", "main", "wide"])
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--out", default=None)
    ap.add_argument("--figures", default=None,
                    help="rebuild figures from an existing results dir")
    args = ap.parse_args()
    if args.figures:
        build_figures(Path(args.figures))
        return
    out = Path(args.out or f"results/{args.stage}")
    dev = jax.devices()[0]
    print(f"device: {dev.platform} ({dev.device_kind})")
    if args.stage != "smoke" and dev.platform == "cpu":
        print("WARNING: stage", args.stage, "is sized for the H200")

    todo = list(stage_configs(args.stage, args.trials))
    done = 0
    for K, arm, trial, cfg in todo:
        tag = f"{arm}_n{cfg.n:03d}_r{cfg.r}_t{trial:02d}"
        path = out / f"{tag}.npz"
        if path.exists():
            done += 1
            continue
        spec = mf.bbg_example(variant="a1", K_blocks=K,
                              profile="replication")
        t0 = time.time()
        res = compare.run_trial(spec, cfg, arm=arm)
        compare.save_trial(res, path)
        done += 1
        gap = res.J_train - res.J_fresh
        opt = res.refs["vstar_avg"] - res.arb_avg
        print(f"[{done}/{len(todo)}] {tag}: gen gap {gap:+8.0f}  "
              f"opt gap {opt:8.0f}  ({time.time() - t0:.0f}s)")
    build_figures(out)


if __name__ == "__main__":
    main()

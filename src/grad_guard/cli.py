"""Command-line interface for grad-guard: pathology audits, determinism checks, and benchmarks."""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from grad_guard.pathologies import REGISTRY, run_pathology_case
from grad_guard.reproducibility import (
    make_deterministic_loader,
    seed_everything,
    verify_checkpoint_parity,
)
from grad_guard.trainer import TrainConfig, Trainer


def _cmd_audit(args: argparse.Namespace) -> int:
    cases = list(REGISTRY.keys()) if args.all or not args.case else [args.case]
    print("=" * 108)
    print(" GRAD-GUARD: SILENT ML PATHOLOGY AUDIT SUITE (BROKEN VS. FIXED)")
    print("=" * 108)
    header = (
        f"{'Pathology Case':<23} | {'Mode':<6} | {'Status':<8} | "
        f"{'Detected Code':<22} | {'Step':>4} | {'Init Loss':>9} | {'Eval Metric':>11}"
    )
    print(header)
    print("-" * 108)

    all_passed = True
    for name in cases:
        broken_res = run_pathology_case(name, mode="broken", seed=args.seed, steps=args.steps)
        fixed_res = run_pathology_case(name, mode="fixed", seed=args.seed, steps=args.steps)

        # In broken mode, we expect an alert to be caught (detected_code == name)
        broken_caught = (not broken_res.healthy) and (broken_res.detected_code == name)
        fixed_clean = fixed_res.healthy and (fixed_res.detected_code is None)
        if not (broken_caught and fixed_clean):
            all_passed = False

        for res in (broken_res, fixed_res):
            status = (
                "CAUGHT"
                if (res.mode == "broken" and broken_caught)
                else ("HEALTHY" if (res.mode == "fixed" and fixed_clean) else "FAILED")
            )
            code_str = res.detected_code if res.detected_code is not None else "None"
            step_str = str(res.detected_step) if res.detected_step is not None else "-"
            print(
                f"{res.case_name:<23} | {res.mode:<6} | {status:<8} | "
                f"{code_str:<22} | {step_str:>4} | {res.initial_loss:9.4f} | {res.eval_metric:11.4f}"
            )
        print("-" * 108)

    print(
        f"Audit Result: {'ALL 6 PATHOLOGIES CAUGHT & REPAIRED (PASS)' if all_passed else 'FAILURE'}"
    )
    print("=" * 108)
    return 0 if all_passed else 1


def _cmd_verify_determinism(args: argparse.Namespace) -> int:
    seed = int(args.seed)
    steps = int(args.steps)

    print("=" * 92)
    print(
        f" GRAD-GUARD: REPRODUCIBILITY & CHECKPOINT PARITY VERIFICATION (SEED={seed}, STEPS={steps})"
    )
    print("=" * 92)

    def _run_trajectory() -> tuple[list[float], dict[str, torch.Tensor]]:
        seed_everything(seed)
        x = torch.randn(128, 16)
        y = torch.randn(128, 2)
        ds = TensorDataset(x, y)
        loader = make_deterministic_loader(ds, batch_size=16, seed=seed)

        model = nn.Sequential(
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 2),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
        trainer = Trainer(
            model,
            opt,
            nn.MSELoss(),
            TrainConfig(seed=seed, max_steps=steps, monitor_mode="lite"),
        )
        res = trainer.fit(loader)
        trainer.close()
        weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        return res.losses, weights

    losses_a, weights_a = _run_trajectory()
    losses_b, weights_b = _run_trajectory()

    max_loss_diff = max(abs(a - b) for a, b in zip(losses_a, losses_b, strict=True))
    max_weight_diff = max(
        float(torch.max(torch.abs(weights_a[k] - weights_b[k])).item()) for k in weights_a
    )

    # Mid-run checkpoint parity verification
    seed_everything(seed)
    batches = [(torch.randn(16, 16), torch.randn(16, 2)) for _ in range(steps)]
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "parity_check.pt"
        parity = verify_checkpoint_parity(
            model_fn=lambda: nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 2)),
            optimizer_fn=lambda m: torch.optim.AdamW(m.parameters(), lr=1e-2),
            batches=batches,
            checkpoint_path=ckpt_path,
            seed=seed,
        )

    print(f"1. Independent Multi-Step Run A vs Run B Max Loss Diff   : {max_loss_diff:.3e}")
    print(f"2. Independent Multi-Step Run A vs Run B Max Weight Diff : {max_weight_diff:.3e}")
    print(
        f"3. Mid-Run Checkpoint Resume (Step {parity['split_step']}/{parity['total_steps']}) "
        f"Max Loss Diff : {parity['max_loss_diff']:.3e}"
    )
    print(
        f"4. Mid-Run Checkpoint Resume (Step {parity['split_step']}/{parity['total_steps']}) "
        f"Max Param Diff: {parity['max_param_diff']:.3e}"
    )
    print("-" * 92)
    ok = max_loss_diff == 0.0 and max_weight_diff == 0.0 and parity["parity_verified"]
    print(f"Determinism Verdict: {'BIT-IDENTICAL PARITY VERIFIED (0.000e+00)' if ok else 'FAILED'}")
    print("=" * 92)
    return 0 if ok else 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    steps = int(args.steps)
    seed = 42
    print("=" * 92)
    print(f" GRAD-GUARD: TELEMETRY HOOK THROUGHPUT & MEMORY OVERHEAD BENCHMARK ({steps} STEPS)")
    print("=" * 92)
    print(
        f"{'Monitor Mode':<16} | {'ms / step':>10} | {'steps / sec':>12} | "
        f"{'Overhead (%)':>13} | {'Peak Memory (KB)':>16}"
    )
    print("-" * 92)

    modes: list[str | None] = [None, "lite", "full"]
    baseline_ms = 0.0

    for mode in modes:
        seed_everything(seed)
        x = torch.randn(256, 64)
        y = torch.randn(256, 8)
        ds = TensorDataset(x, y)
        loader = make_deterministic_loader(ds, batch_size=32, seed=seed)

        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 8),
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model,
            opt,
            nn.MSELoss(),
            TrainConfig(seed=seed, max_steps=steps, monitor_mode=mode),
        )

        # Warmup 3 steps
        warmup_loader = [(x[:32], y[:32]) for _ in range(3)]
        for xb, yb in warmup_loader:
            opt.zero_grad()
            loss = nn.MSELoss()(model(xb), yb)
            loss.backward()
            opt.step()

        tracemalloc.start()
        t0 = time.perf_counter()
        trainer.fit(loader)
        elapsed = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        trainer.close()

        ms_per_step = (elapsed * 1e3) / float(steps)
        steps_per_sec = float(steps) / max(elapsed, 1e-9)
        if mode is None:
            baseline_ms = ms_per_step
            overhead_pct = 0.0
        else:
            overhead_pct = ((ms_per_step - baseline_ms) / max(baseline_ms, 1e-9)) * 100.0

        label = "disabled (None)" if mode is None else f"{mode}"
        print(
            f"{label:<16} | {ms_per_step:10.3f} | {steps_per_sec:12.1f} | "
            f"{overhead_pct:+12.1f}% | {peak_bytes / 1024.0:16.1f}"
        )

    print("=" * 92)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grad-guard",
        description="Reproducible PyTorch Training & Numerical Pathology Diagnostics Harness.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_audit = sub.add_parser("audit", help="Run broken vs. fixed silent ML pathology diagnostics.")
    p_audit.add_argument("--all", action="store_true", help="Run all 6 pathology cases.")
    p_audit.add_argument("--case", type=str, default=None, choices=list(REGISTRY.keys()))
    p_audit.add_argument("--seed", type=int, default=42)
    p_audit.add_argument("--steps", type=int, default=25)
    p_audit.set_defaults(func=_cmd_audit)

    p_det = sub.add_parser(
        "verify-determinism",
        help="Verify bit-identical multi-step training and mid-run checkpoint resume parity.",
    )
    p_det.add_argument("--seed", type=int, default=42)
    p_det.add_argument("--steps", type=int, default=25)
    p_det.set_defaults(func=_cmd_verify_determinism)

    p_bench = sub.add_parser(
        "benchmark",
        help="Benchmark training throughput and memory overhead for GradMonitor modes.",
    )
    p_bench.add_argument("--steps", type=int, default=60)
    p_bench.set_defaults(func=_cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

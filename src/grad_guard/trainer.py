"""Reproducible PyTorch Trainer with integrated GradMonitor telemetry and checkpointing."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn

from grad_guard.reproducibility import capture_rng_state, restore_rng_state, seed_everything
from grad_guard.telemetry import DiagnosticReport, GradMonitor


@dataclass
class TrainConfig:
    """Configuration for deterministic `Trainer` execution."""

    seed: int = 42
    max_steps: int = 50
    max_grad_norm: float | None = 1.0
    monitor_mode: str | None = "full"  # "full" | "lite" | None
    strict_finite_check: bool = True
    loss_type: str = "regression"  # "regression" | "classification"
    device: str = "cpu"


@dataclass
class TrainResult:
    """Summary and telemetry produced by `Trainer.fit()`."""

    losses: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    report: DiagnosticReport | None = None

    @property
    def initial_loss(self) -> float:
        return self.losses[0] if self.losses else 0.0

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else 0.0


class Trainer:
    """Deterministic PyTorch training harness with optional `GradMonitor` instrumentation."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        config: TrainConfig | None = None,
    ) -> None:
        self.config = config if config is not None else TrainConfig()
        seed_everything(self.config.seed)
        self.device = torch.device(self.config.device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.step_count: int = 0
        self.monitor: GradMonitor | None = (
            GradMonitor(self.model, mode=self.config.monitor_mode)
            if self.config.monitor_mode is not None
            else None
        )

    def fit(self, loader: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> TrainResult:
        """Run a deterministic training loop for up to `config.max_steps` iterations."""
        self.model.train()
        losses: list[float] = []
        grad_norms: list[float] = []

        while self.step_count < self.config.max_steps:
            progressed = False
            for x_batch, y_batch in loader:
                if self.step_count >= self.config.max_steps:
                    break
                progressed = True
                self.step_count += 1
                x = x_batch.to(self.device)
                y = y_batch.to(self.device)

                self.optimizer.zero_grad()
                if self.monitor is not None:
                    self.monitor.begin_step(self.step_count)

                preds = self.model(x)
                if self.monitor is not None:
                    self.monitor.check_loss_shapes(preds, y, loss_type=self.config.loss_type)

                loss = self.loss_fn(preds, y)
                loss.backward()

                if self.monitor is not None:
                    gnorm = self.monitor.check_grad_flow()
                else:
                    total_sq = sum(
                        float(p.grad.detach().float().norm(2).item()) ** 2
                        for p in self.model.parameters()
                        if p.grad is not None
                    )
                    gnorm = float(total_sq**0.5)

                if self.config.max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

                self.optimizer.step()

                loss_val = float(loss.item())
                if self.monitor is not None:
                    self.monitor.end_step(loss_val)
                    if self.config.strict_finite_check:
                        self.monitor.assert_finite_step()

                losses.append(loss_val)
                grad_norms.append(gnorm)

            if not progressed:
                break

        return TrainResult(
            losses=losses,
            grad_norms=grad_norms,
            report=self.monitor.report if self.monitor is not None else None,
        )

    @torch.no_grad()
    def evaluate(self, loader: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Evaluate average loss over `loader` in deterministic `.eval()` mode."""
        self.model.eval()
        if self.monitor is not None:
            self.monitor.check_eval_mode_integrity()

        total_loss = 0.0
        count = 0
        for x_batch, y_batch in loader:
            x = x_batch.to(self.device)
            y = y_batch.to(self.device)
            preds = self.model(x)
            loss = self.loss_fn(preds, y)
            total_loss += float(loss.item())
            count += 1
        return total_loss / max(count, 1)

    def save_checkpoint(self, path: str | Path) -> None:
        """Serialize model weights, optimizer state, step counter, and full RNG state."""
        ckpt_path = Path(path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "step_count": self.step_count,
            "model_state_dict": {k: v.detach().clone() for k, v in self.model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rng_state": capture_rng_state(),
        }
        torch.save(payload, ckpt_path)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore model weights, optimizer state, step counter, and RNG state from disk."""
        payload = torch.load(Path(path), weights_only=False)
        self.step_count = int(payload["step_count"])
        self.model.load_state_dict(payload["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        restore_rng_state(payload["rng_state"])

    def export_telemetry_json(self, path: str | Path) -> None:
        """Write the `DiagnosticReport` to a formatted JSON file."""
        if self.monitor is None:
            raise RuntimeError("Cannot export telemetry when monitor_mode is None.")
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(self.monitor.report.to_dict(), indent=2), encoding="utf-8")

    def close(self) -> None:
        """Detach any active `GradMonitor` hooks."""
        if self.monitor is not None:
            self.monitor.close()

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

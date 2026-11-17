"""Structured registry of 6 classic silent ML engineering failures and automated root-cause detectors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from grad_guard.reproducibility import seed_everything
from grad_guard.telemetry import DiagnosticAlert, DiagnosticReport, GradMonitor

Mode = Literal["broken", "fixed"]


@dataclass(frozen=True)
class PathologyAuditResult:
    """Result of executing a single pathology case in either `'broken'` or `'fixed'` mode."""

    case_name: str
    mode: Mode
    healthy: bool
    detected_code: str | None
    detected_step: int | None
    detected_layer: str | None
    initial_loss: float
    final_loss: float
    eval_metric: float
    summary: str
    report: DiagnosticReport


@dataclass(frozen=True)
class BugCase:
    """Metadata and runner for a silent ML training pathology."""

    name: str
    title: str
    symptom: str
    root_cause: str
    detection_signal: str
    fix_description: str
    runner: Callable[[Mode, int, int], PathologyAuditResult]


def _make_synthetic_regression(
    n_samples: int = 128,
    in_dim: int = 16,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    seed_everything(seed)
    x = torch.randn(n_samples, in_dim)
    true_w = torch.linspace(-1.2, 1.2, steps=in_dim).unsqueeze(1)
    y = (x @ true_w) + 0.25 * torch.sin(x[:, :1])
    return x, y


# -----------------------------------------------------------------------------
# Case 1: zero_grad_omission
# -----------------------------------------------------------------------------
def _run_zero_grad_omission(mode: Mode, seed: int = 42, steps: int = 25) -> PathologyAuditResult:
    x, y = _make_synthetic_regression(n_samples=96, in_dim=12, seed=seed)
    seed_everything(seed)
    model = nn.Sequential(
        nn.Linear(12, 24),
        nn.ReLU(),
        nn.Linear(24, 1),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    monitor = GradMonitor(model, mode="full")

    initial_loss = 0.0
    final_loss = 0.0

    for step in range(1, steps + 1):
        if mode == "fixed":
            optimizer.zero_grad()

        monitor.begin_step(step)
        preds = model(x)
        loss = torch.mean((preds - y) ** 2)
        if step == 1:
            initial_loss = float(loss.item())
        loss.backward()
        monitor.check_grad_flow()
        optimizer.step()
        monitor.end_step(float(loss.item()))
        final_loss = float(loss.item())

    monitor.close()
    err = monitor.report.first_error
    return PathologyAuditResult(
        case_name="zero_grad_omission",
        mode=mode,
        healthy=monitor.report.healthy and final_loss < initial_loss * 0.25,
        detected_code=err.code if err else None,
        detected_step=err.step if err else None,
        detected_layer=err.layer_name if err else None,
        initial_loss=initial_loss,
        final_loss=final_loss,
        eval_metric=final_loss,
        summary=err.message if err else "Gradients zeroed cleanly before each backward pass.",
        report=monitor.report,
    )


# -----------------------------------------------------------------------------
# Case 2: broadcast_loss_bug
# -----------------------------------------------------------------------------
def _run_broadcast_loss_bug(mode: Mode, seed: int = 42, steps: int = 25) -> PathologyAuditResult:
    x, y_col = _make_synthetic_regression(n_samples=64, in_dim=10, seed=seed)
    y_1d = y_col.squeeze(-1)  # Shape (B,) instead of (B, 1)

    seed_everything(seed)
    model = nn.Sequential(
        nn.Linear(10, 24),
        nn.ReLU(),
        nn.Linear(24, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.035)
    monitor = GradMonitor(model, mode="full")

    initial_loss = 0.0
    final_loss = 0.0

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        monitor.begin_step(step)
        preds = model(x)  # Shape (B, 1)
        target = y_1d if mode == "broken" else y_col
        monitor.check_loss_shapes(preds, target, loss_type="regression")

        # (B, 1) - (B,) silently broadcasts to (B, B), forcing predictions toward constant mean(y)!
        loss = torch.mean((preds - target) ** 2)
        if step == 1:
            initial_loss = float(loss.item())
        loss.backward()
        monitor.check_grad_flow()
        optimizer.step()
        monitor.end_step(float(loss.item()))
        final_loss = float(loss.item())

    # True per-sample MSE against ground-truth column target
    with torch.no_grad():
        true_mse = float(torch.mean((model(x) - y_col) ** 2).item())

    monitor.close()
    err = monitor.report.first_error
    return PathologyAuditResult(
        case_name="broadcast_loss_bug",
        mode=mode,
        healthy=monitor.report.healthy and true_mse < initial_loss * 0.2,
        detected_code=err.code if err else None,
        detected_step=err.step if err else None,
        detected_layer=err.layer_name if err else None,
        initial_loss=initial_loss,
        final_loss=final_loss,
        eval_metric=true_mse,
        summary=err.message if err else f"Exact target shape matched; true MSE={true_mse:.4f}.",
        report=monitor.report,
    )


# -----------------------------------------------------------------------------
# Case 3: broken_grad_flow
# -----------------------------------------------------------------------------
class _BackboneHeadModel(nn.Module):
    def __init__(self, detach_encoder: bool) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(12, 24), nn.GELU())
        self.head = nn.Linear(24, 1)
        self.detach_encoder = detach_encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        if self.detach_encoder:
            features = features.detach()
        return self.head(features)


def _run_broken_grad_flow(mode: Mode, seed: int = 42, steps: int = 25) -> PathologyAuditResult:
    x, y = _make_synthetic_regression(n_samples=80, in_dim=12, seed=seed)
    seed_everything(seed)
    model = _BackboneHeadModel(detach_encoder=(mode == "broken"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    monitor = GradMonitor(model, mode="full")

    initial_loss = 0.0
    final_loss = 0.0

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        monitor.begin_step(step)
        preds = model(x)
        loss = torch.mean((preds - y) ** 2)
        if step == 1:
            initial_loss = float(loss.item())
        loss.backward()
        monitor.check_grad_flow()
        optimizer.step()
        monitor.end_step(float(loss.item()))
        final_loss = float(loss.item())

    monitor.close()
    err = monitor.report.first_error
    return PathologyAuditResult(
        case_name="broken_grad_flow",
        mode=mode,
        healthy=monitor.report.healthy and final_loss < initial_loss * 0.2,
        detected_code=err.code if err else None,
        detected_step=err.step if err else None,
        detected_layer=err.layer_name if err else None,
        initial_loss=initial_loss,
        final_loss=final_loss,
        eval_metric=final_loss,
        summary=err.message
        if err
        else "All backbone and head parameters received non-zero gradients.",
        report=monitor.report,
    )


# -----------------------------------------------------------------------------
# Case 4: amp_underflow
# -----------------------------------------------------------------------------
def _run_amp_underflow(mode: Mode, seed: int = 42, steps: int = 25) -> PathologyAuditResult:
    x, y = _make_synthetic_regression(n_samples=64, in_dim=12, seed=seed)
    seed_everything(seed)
    model = nn.Sequential(
        nn.Linear(12, 24),
        nn.ReLU(),
        nn.Linear(24, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025)
    monitor = GradMonitor(model, mode="full", vanishing_threshold=1e-6)

    # Simulate FP16 backward pass with small-scale upstream factor (2^-26 underflows FP16 to 0.0)
    # without dynamic GradScaler; with GradScaler (loss_scale = 2^26), FP16 preserves full gradients.
    underflow_factor = 2.0**-26
    loss_scale = 2.0**26 if mode == "fixed" else 1.0

    initial_loss = 0.0
    final_loss = 0.0

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        monitor.begin_step(step)
        preds = model(x)
        raw_loss = torch.mean((preds - y) ** 2)
        if step == 1:
            initial_loss = float(raw_loss.item())

        scaled_loss = raw_loss * loss_scale
        scaled_loss.backward()

        # Simulate FP16 communication/buffer compression where unscaled gradients experience
        # 2^-26 attenuation in FP16 before unscaling back to master FP32 weights
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    fp16_grad = (p.grad * underflow_factor).to(torch.float16)
                    p.grad.copy_(fp16_grad.to(torch.float32) / (underflow_factor * loss_scale))

        monitor.check_grad_flow()
        optimizer.step()
        monitor.end_step(float(raw_loss.item()))
        final_loss = float(raw_loss.item())

    monitor.close()
    err = monitor.report.first_error
    return PathologyAuditResult(
        case_name="amp_underflow",
        mode=mode,
        healthy=monitor.report.healthy and final_loss < initial_loss * 0.25,
        detected_code=err.code if err else None,
        detected_step=err.step if err else None,
        detected_layer=err.layer_name if err else None,
        initial_loss=initial_loss,
        final_loss=final_loss,
        eval_metric=final_loss,
        summary=err.message if err else "Dynamic loss scaling preserved FP16 gradient precision.",
        report=monitor.report,
    )


# -----------------------------------------------------------------------------
# Case 5: eval_mode_leakage
# -----------------------------------------------------------------------------
def _run_eval_mode_leakage(mode: Mode, seed: int = 42, steps: int = 25) -> PathologyAuditResult:
    x, y = _make_synthetic_regression(n_samples=64, in_dim=12, seed=seed)
    seed_everything(seed)
    model = nn.Sequential(
        nn.Linear(12, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(32, 1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
    monitor = GradMonitor(model, mode="full")

    model.train()
    initial_loss = 0.0
    final_loss = 0.0

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        monitor.begin_step(step)
        preds = model(x)
        loss = torch.mean((preds - y) ** 2)
        if step == 1:
            initial_loss = float(loss.item())
        loss.backward()
        monitor.check_grad_flow()
        optimizer.step()
        monitor.end_step(float(loss.item()))
        final_loss = float(loss.item())

    # Evaluation phase: broken mode forgets model.eval()
    if mode == "fixed":
        model.eval()

    monitor.check_eval_mode_integrity()
    with torch.no_grad():
        eval_out_1 = model(x)
        eval_out_2 = model(x)
        eval_discrepancy = float(torch.max(torch.abs(eval_out_1 - eval_out_2)).item())
        eval_mse = float(torch.mean((eval_out_1 - y) ** 2).item())

    if eval_discrepancy > 1e-7 and monitor.report.healthy:
        monitor._record_alert(
            code="eval_mode_leakage",
            severity="ERROR",
            layer_name="model",
            tensor_shape=tuple(eval_out_1.shape),
            message=f"Non-deterministic evaluation outputs (max_diff={eval_discrepancy:.4e}).",
        )

    monitor.close()
    err = monitor.report.first_error
    return PathologyAuditResult(
        case_name="eval_mode_leakage",
        mode=mode,
        healthy=monitor.report.healthy and eval_discrepancy == 0.0,
        detected_code=err.code if err else None,
        detected_step=err.step if err else None,
        detected_layer=err.layer_name if err else None,
        initial_loss=initial_loss,
        final_loss=final_loss,
        eval_metric=eval_mse,
        summary=err.message
        if err
        else f"Deterministic eval mode verified (discrepancy=0.0, MSE={eval_mse:.4f}).",
        report=monitor.report,
    )


# -----------------------------------------------------------------------------
# Case 6: preprocessing_leakage
# -----------------------------------------------------------------------------
def check_split_normalization_integrity(
    train_features: torch.Tensor,
    val_features: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_std: torch.Tensor,
    tol: float = 1e-4,
) -> DiagnosticAlert | None:
    """Verify standardization statistics `(scaler_mean, scaler_std)` were computed strictly on `train_features`."""
    expected_mean = train_features.mean(dim=0, keepdim=True)
    expected_std = train_features.std(dim=0, unbiased=False, keepdim=True)

    mean_diff = float(torch.max(torch.abs(scaler_mean - expected_mean)).item())
    std_diff = float(torch.max(torch.abs(scaler_std - expected_std)).item())

    if mean_diff > tol or std_diff > tol:
        return DiagnosticAlert(
            code="preprocessing_leakage",
            severity="ERROR",
            step=0,
            layer_name="feature_standardizer",
            tensor_shape=tuple(val_features.shape),
            message=(
                f"Feature scaler statistics deviate from train-only split "
                f"(mean_diff={mean_diff:.3e}, std_diff={std_diff:.3e}): "
                f"validation distribution leaked into preprocessing fit."
            ),
        )
    return None


def _run_preprocessing_leakage(mode: Mode, seed: int = 42, steps: int = 25) -> PathologyAuditResult:
    seed_everything(seed)
    n_train, n_val, in_dim = 96, 48, 12
    train_raw = torch.randn(n_train, in_dim) * 1.5 + 2.0
    # Validation set has a strong covariate shift (+8.0 mean offset, 3.5x scale)
    val_raw = torch.randn(n_val, in_dim) * 3.5 + 10.0

    true_w = torch.linspace(-0.8, 0.8, steps=in_dim).unsqueeze(1)
    y_train = train_raw @ true_w
    y_val = val_raw @ true_w

    if mode == "broken":
        # Bug: fit mean/std across combined train + val before splitting
        combined = torch.cat([train_raw, val_raw], dim=0)
        scaler_mean = combined.mean(dim=0, keepdim=True)
        scaler_std = combined.std(dim=0, unbiased=False, keepdim=True)
    else:
        # Fix: fit mean/std strictly on train_raw
        scaler_mean = train_raw.mean(dim=0, keepdim=True)
        scaler_std = train_raw.std(dim=0, unbiased=False, keepdim=True)

    x_train = (train_raw - scaler_mean) / (scaler_std + 1e-6)
    x_val = (val_raw - scaler_mean) / (scaler_std + 1e-6)

    model = nn.Sequential(nn.Linear(in_dim, 24), nn.ReLU(), nn.Linear(24, 1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03)
    monitor = GradMonitor(model, mode="full")

    leak_alert = check_split_normalization_integrity(train_raw, val_raw, scaler_mean, scaler_std)
    if leak_alert is not None:
        monitor.report.alerts.append(leak_alert)

    initial_loss = 0.0
    final_loss = 0.0
    for step in range(1, steps + 1):
        optimizer.zero_grad()
        monitor.begin_step(step)
        preds = model(x_train)
        loss = torch.mean((preds - y_train) ** 2)
        if step == 1:
            initial_loss = float(loss.item())
        loss.backward()
        monitor.check_grad_flow()
        optimizer.step()
        monitor.end_step(float(loss.item()))
        final_loss = float(loss.item())

    with torch.no_grad():
        val_mse = float(torch.mean((model(x_val) - y_val) ** 2).item())

    monitor.close()
    err = monitor.report.first_error
    return PathologyAuditResult(
        case_name="preprocessing_leakage",
        mode=mode,
        healthy=monitor.report.healthy and final_loss < initial_loss * 0.25,
        detected_code=err.code if err else None,
        detected_step=err.step if err else None,
        detected_layer=err.layer_name if err else None,
        initial_loss=initial_loss,
        final_loss=final_loss,
        eval_metric=val_mse,
        summary=err.message
        if err
        else "Train-only feature standardization verified with zero split leakage.",
        report=monitor.report,
    )


REGISTRY: dict[str, BugCase] = {
    "zero_grad_omission": BugCase(
        name="zero_grad_omission",
        title="Missing optimizer.zero_grad() Accumulation",
        symptom="Exploding step size, oscillatory loss divergence, and linear/quadratic gradient norm inflation.",
        root_cause="Omitting optimizer.zero_grad() causes backward() to accumulate into stale .grad buffers across steps.",
        detection_signal="GradMonitor.begin_step() detects non-zero parameter .grad buffers prior to backward().",
        fix_description="Call optimizer.zero_grad(set_to_none=True) before every forward/backward pass.",
        runner=_run_zero_grad_omission,
    ),
    "broadcast_loss_bug": BugCase(
        name="broadcast_loss_bug",
        title="Silent (B, 1) vs (B,) Loss Broadcasting",
        symptom="Training loss decreases smoothly to a plateau, but model predicts a constant mean for every sample.",
        root_cause="Subtracting target shape (B,) from prediction shape (B, 1) broadcasts into a (B, B) pairwise matrix.",
        detection_signal="GradMonitor.check_loss_shapes() flags predictions.shape != targets.shape before loss reduction.",
        fix_description="Align target tensor shape with predictions via targets.reshape_as(preds) before computing loss.",
        runner=_run_broadcast_loss_bug,
    ),
    "broken_grad_flow": BugCase(
        name="broken_grad_flow",
        title="Disconnected Computational Graph / Frozen Backbone",
        symptom="Loss decreases slightly (head-only fitting) while upstream encoder weights remain completely static.",
        root_cause="Accidental tensor.detach() or requires_grad=False severs the autograd graph between encoder and head.",
        detection_signal="GradMonitor.check_grad_flow() detects grad is None or zero update ratio on upstream parameters.",
        fix_description="Remove unintended .detach() calls and ensure trainable backbone parameters have requires_grad=True.",
        runner=_run_broken_grad_flow,
    ),
    "amp_underflow": BugCase(
        name="amp_underflow",
        title="Unscaled Low-Precision (FP16) Gradient Underflow",
        symptom="Loss stalls completely at initialization value; parameter update ratios drop to exact 0.0.",
        root_cause="Small loss gradients underflow the FP16 representable range (< 6e-8) without dynamic GradScaler.",
        detection_signal="GradMonitor.check_grad_flow() detects exact-zero gradients and global_grad_norm < vanishing_threshold.",
        fix_description="Wrap backward pass with dynamic loss scaling (torch.amp.GradScaler) and unscale before step().",
        runner=_run_amp_underflow,
    ),
    "eval_mode_leakage": BugCase(
        name="eval_mode_leakage",
        title="Train-Mode Dropout / BatchNorm Leakage in Evaluation",
        symptom="Stochastic validation metrics across identical inputs and corrupted BatchNorm running statistics.",
        root_cause="Failing to call model.eval() before validation leaves Dropout active and BatchNorm updating stats.",
        detection_signal="GradMonitor.check_eval_mode_integrity() flags submodules with training=True during evaluation.",
        fix_description="Explicitly call model.eval() and use torch.no_grad() context during validation/inference.",
        runner=_run_eval_mode_leakage,
    ),
    "preprocessing_leakage": BugCase(
        name="preprocessing_leakage",
        title="Train/Validation Split Feature Normalization Leakage",
        symptom="Optimistic validation metrics during development followed by severe degradation on unseen test data.",
        root_cause="Fitting standardization (mean/std) on the concatenated train+val dataset leaks test distribution shift.",
        detection_signal="check_split_normalization_integrity() detects mismatch between scaler stats and train-only split.",
        fix_description="Fit preprocessing statistics exclusively on the training split and transform validation/test splits.",
        runner=_run_preprocessing_leakage,
    ),
}


def run_pathology_case(
    name: str,
    mode: Mode = "broken",
    seed: int = 42,
    steps: int = 25,
) -> PathologyAuditResult:
    """Execute a registered pathology case by name in `'broken'` or `'fixed'` mode."""
    if name not in REGISTRY:
        raise KeyError(f"Unknown pathology case {name!r}. Available: {sorted(REGISTRY.keys())}")
    return REGISTRY[name].runner(mode, seed, steps)

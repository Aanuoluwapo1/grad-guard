"""Lightweight forward/backward hook telemetry and numerical pathology diagnostics."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle


@dataclass(frozen=True)
class DiagnosticAlert:
    """Structured diagnostic event pinpointing a training or numerical pathology."""

    code: str
    severity: str  # "ERROR" | "WARNING"
    step: int
    layer_name: str
    tensor_shape: tuple[int, ...]
    message: str


@dataclass
class StepMetrics:
    """Per-step activation, gradient, and weight-update telemetry snapshot."""

    step: int
    loss: float
    global_grad_norm: float
    activation_rms: dict[str, float] = field(default_factory=dict)
    dead_neuron_fraction: dict[str, float] = field(default_factory=dict)
    grad_norms: dict[str, float] = field(default_factory=dict)
    update_ratios: dict[str, float] = field(default_factory=dict)


@dataclass
class DiagnosticReport:
    """Aggregated telemetry report across a training or evaluation run."""

    mode: str
    steps_recorded: int
    alerts: list[DiagnosticAlert] = field(default_factory=list)
    step_history: list[StepMetrics] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        """Return `True` if zero `ERROR`-level diagnostic alerts were raised."""
        return not any(a.severity == "ERROR" for a in self.alerts)

    @property
    def first_error(self) -> DiagnosticAlert | None:
        """Return the earliest `ERROR` alert raised during the run, if any."""
        for alert in self.alerts:
            if alert.severity == "ERROR":
                return alert
        return None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the diagnostic report into a JSON-compatible dictionary."""
        return {
            "mode": self.mode,
            "steps_recorded": self.steps_recorded,
            "healthy": self.healthy,
            "alerts": [asdict(a) for a in self.alerts],
            "step_history": [asdict(s) for s in self.step_history],
        }


class GradMonitor:
    """Non-intrusive PyTorch module hook monitor for activations, gradients, and weight updates.

    Supports two instrumentation levels:
    - `mode="full"`: Captures forward activation RMS, dead-neuron fraction, `NaN`/`Inf` checks,
      per-parameter gradient L2 norms, vanishing/exploding gradient detection, and post-step
      parameter update ratios (`||Delta w|| / ||w||`).
    - `mode="lite"`: Captures fast `NaN`/`Inf` checks and per-parameter gradient norms without
      cloning pre-step parameter tensors on every iteration.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "full",
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 500.0,
        dead_neuron_threshold: float = 0.98,
        accumulation_growth_ratio: float = 1.6,
    ) -> None:
        if mode not in ("full", "lite"):
            raise ValueError(f"Unsupported GradMonitor mode: {mode!r}. Expected 'full' or 'lite'.")

        self.model = model
        self.mode = mode
        self.vanishing_threshold = float(vanishing_threshold)
        self.exploding_threshold = float(exploding_threshold)
        self.dead_neuron_threshold = float(dead_neuron_threshold)
        self.accumulation_growth_ratio = float(accumulation_growth_ratio)

        self._handles: list[RemovableHandle] = []
        self._current_step: int = 0
        self._act_rms: dict[str, float] = {}
        self._dead_frac: dict[str, float] = {}
        self._pre_step_weights: dict[str, torch.Tensor] = {}
        self._pre_backward_grad_norms: dict[str, float] = {}
        self._nonzero_pre_backward_steps: int = 0
        self.report = DiagnosticReport(mode=mode, steps_recorded=0)

        self._register_hooks()

    def _register_hooks(self) -> None:
        for name, module in self.model.named_modules():
            hook_name = name if name != "" else module.__class__.__name__
            is_leaf = len(list(module.children())) == 0
            handle = module.register_forward_hook(self._make_forward_hook(hook_name, is_leaf))
            self._handles.append(handle)

    def _make_forward_hook(self, layer_name: str, is_leaf: bool):
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, torch.Tensor):
                return
            with torch.no_grad():
                detached = output.detach()
                if not torch.isfinite(detached).all():
                    self._record_alert(
                        code="non_finite_activation",
                        severity="ERROR",
                        layer_name=layer_name,
                        tensor_shape=tuple(detached.shape),
                        message=f"NaN or Inf detected in forward activation of layer '{layer_name}'",
                    )
                    return

                if is_leaf and self.mode == "full":
                    rms = float(torch.sqrt(torch.mean(detached.float() ** 2)).item())
                    self._act_rms[layer_name] = rms
                    dead = float((detached == 0).float().mean().item())
                    self._dead_frac[layer_name] = dead
                    if dead >= self.dead_neuron_threshold and detached.numel() >= 8:
                        self._record_alert(
                            code="dead_activation_saturation",
                            severity="WARNING",
                            layer_name=layer_name,
                            tensor_shape=tuple(detached.shape),
                            message=(
                                f"Layer '{layer_name}' has {dead:.1%} exact-zero activations "
                                f"(threshold={self.dead_neuron_threshold:.1%})"
                            ),
                        )

        return _hook

    def _record_alert(
        self,
        code: str,
        severity: str,
        layer_name: str,
        tensor_shape: tuple[int, ...],
        message: str,
    ) -> None:
        # Deduplicate identical (code, layer_name) alerts on the same step
        for existing in self.report.alerts:
            if (
                existing.step == self._current_step
                and existing.code == code
                and existing.layer_name == layer_name
            ):
                return
        self.report.alerts.append(
            DiagnosticAlert(
                code=code,
                severity=severity,
                step=self._current_step,
                layer_name=layer_name,
                tensor_shape=tensor_shape,
                message=message,
            )
        )

    @torch.no_grad()
    def begin_step(self, step: int) -> None:
        """Call immediately before `forward()` / `backward()` on training step `step`.

        Captures pre-step weights (in `full` mode) and detects residual un-zeroed
        gradients left over from previous iterations (`zero_grad_omission`).
        """
        self._current_step = int(step)
        self._act_rms.clear()
        self._dead_frac.clear()

        has_stale_grad = False
        stale_layer = ""
        stale_shape: tuple[int, ...] = ()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if self.mode == "full":
                self._pre_step_weights[name] = param.detach().clone()
            if param.grad is not None:
                gnorm = float(param.grad.detach().float().norm(2).item())
                self._pre_backward_grad_norms[name] = gnorm
                if gnorm > 0.0 and not has_stale_grad:
                    has_stale_grad = True
                    stale_layer = name
                    stale_shape = tuple(param.shape)
            else:
                self._pre_backward_grad_norms[name] = 0.0

        if has_stale_grad:
            self._nonzero_pre_backward_steps += 1
            if self._nonzero_pre_backward_steps >= 1:
                self._record_alert(
                    code="zero_grad_omission",
                    severity="ERROR",
                    layer_name=stale_layer,
                    tensor_shape=stale_shape,
                    message=(
                        f"Non-zero gradient present before backward() on parameter '{stale_layer}' "
                        f"at step {self._current_step} (optimizer.zero_grad() omitted)."
                    ),
                )

    def check_loss_shapes(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        loss_type: str = "regression",
    ) -> None:
        """Detect silent broadcasting bugs between `predictions` and `targets` before loss reduction.

        For regression/binary losses (`MSELoss`, `L1Loss`, `SmoothL1Loss`), a mismatch such as
        `predictions.shape == (B, 1)` and `targets.shape == (B,)` broadcasts into `(B, B)`.
        """
        if loss_type == "regression" and predictions.shape != targets.shape:
            try:
                bcast_shape = torch.broadcast_shapes(predictions.shape, targets.shape)
            except RuntimeError:
                return
            self._record_alert(
                code="broadcast_loss_bug",
                severity="ERROR",
                layer_name="loss_fn",
                tensor_shape=tuple(bcast_shape),
                message=(
                    f"Loss input shape mismatch: predictions {tuple(predictions.shape)} vs "
                    f"targets {tuple(targets.shape)} broadcasts to {tuple(bcast_shape)}."
                ),
            )

    @torch.no_grad()
    def check_grad_flow(self) -> float:
        """Inspect all parameters after `loss.backward()` and return global L2 gradient norm."""
        grad_norms: dict[str, float] = {}
        total_sq = 0.0

        trainable_count = 0
        missing_or_zero_layers: list[tuple[str, tuple[int, ...], str]] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                missing_or_zero_layers.append((name, tuple(param.shape), "requires_grad=False"))
                continue

            trainable_count += 1
            if param.grad is None:
                grad_norms[name] = 0.0
                missing_or_zero_layers.append((name, tuple(param.shape), "grad is None (detached)"))
                continue

            g = param.grad.detach()
            if not torch.isfinite(g).all():
                self._record_alert(
                    code="non_finite_gradient",
                    severity="ERROR",
                    layer_name=name,
                    tensor_shape=tuple(param.shape),
                    message=f"NaN or Inf gradient detected in parameter '{name}'",
                )
                grad_norms[name] = float("nan")
                continue

            gnorm = float(g.float().norm(2).item())
            grad_norms[name] = gnorm
            total_sq += gnorm * gnorm

            if gnorm == 0.0:
                missing_or_zero_layers.append((name, tuple(param.shape), "exact zero gradient"))

        global_norm = float(total_sq**0.5)

        # Check for broken gradient flow (disconnected subgraph or frozen upstream layer)
        if missing_or_zero_layers:
            first_name, first_shape, reason = missing_or_zero_layers[0]
            if reason in ("grad is None (detached)", "requires_grad=False"):
                self._record_alert(
                    code="broken_grad_flow",
                    severity="ERROR",
                    layer_name=first_name,
                    tensor_shape=first_shape,
                    message=(
                        f"Broken gradient flow at parameter '{first_name}' "
                        f"({reason}; {len(missing_or_zero_layers)} parameter tensors affected)."
                    ),
                )
            elif trainable_count > 0 and global_norm < self.vanishing_threshold:
                self._record_alert(
                    code="amp_underflow",
                    severity="ERROR",
                    layer_name=first_name,
                    tensor_shape=first_shape,
                    message=(
                        f"Gradient underflow to zero in '{first_name}' "
                        f"(global_grad_norm={global_norm:.3e} < {self.vanishing_threshold:.1e})."
                    ),
                )

        if 0.0 < global_norm < self.vanishing_threshold:
            first_param_name, first_param = next(iter(self.model.named_parameters()))
            self._record_alert(
                code="vanishing_gradient",
                severity="ERROR",
                layer_name=first_param_name,
                tensor_shape=tuple(first_param.shape),
                message=(
                    f"Global gradient L2 norm {global_norm:.3e} below vanishing threshold "
                    f"{self.vanishing_threshold:.1e}"
                ),
            )
        elif global_norm > self.exploding_threshold:
            max_layer = max(grad_norms, key=lambda k: grad_norms[k])
            param_shape = tuple(dict(self.model.named_parameters())[max_layer].shape)
            self._record_alert(
                code="exploding_gradient",
                severity="ERROR",
                layer_name=max_layer,
                tensor_shape=param_shape,
                message=(
                    f"Exploding global gradient L2 norm {global_norm:.2f} > "
                    f"{self.exploding_threshold:.1f} (peak layer '{max_layer}')"
                ),
            )

        self._last_grad_norms = grad_norms
        self._last_global_norm = global_norm
        return global_norm

    @torch.no_grad()
    def end_step(self, loss: float) -> StepMetrics:
        """Call immediately after `optimizer.step()` to record update ratios and step telemetry."""
        grad_norms = getattr(self, "_last_grad_norms", {})
        global_norm = getattr(self, "_last_global_norm", 0.0)
        update_ratios: dict[str, float] = {}

        if self.mode == "full":
            for name, param in self.model.named_parameters():
                if name in self._pre_step_weights:
                    prev_w = self._pre_step_weights[name]
                    curr_w = param.detach()
                    if not torch.isfinite(curr_w).all():
                        self._record_alert(
                            code="non_finite_parameter",
                            severity="ERROR",
                            layer_name=name,
                            tensor_shape=tuple(curr_w.shape),
                            message=f"NaN or Inf parameter values after optimizer.step() in '{name}'",
                        )
                    delta_norm = float((curr_w.float() - prev_w.float()).norm(2).item())
                    weight_norm = float(prev_w.float().norm(2).item())
                    update_ratios[name] = delta_norm / max(weight_norm, 1e-12)

        metrics = StepMetrics(
            step=self._current_step,
            loss=float(loss),
            global_grad_norm=global_norm,
            activation_rms=dict(self._act_rms),
            dead_neuron_fraction=dict(self._dead_frac),
            grad_norms=dict(grad_norms),
            update_ratios=update_ratios,
        )
        self.report.step_history.append(metrics)
        self.report.steps_recorded = len(self.report.step_history)
        return metrics

    def assert_finite_step(self) -> None:
        """Raise a `RuntimeError` immediately if any non-finite activation, gradient, or parameter alert fired."""
        for alert in self.report.alerts:
            if alert.step == self._current_step and alert.code in (
                "non_finite_activation",
                "non_finite_gradient",
                "non_finite_parameter",
            ):
                raise RuntimeError(
                    f"[GradMonitor Step {alert.step}] {alert.code} in '{alert.layer_name}' "
                    f"(shape={alert.tensor_shape}): {alert.message}"
                )

    def check_eval_mode_integrity(self) -> bool:
        """Verify all `Dropout` and `BatchNorm` submodules are in `.eval()` (`training=False`) mode."""
        leaking_modules: list[tuple[str, tuple[int, ...]]] = []
        for name, module in self.model.named_modules():
            if (
                isinstance(
                    module,
                    (
                        nn.Dropout,
                        nn.Dropout1d,
                        nn.Dropout2d,
                        nn.Dropout3d,
                        nn.BatchNorm1d,
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                    ),
                )
                and module.training
            ):
                shape = (
                    tuple(module.weight.shape)
                    if hasattr(module, "weight") and module.weight is not None
                    else ()
                )
                leaking_modules.append((name or module.__class__.__name__, shape))

        if leaking_modules:
            first_name, first_shape = leaking_modules[0]
            self._record_alert(
                code="eval_mode_leakage",
                severity="ERROR",
                layer_name=first_name,
                tensor_shape=first_shape,
                message=(
                    f"Module '{first_name}' left in training=True mode during evaluation "
                    f"({len(leaking_modules)} stochastic/norm layers affected)."
                ),
            )
            return False
        return True

    def remove_hooks(self) -> None:
        """Remove all registered forward hooks and release pre-step tensor snapshots."""
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._pre_step_weights.clear()

    def close(self) -> None:
        """Alias for `remove_hooks()`."""
        self.remove_hooks()

    def __enter__(self) -> GradMonitor:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_hooks()

"""Tests for GradMonitor forward/backward hooks and numerical pathology assertions."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from grad_guard import GradMonitor, seed_everything


class _NanInjectionModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(6, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc(x)
        out = out.clone()
        out[0, 0] = float("nan")
        return out


class TestGradMonitorTelemetry:
    """Verify GradMonitor captures activation stats, NaN/Inf faults, and gradient/update ratios."""

    def test_healthy_step_metrics_and_update_ratios(self) -> None:
        seed_everything(42)
        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        monitor = GradMonitor(model, mode="full")

        x = torch.randn(16, 8)
        y = torch.randn(16, 2)

        opt.zero_grad()
        monitor.begin_step(1)
        preds = model(x)
        monitor.check_loss_shapes(preds, y, loss_type="regression")
        loss = torch.mean((preds - y) ** 2)
        loss.backward()
        gnorm = monitor.check_grad_flow()
        opt.step()
        step_metrics = monitor.end_step(float(loss.item()))
        monitor.assert_finite_step()
        monitor.close()

        assert monitor.report.healthy is True
        assert gnorm > 0.0
        assert len(step_metrics.activation_rms) == 3
        assert len(step_metrics.grad_norms) == 4
        assert all(r > 0.0 for r in step_metrics.update_ratios.values())

    def test_nan_activation_detection_and_assert_finite_step(self) -> None:
        seed_everything(42)
        model = _NanInjectionModule()
        monitor = GradMonitor(model, mode="full")

        x = torch.randn(4, 6)
        monitor.begin_step(1)
        _ = model(x)

        assert not monitor.report.healthy
        err = monitor.report.first_error
        assert err is not None
        assert err.code == "non_finite_activation"
        assert err.tensor_shape == (4, 4)

        with pytest.raises(RuntimeError, match="non_finite_activation"):
            monitor.assert_finite_step()
        monitor.close()

    def test_exploding_and_vanishing_gradient_detection(self) -> None:
        seed_everything(42)
        model = nn.Linear(6, 2)
        monitor = GradMonitor(
            model, mode="lite", vanishing_threshold=1e-5, exploding_threshold=10.0
        )

        # 1. Exploding gradient
        x = torch.randn(8, 6)
        y = torch.randn(8, 2) * 1000.0
        model.zero_grad()
        monitor.begin_step(1)
        loss_big = torch.mean((model(x) - y) ** 2) * 500.0
        loss_big.backward()
        monitor.check_grad_flow()
        assert any(a.code == "exploding_gradient" for a in monitor.report.alerts)

        # 2. Vanishing gradient
        model.zero_grad()
        monitor.begin_step(2)
        loss_tiny = torch.mean((model(x) - y) ** 2) * 1e-12
        loss_tiny.backward()
        monitor.check_grad_flow()
        assert any(a.code == "vanishing_gradient" for a in monitor.report.alerts)
        monitor.close()

    def test_context_manager_hook_cleanup_and_zero_weight_update_ratio(self) -> None:
        seed_everything(42)
        model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
        # Initialize first layer weights and bias to exact zeros (||w|| == 0)
        with torch.no_grad():
            model[0].weight.zero_()
            model[0].bias.zero_()

        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        with GradMonitor(model, mode="full") as monitor:
            assert len(monitor._handles) > 0
            x = torch.ones(2, 4)
            y = torch.ones(2, 4)
            opt.zero_grad()
            monitor.begin_step(1)
            loss = torch.mean((model(x) - y) ** 2)
            loss.backward()
            monitor.check_grad_flow()
            opt.step()
            metrics = monitor.end_step(float(loss.item()))
            for ratio in metrics.update_ratios.values():
                assert ratio >= 0.0 and torch.isfinite(torch.tensor(ratio))

        # After exiting context manager, all RemovableHandles and weight snapshots are cleared
        assert len(monitor._handles) == 0
        for module in model.modules():
            assert len(module._forward_hooks) == 0

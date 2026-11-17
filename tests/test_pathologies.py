"""Parametrized tests for all 6 silent ML pathologies in REGISTRY (broken vs. fixed)."""

from __future__ import annotations

import pytest

from grad_guard import REGISTRY, run_pathology_case


class TestPathologyRegistrySuite:
    """Verify every registered pathology triggers its exact diagnostic in broken mode and passes in fixed mode."""

    def test_registry_contains_all_six_canonical_cases(self) -> None:
        expected = {
            "zero_grad_omission",
            "broadcast_loss_bug",
            "broken_grad_flow",
            "amp_underflow",
            "eval_mode_leakage",
            "preprocessing_leakage",
        }
        assert set(REGISTRY.keys()) == expected

    @pytest.mark.parametrize("case_name", list(REGISTRY.keys()))
    def test_broken_mode_triggers_exact_diagnostic_alert(self, case_name: str) -> None:
        res = run_pathology_case(case_name, mode="broken", seed=42, steps=25)
        assert res.healthy is False, (
            f"Expected broken mode of '{case_name}' to be flagged unhealthy"
        )
        assert res.detected_code == case_name, (
            f"Expected alert code '{case_name}', got '{res.detected_code}'"
        )
        assert res.detected_layer is not None and len(res.detected_layer) > 0

    @pytest.mark.parametrize("case_name", list(REGISTRY.keys()))
    def test_fixed_mode_passes_all_health_checks_and_converges(self, case_name: str) -> None:
        res = run_pathology_case(case_name, mode="fixed", seed=42, steps=25)
        assert res.healthy is True, (
            f"Expected fixed mode of '{case_name}' to be healthy, got alerts={res.report.alerts}"
        )
        assert res.detected_code is None
        assert res.final_loss < res.initial_loss * 0.25

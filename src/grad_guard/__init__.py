"""grad-guard: Reproducible Training & Numerical Pathology Diagnostics Harness."""

from grad_guard.pathologies import REGISTRY, BugCase, PathologyAuditResult, run_pathology_case
from grad_guard.reproducibility import (
    DeterminismConfig,
    capture_rng_state,
    make_deterministic_loader,
    restore_rng_state,
    seed_everything,
    verify_checkpoint_parity,
)
from grad_guard.telemetry import DiagnosticAlert, DiagnosticReport, GradMonitor, StepMetrics
from grad_guard.trainer import TrainConfig, Trainer, TrainResult

__version__ = "0.1.0"

__all__ = [
    "REGISTRY",
    "BugCase",
    "DeterminismConfig",
    "DiagnosticAlert",
    "DiagnosticReport",
    "GradMonitor",
    "PathologyAuditResult",
    "StepMetrics",
    "TrainConfig",
    "TrainResult",
    "Trainer",
    "capture_rng_state",
    "make_deterministic_loader",
    "restore_rng_state",
    "run_pathology_case",
    "seed_everything",
    "verify_checkpoint_parity",
]

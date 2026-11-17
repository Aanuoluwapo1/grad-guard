"""Deterministic execution controller, RNG state snapshots, and checkpoint parity verification."""

from __future__ import annotations

import os
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class DeterminismConfig:
    """Immutable snapshot of active deterministic execution settings."""

    seed: int
    warn_only: bool
    python_hash_seed: str
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cublas_workspace_config: str
    deterministic_algorithms: bool


def seed_everything(seed: int = 42, warn_only: bool = False) -> DeterminismConfig:
    """Seed all RNG sources (Python, NumPy, PyTorch CPU & CUDA) and enforce deterministic kernels.

    Args:
        seed: Global integer seed in `[0, 2**32 - 1]`.
        warn_only: Passed to `torch.use_deterministic_algorithms(True, warn_only=warn_only)`.

    Returns:
        `DeterminismConfig` describing the configured environment state.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.use_deterministic_algorithms(True, warn_only=warn_only)

    return DeterminismConfig(
        seed=seed,
        warn_only=warn_only,
        python_hash_seed=os.environ["PYTHONHASHSEED"],
        cudnn_deterministic=bool(getattr(torch.backends.cudnn, "deterministic", True)),
        cudnn_benchmark=bool(getattr(torch.backends.cudnn, "benchmark", False)),
        cublas_workspace_config=os.environ["CUBLAS_WORKSPACE_CONFIG"],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
    )


def _seed_worker(worker_id: int) -> None:
    """Deterministic DataLoader worker initialization callback."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_deterministic_loader(
    dataset: Dataset[Any],
    batch_size: int,
    seed: int = 42,
    num_workers: int = 0,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader[Any]:
    """Create a `torch.utils.data.DataLoader` with isolated generator and worker seeding.

    Guarantees bit-for-bit identical batch ordering across repeated iterations and processes.
    """
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=drop_last,
    )


def capture_rng_state() -> dict[str, Any]:
    """Capture complete RNG states for Python `random`, NumPy, and PyTorch (CPU + CUDA)."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(rng_state: dict[str, Any]) -> None:
    """Restore Python, NumPy, and PyTorch RNG states captured via `capture_rng_state()`."""
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy"])
    torch.set_rng_state(rng_state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in rng_state:
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


def verify_checkpoint_parity(
    model_fn: Callable[[], nn.Module],
    optimizer_fn: Callable[[nn.Module], torch.optim.Optimizer],
    batches: list[tuple[torch.Tensor, torch.Tensor]],
    checkpoint_path: str | Path,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Verify bit-identical training continuation from a serialized mid-run checkpoint.

    Runs `k = len(batches) // 2` warmup steps, saves model, optimizer, and RNG states
    to `checkpoint_path`, continues uninterrupted for the remaining steps, and then
    instantiates a fresh model + optimizer, loads the checkpoint, and asserts exact
    tensor equality (`atol=0.0, rtol=0.0`) on all subsequent losses, gradients, and weights.
    """
    if len(batches) < 2:
        raise ValueError("verify_checkpoint_parity requires at least 2 batches.")

    criterion = loss_fn if loss_fn is not None else nn.MSELoss()
    ckpt_file = Path(checkpoint_path)
    ckpt_file.parent.mkdir(parents=True, exist_ok=True)
    split_step = len(batches) // 2

    # 1. Uninterrupted reference trajectory
    seed_everything(seed)
    model_ref = model_fn()
    opt_ref = optimizer_fn(model_ref)
    model_ref.train()

    for step_idx in range(split_step):
        x, y = batches[step_idx]
        opt_ref.zero_grad()
        out = model_ref(x)
        loss = criterion(out, y)
        loss.backward()
        opt_ref.step()

    # Save full state at split_step
    checkpoint_payload = {
        "model_state_dict": {k: v.detach().clone() for k, v in model_ref.state_dict().items()},
        "optimizer_state_dict": opt_ref.state_dict(),
        "rng_state": capture_rng_state(),
        "step": split_step,
    }
    torch.save(checkpoint_payload, ckpt_file)

    ref_post_losses: list[float] = []
    for step_idx in range(split_step, len(batches)):
        x, y = batches[step_idx]
        opt_ref.zero_grad()
        out = model_ref(x)
        loss = criterion(out, y)
        loss.backward()
        opt_ref.step()
        ref_post_losses.append(float(loss.item()))

    # 2. Perturb global RNG state, then reload into a fresh model & optimizer
    seed_everything(seed + 999)
    loaded = torch.load(ckpt_file, weights_only=False)
    restore_rng_state(loaded["rng_state"])

    model_resumed = model_fn()
    opt_resumed = optimizer_fn(model_resumed)
    model_resumed.load_state_dict(loaded["model_state_dict"], strict=True)
    opt_resumed.load_state_dict(loaded["optimizer_state_dict"])
    model_resumed.train()

    resumed_post_losses: list[float] = []
    for step_idx in range(split_step, len(batches)):
        x, y = batches[step_idx]
        opt_resumed.zero_grad()
        out = model_resumed(x)
        loss = criterion(out, y)
        loss.backward()
        opt_resumed.step()
        resumed_post_losses.append(float(loss.item()))

    # 3. Verify bit-for-bit equality (atol=0.0, rtol=0.0)
    max_param_diff = 0.0
    for (name_a, p_a), (name_b, p_b) in zip(
        model_ref.named_parameters(), model_resumed.named_parameters(), strict=True
    ):
        assert name_a == name_b
        diff = float(torch.max(torch.abs(p_a.detach() - p_b.detach())).item())
        max_param_diff = max(max_param_diff, diff)
        torch.testing.assert_close(p_a, p_b, atol=0.0, rtol=0.0)
        if p_a.grad is not None and p_b.grad is not None:
            torch.testing.assert_close(p_a.grad, p_b.grad, atol=0.0, rtol=0.0)

    max_loss_diff = max(
        abs(a - b) for a, b in zip(ref_post_losses, resumed_post_losses, strict=True)
    )
    if max_loss_diff != 0.0:
        raise AssertionError(f"Checkpoint resume loss mismatch: max_loss_diff={max_loss_diff:.3e}")

    return {
        "parity_verified": True,
        "split_step": split_step,
        "total_steps": len(batches),
        "max_loss_diff": max_loss_diff,
        "max_param_diff": max_param_diff,
        "ref_post_losses": ref_post_losses,
        "resumed_post_losses": resumed_post_losses,
    }

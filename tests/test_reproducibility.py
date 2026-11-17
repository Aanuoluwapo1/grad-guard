"""Tests for deterministic seeding, reproducible DataLoaders, and mid-run checkpoint parity."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import TensorDataset

from grad_guard import (
    TrainConfig,
    Trainer,
    capture_rng_state,
    make_deterministic_loader,
    restore_rng_state,
    seed_everything,
    verify_checkpoint_parity,
)


class TestReproducibilityAndCheckpointing:
    """Verify bit-for-bit reproducibility across seeds, DataLoaders, and checkpoint resumes."""

    def test_seed_everything_configures_deterministic_flags(self) -> None:
        cfg = seed_everything(2026, warn_only=False)
        assert cfg.seed == 2026
        assert cfg.python_hash_seed == "2026"
        assert cfg.cudnn_deterministic is True
        assert cfg.cudnn_benchmark is False
        assert cfg.cublas_workspace_config == ":4096:8"
        assert cfg.deterministic_algorithms is True

    def test_multi_step_training_is_bit_identical_across_runs(self) -> None:
        def _train_once(seed: int) -> tuple[list[float], list[float], dict[str, torch.Tensor]]:
            seed_everything(seed)
            x = torch.randn(64, 12)
            y = torch.randn(64, 3)
            loader = make_deterministic_loader(TensorDataset(x, y), batch_size=16, seed=seed)

            model = nn.Sequential(
                nn.Linear(12, 24),
                nn.LayerNorm(24),
                nn.GELU(),
                nn.Dropout(p=0.2),
                nn.Linear(24, 3),
            )
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
            trainer = Trainer(
                model,
                opt,
                nn.MSELoss(),
                TrainConfig(seed=seed, max_steps=15, monitor_mode="full"),
            )
            result = trainer.fit(loader)
            trainer.close()
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            return result.losses, result.grad_norms, state

        losses_1, gnorms_1, state_1 = _train_once(77)
        losses_2, gnorms_2, state_2 = _train_once(77)
        losses_diff_seed, _, _ = _train_once(78)

        assert losses_1 == losses_2
        assert gnorms_1 == gnorms_2
        assert losses_1 != losses_diff_seed
        for k in state_1:
            torch.testing.assert_close(state_1[k], state_2[k], atol=0.0, rtol=0.0)

    def test_deterministic_dataloader_batch_order_parity(self) -> None:
        seed_everything(42)
        ds = TensorDataset(torch.arange(40, dtype=torch.float32).unsqueeze(1))
        loader_a = make_deterministic_loader(ds, batch_size=8, seed=99, shuffle=True)
        loader_b = make_deterministic_loader(ds, batch_size=8, seed=99, shuffle=True)

        batches_a = [b[0] for b in loader_a]
        batches_b = [b[0] for b in loader_b]
        for ba, bb in zip(batches_a, batches_b, strict=True):
            torch.testing.assert_close(ba, bb, atol=0.0, rtol=0.0)

    def test_rng_state_capture_and_restore(self) -> None:
        seed_everything(123)
        snap = capture_rng_state()
        t1 = torch.randn(5, 5)

        restore_rng_state(snap)
        t2 = torch.randn(5, 5)
        torch.testing.assert_close(t1, t2, atol=0.0, rtol=0.0)

    def test_verify_checkpoint_parity_mid_run(self, tmp_path: Path) -> None:
        seed_everything(42)
        batches = [(torch.randn(12, 8), torch.randn(12, 2)) for _ in range(10)]
        ckpt_file = tmp_path / "mid_run.pt"

        res = verify_checkpoint_parity(
            model_fn=lambda: nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 2)),
            optimizer_fn=lambda m: torch.optim.AdamW(m.parameters(), lr=1e-2),
            batches=batches,
            checkpoint_path=ckpt_file,
            seed=42,
        )
        assert res["parity_verified"] is True
        assert res["max_loss_diff"] == 0.0
        assert res["max_param_diff"] == 0.0

    def test_trainer_checkpoint_and_json_telemetry_export(self, tmp_path: Path) -> None:
        seed_everything(42)
        x = torch.randn(32, 8)
        y = torch.randn(32, 1)
        loader = make_deterministic_loader(TensorDataset(x, y), batch_size=8, seed=42)

        model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 1))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        trainer = Trainer(model, opt, nn.MSELoss(), TrainConfig(seed=42, max_steps=6))
        trainer.fit(loader)

        ckpt_path = tmp_path / "trainer_ckpt.pt"
        json_path = tmp_path / "telemetry.json"
        trainer.save_checkpoint(ckpt_path)
        trainer.export_telemetry_json(json_path)
        trainer.close()

        assert ckpt_path.exists()
        assert json_path.exists()
        assert '"healthy": true' in json_path.read_text(encoding="utf-8")

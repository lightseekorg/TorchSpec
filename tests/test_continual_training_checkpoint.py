from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from torchspec.training import checkpoint


class DummyBF16Optimizer:
    def __init__(self, *, lr: float, betas: tuple[float, float], weight_decay: float) -> None:
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.fp32_params = [torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))]
        self.optimizer = torch.optim.AdamW(
            self.fp32_params,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
        )

    def sync_fp32_params_from_model(self):
        raise AssertionError("fallback path should not run in this test")


def test_restore_fp32_master_params_keeps_fresh_optimizer_hparams():
    optimizer = DummyBF16Optimizer(lr=0.123, betas=(0.7, 0.8), weight_decay=0.456)
    actor = SimpleNamespace(model=mock.MagicMock(), optimizer=optimizer)

    def fake_dcp_load(*, state_dict, checkpoint_id):
        assert checkpoint_id == "/tmp/fake_optim"
        assert state_dict["optim_state"].keys() == {"fp32_params"}
        state_dict["optim_state"]["fp32_params"]["weight"].fill_(42.0)

    def fake_exists(self):
        return str(self) in {"/tmp/fake_optim", "/tmp/fake_optim/.metadata"}

    with (
        mock.patch("torchspec.training.checkpoint.dcp.load", side_effect=fake_dcp_load),
        mock.patch("pathlib.Path.exists", new=fake_exists),
    ):
        checkpoint._restore_fp32_master_params(actor, Path("/tmp/fake_optim"))

    group = optimizer.optimizer.param_groups[0]
    assert group["lr"] == 0.123
    assert group["betas"] == (0.7, 0.8)
    assert group["weight_decay"] == 0.456
    assert optimizer.optimizer.state == {}
    assert torch.equal(optimizer.fp32_params[0].detach(), torch.tensor([42.0], dtype=torch.float32))

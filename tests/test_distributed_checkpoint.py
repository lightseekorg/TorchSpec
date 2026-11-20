# Copyright (c) 2026 LightSeek Foundation
# MIT License

from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp

from torchspec.training import checkpoint
from torchspec.utils.distributed import init_gloo_group


def _check_collective_save(rank, root):
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=2,
        init_method=(root / "rendezvous").as_uri(),
        timeout=timedelta(seconds=45),
    )
    try:
        group = init_gloo_group()
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters())
        actor = SimpleNamespace(
            args=SimpleNamespace(checkpoint_dir=root),
            model=model,
            optimizer=optimizer,
            lr_scheduler=torch.optim.lr_scheduler.StepLR(optimizer, 1),
            global_step=1,
        )
        with (
            patch.object(torch.cuda, "synchronize"),
            patch.object(torch.cuda, "get_rng_state_all", return_value=[]),
            patch.object(dcp, "save", wraps=dcp.save) as write,
        ):
            torch.manual_seed(7100 + rank)
            expected_rng = torch.get_rng_state()
            checkpoint.save(actor, step=0)
            assert [Path(call.kwargs["checkpoint_id"]).name for call in write.call_args_list] == [
                "model",
                "optimizer",
                "lr_scheduler",
            ]
            assert all(call.kwargs["process_group"] is group for call in write.call_args_list)
            tracker = root / "latest_checkpointed_iteration.txt"

            reader = SimpleNamespace(
                args=SimpleNamespace(load_path=root), model=model, global_step=0
            )
            torch.manual_seed(0)
            checkpoint.finalize_load(reader, checkpoint.load(reader))
            assert torch.equal(torch.get_rng_state(), expected_rng)

            # Old rng.pt files contain a single shared stream.
            if rank == 0:
                rng_file = root / "iter_0000001" / "rng.pt"
                rng_state = torch.load(rng_file)
                rng_state.pop("other_ranks")
                torch.save(rng_state, rng_file)
            dist.barrier(group=group)
            checkpoint.finalize_load(reader, checkpoint.load(reader))
            assert torch.equal(
                torch.get_rng_state(), torch.Generator().manual_seed(7100).get_state()
            )

            # Fail inside DCP on a worker, then after DCP on the publishing rank.
            writer_failure = patch.object(
                dcp.FileSystemWriter, "write_data", side_effect=OSError("worker write")
            )
            with writer_failure if rank == 1 else nullcontext():
                with pytest.raises(dcp.CheckpointException, match="worker write"):
                    checkpoint.save(actor, step=1)
            assert tracker.read_text() == "1"

            with patch.object(
                checkpoint, "_write_checkpoint_metadata", side_effect=OSError("metadata write")
            ):
                with pytest.raises(RuntimeError, match="metadata write"):
                    checkpoint.save(actor, step=1)
            assert tracker.read_text() == "1"

            checkpoint.save(actor, step=1)
            assert tracker.read_text() == "2"
            dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="requires Gloo")
def test_checkpoint_save_propagates_errors_and_reuses_trainer_group(tmp_path):
    mp.spawn(_check_collective_save, args=(tmp_path,), nprocs=2)

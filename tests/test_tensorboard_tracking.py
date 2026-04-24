from types import SimpleNamespace

from torchspec.utils import tensorboard as tensorboard_utils
from torchspec.utils.logging import finish_tracking, init_tracking


def test_init_tracking_creates_tensorboard_event_file(tmp_path):
    args = SimpleNamespace(
        use_wandb=False,
        use_tensorboard=True,
        tensorboard_dir=str(tmp_path / "tb"),
        output_dir=str(tmp_path / "output"),
        wandb_mode=None,
        wandb_key=None,
        wandb_host=None,
        wandb_group=None,
        wandb_project=None,
        wandb_random_suffix=False,
        wandb_dir=None,
        wandb_team=None,
        rank=0,
    )

    init_tracking(args)
    tensorboard_utils.log_metrics({"train/step": 1, "train/avg_loss": 1.23, "perf/step_time": 4.56})
    finish_tracking()

    event_files = list((tmp_path / "tb").glob("events.out.tfevents.*"))
    assert event_files


def test_finish_tracking_is_safe_without_writer():
    finish_tracking()

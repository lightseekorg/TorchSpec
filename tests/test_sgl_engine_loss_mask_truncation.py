from torchspec.inference.engine.sgl_engine import SglEngine


def test_truncate_packed_loss_mask_to_actual_seq_len():
    packed = "2,3,2,2,1"  # total length = 10
    truncated = SglEngine._truncate_packed_loss_mask(packed, 8)
    assert truncated == "2,3,2,1"


def test_truncate_packed_loss_mask_noop_when_lengths_match():
    packed = "2,3,2,2,1"
    assert SglEngine._truncate_packed_loss_mask(packed, 10) == packed

from types import SimpleNamespace

import pytest
import torch

from torchspec.data.dataset import (
    _is_pretokenized_dataset,
    _load_pretokenized_dataset,
    load_conversation_dataset,
)

MAX_LENGTH = 4096


class Rows(list):
    """Stands in for an HF dataset with a typed schema."""

    column_names = [
        "data_id",
        "input_ids",
        "loss_mask",
        "packed_loss_mask",
        "seq_len",
        "loss_tokens",
        "corpus",
    ]


def _row(**overrides):
    row = {
        "data_id": "row-1",
        "input_ids": [10, 11, 12, 13, 14],
        "loss_mask": [0, 0, 1, 1, 0],
        "packed_loss_mask": "2,2,1",
        "seq_len": 5,
        "loss_tokens": 2,
        "corpus": "example-corpus",
    }
    row.update(overrides)
    return row


def test_pretokenized_rows_are_loaded_without_rerendering():
    rows = Rows(
        [
            _row(),
            _row(
                data_id="row-2",
                input_ids=[20, 21, 22, 23],
                loss_mask=[0, 1, 0, 1],
                packed_loss_mask="1,1,1,1",
                seq_len=4,
                corpus="other-corpus",
            ),
        ]
    )
    assert _is_pretokenized_dataset(rows)

    loaded = _load_pretokenized_dataset(rows, max_length=MAX_LENGTH)

    assert [row["data_id"] for row in loaded] == ["row-1", "row-2"]
    assert torch.equal(loaded[0]["input_ids"], torch.tensor([10, 11, 12, 13, 14]))
    assert loaded[0]["packed_loss_mask"] == "2,2,1"
    assert loaded[0]["formatted_prompt"] is None
    assert loaded[0]["multimodal_inputs"] is None


def test_producer_columns_are_carried_through_as_metadata():
    loaded = _load_pretokenized_dataset(Rows([_row()]), max_length=MAX_LENGTH)

    # Non-contract columns pass through untouched; contract columns do not leak.
    assert loaded[0]["metadata"] == {"corpus": "example-corpus"}


def test_non_scalar_columns_are_not_pinned_in_metadata():
    loaded = _load_pretokenized_dataset(
        Rows([_row(corpus={"nested": "value"})]), max_length=MAX_LENGTH
    )

    assert loaded[0]["metadata"] == {}


def test_an_explicit_mask_alone_is_enough():
    rows = Rows([_row(packed_loss_mask=None)])

    loaded = _load_pretokenized_dataset(rows, max_length=MAX_LENGTH)

    assert loaded[0]["packed_loss_mask"] == "2,2,1"


def test_missing_ids_or_mask_is_rejected_before_any_row_is_read():
    class IdsOnly(list):
        column_names = ["data_id", "input_ids"]

    with pytest.raises(ValueError, match="input_ids together with"):
        _is_pretokenized_dataset(IdsOnly())


def test_raw_conversation_datasets_are_not_mistaken_for_pretokenized():
    class Conversations(list):
        column_names = ["id", "conversations"]

    assert _is_pretokenized_dataset(Conversations()) is False


def test_pretokenized_rows_never_silently_truncate():
    with pytest.raises(ValueError, match="instead of truncating"):
        _load_pretokenized_dataset(Rows([_row()]), max_length=2)


def test_duplicate_data_ids_are_rejected():
    with pytest.raises(ValueError, match="Duplicate pretokenized data_id"):
        _load_pretokenized_dataset(Rows([_row(), _row()]), max_length=MAX_LENGTH)


@pytest.mark.parametrize(
    "row, message",
    [
        (_row(packed_loss_mask="1,1"), "length does not match"),
        (_row(loss_tokens=1), "loss_tokens metadata"),
        (_row(seq_len=4), "seq_len metadata"),
        (_row(loss_mask=[0, 1, 1, 0, 0]), "disagrees with packed mask"),
        (_row(packed_loss_mask="5", loss_mask=[0, 0, 0, 0, 0]), "no supervised tokens"),
        (_row(input_ids=[]), "empty input_ids"),
        (_row(input_ids=[1, 2, "three", 4, 5]), "invalid input_ids"),
    ],
)
def test_pretokenized_rows_validate_masks(row, message):
    with pytest.raises(ValueError, match=message):
        _load_pretokenized_dataset(Rows([row]), max_length=MAX_LENGTH)


def _write_parquet(tmp_path, rows):
    pytest.importorskip("pyarrow")
    import pyarrow as pa
    import pyarrow.parquet as pq

    source = tmp_path / "pretokenized.parquet"
    pq.write_table(pa.Table.from_pylist(rows), source)
    return source


def _pretokenized_args(source, tmp_path, **overrides):
    args = dict(
        cache_dir=str(tmp_path / "cache"),
        chat_template="llama3",
        defer_tokenization=False,
        max_seq_length=MAX_LENGTH,
        prompt_key="conversations",
        target_model_path="unused-because-no-rendering-happens",
        train_data_path=str(source),
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def test_parquet_dataset_bypasses_rendering_end_to_end(tmp_path):
    source = _write_parquet(tmp_path, [_row(), _row(data_id="row-2")])

    prompts = load_conversation_dataset(_pretokenized_args(source, tmp_path))

    # No tokenizer was loaded and no cache was written: nothing was rendered.
    assert [prompt["data_id"] for prompt in prompts] == ["row-1", "row-2"]
    assert prompts[0]["metadata"] == {"corpus": "example-corpus"}
    assert not (tmp_path / "cache").exists()


def test_pretokenized_dataset_rejects_deferred_tokenization(tmp_path):
    source = _write_parquet(tmp_path, [_row()])
    args = _pretokenized_args(source, tmp_path, defer_tokenization=True)

    with pytest.raises(ValueError, match="require defer_tokenization=False"):
        load_conversation_dataset(args)


def test_pretokenized_dataset_needs_neither_renderer_nor_chat_template(tmp_path):
    source = _write_parquet(tmp_path, [_row()])
    args = _pretokenized_args(source, tmp_path, chat_template=None)

    prompts = load_conversation_dataset(args)

    assert [prompt["data_id"] for prompt in prompts] == ["row-1"]

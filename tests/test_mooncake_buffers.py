from unittest.mock import MagicMock

import pytest

from torchspec.transfer.mooncake.buffers import AsyncPutManager


def test_async_wait_reports_failure_once_and_allows_buffer_reuse():
    store = MagicMock()
    store.batch_put_from.side_effect = [[-200], [0]]
    manager = AsyncPutManager(store)
    try:
        manager.submit(["failed"], [123], [8], owner_buffer_ptr=123)
        with pytest.raises(RuntimeError, match=r"failed \(code=-200\)"):
            manager.wait_for_buffer(123)
        manager.check_last_error()
        manager.check_errors()
        manager.submit(["next"], [123], [8], owner_buffer_ptr=123)
        manager.wait_for_buffer(123)
        manager.check_last_error()
    finally:
        manager.shutdown()
    store.batch_remove.assert_called_once_with(["failed"], force=True)

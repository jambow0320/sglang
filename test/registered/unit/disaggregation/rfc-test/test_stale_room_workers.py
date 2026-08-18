import importlib
import sys
import threading
import types
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock


def _install_mori_stubs():
    """Make Mori control paths importable without the optional Mori package."""
    mori = types.ModuleType("mori")
    mori.__path__ = []
    mori_cpp = types.ModuleType("mori.cpp")
    mori_io = types.ModuleType("mori.io")

    class _Dummy:
        pass

    mori_cpp.TransferStatus = _Dummy
    for name in (
        "BackendType",
        "EngineDesc",
        "IOEngine",
        "IOEngineConfig",
        "MemoryDesc",
        "MemoryLocationType",
        "PollCqMode",
        "RdmaBackendConfig",
        "StatusCode",
    ):
        setattr(mori_io, name, _Dummy)

    mori.cpp = mori_cpp
    mori.io = mori_io
    sys.modules.setdefault("mori", mori)
    sys.modules.setdefault("mori.cpp", mori_cpp)
    sys.modules.setdefault("mori.io", mori_io)


_install_mori_stubs()

KVPoll = importlib.import_module("sglang.srt.disaggregation.base.conn").KVPoll
NixlKVManager = importlib.import_module(
    "sglang.srt.disaggregation.nixl.conn"
).NixlKVManager
MoriKVSender = importlib.import_module(
    "sglang.srt.disaggregation.mori.conn"
).MoriKVSender
register_cpu_ci = importlib.import_module(
    "sglang.test.ci.ci_register"
).register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _OneChunkQueue:
    def __init__(self, room):
        self.chunk = SimpleNamespace(room=room)

    def get(self):
        if self.chunk is not None:
            chunk = self.chunk
            self.chunk = None
            return chunk
        raise KeyboardInterrupt


class TestStaleRoomWorkers(unittest.TestCase):
    def test_nixl_worker_skips_cleared_room(self):
        room = 21
        manager = object.__new__(NixlKVManager)
        manager.request_status = {}
        manager.transfer_infos = {}
        manager._staging_outstanding = {room: 1}
        manager.exceptions = {}
        manager.failure_records = {}
        manager.failure_lock = threading.Lock()

        with self.assertRaises(KeyboardInterrupt):
            manager.transfer_worker(_OneChunkQueue(room))

        self.assertNotIn(room, manager._staging_outstanding)
        self.assertEqual(manager.exceptions, {})

    def test_mori_worker_skips_cleared_room_before_transfer_submission(self):
        room = 21
        add_transfer_request = MagicMock(
            side_effect=AssertionError("stale room reached transfer submission")
        )
        sender = object.__new__(MoriKVSender)
        sender.bootstrap_room = room
        sender.conclude_state = None
        sender.kv_mgr = SimpleNamespace(
            request_status={},
            add_transfer_request=add_transfer_request,
        )
        task = SimpleNamespace(
            wait_event=None,
            kv_indices=[],
            index_slice=slice(0, 0),
            is_last_chunk=False,
            aux_index=None,
            normalized_state=None,
        )

        sender._run_chunk(task)

        add_transfer_request.assert_not_called()


if __name__ == "__main__":
    unittest.main()

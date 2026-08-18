import importlib
import sys
import threading
import types
import unittest
from unittest.mock import patch


def _install_mori_stubs():
    """Make the control-path test importable without the optional Mori package."""
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

MoriKVManager = importlib.import_module(
    "sglang.srt.disaggregation.mori.conn"
).MoriKVManager
KVPoll = importlib.import_module("sglang.srt.disaggregation.base.conn").KVPoll
register_cpu_ci = importlib.import_module(
    "sglang.test.ci.ci_register"
).register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _AbortThenStopSocket:
    def __init__(self, room):
        self.room = room
        self.received = False

    def recv_multipart(self):
        if not self.received:
            self.received = True
            return [b"ABORT", str(self.room).encode("ascii"), b"127.0.0.1", b"9000"]
        raise KeyboardInterrupt


class TestMoriAbortDelivery(unittest.TestCase):
    def test_common_abort_frame_is_handled_before_mori_guard_validation(self):
        room = 21
        manager = object.__new__(MoriKVManager)
        manager.server_socket = _AbortThenStopSocket(room)
        manager.request_status = {room: KVPoll.WaitingForInput}
        manager.failure_records = {}
        manager.failure_lock = threading.Lock()
        captured = {}

        class _FakeThread:
            def __init__(self, target, **_kwargs):
                captured["target"] = target

            def start(self):
                pass

        with (
            patch(
                "sglang.srt.disaggregation.mori.conn.threading.Thread",
                _FakeThread,
            ),
            patch.object(
                manager,
                "_validate_message",
                wraps=manager._validate_message,
            ) as validate_message,
        ):
            manager._start_bootstrap_thread()
            with self.assertRaises(KeyboardInterrupt):
                captured["target"]()

        validate_message.assert_not_called()
        self.assertEqual(manager.request_status[room], KVPoll.Failed)
        self.assertIn("decode-side abort", manager.failure_records[room])


if __name__ == "__main__":
    unittest.main()

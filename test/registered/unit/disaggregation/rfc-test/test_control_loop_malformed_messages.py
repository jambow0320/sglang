import importlib
import threading
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

KVPoll = importlib.import_module("sglang.srt.disaggregation.base.conn").KVPoll
mooncake_conn = importlib.import_module(
    "sglang.srt.disaggregation.mooncake.conn"
)
nixl_conn = importlib.import_module("sglang.srt.disaggregation.nixl.conn")
register_cpu_ci = importlib.import_module(
    "sglang.test.ci.ci_register"
).register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _SequenceSocket:
    def __init__(self, messages):
        self.messages = list(messages)

    def recv_multipart(self):
        if self.messages:
            return self.messages.pop(0)
        raise KeyboardInterrupt


def _capture_thread(module_path, start_thread):
    captured = {}

    class _FakeThread:
        def __init__(self, target, **_kwargs):
            captured["target"] = target

        def start(self):
            pass

    with patch(f"{module_path}.threading.Thread", _FakeThread):
        start_thread()
    return captured["target"]


class TestMalformedControlMessages(unittest.TestCase):
    def test_mooncake_prefill_survives_truncated_abort(self):
        room = 21
        manager = object.__new__(mooncake_conn.MooncakeKVManager)
        manager.server_socket = _SequenceSocket(
            [
                [b"ABORT"],
                [b"ABORT", b"21", b"127.0.0.1", b"9000"],
            ]
        )
        manager.request_status = {room: KVPoll.WaitingForInput}
        manager._send_multipart_locked = MagicMock()

        target = _capture_thread(
            "sglang.srt.disaggregation.mooncake.conn",
            manager.start_prefill_thread,
        )
        with self.assertRaises(KeyboardInterrupt):
            target()

        self.assertEqual(manager.request_status[room], KVPoll.Failed)
        manager._send_multipart_locked.assert_called_once()

    def test_mooncake_decode_survives_unknown_frame_count(self):
        room = 21
        manager = object.__new__(mooncake_conn.MooncakeKVManager)
        manager.server_socket = _SequenceSocket(
            [
                [b"unexpected", b"extra"],
                [b"21", str(KVPoll.Success).encode("ascii"), b"0"],
            ]
        )
        manager.request_status = {room: KVPoll.WaitingForInput}
        manager.prefill_response_tracker = defaultdict(set)
        manager.required_prefill_response_num_table = {room: 1}
        manager.enable_staging = False
        manager._start_heartbeat_checker_thread = MagicMock()

        target = _capture_thread(
            "sglang.srt.disaggregation.mooncake.conn",
            manager.start_decode_thread,
        )
        with self.assertRaises(KeyboardInterrupt):
            target()

        self.assertEqual(manager.request_status[room], KVPoll.Success)

    def test_nixl_prefill_survives_invalid_and_truncated_frames(self):
        room = 21
        manager = object.__new__(nixl_conn.NixlKVManager)
        manager.server_socket = _SequenceSocket(
            [
                [],
                [b"invalid-guard"],
                [nixl_conn.GUARD],
                [b"ABORT", b"21", b"127.0.0.1", b"9000"],
            ]
        )
        manager.request_status = {room: KVPoll.WaitingForInput}
        manager.failure_records = {}
        manager.failure_lock = threading.Lock()

        target = _capture_thread(
            "sglang.srt.disaggregation.nixl.conn",
            manager._start_bootstrap_thread,
        )
        with self.assertRaises(KeyboardInterrupt):
            target()

        self.assertEqual(manager.request_status[room], KVPoll.Failed)

    def test_nixl_staging_survives_empty_and_truncated_frames(self):
        manager = object.__new__(nixl_conn.NixlKVManager)
        manager.server_socket = _SequenceSocket(
            [
                [],
                [b"STAGING_REQ"],
                [b"STAGING_REQ", b"21", b"0", b"1", b"peer"],
            ]
        )
        handled = []

        def handle_staging_req(msg):
            if len(msg) < 5:
                raise IndexError("incomplete STAGING_REQ")
            handled.append(msg)

        manager._handle_staging_req = MagicMock(side_effect=handle_staging_req)

        target = _capture_thread(
            "sglang.srt.disaggregation.nixl.conn",
            manager._start_decode_staging_thread,
        )
        with self.assertRaises(KeyboardInterrupt):
            target()

        self.assertEqual(manager._handle_staging_req.call_count, 2)
        self.assertEqual(len(handled), 1)
        self.assertEqual(handled[0][1], b"21")


if __name__ == "__main__":
    unittest.main()

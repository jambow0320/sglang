"""CPU unit tests for the prefill->decode transfer status notification."""

import importlib.util
import sys
import threading
import types
import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np


def _stub_mori_package() -> None:
    """Bind the names ``mori/conn.py`` imports so it loads without the wheel.

    mori ships only for AMD, so the module is otherwise unimportable here and
    its control plane would have no CPU coverage at all. Only the symbols
    resolved at import time are stubbed; anything that actually drives the
    engine stays out of these tests.
    """
    if importlib.util.find_spec("mori") is not None:
        return
    package = types.ModuleType("mori")
    cpp = types.ModuleType("mori.cpp")
    cpp.TransferStatus = type("TransferStatus", (), {})
    io = types.ModuleType("mori.io")
    for name in (
        "BackendType",
        "EngineDesc",
        "IOEngine",
        "IOEngineConfig",
        "MemoryDesc",
        "MemoryLocationType",
        "PollCqMode",
        "RdmaBackendConfig",
    ):
        setattr(io, name, type(name, (), {}))
    io.StatusCode = type("StatusCode", (), {"SUCCESS": 0, "IN_PROGRESS": 1})
    package.cpp = cpp
    package.io = io
    sys.modules.update({"mori": package, "mori.cpp": cpp, "mori.io": io})


_stub_mori_package()

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.disaggregation.mooncake.conn import TransferInfo as MooncakeTransferInfo
from sglang.srt.disaggregation.mori.conn import (
    MORI_GUARD,
    MoriKVManager,
    MoriKVReceiver,
)
from sglang.srt.disaggregation.mori.conn import TransferInfo as MoriTransferInfo
from sglang.srt.disaggregation.nixl.conn import (
    NixlKVManager,
)
from sglang.srt.disaggregation.nixl.conn import TransferInfo as NixlTransferInfo
from sglang.srt.disaggregation.nixl.conn import (
    TransferKVChunk,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _arm_prefill_manager(mgr, room, endpoints, transfer_info_factory):
    """Give a bare manager the state conclude_transfer() touches."""
    mgr.request_status = {room: KVPoll.WaitingForInput}
    mgr.failure_records = {}
    mgr.failure_lock = threading.Lock()
    mgr.transfer_infos = {
        room: {
            f"peer-{port}": transfer_info_factory(room, ip, port, is_dummy)
            for ip, port, is_dummy in endpoints
        }
    }
    mgr.attn_tp_rank = 0
    mgr.pp_rank = 0
    mgr.attn_cp_rank = 0
    mgr.pp_size = 1
    mgr.attn_cp_size = 1
    mgr._send_multipart_locked = MagicMock()
    return mgr


def _mooncake_transfer_info(room, ip, port, is_dummy):
    return MooncakeTransferInfo(
        room=room,
        endpoint=ip,
        dst_port=port,
        mooncake_session_id=f"{ip}:{port}",
        dst_kv_indices=np.array([], dtype=np.int32),
        dst_aux_index=0,
        dst_state_indices=[],
        required_dst_info_num=1,
        is_dummy=is_dummy,
    )


def _nixl_transfer_info(room, ip, port, is_dummy):
    return NixlTransferInfo(
        room=room,
        endpoint=ip,
        dst_port=port,
        agent_name=f"agent-{port}",
        dst_kv_indices=np.array([2], dtype=np.int32),
        dst_aux_index=0,
        required_dst_info_num=1,
        dst_state_indices=[],
        is_dummy=is_dummy,
    )


def _mori_transfer_info(room, ip, port, is_dummy):
    return MoriTransferInfo(
        room=room,
        endpoint=ip,
        dst_port=port,
        engine_key=f"engine-{port}",
        dst_kv_indices=np.array([2], dtype=np.int32),
        dst_aux_index=0,
        dst_state_indices=[],
        required_dst_info_num=1,
        is_dummy=is_dummy,
    )


def _make_mooncake_prefill_manager(room=7, endpoints=(("10.0.0.1", 5555, False),)):
    mgr = object.__new__(MooncakeKVManager)
    return _arm_prefill_manager(mgr, room, endpoints, _mooncake_transfer_info)


def _make_mori_prefill_manager(room=7, endpoints=(("10.0.0.1", 5555, False),)):
    mgr = object.__new__(MoriKVManager)
    mgr.transfer_lock = threading.Lock()
    return _arm_prefill_manager(mgr, room, endpoints, _mori_transfer_info)


def _make_nixl_prefill_manager(room=7, endpoints=(("10.0.0.1", 5555, False),)):
    mgr = object.__new__(NixlKVManager)
    return _arm_prefill_manager(mgr, room, endpoints, _nixl_transfer_info)


def _sent_frames(mgr):
    return [call.args[1] for call in mgr._send_multipart_locked.call_args_list]


class TestStatusMessageWire(CustomTestCase):
    def test_mooncake_keeps_the_legacy_three_untagged_frames(self):
        mgr = _make_mooncake_prefill_manager()

        mgr.send_kv_status_message(
            targets=[("10.0.0.1", 5555)],
            bootstrap_room=7,
            status=KVPoll.Failed,
            failure_reason="session is dead",
        )

        self.assertEqual(
            _sent_frames(mgr), [[b"7", str(int(KVPoll.Failed)).encode(), b"0"]]
        )

    def test_nixl_tags_the_message_and_carries_the_reason(self):
        mgr = _make_nixl_prefill_manager()

        mgr.send_kv_status_message(
            targets=[("10.0.0.1", 5555)],
            bootstrap_room=7,
            status=KVPoll.Failed,
            failure_reason="NIXL transfer encountered ERR",
        )

        self.assertEqual(
            _sent_frames(mgr),
            [
                [
                    b"KV_STATUS",
                    b"7",
                    str(int(KVPoll.Failed)).encode(),
                    b"0",
                    b"NIXL transfer encountered ERR",
                ]
            ],
        )

    def test_each_backend_parses_back_what_it_encoded(self):
        for name, mgr, reason in (
            ("mooncake", _make_mooncake_prefill_manager(), None),
            ("nixl", _make_nixl_prefill_manager(), "boom"),
        ):
            with self.subTest(backend=name):
                frames = mgr._encode_kv_status_message(
                    bootstrap_room=7, status=KVPoll.Failed, failure_reason="boom"
                )

                self.assertEqual(
                    mgr.parse_kv_status_message(frames),
                    (7, int(KVPoll.Failed), 0, reason),
                )

    def test_legacy_decode_accepts_a_reason_frame_it_never_sends(self):
        mgr = _make_mooncake_prefill_manager()

        parsed = mgr.parse_kv_status_message(
            [b"7", str(int(KVPoll.Failed)).encode(), b"3", b"kv chunk send failed"]
        )

        self.assertEqual(parsed, (7, int(KVPoll.Failed), 3, "kv chunk send failed"))

    def test_unparsable_and_foreign_messages_are_dropped(self):
        mooncake = _make_mooncake_prefill_manager()
        nixl = _make_nixl_prefill_manager()

        self.assertIsNone(mooncake.parse_kv_status_message([b"7", b"2"]))
        self.assertIsNone(
            mooncake.parse_kv_status_message([b"CHUNK_READY", b"7", b"0"])
        )
        self.assertIsNone(nixl.parse_kv_status_message([b"7", b"2", b"0"]))
        self.assertIsNone(nixl.parse_kv_status_message([b"STAGING_REQ", b"7", b"0"]))


class TestMoriOnCommonHelpers(CustomTestCase):
    """Mori keeps its 5-frame layout while sharing the common conclude path."""

    def test_the_wire_layout_is_unchanged_by_the_shared_encoder(self):
        mgr = _make_mori_prefill_manager()

        mgr.conclude_failure(
            bootstrap_room=7, failure_reason="KV transfer failed: rdma error"
        )

        self.assertEqual(
            _sent_frames(mgr),
            [
                [
                    MORI_GUARD,
                    b"7",
                    str(int(KVPoll.Failed)).encode(),
                    b"0",
                    b"KV transfer failed: rdma error",
                ]
            ],
        )

    def test_success_is_downgraded_when_a_failure_was_recorded_in_flight(self):
        mgr = _make_mori_prefill_manager()
        mgr.record_failure(7, "Aborted by AbortReq.")

        concluded = mgr.conclude_transfer(bootstrap_room=7, status=KVPoll.Success)

        self.assertEqual(concluded, KVPoll.Failed)
        self.assertEqual(mgr.request_status[7], KVPoll.Failed)
        frames = _sent_frames(mgr)
        self.assertEqual(frames[0][2], str(int(KVPoll.Failed)).encode())
        self.assertEqual(frames[0][4], b"Aborted by AbortReq.")

    def test_dummy_endpoints_are_not_notified(self):
        mgr = _make_mori_prefill_manager(
            endpoints=(("10.0.0.1", 5555, False), ("10.0.0.2", 5556, True))
        )

        mgr.conclude_failure(bootstrap_room=7, failure_reason="rdma error")

        self.assertEqual(
            [call.args[0] for call in mgr._send_multipart_locked.call_args_list],
            ["tcp://10.0.0.1:5555"],
        )

    def test_decode_side_room_tracking_is_dropped_by_receiver_clear(self):
        mgr = object.__new__(MoriKVManager)
        mgr.request_status = {7: KVPoll.Success}
        mgr.prefill_response_tracker = defaultdict(set, {7: {0, 1}})
        mgr.required_prefill_response_num_table = {7: 2}
        mgr.addr_to_rooms_tracker = defaultdict(set, {"10.0.0.9:8998": {7, 8}})
        receiver = object.__new__(MoriKVReceiver)
        receiver.kv_mgr = mgr
        receiver.bootstrap_room = 7
        receiver.bootstrap_addr = "10.0.0.9:8998"

        receiver.clear()

        self.assertEqual(mgr.request_status, {})
        self.assertEqual(mgr.required_prefill_response_num_table, {})
        self.assertNotIn(7, mgr.prefill_response_tracker)
        # Only this room leaves the tracker; a sibling room on the same prefill
        # keeps its entry.
        self.assertEqual(mgr.addr_to_rooms_tracker, {"10.0.0.9:8998": {8}})


class TestUpdateStatus(CustomTestCase):
    """The sticky-Failed / no-resurrect policy shared by all three backends."""

    def _make_manager(self, request_status):
        mgr = object.__new__(CommonKVManager)
        mgr.request_status = dict(request_status)
        return mgr

    def test_failed_is_never_promoted(self):
        for status in (KVPoll.WaitingForInput, KVPoll.Transferring, KVPoll.Success):
            with self.subTest(status=status):
                mgr = self._make_manager({7: KVPoll.Failed})

                mgr.update_status(7, status)

                self.assertEqual(mgr.request_status[7], KVPoll.Failed)

    def test_only_an_opening_status_may_create_a_room(self):
        for status, expected in (
            (KVPoll.Bootstrapping, KVPoll.Bootstrapping),
            (KVPoll.WaitingForInput, KVPoll.WaitingForInput),
            (KVPoll.Transferring, None),
            (KVPoll.Success, None),
            (KVPoll.Failed, None),
        ):
            with self.subTest(status=status):
                mgr = self._make_manager({})

                mgr.update_status(7, status)

                self.assertEqual(mgr.request_status.get(7), expected)

    def test_a_live_room_still_advances(self):
        mgr = self._make_manager({7: KVPoll.Bootstrapping})

        mgr.update_status(7, KVPoll.Transferring)
        self.assertEqual(mgr.request_status[7], KVPoll.Transferring)

        mgr.update_status(7, KVPoll.WaitingForInput)
        self.assertEqual(mgr.request_status[7], KVPoll.Transferring)

        mgr.update_status(7, KVPoll.Failed)
        self.assertEqual(mgr.request_status[7], KVPoll.Failed)


class TestConcludeTransfer(CustomTestCase):
    def test_failure_reaches_every_non_dummy_endpoint_of_the_room(self):
        mgr = _make_mooncake_prefill_manager(
            endpoints=(
                ("10.0.0.1", 5555, False),
                ("10.0.0.2", 5556, False),
                ("10.0.0.3", 5557, True),
            )
        )

        mgr.conclude_failure(bootstrap_room=7, failure_reason="session is dead")

        self.assertEqual(
            [call.args[0] for call in mgr._send_multipart_locked.call_args_list],
            ["tcp://10.0.0.1:5555", "tcp://10.0.0.2:5556"],
        )
        self.assertEqual(mgr.request_status[7], KVPoll.Failed)
        self.assertEqual(mgr.failure_records[7], "session is dead")

    def test_the_first_root_cause_is_kept(self):
        mgr = _make_mooncake_prefill_manager()

        mgr.conclude_failure(bootstrap_room=7, failure_reason="kv chunk send failed")
        mgr.conclude_failure(
            bootstrap_room=7, failure_reason="state components send failed"
        )

        self.assertEqual(mgr.failure_records[7], "kv chunk send failed")

    def test_success_after_a_recorded_failure_is_downgraded(self):
        mgr = _make_mooncake_prefill_manager()
        mgr.record_failure(7, "aux data send failed")

        concluded = mgr.conclude_transfer(bootstrap_room=7, status=KVPoll.Success)

        self.assertEqual(concluded, KVPoll.Failed)
        self.assertEqual(mgr.request_status[7], KVPoll.Failed)
        self.assertEqual(
            _sent_frames(mgr), [[b"7", str(int(KVPoll.Failed)).encode(), b"0"]]
        )

    def test_success_notifies_the_endpoints_the_caller_passed(self):
        mgr = _make_mooncake_prefill_manager()

        concluded = mgr.conclude_transfer(
            bootstrap_room=7, status=KVPoll.Success, targets=[("10.0.0.9", 5599)]
        )

        self.assertEqual(concluded, KVPoll.Success)
        self.assertEqual(mgr.request_status[7], KVPoll.Success)
        self.assertEqual(
            mgr._send_multipart_locked.call_args.args[0], "tcp://10.0.0.9:5599"
        )

    def test_a_cleared_room_is_not_resurrected_by_a_late_conclude(self):
        mgr = _make_mooncake_prefill_manager()
        mgr.request_status.pop(7)

        self.assertIsNone(
            mgr.conclude_failure(
                bootstrap_room=7, failure_reason="kv chunk send failed"
            )
        )

        self.assertNotIn(7, mgr.request_status)
        self.assertEqual(mgr.failure_records, {})
        mgr._send_multipart_locked.assert_not_called()


class TestApplyPrefillStatus(CustomTestCase):
    def _make_decode_manager(self, room, required=1):
        mgr = object.__new__(MooncakeKVManager)
        mgr.request_status = {room: KVPoll.WaitingForInput}
        mgr.failure_records = {}
        mgr.failure_lock = threading.Lock()
        mgr.prefill_response_tracker = defaultdict(set)
        mgr.required_prefill_response_num_table = {room: required}
        mgr.enable_staging = False
        mgr._staging_handler = None
        return mgr

    @staticmethod
    def _arm_staging(mgr, room, is_staging_room=True):
        """Put a recording staging handler behind the room, as the decode
        transfer queue does once staging is enabled."""
        armed = []
        mgr.enable_staging = True
        mgr._staging_handler = SimpleNamespace(
            is_staging_room=lambda r: is_staging_room,
            submit_last_scatter_async=armed.append,
        )
        return armed

    def test_failure_marks_the_room_and_records_the_reported_reason(self):
        mgr = self._make_decode_manager(7)

        mgr.apply_prefill_status(
            bootstrap_room=7,
            status=KVPoll.Failed,
            prefill_rank=0,
            failure_reason="NIXL transfer encountered ERR",
        )

        self.assertEqual(mgr.request_status[7], KVPoll.Failed)
        self.assertEqual(mgr.failure_records[7], "NIXL transfer encountered ERR")

    def test_failure_without_a_reason_falls_back_to_the_generic_message(self):
        mgr = self._make_decode_manager(7)

        mgr.apply_prefill_status(bootstrap_room=7, status=KVPoll.Failed, prefill_rank=0)

        self.assertEqual(
            mgr.failure_records[7], MooncakeKVManager.DEFAULT_PREFILL_FAILURE_REASON
        )

    def test_a_cleared_room_is_left_untouched(self):
        mgr = self._make_decode_manager(7)
        mgr.request_status.pop(7)

        mgr.apply_prefill_status(
            bootstrap_room=7,
            status=KVPoll.Failed,
            prefill_rank=0,
            failure_reason="too late",
        )

        self.assertEqual(mgr.failure_records, {})
        self.assertNotIn(7, mgr.request_status)

    def test_success_concludes_only_once_every_prefill_rank_reported(self):
        mgr = self._make_decode_manager(7, required=2)
        armed = self._arm_staging(mgr, 7)

        mgr.apply_prefill_status(
            bootstrap_room=7, status=KVPoll.Success, prefill_rank=0
        )
        self.assertEqual(mgr.request_status[7], KVPoll.WaitingForInput)
        self.assertEqual(armed, [])

        # A repeat from the same rank must not count as a second response.
        mgr.apply_prefill_status(
            bootstrap_room=7, status=KVPoll.Success, prefill_rank=0
        )
        self.assertEqual(mgr.request_status[7], KVPoll.WaitingForInput)

        mgr.apply_prefill_status(
            bootstrap_room=7, status=KVPoll.Success, prefill_rank=1
        )
        self.assertEqual(mgr.request_status[7], KVPoll.Success)
        # The staging handler learns the prefill side is done before any poller
        # can observe Success.
        self.assertEqual(armed, [7])

    def test_staging_is_left_alone_when_the_room_does_not_use_it(self):
        mgr = self._make_decode_manager(7)
        armed = self._arm_staging(mgr, 7, is_staging_room=False)

        mgr.apply_prefill_status(
            bootstrap_room=7, status=KVPoll.Success, prefill_rank=0
        )

        self.assertEqual(mgr.request_status[7], KVPoll.Success)
        self.assertEqual(armed, [])

    def test_success_without_staging_needs_no_handler(self):
        mgr = self._make_decode_manager(7)

        mgr.apply_prefill_status(
            bootstrap_room=7, status=KVPoll.Success, prefill_rank=0
        )

        self.assertEqual(mgr.request_status[7], KVPoll.Success)


class TestNixlWorkerFailureNotification(CustomTestCase):
    ROOM = 21

    def _make_manager(self, xfer_state):
        mgr = _make_nixl_prefill_manager(room=self.ROOM)
        mgr.decode_kv_args_table = {
            "agent-5555": SimpleNamespace(
                decode_tp_size=1,
                dst_kv_ptrs=[0],
                dst_aux_ptrs=[0],
                gpu_id=0,
                staging_base_ptr=0,
                staging_total_size=0,
                kv_xfer_segments=None,
                dst_homogeneous_mem_kind="VRAM",
                requires_dcp_relayout=False,
                dcp_dst_region_indices=None,
                dcp_token_item_lens=None,
            )
        }
        mgr.req_to_decode_prefix_len = {self.ROOM: 0}
        mgr.enable_staging = False
        mgr.enable_deferred_decode_kv_release = False
        mgr._staging_ctx = None
        mgr._staging_outstanding = defaultdict(int)
        mgr.is_mla_backend = False
        mgr.is_hybrid_mla_backend = False
        mgr.attn_tp_size = 1
        mgr.transfer_source_rank = 0
        mgr.kv_args = SimpleNamespace(engine_rank=0, kv_data_ptrs=[0])
        mgr.exceptions = {}
        mgr.agent = SimpleNamespace(check_xfer_state=lambda _handle: xfer_state)
        mgr.send_kvcache = MagicMock(return_value="kv_handle")
        return mgr

    def _run_worker_once(self, mgr, is_last_chunk=False):
        chunk = TransferKVChunk(
            room=self.ROOM,
            prefill_kv_indices=np.array([1], dtype=np.int32),
            index_slice=slice(0, 1),
            is_last_chunk=is_last_chunk,
            chunk_id=0,
            prefill_aux_index=0 if is_last_chunk else None,
            state_indices=None,
        )
        queue = SimpleNamespace(get=MagicMock(side_effect=[chunk, SystemExit()]))
        with self.assertRaises(SystemExit):
            mgr.transfer_worker(queue)

    def test_a_transfer_error_is_reported_to_decode(self):
        mgr = self._make_manager("ERR")

        self._run_worker_once(mgr)

        self.assertEqual(mgr.request_status[self.ROOM], KVPoll.Failed)
        frames = _sent_frames(mgr)
        self.assertEqual(len(frames), 1)
        self.assertEqual(
            frames[0][:4],
            [b"KV_STATUS", b"21", str(int(KVPoll.Failed)).encode(), b"0"],
        )
        self.assertIn(b"ERR", frames[0][4])

    def test_the_error_is_held_back_until_every_sibling_settled(self):
        # One handle fails immediately; the other keeps writing for a few polls.
        # Reporting Failed while it runs would let decode reuse pages it is
        # still writing into, so the worker must wait it out first.
        mgr = self._make_manager("ERR")
        polls = {"kv_handle": 0, "aux_handle": 0}
        settled_at_send = []

        def check_xfer_state(handle):
            polls[handle] += 1
            if handle == "kv_handle":
                return "ERR"
            return "PROC" if polls[handle] < 4 else "DONE"

        mgr.agent = SimpleNamespace(check_xfer_state=check_xfer_state)
        mgr.send_aux = MagicMock(return_value="aux_handle")
        mgr._send_multipart_locked = MagicMock(
            side_effect=lambda *a, **kw: settled_at_send.append(polls["aux_handle"])
        )

        self._run_worker_once(mgr, is_last_chunk=True)

        self.assertEqual(mgr.request_status[self.ROOM], KVPoll.Failed)
        self.assertEqual(settled_at_send, [4])


if __name__ == "__main__":
    unittest.main()

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.nixl.conn import NixlKVManager, NixlKVSender
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestNixlSenderBootstrapTimeout(unittest.TestCase):
    ROOM = 21

    def _make_manager(self):
        manager = object.__new__(NixlKVManager)
        manager.request_status = {}
        manager.failure_records = {}
        manager.failure_lock = threading.Lock()
        manager.bootstrap_timeout = 5
        manager._staging_outstanding = {}
        manager.is_dummy_cp_rank = False
        return manager

    @patch("sglang.srt.disaggregation.common.conn.get_parallel")
    @patch("sglang.srt.disaggregation.common.conn.time.time")
    def test_missing_metadata_times_out(self, mock_time, mock_get_parallel):
        mock_time.return_value = 10.0
        mock_get_parallel.return_value = SimpleNamespace(dp_size=1)
        manager = self._make_manager()

        sender = NixlKVSender(manager, "prefill:8998", self.ROOM, [0], 0)

        self.assertEqual(sender.init_time, 10.0)
        mock_time.return_value = 20.0
        self.assertEqual(sender.poll(), KVPoll.Failed)
        self.assertEqual(manager.request_status[self.ROOM], KVPoll.Failed)
        self.assertIn("timed out", manager.failure_records[self.ROOM])


if __name__ == "__main__":
    unittest.main()

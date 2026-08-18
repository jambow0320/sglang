import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

CommonKVManager = importlib.import_module(
    "sglang.srt.disaggregation.common.conn"
).CommonKVManager
MooncakeKVManager = importlib.import_module(
    "sglang.srt.disaggregation.mooncake.conn"
).MooncakeKVManager
DisaggregationMode = importlib.import_module(
    "sglang.srt.disaggregation.utils"
).DisaggregationMode
register_cpu_ci = importlib.import_module(
    "sglang.test.ci.ci_register"
).register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _StopAfterControlThreadStart(Exception):
    pass


class TestMooncakeStartupOrder(unittest.TestCase):
    def test_session_state_exists_before_prefill_control_thread_starts(self):
        manager = object.__new__(MooncakeKVManager)
        manager.disaggregation_mode = DisaggregationMode.PREFILL
        observed = {}

        def inspect_session_state():
            observed["session_failures"] = hasattr(manager, "session_failures")
            observed["failed_sessions"] = hasattr(manager, "failed_sessions")
            observed["session_lock"] = hasattr(manager, "session_lock")
            raise _StopAfterControlThreadStart

        with (
            patch.object(CommonKVManager, "__init__", return_value=None),
            patch.object(MooncakeKVManager, "init_engine"),
            patch.object(MooncakeKVManager, "register_buffer_to_engine"),
            patch.object(
                MooncakeKVManager,
                "start_prefill_thread",
                side_effect=inspect_session_state,
            ),
        ):
            with self.assertRaises(_StopAfterControlThreadStart):
                MooncakeKVManager.__init__(
                    manager,
                    args=object(),
                    disaggregation_mode=DisaggregationMode.PREFILL,
                    server_args=SimpleNamespace(enable_trace=False),
                )

        self.assertEqual(
            observed,
            {
                "session_failures": True,
                "failed_sessions": True,
                "session_lock": True,
            },
        )


if __name__ == "__main__":
    unittest.main()

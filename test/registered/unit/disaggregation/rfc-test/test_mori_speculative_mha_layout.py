import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


def _install_mori_stubs():
    """Make Mori planner code importable without the optional Mori package."""
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

CommonKVManager = importlib.import_module(
    "sglang.srt.disaggregation.common.conn"
).CommonKVManager
MoriKVManager = importlib.import_module(
    "sglang.srt.disaggregation.mori.conn"
).MoriKVManager
register_cpu_ci = importlib.import_module(
    "sglang.test.ci.ci_register"
).register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestMoriSpeculativeMHALayout(unittest.TestCase):
    def _assert_mha_matches_common(self, src_descs, dst_descs, start_layer):
        kv_args = SimpleNamespace(prefill_start_layer=start_layer)

        common = object.__new__(CommonKVManager)
        common.kv_args = kv_args
        _, _, expected_k, expected_v, expected_layers = (
            common.get_mha_kv_ptrs_with_pp(src_descs, dst_descs)
        )

        mori = object.__new__(MoriKVManager)
        mori.kv_args = kv_args
        mori.kv_mem_descs = src_descs
        _, _, actual_k, actual_v, actual_layers = mori._get_mha_mem_desc_slices(
            dst_descs
        )

        self.assertEqual(actual_layers, expected_layers)
        self.assertEqual(actual_k, expected_k)
        self.assertEqual(actual_v, expected_v)
        return actual_k, actual_v

    def test_destination_regions_match_common_planner(self):
        # Prefill PP stage owns target layers 2..3. Decode additionally has one
        # draft layer: [K_main x4, V_main x4, draft_K, draft_V].
        src_descs = ["src_k2", "src_k3", "src_v2", "src_v3"]
        dst_descs = [
            "dst_k0",
            "dst_k1",
            "dst_k2",
            "dst_k3",
            "dst_v0",
            "dst_v1",
            "dst_v2",
            "dst_v3",
            "draft_k",
            "draft_v",
        ]
        _, actual_v = self._assert_mha_matches_common(src_descs, dst_descs, 2)
        self.assertEqual(actual_v, ["dst_v2", "dst_v3"])

    def test_same_pp_uses_matching_local_mha_descriptors(self):
        src_descs = ["src_k2", "src_k3", "src_v2", "src_v3"]
        dst_descs = ["dst_k2", "dst_k3", "dst_v2", "dst_v3"]

        actual_k, actual_v = self._assert_mha_matches_common(
            src_descs, dst_descs, 2
        )

        self.assertEqual(actual_k, ["dst_k2", "dst_k3"])
        self.assertEqual(actual_v, ["dst_v2", "dst_v3"])

    def test_decode_pp_one_uses_global_mha_descriptors(self):
        src_descs = ["src_k2", "src_k3", "src_v2", "src_v3"]
        dst_descs = [
            "dst_k0",
            "dst_k1",
            "dst_k2",
            "dst_k3",
            "dst_v0",
            "dst_v1",
            "dst_v2",
            "dst_v3",
        ]

        actual_k, actual_v = self._assert_mha_matches_common(
            src_descs, dst_descs, 2
        )

        self.assertEqual(actual_k, ["dst_k2", "dst_k3"])
        self.assertEqual(actual_v, ["dst_v2", "dst_v3"])

    def test_same_pp_uses_matching_local_mla_descriptors(self):
        src_descs = ["src_kv2", "src_kv3"]
        dst_descs = ["dst_kv2", "dst_kv3"]
        kv_args = SimpleNamespace(prefill_start_layer=2)

        common = object.__new__(CommonKVManager)
        common.kv_args = kv_args
        _, expected_dst, expected_layers = common.get_mla_kv_ptrs_with_pp(
            src_descs, dst_descs
        )

        mori = object.__new__(MoriKVManager)
        mori.kv_args = kv_args
        mori.kv_mem_descs = src_descs
        _, actual_dst, actual_layers = mori._get_mla_mem_desc_slices(dst_descs)

        self.assertEqual(actual_layers, expected_layers)
        self.assertEqual(actual_dst, expected_dst)
        self.assertEqual(actual_dst, ["dst_kv2", "dst_kv3"])

    def test_hybrid_mla_uses_flat_mla_planner(self):
        mori = object.__new__(MoriKVManager)
        mori.is_mla_backend = False
        mori.is_hybrid_mla_backend = True
        mori.kv_args = SimpleNamespace(kv_item_lens=[16])
        peer_info = SimpleNamespace(dst_kv_mem_descs=["dst_kv"])

        with (
            patch.object(
                mori,
                "_get_mla_mem_desc_slices",
                return_value=([], [], 0),
            ) as get_mla,
            patch.object(mori, "_get_mha_mem_desc_slices") as get_mha,
        ):
            statuses = mori.send_kvcache(
                peer_info,
                np.array([0], dtype=np.int32),
                np.array([0], dtype=np.int32),
            )

        self.assertEqual(statuses, [])
        get_mla.assert_called_once_with(peer_info.dst_kv_mem_descs)
        get_mha.assert_not_called()


if __name__ == "__main__":
    unittest.main()

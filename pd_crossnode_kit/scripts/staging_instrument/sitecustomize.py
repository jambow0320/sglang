"""Count NIXL / Mooncake staging transfers and deferrals without editing the sglang tree.

Enabled by putting this directory on PYTHONPATH. Prints cumulative counters to
stderr each time a counter reaches a power of two.
"""

import importlib.abc
import importlib.machinery
import os
import sys
import threading

_lock = threading.Lock()
_counts = {}


def _bump(key):
    with _lock:
        _counts[key] = _counts.get(key, 0) + 1
        n = _counts[key]
        snapshot = dict(_counts)
    if n & (n - 1) == 0:
        print(f"[staging-instrument pid={os.getpid()}] {snapshot}", file=sys.stderr, flush=True)


def _patch_nixl(module):
    cls = module.NixlKVManager
    original = cls._do_staging_transfer

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        handle, deferred = result[0], result[1]
        if deferred:
            _bump("nixl_deferred")
        elif handle is not None:
            _bump("nixl_staged")
        else:
            _bump("nixl_fallback")
        return result

    cls._do_staging_transfer = wrapped


def _patch_mooncake(module):
    cls = module.MooncakeKVManager
    original = cls._do_staging_transfer

    def wrapped(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        ret, deferred = result[0], result[1]
        if deferred:
            _bump("mc_deferred")
        elif ret == -1:
            _bump("mc_fallback")
        else:
            _bump("mc_staged")
        return result

    cls._do_staging_transfer = wrapped


_TARGETS = {
    "sglang.srt.disaggregation.nixl.conn": _patch_nixl,
    "sglang.srt.disaggregation.mooncake.conn": _patch_mooncake,
}


class _Loader(importlib.abc.Loader):
    def __init__(self, inner, patch):
        self._inner = inner
        self._patch = patch

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        self._patch(module)
        print(f"[staging-instrument pid={os.getpid()}] patched {module.__name__}", file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        patch = _TARGETS.get(fullname)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader, patch)
        return spec


sys.meta_path.insert(0, _Finder())

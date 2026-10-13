"""NIXL staging shim plus optional timing probes, applied identically to base and PR.

The shim drops the slot_layer_ids kwarg that NixlKVManager.set_kv_buffer_tensors does
not accept at these commits (identity for models without a draft pool). With
P4DIAG_DIR set, each scheduler process appends timing events to <dir>/<host>-<pid>.jsonl.
"""

import importlib.abc
import importlib.machinery
import json
import os
import socket
import sys
import threading
import time

_OUT_DIR = os.environ.get("P4DIAG_DIR")
_lock = threading.Lock()
_events = []
_deferred = {}
_agg = {}
_flusher_started = False


def _emit(kind, **fields):
    fields["k"] = kind
    fields.setdefault("t", time.time())
    fields["th"] = threading.current_thread().name
    with _lock:
        _events.append(fields)


def _add_agg(key, dur):
    with _lock:
        a = _agg.setdefault(key, [0, 0.0])
        a[0] += 1
        a[1] += dur


def _flush():
    with _lock:
        events = list(_events)
        _events.clear()
        agg = {k: list(v) for k, v in _agg.items()}
    if not events and not agg:
        return
    with open(os.path.join(_OUT_DIR, f"{socket.gethostname()}-{os.getpid()}.jsonl"), "a") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
        f.write(json.dumps({"k": "agg", "t": time.time(), "agg": agg}) + "\n")


def _start_flusher():
    global _flusher_started
    if _flusher_started:
        return
    _flusher_started = True

    def loop():
        while True:
            time.sleep(2)
            try:
                _flush()
            except Exception as e:
                print(f"[p4diag] flush failed: {e}", file=sys.stderr, flush=True)

    threading.Thread(target=loop, name="p4diag-flush", daemon=True).start()


def _patch_nixl(module):
    cls = module.NixlKVManager
    orig_set = cls.set_kv_buffer_tensors

    def set_kv_buffer_tensors(self, k_buffers, v_buffers, page_size, slot_layer_ids=None):
        return orig_set(self, k_buffers, v_buffers, page_size)

    cls.set_kv_buffer_tensors = set_kv_buffer_tensors
    print(f"[p4diag pid={os.getpid()}] shim applied", file=sys.stderr, flush=True)

    if os.environ.get("P4_DECODE_PROGRESS") == "1":
        orig_init = cls.__init__

        def __init__(self, *args, **kwargs):
            orig_init(self, *args, **kwargs)
            if self.disaggregation_mode != module.DisaggregationMode.DECODE:
                return

            def poll_notifs():
                while True:
                    self.agent.get_new_notifs()
                    time.sleep(0)

            threading.Thread(target=poll_notifs, name="p4-decode-progress", daemon=True).start()
            print(f"[p4diag pid={os.getpid()}] decode progress thread started", file=sys.stderr, flush=True)

        cls.__init__ = __init__

    if _OUT_DIR is None:
        return
    _start_flusher()
    success = module.KVPoll.Success

    orig_stg = cls._do_staging_transfer

    def _do_staging_transfer(self, *args, **kwargs):
        kv_chunk = args[1]
        t0 = time.time()
        res = orig_stg(self, *args, **kwargs)
        t1 = time.time()
        key = (kv_chunk.room, kv_chunk.index_slice.start)
        if res[1]:
            with _lock:
                d = _deferred.get(key)
                if d is None:
                    _deferred[key] = [t0, 1, t1 - t0]
                else:
                    d[1] += 1
                    d[2] += t1 - t0
        else:
            with _lock:
                d = _deferred.pop(key, None)
            _emit(
                "stg_post",
                t=t0,
                end=t1,
                room=kv_chunk.room,
                ps=kv_chunk.index_slice.start,
                np=len(kv_chunk.prefill_kv_indices),
                last=bool(kv_chunk.is_last_chunk),
                first_defer=d[0] if d else None,
                ndefer=d[1] if d else 0,
                defer_s=d[2] if d else 0.0,
            )
        return res

    cls._do_staging_transfer = _do_staging_transfer

    orig_await = cls._await_handles

    def _await_handles(self, handles, *args, **kwargs):
        if not handles:
            return orig_await(self, handles, *args, **kwargs)
        t0, c0 = time.time(), time.thread_time()
        res = orig_await(self, handles, *args, **kwargs)
        _emit("await", t=t0, end=time.time(), n=len(handles), cpu=time.thread_time() - c0)
        return res

    cls._await_handles = _await_handles

    orig_add = cls.add_transfer_request

    def add_transfer_request(self, *args, **kwargs):
        _emit("enqueue", room=args[0], last=bool(args[3]))
        return orig_add(self, *args, **kwargs)

    cls.add_transfer_request = add_transfer_request

    orig_update = cls.update_status

    def update_status(self, bootstrap_room, status):
        if status == success:
            _emit("status_success", room=bootstrap_room)
        return orig_update(self, bootstrap_room, status)

    cls.update_status = update_status

    if "send_chunk_ready" in dir(cls):
        orig_ready = cls.send_chunk_ready

        def send_chunk_ready(self, **kwargs):
            t0 = time.time()
            res = orig_ready(self, **kwargs)
            _emit(
                "chunk_ready_sent",
                t=t0,
                end=time.time(),
                room=kwargs["room"],
                ps=kwargs["page_start"],
            )
            return res

        cls.send_chunk_ready = send_chunk_ready

    if "update_transfer_status" in dir(cls):
        orig_uts = cls.update_transfer_status

        def update_transfer_status(self):
            t0 = time.time()
            res = orig_uts(self)
            _add_agg("decode_update_transfer_status", time.time() - t0)
            return res

        cls.update_transfer_status = update_transfer_status

    orig_wm = module.handle_watermark_msg

    def handle_watermark_msg(staging_ctx, msg_parts):
        _emit("wm_recv", round=int(msg_parts[1]), tail=int(msg_parts[2]))
        return orig_wm(staging_ctx, msg_parts)

    module.handle_watermark_msg = handle_watermark_msg

    import nixl._api as nixl_api

    agent_cls = nixl_api.nixl_agent
    for name in ("initialize_xfer", "transfer"):
        _wrap_timed(agent_cls, name, f"nixl_{name}")


def _wrap_timed(owner, name, kind):
    orig = getattr(owner, name)

    def timed(*args, **kwargs):
        if "transfer_worker" not in threading.current_thread().name:
            return orig(*args, **kwargs)
        t0, c0 = time.time(), time.thread_time()
        res = orig(*args, **kwargs)
        _emit(kind, t=t0, end=time.time(), cpu=time.thread_time() - c0)
        return res

    setattr(owner, name, timed)


def _patch_staging_buffer(module):
    if _OUT_DIR is None:
        return
    _wrap_timed(module, "gather_all_layers_to_staging", "gather")


def _patch_staging_handler(module):
    if _OUT_DIR is None:
        return
    _start_flusher()
    cls = module.DecodeStagingHandler

    orig_arrived = cls.handle_chunk_arrived

    def handle_chunk_arrived(self, room, chunk_idx, page_start, num_pages, writer_id):
        t0 = time.time()
        res = orig_arrived(self, room, chunk_idx, page_start, num_pages, writer_id)
        _emit(
            "arrived",
            t=t0,
            end=time.time(),
            room=room,
            ci=chunk_idx,
            ps=page_start,
            writer=writer_id,
            submitted=bool(res),
        )
        return res

    cls.handle_chunk_arrived = handle_chunk_arrived

    orig_submit = cls.submit_chunk_scatter

    def submit_chunk_scatter(self, room, chunk_idx, page_start, num_pages):
        receiver = self._room_to_receiver.get(room)
        infos = receiver.chunk_staging_infos if receiver is not None else []
        alloc_id = infos[chunk_idx][0] if chunk_idx < len(infos) else None
        t0 = time.time()
        res = orig_submit(self, room, chunk_idx, page_start, num_pages)
        _emit(
            "scatter_submit",
            t=t0,
            end=time.time(),
            room=room,
            ci=chunk_idx,
            ps=page_start,
            alloc=alloc_id,
        )
        return res

    cls.submit_chunk_scatter = submit_chunk_scatter

    orig_free = cls._free_and_send_watermark

    def _free_and_send_watermark(self, alloc_id, decode_req):
        t0 = time.time()
        res = orig_free(self, alloc_id, decode_req)
        _emit(
            "wm_send",
            t=t0,
            end=time.time(),
            alloc=alloc_id,
            room=decode_req.req.bootstrap_room,
        )
        return res

    cls._free_and_send_watermark = _free_and_send_watermark


_TARGETS = {
    "sglang.srt.disaggregation.nixl.conn": _patch_nixl,
    "sglang.srt.disaggregation.common.staging_handler": _patch_staging_handler,
    "sglang.srt.disaggregation.common.staging_buffer": _patch_staging_buffer,
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

"""Decompose staged-chunk timing for P4c32 rooms from p4diag probe files.

usage: p4_analyze.py <run-dir> [<run-dir> ...]   (each run-dir holds the <pid>.jsonl files of one launch)
Rooms are selected by their total staged pages per prefill rank (at least 12000: the P4c32 rooms).
"""

import bisect
import collections
import json
import statistics
import sys
from pathlib import Path

ROOM_MIN_PAGES = 12000


def load(run_dir):
    prefill, decode = [], None
    for f in sorted(Path(run_dir).glob("*.jsonl")):
        events = [json.loads(line) for line in f.open()]
        kinds = {e["k"] for e in events}
        if "stg_post" in kinds:
            prefill.append(events)
        elif "arrived" in kinds:
            decode = events
    return prefill, decode


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else float("nan")


def summarize(name, values_s):
    ms = [v * 1000 for v in values_s if v is not None]
    if not ms:
        return f"{name:44s} n=0"
    return (
        f"{name:44s} n={len(ms):5d} mean={statistics.mean(ms):9.2f} "
        f"p50={pct(ms, 0.5):9.2f} p90={pct(ms, 0.9):9.2f} max={max(ms):9.2f} ms"
    )


def analyze(run_dir):
    prefill, decode = load(run_dir)
    chunks = {}  # (rank, room, ps) -> dict
    room_pages = collections.defaultdict(int)
    room_success = {}
    room_last_enqueue = {}
    for rank, events in enumerate(prefill):
        by_thread = collections.defaultdict(list)
        enq = collections.defaultdict(list)
        for e in events:
            k = e["k"]
            if k == "stg_post":
                chunks[(rank, e["room"], e["ps"])] = dict(e)
                room_pages[(rank, e["room"])] += e["np"]
                by_thread[e["th"]].append(("post", e["end"], (rank, e["room"], e["ps"])))
            elif k == "await":
                by_thread[e["th"]].append(("await", e["t"], e))
            elif k == "chunk_ready_sent":
                c = chunks.get((rank, e["room"], e["ps"]))
                if c is not None:
                    c["ready_sent"] = e["end"]
            elif k == "enqueue":
                enq[e["room"]].append(e["t"])
                if e["last"]:
                    room_last_enqueue[(rank, e["room"])] = e["t"]
            elif k == "status_success":
                room_success.setdefault((rank, e["room"]), e["t"])
        for seq in by_thread.values():
            seq.sort(key=lambda x: x[1])
            for i, (kind, _, ref) in enumerate(seq):
                if kind != "post":
                    continue
                nxt = next((s for s in seq[i + 1 :] if s[0] == "await"), None)
                if nxt is not None:
                    chunks[ref]["write_wait"] = nxt[2]["end"] - nxt[2]["t"]
                    chunks[ref]["write_done"] = nxt[2]["end"]
        for room, times in enq.items():
            mine = sorted(c for c in chunks if c[0] == rank and c[1] == room)
            for key, t in zip(mine, sorted(times)):
                chunks[key]["enqueue"] = t

    target_rooms = {room for (rank, room), pages in room_pages.items() if pages >= ROOM_MIN_PAGES}
    ranks = len(prefill)

    # decode side
    submit = {}
    wm_send_by_alloc = collections.defaultdict(list)
    arrivals = collections.defaultdict(list)
    for e in decode or []:
        if e["k"] == "scatter_submit":
            submit[(e["room"], e["ps"])] = e
        elif e["k"] == "wm_send":
            wm_send_by_alloc[e["alloc"]].append(e["t"])
        elif e["k"] == "arrived":
            arrivals[(e["room"], e["ps"])].append(e)
    for v in wm_send_by_alloc.values():
        v.sort()
    wm_recv = [sorted(e["t"] for e in events if e["k"] == "wm_recv") for events in prefill]

    rows = collections.defaultdict(list)
    for (rank, room, ps), c in chunks.items():
        if room not in target_rooms:
            continue
        first_try = c["first_defer"] if c["first_defer"] is not None else c["t"]
        if "enqueue" in c:
            rows["1 queue (enqueue -> first try)"].append(first_try - c["enqueue"])
        rows["2 ring wait (first defer -> post)"].append(c["t"] - first_try)
        rows["   defer attempts per chunk"].append(c["ndefer"] / 1000.0)
        rows["   post (gather + submit)"].append(c["end"] - c["t"])
        if "write_wait" in c:
            rows["3 write wait (post -> DONE)"].append(c["write_wait"])
        if "ready_sent" in c and "write_done" in c:
            rows["   DONE -> CHUNK_READY sent"].append(c["ready_sent"] - c["write_done"])
        if "write_done" in c:
            busy_end = max(c["write_done"], c.get("ready_sent", 0.0))
            c["busy"] = c["defer_s"] + (busy_end - c["t"])
            rows["S worker time per chunk (defer+post+write+notify)"].append(c["busy"])

    for (room, ps), s in submit.items():
        if room not in target_rooms:
            continue
        dones = [chunks.get((r, room, ps), {}).get("write_done") for r in range(ranks)]
        if all(d is not None for d in dones):
            rows["4 both DONE -> decode scatter submit"].append(s["t"] - max(dones))
            arr = arrivals.get((room, ps), [])
            if arr:
                rows["   both DONE -> last arrival seen"].append(max(a["t"] for a in arr) - max(dones))
        sends = wm_send_by_alloc.get(s["alloc"], [])
        i = bisect.bisect_left(sends, s["end"])
        if i < len(sends):
            freed = sends[i]
            rows["5 scatter submit -> watermark sent"].append(freed - s["end"])
            lat = []
            for recv in wm_recv:
                j = bisect.bisect_left(recv, freed)
                if j < len(recv):
                    lat.append(recv[j] - freed)
            if lat:
                rows["6 watermark sent -> prefill recv"].append(max(lat))

    for (rank, room), t in room_success.items():
        if room in target_rooms and (rank, room) in room_last_enqueue:
            rows["R last enqueue -> worker Success (per rank)"].append(t - room_last_enqueue[(rank, room)])
    per_room_ring = collections.defaultdict(float)
    for (rank, room, ps), c in chunks.items():
        if room in target_rooms and c["first_defer"] is not None:
            per_room_ring[(rank, room)] += c["t"] - c["first_defer"]
    rows["R ring wait summed per room (per rank)"] = list(per_room_ring.values()) + [
        0.0 for key in room_success if key[1] in target_rooms and key not in per_room_ring
    ]

    print(f"\n### {run_dir}  rooms={len(target_rooms)} prefill_ranks={ranks}")
    target_starts = [c.get("enqueue", c["t"]) for (r, room, _), c in chunks.items() if room in target_rooms]
    window_start = min(target_starts) if target_starts else 0.0
    for kind in ("gather", "nixl_initialize_xfer", "nixl_transfer", "await"):
        evs = [e for events in prefill for e in events if e["k"] == kind and e["t"] >= window_start]
        if evs and "cpu" in evs[0]:
            wall = [e["end"] - e["t"] for e in evs]
            cpu = [e["cpu"] for e in evs]
            print(
                f"{'T ' + kind:44s} n={len(evs):5d} wall mean={statistics.mean(wall) * 1000:9.2f} "
                f"p50={pct(wall, 0.5) * 1000:9.2f}  cpu mean={statistics.mean(cpu) * 1000:9.2f} ms"
            )
    for rank in range(ranks):
        mine = [c for (r, room, _), c in chunks.items() if r == rank and room in target_rooms]
        if not mine:
            continue
        start = min(c.get("enqueue", c["t"]) for c in mine)
        end = max(c.get("write_done", c["end"]) for c in mine)
        busy = sum(c.get("busy", 0.0) for c in mine)
        threads = len({c["th"] for c in mine})
        print(
            f"rank {rank}: P4c32 window {end - start:7.2f}s, transfer threads {threads}, "
            f"worker busy {busy:7.2f}s = {busy / max(end - start, 1e-9) / threads:5.1%} utilization"
        )
    for name in sorted(rows):
        values = rows[name]
        if name.startswith("   defer attempts"):
            print(f"{name:44s} n={len(values):5d} mean={statistics.mean(values) * 1000:9.1f} attempts")
        else:
            print(summarize(name, values))


for d in sys.argv[1:]:
    analyze(d)

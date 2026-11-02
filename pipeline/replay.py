"""Replay a recorded feed of events against the app, as if the news were arriving live.

Events whose timestamps fall within `window` seconds of the first event of a burst are sent as ONE request, so the app recomputes every value
once per burst instead of once per event. `speed` compresses the pauses between bursts (60 = one minute of recorded time per second); 0 sends
everything immediately. Replaying the same file twice is safe: events are idempotent, so the second run reports duplicates."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from .events import normalize_ts


def load_events(path) -> list[dict]:
    """One JSON object per line. Blank lines and lines starting with # are skipped. Errors name the line."""
    events = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"line {n}: not valid JSON ({e.msg}).")
        if not isinstance(obj, dict):
            raise ValueError(f"line {n}: each line must be a JSON object.")
        events.append(obj)
    return events


def _when(ev: dict) -> datetime:
    return datetime.strptime(normalize_ts(ev.get("ts")), "%Y-%m-%dT%H:%M:%S.%fZ")


def bursts(events: list[dict], window: float = 1.0) -> list[list[dict]]:
    """Sort by time and group: a new burst starts when an event is more than `window` seconds after the burst's first event."""
    ordered = sorted(events, key=_when)
    out, start = [], None
    for ev in ordered:
        t = _when(ev)
        if start is None or (t - start).total_seconds() > window:
            out.append([])
            start = t
        out[-1].append(ev)
    return out


def replay(events: list[dict], post, window: float = 1.0, speed: float = 0.0, sleep=time.sleep, on_burst=None) -> dict:
    """post(list_of_events) -> the server's response dict ({"accepted": [...], "duplicates": [...], "rejected": [...], "recomputes": n})."""
    groups = bursts(events, window)
    total = {"bursts": len(groups), "events": len(events), "accepted": 0, "duplicates": 0, "rejected": 0, "recomputes": 0}
    prev = None
    for i, group in enumerate(groups):
        t0 = _when(group[0])
        if speed > 0 and prev is not None:
            sleep(max((t0 - prev).total_seconds(), 0) / speed)
        prev = t0
        resp = post(group)
        total["accepted"] += len(resp.get("accepted", []))
        total["duplicates"] += len(resp.get("duplicates", []))
        total["rejected"] += len(resp.get("rejected", []))
        total["recomputes"] += int(resp.get("recomputes", 0))
        if on_burst:
            on_burst(i + 1, group, resp)
    return total

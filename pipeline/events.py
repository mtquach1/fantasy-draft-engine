"""Events: small, validated, idempotent facts about players (injury news, minutes changes, trades) and the pure function that folds a log
of them into the adjustments the app applies.

Ordering rule: events are applied in (timestamp, arrival order). A late-arriving event with an OLDER timestamp therefore cannot overwrite
newer information, and replaying the same log always gives the same result."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import NamedTuple

TYPES = ("games_out", "games", "mpg", "team", "clear")
MAX_BATCH = 500
_ID = re.compile(r"^[A-Za-z0-9._:\-]{1,80}$")


@dataclass(frozen=True)
class Event:
    event_id: str
    type: str
    player: str                  # as sent: a name, a player_id, or "br:<basketball-reference id>"
    value: object                # int games_out | int games | float mpg | str team | None
    ts: str                      # normalized UTC ISO-8601, sorts chronologically as plain text
    source: str = ""
    note: str = ""


class StoredEvent(NamedTuple):
    seq: int
    event_id: str
    ts: str
    type: str
    player_id: str
    value: object


def normalize_ts(ts, now: datetime | None = None) -> str:
    """Any ISO-8601 timestamp (with or without 'Z' or an offset; naive means UTC) -> 'YYYY-MM-DDTHH:MM:SS.ffffffZ'."""
    if ts is None or ts == "":
        dt = now or datetime.now(timezone.utc)
    elif isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
        except ValueError:
            raise ValueError(f"ts '{ts}' is not an ISO-8601 timestamp (for example 2026-10-28T14:30:00Z).")
    else:
        raise ValueError("ts must be a text timestamp such as 2026-10-28T14:30:00Z.")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _number(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{what} must be a number, got {v!r}.")
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError(f"{what} must be a finite number.")
    return float(v)


def parse_event(raw, now: datetime | None = None) -> Event:
    """Validate one incoming event. Raises ValueError with a message a person can act on."""
    if not isinstance(raw, dict):
        raise ValueError("An event must be an object like {\"event_id\": ..., \"type\": ..., \"player\": ..., \"value\": ...}.")
    eid = raw.get("event_id")
    if not isinstance(eid, str) or not _ID.match(eid):
        raise ValueError("event_id is required: 1 to 80 letters, digits or . _ : - (it makes retries safe).")
    typ = raw.get("type")
    if typ not in TYPES:
        raise ValueError(f"type must be one of {', '.join(TYPES)}; got {typ!r}.")
    player = raw.get("player")
    if not isinstance(player, str) or not player.strip() or len(player) > 80:
        raise ValueError("player is required (a name, a player_id, or br:<id>), at most 80 characters.")
    value = raw.get("value")
    if typ == "games_out":
        v = _number(value, "value (games out)")
        if not v.is_integer() or not 0 <= v <= 82:
            raise ValueError("For games_out, value must be a whole number of games from 0 to 82.")
        value = int(v)
    elif typ == "games":
        v = _number(value, "value (games)")
        if not v.is_integer() or not 1 <= v <= 82:
            raise ValueError("For games, value must be a whole number from 1 to 82.")
        value = int(v)
    elif typ == "mpg":
        v = _number(value, "value (minutes per game)")
        if not 1 <= v <= 48:
            raise ValueError("For mpg, value must be between 1 and 48.")
        value = round(v, 2)
    elif typ == "team":
        if not isinstance(value, str) or not 1 <= len(value.strip()) <= 30:
            raise ValueError("For team, value must be the new team's name or abbreviation (1 to 30 characters).")
        value = value.strip()
    else:
        value = None
    source, note = raw.get("source", ""), raw.get("note", "")
    if not isinstance(source, str) or len(source) > 40:
        raise ValueError("source must be text of at most 40 characters.")
    if not isinstance(note, str) or len(note) > 200:
        raise ValueError("note must be text of at most 200 characters.")
    return Event(eid, typ, player.strip(), value, normalize_ts(raw.get("ts"), now), source.strip(), note.strip())


def fold(stored: list[StoredEvent]) -> dict:
    """Pure function: events -> {player_id: {"games": {"kind": "out"|"abs", "n": int}, "mpg": float, "team": str}}.
    Last write wins per field, in (ts, seq) order. games_out and games are one field: whichever came last decides.
    A 'clear' removes everything known about that player up to that point."""
    out: dict = {}
    for e in sorted(stored, key=lambda s: (s.ts, s.seq)):
        slot = out.setdefault(e.player_id, {})
        if e.type == "games_out":
            slot["games"] = {"kind": "out", "n": int(e.value)}
        elif e.type == "games":
            slot["games"] = {"kind": "abs", "n": int(e.value)}
        elif e.type == "mpg":
            slot["mpg"] = float(e.value)
        elif e.type == "team":
            slot["team"] = str(e.value)
        elif e.type == "clear":
            out[e.player_id] = {}
    return {pid: v for pid, v in out.items() if v}

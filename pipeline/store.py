"""SQLite store: events, versioned data snapshots, effective-dated team history and a player-id crosswalk.

One connection per Store, guarded by a lock, so many threads can use it safely. ':memory:' gives a private store (hosted sessions);
a file path gives a durable one (single-user mode and the command-line tool)."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from .events import Event, StoredEvent

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT UNIQUE NOT NULL,
    ts TEXT NOT NULL, type TEXT NOT NULL,
    player_id TEXT NOT NULL, player_name TEXT,
    value TEXT, source TEXT, note TEXT, received_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL, source TEXT NOT NULL, created_at TEXT NOT NULL,
    rows INTEGER NOT NULL, sha256 TEXT NOT NULL, quality TEXT NOT NULL, csv TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS current_snapshot (kind TEXT PRIMARY KEY, snapshot_id INTEGER NOT NULL REFERENCES snapshots(id));
CREATE TABLE IF NOT EXISTS team_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id TEXT NOT NULL, team TEXT NOT NULL, effective_from TEXT NOT NULL,
    source TEXT NOT NULL, event_id TEXT,
    UNIQUE (player_id, effective_from, source));
CREATE INDEX IF NOT EXISTS team_history_player ON team_history (player_id, effective_from);
CREATE TABLE IF NOT EXISTS crosswalk (source TEXT NOT NULL, source_id TEXT NOT NULL, player_key TEXT NOT NULL, PRIMARY KEY (source, source_id));
"""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class Store:
    def __init__(self, path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout = 10000")
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")        # readers (the app) and a writer (the CLI) don't block each other
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    # ------------------------------------------------------------------------------------------------------------- events
    def add_event(self, ev: Event, player_id: str, player_name: str = "") -> bool:
        """Store one event. False when the event_id was already stored (a retry): nothing changes, nothing is applied twice."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO events (event_id, ts, type, player_id, player_name, value, source, note, received_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (ev.event_id, ev.ts, ev.type, player_id, player_name, json.dumps(ev.value), ev.source, ev.note, _now()))
            return cur.rowcount == 1

    def events(self) -> list[StoredEvent]:
        with self._lock:
            rows = self._conn.execute("SELECT seq, event_id, ts, type, player_id, value FROM events ORDER BY ts, seq").fetchall()
        return [StoredEvent(r["seq"], r["event_id"], r["ts"], r["type"], r["player_id"], json.loads(r["value"])) for r in rows]

    def recent_events(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM events ORDER BY seq DESC LIMIT ?", (int(limit),)).fetchall()
        return [{**dict(r), "value": json.loads(r["value"])} for r in rows]

    def event_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]

    def clear_events(self) -> int:
        with self._lock, self._conn:
            n = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
            self._conn.execute("DELETE FROM events")
            self._conn.execute("DELETE FROM team_history WHERE event_id IS NOT NULL")      # facts that came from events go with them
            return n

    # ------------------------------------------------------------------------------------------------------------- team history
    def add_team_fact(self, player_id: str, team: str, effective_from: str, source: str, event_id: str | None = None) -> bool:
        with self._lock, self._conn:
            cur = self._conn.execute("INSERT OR IGNORE INTO team_history (player_id, team, effective_from, source, event_id) VALUES (?,?,?,?,?)",
                                     (player_id, team, effective_from, source, event_id))
            return cur.rowcount == 1

    def team_on(self, player_id: str, when: str) -> str | None:
        """The team a player belonged to at a given moment: the latest fact that took effect on or before it."""
        with self._lock:
            r = self._conn.execute("SELECT team FROM team_history WHERE player_id = ? AND effective_from <= ? "
                                   "ORDER BY effective_from DESC, id DESC LIMIT 1", (player_id, when)).fetchone()
        return r["team"] if r else None

    def team_history(self, player_id: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute("SELECT team, effective_from, source, event_id FROM team_history WHERE player_id = ? "
                                      "ORDER BY effective_from, id", (player_id,)).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------------------------------------------------- snapshots
    def add_snapshot(self, kind: str, source: str, csv_text: str, rows: int, quality: dict) -> tuple[int, bool]:
        """Store an immutable copy of a table. Returns (snapshot id, created). Identical content to the current snapshot is not stored again."""
        sha = hashlib.sha256(csv_text.encode("utf-8")).hexdigest()
        with self._lock, self._conn:
            cur = self.current_snapshot(kind)
            if cur and cur["sha256"] == sha:
                return cur["id"], False
            c = self._conn.execute("INSERT INTO snapshots (kind, source, created_at, rows, sha256, quality, csv) VALUES (?,?,?,?,?,?,?)",
                                   (kind, source, _now(), rows, sha, json.dumps(quality), csv_text))
            self._conn.execute("INSERT INTO current_snapshot (kind, snapshot_id) VALUES (?, ?) "
                               "ON CONFLICT(kind) DO UPDATE SET snapshot_id = excluded.snapshot_id", (kind, c.lastrowid))
            return c.lastrowid, True

    def current_snapshot(self, kind: str) -> dict | None:
        with self._lock:
            r = self._conn.execute("SELECT s.id, s.kind, s.source, s.created_at, s.rows, s.sha256, s.quality FROM snapshots s "
                                   "JOIN current_snapshot c ON c.snapshot_id = s.id WHERE c.kind = ?", (kind,)).fetchone()
        return {**dict(r), "quality": json.loads(r["quality"])} if r else None

    def snapshots(self, kind: str | None = None) -> list[dict]:
        q, args = "SELECT id, kind, source, created_at, rows, sha256 FROM snapshots", ()
        if kind:
            q, args = q + " WHERE kind = ?", (kind,)
        with self._lock:
            rows = self._conn.execute(q + " ORDER BY id", args).fetchall()
            cur = {r["kind"]: r["snapshot_id"] for r in self._conn.execute("SELECT kind, snapshot_id FROM current_snapshot").fetchall()}
        return [{**dict(r), "current": cur.get(r["kind"]) == r["id"]} for r in rows]

    def set_current(self, kind: str, snapshot_id: int) -> None:
        with self._lock, self._conn:
            r = self._conn.execute("SELECT kind FROM snapshots WHERE id = ?", (int(snapshot_id),)).fetchone()
            if r is None:
                raise ValueError(f"There is no snapshot #{snapshot_id}.")
            if r["kind"] != kind:
                raise ValueError(f"Snapshot #{snapshot_id} is a '{r['kind']}' snapshot, not '{kind}'.")
            self._conn.execute("INSERT INTO current_snapshot (kind, snapshot_id) VALUES (?, ?) "
                               "ON CONFLICT(kind) DO UPDATE SET snapshot_id = excluded.snapshot_id", (kind, int(snapshot_id)))

    def snapshot_csv(self, snapshot_id: int) -> str:
        with self._lock:
            r = self._conn.execute("SELECT csv FROM snapshots WHERE id = ?", (int(snapshot_id),)).fetchone()
        if r is None:
            raise ValueError(f"There is no snapshot #{snapshot_id}.")
        return r["csv"]

    # ------------------------------------------------------------------------------------------------------------- crosswalk
    def link_id(self, source: str, source_id: str, player_key: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT INTO crosswalk (source, source_id, player_key) VALUES (?,?,?) "
                               "ON CONFLICT(source, source_id) DO UPDATE SET player_key = excluded.player_key", (source, source_id, player_key))

    def key_for(self, source: str, source_id: str) -> str | None:
        with self._lock:
            r = self._conn.execute("SELECT player_key FROM crosswalk WHERE source = ? AND source_id = ?", (source, source_id)).fetchone()
        return r["player_key"] if r else None

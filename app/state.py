
"""Pick log is the single source of truth. Everything else is derived from it."""
import contextvars
import copy
import itertools
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import pandas as pd

from .config import LeagueConfig


@dataclass
class GlobalState:
    config: LeagueConfig = field(default_factory=LeagueConfig)
    raw_df: pd.DataFrame = field(default_factory=pd.DataFrame)     # base projections with live overrides applied (rebuilt, never edited in place)
    base_raw_df: pd.DataFrame = field(default_factory=pd.DataFrame)  # the projections exactly as loaded from disk
    overrides: dict = field(default_factory=dict)                  # player_id -> {games, mpg, note}: live what-if edits (NBA)
    lock: object = field(default_factory=threading.RLock, repr=False, compare=False)       # held while the draft is changed: one writer at a time
    idem: OrderedDict = field(default_factory=OrderedDict, repr=False, compare=False)      # Idempotency-Key -> the pick it created
    blend: float | None = None                                     # live blend weight toward the reference list (None = not available)
    data_version: int = 0                                          # bumps whenever raw_df changes, so every cache is rebuilt
    draft_log: list[dict] = field(default_factory=list)            # [{pick_number, player_id, team_id, source}]
    my_team_id: int = 1
    source: object = None                                          # DraftSource
    setup: object = None                                           # draft.DraftSetup once the user saves it
    names: dict = field(default_factory=dict)                      # player_id -> name
    events: list = field(default_factory=list)                     # recent picks seen by the live sync
    sync: dict = field(default_factory=lambda: {"platform": "manual", "connected": False, "last_ok": None,
                                                "last_error": None, "unmapped": [],
                                                "message": "Manual mode: enter picks yourself"})
    adp: dict = field(default_factory=dict)                        # player_id -> average draft position (optional adp.csv)
    adp_note: str | None = None                                    # what happened when adp.csv was read
    cleaning: list = field(default_factory=list)                   # what clean_projections changed at startup
    history: dict = field(default_factory=dict)                    # {'H': df, 'P': df} past-season stats
    platform_map: dict = field(default_factory=dict)               # platform player_id -> internal player_id
    name_map: dict = field(default_factory=dict)                   # clean player name -> player_id (unambiguous names only)
    platform_names: dict = field(default_factory=dict)             # platform player_id -> name, looked up for live picks
    saved: dict = field(default_factory=lambda: {"path": None, "saved_at": None, "restored": False,
                                                 "picks_restored": 0, "warning": None, "error": None})   # disk-save status

    def drafted_ids(self) -> set:
        return {p["player_id"] for p in self.draft_log}

    def team_player_ids(self, team_id: int) -> list:
        return [p["player_id"] for p in self.draft_log if p["team_id"] == team_id]

    def my_open_il_slots(self) -> int:
        """IL slots still free on YOUR roster: the league's IL slots minus injured players you already hold.
        (It decides how much an injured player is discounted when you consider drafting him.)"""
        total = max(int(self.config.roster.il_slots or 0), 0)
        if total == 0 or self.raw_df.empty or "is_injured" not in self.raw_df.columns:
            return total
        mine = set(self.team_player_ids(self.my_team_id))
        if not mine:
            return total
        held = self.raw_df[self.raw_df["player_id"].isin(mine)]
        return max(total - int(held["is_injured"].fillna(False).astype(bool).sum()), 0)


# ---------------------------------------------------------------------------------------------------------------------
# Sessions. `state` is a proxy: it forwards every read and write to the draft that belongs to the current request.
#  - Single-user mode (the default, and how you run it on your own computer): one draft, saved to disk, exactly as before.
#  - Multi-session mode (MULTI_SESSION=1, for a hosted copy): each browser gets its own private draft, kept in memory only,
#    dropped after it sits idle, and never written to disk.
# Code outside this file keeps writing `state.config`, `state.draft_log` and so on, and needs no change.
# ---------------------------------------------------------------------------------------------------------------------
_default = GlobalState()                     # the draft in single-user mode, and the template new sessions are cloned from
_current: contextvars.ContextVar = contextvars.ContextVar("draft_session", default=None)
_versions = itertools.count(1)


def next_version() -> int:
    """A data version no other session can ever have, so cached valuations are never shared by mistake."""
    return next(_versions)


class StateProxy:
    def __getattr__(self, name):
        return getattr(_current.get() or _default, name)

    def __setattr__(self, name, value):
        setattr(_current.get() or _default, name, value)


state = StateProxy()


def _clone_for_session() -> GlobalState:
    d = _default
    s = GlobalState(config=copy.deepcopy(d.config), raw_df=d.raw_df, base_raw_df=d.base_raw_df, my_team_id=d.my_team_id)
    s.names, s.name_map, s.adp, s.adp_note, s.cleaning, s.history = d.names, d.name_map, d.adp, d.adp_note, d.cleaning, d.history
    s.sync.update(platform="manual", connected=False, message="Manual mode: enter picks yourself")   # live sync is per-person credentials: off when hosted
    s.blend = d.blend
    s.data_version = next_version()
    return s


class Sessions:
    def __init__(self):
        self.enabled, self.max_sessions, self.ttl = False, 100, 3 * 3600.0
        self._items: OrderedDict = OrderedDict()           # id -> [GlobalState, last_seen]
        self._lock = threading.Lock()

    def configure(self, enabled: bool, max_sessions: int = 100, ttl_hours: float = 3.0) -> None:
        self.enabled, self.max_sessions, self.ttl = bool(enabled), max(int(max_sessions), 1), max(float(ttl_hours), 0.01) * 3600.0
        with self._lock:
            self._items.clear()

    def _prune(self, now: float) -> None:
        for sid in [k for k, (_, seen) in self._items.items() if now - seen > self.ttl]:
            del self._items[sid]
        while len(self._items) > self.max_sessions:
            self._items.popitem(last=False)                # the least recently used goes first

    def get_or_create(self, sid: str | None):
        """Returns (session id, state, created). An unknown, expired or tampered id simply gets a fresh session."""
        now = time.time()
        with self._lock:
            self._prune(now)
            if sid and sid in self._items:
                self._items[sid][1] = now
                self._items.move_to_end(sid)
                return sid, self._items[sid][0], False
            new_id = secrets.token_urlsafe(24)
            self._items[new_id] = [_clone_for_session(), now]
            self._prune(now)
            return new_id, self._items[new_id][0], True

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


sessions = Sessions()

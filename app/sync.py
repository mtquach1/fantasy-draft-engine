
"""Keeps the pick log in step with the real draft (Yahoo, ESPN, or the built-in mock).
Each poll re-reads the FULL pick list, so commissioner undos/corrections heal themselves.

Matching a platform pick to your projections, in order: your id crosswalk (data/id_crosswalk.csv), then the
player's name (only when the name belongs to exactly one player in your data). Anything else keeps its pick
slot with player_id None and is listed as unmatched."""
from __future__ import annotations

import asyncio
import contextlib
import time

from .connectors.base import SourceError
from .ingestion import clean_name

MAX_BACKOFF = 30.0
NAME_LOOKUP_BATCH = 25


def _resolve(rp, platform_map: dict, name_map: dict | None):
    """-> (internal player_id or None, 'id' | 'name' | None)"""
    pid = platform_map.get(str(rp.platform_player_id))
    if pid is not None:
        return pid, "id"
    if name_map and getattr(rp, "player_name", ""):
        pid = name_map.get(clean_name(rp.player_name))
        if pid is not None:
            return pid, "name"
    return None, None


def build_log(remote, platform_map: dict, team_index, name_map: dict | None = None) -> tuple[list[dict], list[str]]:
    """Platform picks -> our pick log. A pick we can't match keeps its place (player_id None) so pick numbers,
    the on-the-clock math and 'picks until my turn' stay correct; the platform id is reported in `unmapped`."""
    log, unmapped = [], []
    for rp in remote:
        pid, _ = _resolve(rp, platform_map, name_map)
        if pid is None:
            unmapped.append(str(rp.platform_player_id))
        entry = {"pick_number": rp.pick_number, "player_id": pid,
                 "team_id": team_index(rp.platform_team_id), "source": "live"}
        if getattr(rp, "cost", None) is not None:
            entry["price"] = int(rp.cost)                         # auction price, when the platform reports it
        log.append(entry)
    return log, unmapped


def diff_events(old: list[dict], new: list[dict], my_team_id, names: dict) -> list[dict]:
    have = {p["pick_number"]: p["player_id"] for p in old}
    events = []
    if len(new) < len(old):
        events.append({"type": "undo", "time": time.time(),
                       "text": f"The pick count dropped from {len(old)} to {len(new)}: a pick was undone or corrected"})
    for p in new:
        if p["pick_number"] not in have or have[p["pick_number"]] != p["player_id"]:
            events.append({"type": "pick", "time": time.time(), "pick_number": p["pick_number"],
                           "team_id": p["team_id"], "player_id": p["player_id"],
                           "player": names.get(p["player_id"], "Unmatched player"),
                           "mine": my_team_id is not None and p["team_id"] == my_team_id})
    return events


async def _fill_names(state, source, remote) -> None:
    """Give name-less picks a name (asking the platform once per id) so the name fallback can work."""
    cache = state.platform_names
    need = [str(p.platform_player_id) for p in remote
            if not p.player_name and str(p.platform_player_id) not in state.platform_map
            and str(p.platform_player_id) not in cache]
    if need and getattr(state, "name_map", None):
        found = await asyncio.to_thread(source.resolve_names, need[:NAME_LOOKUP_BATCH])
        for i in need[:NAME_LOOKUP_BATCH]:
            cache[i] = found.get(i, "")                           # "" = asked, no answer: don't ask again
    for p in remote:
        if not p.player_name:
            p.player_name = cache.get(str(p.platform_player_id), "")


async def run(state, make_source, make_map, interval: float = 3.0, retry_delay: float = 5.0, on_change=None) -> None:
    source, fails = None, 0
    while True:
        delay = None
        try:
            if source is None:                                   # (re)connect; blocking calls go to a thread
                source = await asyncio.to_thread(make_source)
                state.platform_map = await asyncio.to_thread(make_map)
                state.source = source
                mine = await asyncio.to_thread(source.my_team_id)
                if mine:
                    state.my_team_id = mine
            remote = await source.fetch_picks()
            await _fill_names(state, source, remote)
            name_map = getattr(state, "name_map", None)
            unmapped = sorted({str(p.platform_player_id) for p in remote
                               if _resolve(p, state.platform_map, name_map)[0] is None})
            by_name = sum(1 for p in remote if _resolve(p, state.platform_map, name_map)[1] == "name")
            if remote or not state.draft_log:                    # an empty reply while we hold picks is treated as a glitch
                new_log, _ = build_log(remote, state.platform_map, source.team_index, name_map)
                with (getattr(state, "lock", None) or contextlib.nullcontext()):    # same lock as manual picks: one writer at a time
                    events = diff_events(state.draft_log, new_log, state.my_team_id, state.names)
                    if not state.draft_log:
                        events = events[-5:]                     # first sync: don't flood the feed
                    changed = new_log != state.draft_log
                    state.draft_log[:] = new_log
                    state.events.extend(events)
                    del state.events[:-40]
                if changed and on_change is not None:
                    try:
                        await asyncio.to_thread(on_change)       # e.g. save to disk; never lets a failure stop the sync
                    except Exception:
                        pass
            msg = f"Connected: {len(state.draft_log)} picks synced"
            if by_name:
                msg += f" ({by_name} matched by player name)"
            if unmapped:
                msg += f". {len(unmapped)} pick(s) couldn't be matched to your projections"
            unmatched = [{"id": str(p.platform_player_id), "name": p.player_name} for p in remote
                         if _resolve(p, state.platform_map, name_map)[0] is None][:20]
            state.sync.update(connected=True, last_ok=time.time(), last_error=None, unmapped=unmapped,
                              unmatched=unmatched, name_matched=by_name, message=msg)
            fails = 0
        except Exception as e:
            fails += 1
            text = str(e) if isinstance(e, SourceError) else f"{type(e).__name__}: {e}"
            state.sync.update(connected=False, last_error=text, message="Not connected")
            # back off while the platform is failing (it may be rate-limiting us), up to MAX_BACKOFF
            delay = min(max(interval, retry_delay) * (1 + fails), MAX_BACKOFF)
            if fails >= 3:
                source, fails = None, 0                          # rebuild the connection from scratch
        wait = delay if delay is not None else (interval if source else max(interval, retry_delay))
        waited = 0.0
        while waited < wait and not getattr(state, "sync_wake", False):   # saving the draft setup wakes us early
            step = min(0.25, wait - waited)
            await asyncio.sleep(step)
            waited += step
        state.sync_wake = False



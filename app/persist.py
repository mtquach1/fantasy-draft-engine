
"""Saves the draft (league settings, draft setup, pick log) to disk so a crash or restart doesn't wipe it.

One file per sport: data/draft_state_<sport>.json. Writes are atomic (temp file, then rename), so a crash in the
middle of a save can never leave a half-written file. A damaged or unusable file is renamed to
draft_state_<sport>.bad-<time>.json (never deleted) and the app starts clean with a warning.
Saving never raises into the draft: callers get the error back and the draft carries on in memory.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, fields
from pathlib import Path

from . import draft
from .config import LeagueConfig, RosterConfig, ScoringType
from .validate import check_config

VERSION = 1
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_lock = threading.Lock()
_PICK_KEYS = ("pick_number", "player_id", "team_id", "source")


def path_for(sport: str, directory=None) -> Path:
    return Path(directory or DATA_DIR) / f"draft_state_{sport}.json"


# ---------- league settings <-> plain dict ----------
def config_to_dict(cfg: LeagueConfig) -> dict:
    d = asdict(cfg)
    d["scoring_type"] = ScoringType(cfg.scoring_type).value
    return d


def config_from_dict(d: dict, sport: str) -> LeagueConfig:
    """Raises on anything malformed; the caller falls back to the sport's default league."""
    r = d["roster"]
    roster = RosterConfig(teams=int(r["teams"]), slots={str(k): int(v) for k, v in r["slots"].items()},
                          il_slots=int(r["il_slots"]))
    simple = {f.name for f in fields(LeagueConfig)} - {"sport", "scoring_type", "roster"}
    kw = {k: v for k, v in d.items() if k in simple}
    if "categories" in kw:
        kw["categories"] = {str(k): [str(c) for c in v] for k, v in kw["categories"].items()}
    for k in ("category_weights", "point_weights"):
        if k in kw:
            kw[k] = {str(a): float(b) for a, b in kw[k].items()}
    return LeagueConfig(sport=sport, scoring_type=ScoringType(d["scoring_type"]), roster=roster, **kw)


# ---------- save ----------
def _clean_pick(p: dict) -> dict:
    tid = p.get("team_id")
    pid = p.get("player_id")
    out = {"pick_number": int(p["pick_number"]), "player_id": None if pid is None else str(pid),
           "team_id": None if tid is None else int(tid), "source": str(p.get("source", "manual"))}
    if p.get("price") is not None:                    # auction price; absent for snake/linear picks and older files
        out["price"] = int(p["price"])
    return out


def snapshot(state, include_picks: bool = True) -> dict:
    return {"version": VERSION, "sport": state.config.sport,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "setup": asdict(state.setup) if state.setup else None,
            "draft_log": [_clean_pick(p) for p in list(state.draft_log)] if include_picks else [],
            "config": config_to_dict(state.config)}


def _replace(tmp: str, dest: Path) -> None:
    """os.replace, retried briefly: on Windows OneDrive/antivirus can hold the file for a moment."""
    for attempt in range(8):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(0.15)


def save(state, include_picks: bool = True, directory=None) -> Path:
    """Write the current draft atomically. Raises OSError etc. on failure; use try_save() from request code."""
    path = path_for(state.config.sport, directory)
    with _lock:                                   # snapshot inside the lock so the last writer has the newest data
        text = json.dumps(snapshot(state, include_picks), indent=1)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
                f.flush()
                os.fsync(f.fileno())
            _replace(tmp, path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
    return path


def try_save(state, include_picks: bool = True, directory=None) -> str | None:
    """Save and record the outcome in state.saved. Returns an error message, or None. Never raises."""
    try:
        path = save(state, include_picks, directory)
        state.saved.update(path=str(path), saved_at=time.time(), error=None)
        return None
    except Exception as e:                        # the draft must keep working even if the disk doesn't
        msg = f"Couldn't save the draft to disk ({type(e).__name__}: {e}). The draft is still running in memory."
        state.saved["error"] = msg
        return msg


def delete(sport: str, directory=None) -> None:
    try:
        path_for(sport, directory).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


# ---------- restore ----------
def _quarantine(path: Path, move: bool = True) -> Path | None:
    dest = path.with_name(f"{path.stem}.bad-{time.strftime('%Y%m%d-%H%M%S')}{path.suffix}")
    try:
        if move:
            os.replace(path, dest)
        else:
            dest.write_bytes(path.read_bytes())
        return dest
    except OSError:
        return None


def load(sport: str, directory=None) -> tuple[dict | None, str | None]:
    """(data, warning). A file that can't be used is set aside and reported; it is never deleted."""
    path = path_for(sport, directory)
    if not path.exists():
        return None, None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
        if data.get("version") != VERSION:
            raise ValueError(f"unsupported file version {data.get('version')!r}")
        if data.get("sport") != sport:
            raise ValueError(f"file is for sport {data.get('sport')!r}, not {sport!r}")
        if not isinstance(data.get("draft_log", []), list):
            raise ValueError("draft_log is not a list")
    except Exception as e:
        moved = _quarantine(path)
        where = f" It was kept as {moved.name}." if moved else ""
        return None, f"The saved draft couldn't be read ({e}), so you're starting fresh.{where}"
    return data, None


def check_picks(picks, valid_ids: set, teams: int, max_picks: int | None, auction: bool = False) -> str | None:
    """None if the saved pick log is usable, otherwise a short reason."""
    if not isinstance(picks, list):
        return "pick log is not a list"
    seen, last = set(), 0
    for p in picks:
        if not isinstance(p, dict):
            return "a pick is not an object"
        n, pid, tid, src = p.get("pick_number"), p.get("player_id"), p.get("team_id"), p.get("source")
        if isinstance(n, bool) or not isinstance(n, int) or n <= last:
            return "pick numbers are not in increasing order"
        last = n
        live = src == "live"
        price = p.get("price")
        if price is not None and (isinstance(price, bool) or not isinstance(price, int) or price < 1):
            return f"pick {n} has an invalid price"
        if auction and price is None and not live:
            return f"pick {n} has no auction price"
        if pid is None:
            if not live:
                return f"pick {n} has no player"
        else:
            if not isinstance(pid, str) or pid not in valid_ids:
                return f"pick {n} is a player that isn't in your projections (did the data files change?)"
            if pid in seen:
                return f"pick {n}: that player was already drafted"
            seen.add(pid)
        if tid is None:
            if not live:
                return f"pick {n} has no team"
        elif isinstance(tid, bool) or not isinstance(tid, int) or not 1 <= tid <= teams:
            return f"pick {n} is for team {tid}, outside 1..{teams}"
    if max_picks is not None and len(picks) > max_picks:
        return f"{len(picks)} picks is more than this draft allows ({max_picks})"
    return None


def restore(state, restore_picks: bool = True, set_my_team: bool = True, directory=None) -> dict:
    """Load the saved draft into `state` (call right after the sport's data is loaded).
    Returns {restored, picks, setup, config, warning}. Anything unusable is skipped with a warning."""
    sport = state.config.sport
    out = {"restored": False, "picks": 0, "setup": False, "config": False, "warning": None}
    for stale in Path(directory or DATA_DIR).glob(f"draft_state_{sport}.json.*.tmp"):   # leftovers of a crashed save
        try:
            stale.unlink()
        except OSError:
            pass
    data, warning = load(sport, directory)
    warns = [warning] if warning else []
    if data is None:
        out["warning"] = " ".join(warns) or None
        state.saved.update(restored=False, picks_restored=0, warning=out["warning"], path=str(path_for(sport, directory)))
        return out

    cfg = state.config
    if isinstance(data.get("config"), dict):
        try:
            cand = config_from_dict(data["config"], sport)
            problems = check_config(state.raw_df, cand)
            if problems:
                raise ValueError("; ".join(problems[:2]))
            cfg, out["config"] = cand, True
        except Exception as e:
            warns.append(f"Saved league settings weren't used ({type(e).__name__}: {e}); the defaults are in place.")

    setup = None
    if isinstance(data.get("setup"), dict):
        setup, errs = draft.validate_setup(data["setup"], len(cfg.roster.slot_list()))
        if errs:
            warns.append("The saved draft setup wasn't valid (" + "; ".join(errs.values()) + "), so it wasn't restored.")
            setup = None
    teams = setup.teams if setup else cfg.roster.teams

    picks = []
    if restore_picks and data.get("draft_log"):
        reason = check_picks(data["draft_log"], set(state.raw_df["player_id"]), teams,
                             setup.teams * setup.rounds if setup else None,
                             auction=bool(setup and setup.draft_type == "auction"))
        if reason:
            kept = _quarantine(path_for(sport, directory), move=False)
            warns.append(f"The saved picks weren't restored: {reason}."
                         + (f" A copy was kept as {kept.name}." if kept else ""))
        else:
            picks = [_clean_pick(p) for p in data["draft_log"]]

    state.config = cfg
    state.setup = setup
    if setup:
        cfg.roster.teams = setup.teams
        if set_my_team and setup.my_slot:
            state.my_team_id = setup.my_slot
    state.draft_log[:] = picks
    out.update(restored=bool(setup or picks or out["config"]), picks=len(picks), setup=setup is not None,
               warning=" ".join(warns) or None)
    state.saved.update(restored=out["restored"], picks_restored=len(picks), warning=out["warning"],
                       path=str(path_for(sport, directory)))
    return out


def other_saved_drafts(current_sport: str, directory=None) -> list[str]:
    """Other sports that have a saved draft with a setup or picks (to remind you to set SPORT in .env)."""
    found = []
    for p in sorted(Path(directory or DATA_DIR).glob("draft_state_*.json")):
        key = p.stem.removeprefix("draft_state_")
        if key == current_sport or ".bad-" in key:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("setup") or d.get("draft_log"):
                found.append(key)
        except Exception:
            pass
    return found



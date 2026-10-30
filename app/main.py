import dataclasses


import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from pathlib import Path

from fastapi import Body, FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import auction, draft, ingestion, live_blend, lookahead, optimizer, persist, recommend as rec, scoring, sports, stats, sync, validate, valuation, views
from .config import LeagueConfig, ScoringType
from .connectors import PLATFORMS, check_env, get_source
from .state import _current, next_version, sessions, state

load_dotenv()
try:
    POLL_SECONDS = float(os.getenv("POLL_SECONDS", "3"))
except ValueError:
    POLL_SECONDS = 3.0
MIN_LIVE_POLL = 2.0                  # Yahoo/ESPN are other people's servers: don't poll them faster than this
LIVE_PLATFORMS = ("yahoo", "espn", "mock")


_predraft_cache: dict = {}


def _remember(cache: dict, key, value, cap: int = 8) -> None:
    """Keep the few most recent entries (several sessions may be alive), dropping the oldest."""
    cache[key] = value
    while len(cache) > cap:
        cache.pop(next(iter(cache)))


def predraft_full():
    """(board, replacement levels) with nobody drafted. Cached until the league settings change."""
    key = (repr(state.config), len(state.raw_df), state.data_version)   # the pre-draft board is a fixed ruler: all IL slots open
    if key not in _predraft_cache:
        _remember(_predraft_cache, key, valuation.calculate_player_valuations(
            state.raw_df, state.config, set(), state.config.roster.il_slots))
    return _predraft_cache[key]


def predraft_board():
    return predraft_full()[0]


_scorer_cache: dict = {}
_bench = scoring.BenchmarkStore()


def predraft_scorer():
    key = (repr(state.config), len(state.raw_df), state.data_version)
    if key not in _scorer_cache:
        _remember(_scorer_cache, key, scoring.TeamScorer(*predraft_full(), state.config))
    return _scorer_cache[key]


def valuated():
    with state.lock:                                   # a consistent snapshot: never a half-applied pick or edit
        raw, cfg, drafted, il = state.raw_df, state.config, state.drafted_ids(), state.my_open_il_slots()
    df, repl = valuation.calculate_player_valuations(raw, cfg, drafted, il)
    df = valuation.add_ranks(df, predraft_board())
    if state.adp:
        df["adp"] = df["player_id"].map(state.adp)
    return df, repl


def _is_auction() -> bool:
    return state.setup is not None and state.setup.draft_type == "auction"


def _auction_columns(df):
    """Adds `bid` (suggested price now, from the money and players that are left) and `pre_bid` (the price list
    from before the draft) to a valuated frame. Only meaningful in an auction."""
    s = state.setup
    pre = predraft_board()
    free0 = max(s.teams * (s.budget - s.rounds), 0)
    pre_bid = auction.bid_values(pre, free0)
    pre_by_id = dict(zip(pre["player_id"], pre_bid))
    out = df.copy()
    out["pre_bid"] = out["player_id"].map(pre_by_id)
    free_now = auction.free_money(auction.team_budgets(s, state.draft_log)["teams"])
    out["bid"] = auction.bid_values(out, free_now)
    return out


def _auction_payload(df=None) -> dict | None:
    if not _is_auction():
        return None
    if df is None:
        df = _auction_columns(valuated()[0])
    return auction.summary(state.setup, state.draft_log, state.my_team_id, df, df["pre_bid"])


def _save() -> None:
    """Write the draft to disk. Mock drafts restart from pick 1 every run, so their picks aren't kept."""
    if sessions.enabled:
        return                                                       # hosted sessions live in memory only
    persist.try_save(state, include_picks=state.sync.get("platform") != "mock")


def _restore(platform: str) -> None:
    """Load the saved draft after the sport's data is loaded. Can never stop the server from starting."""
    try:
        res = persist.restore(state, restore_picks=platform != "mock", set_my_team=platform != "yahoo")
        if res["restored"]:
            print(f"[draft] Restored your saved draft: {res['picks']} pick(s), "
                  f"{'setup' if res['setup'] else 'no setup'}, league settings {'restored' if res['config'] else 'default'}.")
        for other in persist.other_saved_drafts(state.config.sport):
            print(f"[draft] A saved {other.upper()} draft exists. To resume it, set SPORT={other} in .env and restart.")
        if res["warning"]:
            print(f"[draft] WARNING: {res['warning']}")
    except Exception as e:
        state.saved["warning"] = f"Saved draft couldn't be restored ({type(e).__name__}: {e}); starting fresh."
        print(f"[draft] WARNING: {state.saved['warning']}")


def _make_mock():
    if state.setup is None:
        raise RuntimeError("Save your draft setup first: the mock draft needs your teams and pick")
    pre = predraft_board().sort_values("vorp", ascending=False)
    from .connectors.mock import MockSource
    s, prices = state.setup, None
    if s.draft_type == "auction":
        prices = dict(zip(pre["player_id"], auction.bid_values(pre, max(s.teams * (s.budget - s.rounds), 0))))
    return MockSource(pre["player_id"].tolist(), s.teams, s.rounds, s.draft_type, s.my_slot,
                      float(os.getenv("MOCK_SECONDS_PER_PICK", "4")), budget=s.budget, prices=prices)


# ---------------------------------------------------------------------------------------------------------------------
# Live per-player overrides (basketball): "he is out 20 games", "his minutes dropped". They rebuild raw_df from the pristine
# projections, so every value, rank, recommendation, team score and the pace benchmark is recomputed from the edited numbers.
# ---------------------------------------------------------------------------------------------------------------------
def _override_file() -> Path:
    return persist.DATA_DIR / f"overrides_{state.config.sport}.json"


def _apply_overrides() -> None:
    base = state.base_raw_df
    df = base.copy()
    sp = sports.get_sport(state.config.sport)
    comps = [c for g in sp.groups for c in g.components if c in df.columns and c != "GP"]
    if state.overrides:
        df[comps + ["GP"]] = df[comps + ["GP"]].astype(float)      # whole-number columns can't hold scaled values (pandas 3 refuses)
    for pid, o in state.overrides.items():
        hit = df.index[df["player_id"] == pid]
        if len(hit) == 0:
            continue
        i = hit[0]
        gp0, mpg0 = float(df.at[i, "GP"]), float(df.at[i, "MIN"]) / float(df.at[i, "GP"])
        g = o.get("games") if o.get("games") is not None else gp0
        m = o.get("mpg") if o.get("mpg") is not None else mpg0
        f = (g / gp0) * (m / mpg0)
        for c in comps:
            df.at[i, c] = float(df.at[i, c]) * f
        df.at[i, "GP"] = g
        df.at[i, "MIN"] = m * g
    state.raw_df = df
    state.data_version = next_version()


def _save_overrides() -> None:
    if sessions.enabled:
        return                                                       # hosted sessions live in memory only
    try:
        path = _override_file()
        if not state.overrides:
            path.unlink(missing_ok=True)
            return
        import json, os as _os, tempfile
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        with _os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state.overrides, f, indent=1)
        _os.replace(tmp, path)
    except Exception as e:                                           # never let a failed save break a draft
        print(f"[overrides] couldn't save: {type(e).__name__}: {e}")


def _load_saved_overrides() -> None:
    if sessions.enabled:
        return
    try:
        path = _override_file()
        if state.config.sport != "nba" or not path.exists():
            return
        import json
        saved = json.loads(path.read_text(encoding="utf-8"))
        known = set(state.base_raw_df["player_id"])
        state.overrides = {k: v for k, v in saved.items() if k in known and isinstance(v, dict)}
        if state.overrides:
            _apply_overrides()
            print(f"[overrides] restored {len(state.overrides)} saved override(s).")
    except Exception as e:
        print(f"[overrides] couldn't restore saved overrides ({type(e).__name__}: {e}); starting without them.")


def _find_player(q: str) -> str:
    from .own_model import norm_name
    q = (q or "").strip()
    if not q:
        raise HTTPException(400, {"message": "Say which player (a name or player_id)."})
    if q in state.names:
        return q
    hits = [pid for pid, n in state.names.items() if norm_name(n) == norm_name(q)]
    if not hits:
        hits = [pid for pid, n in state.names.items() if norm_name(q) in norm_name(n)]
    if not hits:
        raise HTTPException(404, {"message": f"No player matches '{q}'."})
    if len(hits) > 1:
        raise HTTPException(400, {"message": f"'{q}' matches {len(hits)} players: " + ", ".join(state.names[h] for h in hits[:5]) + ". Use the full name."})
    return hits[0]


# ---------------------------------------------------------------------------------------------------------------------
# Live blend (basketball): mix the pure own model with a reference list while the app is running. The two sources stay in
# memory; each session has its own weight. Needs data/nba/players_own.csv, players_third_party.csv and blend.json
# (python build_own_model.py writes the first and the last).
# ---------------------------------------------------------------------------------------------------------------------
_blend_src: dict = {}
_blend_reason: str = ""


def _blend_file() -> Path:
    return persist.DATA_DIR / f"blend_{state.config.sport}.json"


def _blend_comps() -> list:
    sp = sports.get_sport(state.config.sport)
    return [c for g in sp.groups for c in g.components if c in state.base_raw_df.columns]


def _init_blend() -> None:
    """Called after a sport loads. Switches the live blend on when its files exist and match the loaded players."""
    global _blend_reason
    _blend_src.clear()
    state.blend = None
    _blend_reason = ""
    if state.config.sport != "nba":
        _blend_reason = "Live blending is available for basketball."
        return
    base = ingestion.data_dir("nba")
    needed = {"blend.json": base / "blend.json", "players_own.csv": base / "players_own.csv",
              "players_third_party.csv": base / "players_third_party.csv"}
    missing = [n for n, p in needed.items() if not p.exists()]
    if missing:
        _blend_reason = (f"Missing in {base}: {', '.join(missing)}. Run python build_own_model.py --blend 0.5 "
                         "(with players_third_party.csv in that folder), then restart the server.")
        print(f"[blend] live blend is off. {_blend_reason}")
        return
    try:
        import json
        w = float(json.loads(needed["blend.json"].read_text(encoding="utf-8"))["weight"])
        if not sessions.enabled and _blend_file().exists():
            w = float(json.loads(_blend_file().read_text(encoding="utf-8"))["weight"])      # the weight you last chose
        platform = state.sync.get("platform")
        own, _ = validate.clean_projections(ingestion.load_projections("nba", platform, {"players.csv": "players_own.csv"}), "nba")
        ref, _ = validate.clean_projections(ingestion.load_projections("nba", platform, {"players.csv": "players_third_party.csv"}), "nba")
        if set(own["player_id"]) != set(state.base_raw_df["player_id"]):
            _blend_reason = "players_own.csv doesn't match players.csv. Rebuild with python build_own_model.py --blend 0.5, then restart the server."
            print(f"[blend] live blend is off. {_blend_reason}")
            return
        if not 0.0 <= w <= 1.0:
            w = 0.0
        comps = _blend_comps()
        _blend_src.update(own=own, ref=ref, comps=comps, built=w)
        state.blend = w
        state.base_raw_df = live_blend.blend_frame(own, ref, w, comps)
        _apply_overrides()
        print(f"[blend] live blend ready: {w:.0%} toward the reference list.")
    except Exception as e:
        _blend_src.clear()
        state.blend = None
        _blend_reason = f"The blend files couldn't be read ({type(e).__name__}: {e}). Rebuild with python build_own_model.py --blend 0.5."
        print(f"[blend] live blend is off. {_blend_reason}")


def _save_blend() -> None:
    if sessions.enabled:
        return
    try:
        import json, os as _os, tempfile
        path = _blend_file()
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
        with _os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"weight": state.blend}, f)
        _os.replace(tmp, path)
    except Exception as e:
        print(f"[blend] couldn't save the weight: {type(e).__name__}: {e}")


class BlendBody(BaseModel):
    weight: float


class OverrideBody(BaseModel):
    player: str
    games: float | None = None
    mpg: float | None = None
    note: str | None = None


def _override_rows() -> list:
    out = []
    for pid, o in state.overrides.items():
        base = state.base_raw_df[state.base_raw_df["player_id"] == pid].iloc[0]
        out.append({"player_id": pid, "name": state.names.get(pid), "games": o.get("games"), "mpg": o.get("mpg"), "note": o.get("note"),
                    "was_games": float(base["GP"]), "was_mpg": round(float(base["MIN"]) / float(base["GP"]), 1)})
    return out


def load_sport(key: str, preset: str | None = None) -> dict:
    """Load a sport's data and reset the draft. Raises BEFORE changing anything if the files can't be loaded."""
    cfg = LeagueConfig.for_sport(key, preset)
    raw, cleaning = validate.clean_projections(ingestion.load_projections(key, state.sync.get("platform")), key)
    history = ingestion.load_stats_history(key)
    state.config, state.raw_df, state.cleaning, state.history = cfg, raw, cleaning, history
    state.base_raw_df, state.overrides = raw.copy(), {}
    state.data_version = next_version()
    state.names = dict(zip(raw["player_id"], raw["name"]))
    state.name_map = ingestion.build_name_map(raw)
    state.adp, state.adp_note = ingestion.load_adp(key, raw, state.name_map)
    if state.adp_note:
        print(f"[data] {state.adp_note}")
    state.platform_names.clear()
    state.setup = None
    state.draft_log.clear()
    state.events.clear()
    state.saved.update(restored=False, picks_restored=0, warning=None, error=None)
    _predraft_cache.clear()
    _scorer_cache.clear()
    _init_blend()
    _load_saved_overrides()
    report = validate.build_report(state.raw_df, cfg, history, cleaning)
    validate.print_summary(report)
    return report


@asynccontextmanager
async def lifespan(app: FastAPI):
    _multi = os.getenv("MULTI_SESSION", "").strip().lower() in ("1", "true", "yes", "on")
    sessions.configure(_multi, os.getenv("MAX_SESSIONS", "100") or "100", os.getenv("SESSION_HOURS", "3") or "3")
    if _multi:
        os.environ["DRAFT_PLATFORM"] = "manual"              # live Yahoo/ESPN sync is for one person's own league, not a shared host
        print(f"[sessions] multi-session mode: private in-memory drafts, up to {sessions.max_sessions}, idle for {sessions.ttl / 3600:g} h are dropped.")
    try:
        state.my_team_id = int(os.getenv("MY_TEAM_ID", "1"))
    except ValueError:
        state.my_team_id = 1
    platform = os.getenv("DRAFT_PLATFORM", "manual").strip().lower()
    problems, warnings = check_env(platform)
    if platform not in PLATFORMS:
        platform = "manual"                                  # unknown value: run manual, and say so
    state.sync.update(platform=platform, setup_problems=problems, setup_warnings=warnings)
    _preset = os.getenv("LEAGUE_PRESET", "").strip().lower() or None     # e.g. espn_default (NBA); blank = the sport's standard league
    try:
        load_sport(os.getenv("SPORT", "mlb").strip().lower(), _preset)    # SPORT=mlb | nba | nfl in .env
    except ValueError as _e:                                              # unknown preset: say so, then start with the standard league
        print(f"[config] {_e} Starting with the standard league instead.")
        load_sport(os.getenv("SPORT", "mlb").strip().lower())
    if not _multi:
        _restore(platform)                                   # bring back the saved setup and picks, if any
    for line in problems + warnings:
        print(f"[draft] {'SETUP NEEDED' if line in problems else 'note'}: {line}")
    task = None
    poll = max(POLL_SECONDS, MIN_LIVE_POLL)
    if problems:
        state.sync.update(connected=False, message="Setup needed: " + " ".join(problems))
        if platform in ("yahoo", "espn"):
            platform = "manual-fallback"                     # no sync task; the page explains what to fix
    if platform in ("yahoo", "espn"):
        def make_map(platform=platform):
            found, note = ingestion.load_platform_map(state.config.sport, platform)
            state.sync["map_note"] = note
            return found
        task = asyncio.create_task(sync.run(state, lambda: get_source(state.config.sport), make_map, poll,
                                            on_change=_save))
    elif platform == "mock":
        make_map = lambda: {i: i for i in state.raw_df["player_id"]}
        task = asyncio.create_task(sync.run(state, _make_mock, make_map, POLL_SECONDS))
    yield
    if task:
        task.cancel()


app = FastAPI(title="Fantasy Draft Engine", lifespan=lifespan)


class SessionMiddleware:
    """Pure ASGI middleware: in multi-session mode, find (or create) this browser's draft and make it the current one."""
    COOKIE = "draft_session"
    NO_SESSION = ("/health", "/docs", "/redoc", "/openapi.json", "/session")

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not sessions.enabled or scope.get("path", "").startswith(self.NO_SESSION):
            return await self.app(scope, receive, send)       # health checks and the API docs must not use up session slots
        headers = dict(scope.get("headers") or [])
        sid = None
        for part in headers.get(b"cookie", b"").decode("latin-1").split(";"):
            k, _, v = part.strip().partition("=")
            if k == self.COOKIE:
                sid = v[:100]
        sid, sess, created = sessions.get_or_create(sid)
        secure = headers.get(b"x-forwarded-proto", b"").decode() == "https"

        async def send_with_cookie(message):
            if created and message["type"] == "http.response.start":
                cookie = f"{self.COOKIE}={sid}; Path=/; HttpOnly; SameSite=Lax; Max-Age={int(sessions.ttl)}" + ("; Secure" if secure else "")
                message["headers"] = list(message.get("headers") or []) + [(b"set-cookie", cookie.encode())]
            await send(message)

        token = _current.set(sess)
        try:
            await self.app(scope, receive, send_with_cookie)
        finally:
            _current.reset(token)


class SlowRequestLog:
    """Prints a line in the server window for any request slower than SLOW_REQUEST_SECONDS (default 1.0), so a slow draft can be
    diagnosed: is it the calculation, or the computer?"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        import time as _t
        t0 = _t.perf_counter()
        try:
            await self.app(scope, receive, send)
        finally:
            took = _t.perf_counter() - t0
            try:
                limit = float(os.getenv("SLOW_REQUEST_SECONDS", "1.0"))
            except ValueError:
                limit = 1.0
            if took >= limit:
                print(f"[slow] {scope.get('method')} {scope.get('path')} took {took:.1f}s")


app.add_middleware(SessionMiddleware)
app.add_middleware(SlowRequestLog)


@app.get("/session")
def session_info():
    """Is this a shared host with private drafts per browser, or your own single-draft copy?"""
    return {"mode": "multi" if sessions.enabled else "single", "active_sessions": len(sessions) if sessions.enabled else 1,
            "saved_to_disk": not sessions.enabled, "idle_hours_before_drop": round(sessions.ttl / 3600, 2) if sessions.enabled else None}


class PickRequest(BaseModel):
    player_id: str
    team_id: int | None = None      # omit to use the team on the clock (needs a saved draft setup)
    price: int | None = None        # auction only: what the team paid, in whole dollars
    request_id: str | None = None   # optional idempotency key (the Idempotency-Key header does the same)


# ---------- league format ----------
@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).parent / "static" / "index.html", headers={"Cache-Control": "no-cache"})   # always the newest page


def _manual_entry_blocked() -> str | None:
    """Hand-entered picks would be overwritten at the next poll while a live connection is working."""
    plat = state.sync.get("platform")
    if plat in LIVE_PLATFORMS and state.sync.get("connected"):
        name = "the mock draft" if plat == "mock" else plat.upper() if plat == "espn" else plat.capitalize()
        return (f"Picks are coming in automatically from {name}, so entering them by hand is turned off. "
                "If the connection drops, hand entry turns back on.")
    return None


def _setup_payload():
    return {"setup": state.setup, "options": draft.TYPES, "limits": draft.LIMITS,
            "sport": state.config.sport, "sports": [{"key": k, "label": v.label} for k, v in sports.SPORTS.items()],
            "default_rounds": len(state.config.roster.slot_list()), "saved": state.saved,
            "format": ScoringType(state.config.scoring_type).value, "platform": state.sync.get("platform", "manual"),
            "positions": sports.board_positions(state.config.sport), "manual_entry_blocked": _manual_entry_blocked(),
            "status": draft.status(state.setup, len(state.draft_log)) if state.setup else None,
            "auction": _auction_payload()}


@app.get("/sports")
def list_sports():
    """Supported sports, their scoring presets and the files each needs (inside data/ for MLB, data/<sport>/ otherwise)."""
    return {"current": state.config.sport,
            "sports": [{"key": k, "label": v.label, "default_scoring": v.default_scoring, "presets": list(v.presets),
                        "needs_files": [g.file for g in v.groups], "category_scoring": v.has_categories,
                        "slots": list(v.slot_rules), "default_slots": dict(v.default_slots)}
                       for k, v in sports.SPORTS.items()]}


@app.put("/league/sport")
def set_sport(body: dict = Body(...)):
    """Switch sport (and optionally a scoring preset, e.g. {"sport": "nfl", "preset": "ppr"}).
    Loads that sport's data and clears the draft setup and picks, so the UI asks you to confirm first."""
    key = str(body.get("sport", "")).lower()
    with state.lock:
        old = state.config.sport
        try:
            report = load_sport(key, body.get("preset"))
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))
        state.idem.clear()
        if old != key:
            persist.delete(old)                  # the page asked you to confirm clearing that draft
        _save()                                  # new sport starts empty, but keeps its chosen scoring preset
        return {"ok": True, "sport": key, "data_ok": report["ok"], "errors": report["errors"],
                "warnings": report["warnings"][:5], **_setup_payload()}


@app.get("/draft/setup")
def get_setup():
    return _setup_payload()


@app.put("/draft/setup")
def put_setup(body: dict = Body(...)):
    """Format, number of teams and your pick. Field-level errors come back as {"errors": {field: message}}."""
    with state.lock:                                 # validated against the picks that exist right now, then applied
        setup, errs = draft.validate_setup(body, len(state.config.roster.slot_list()))
        n, cur = len(state.draft_log), state.setup
        if setup and n:
            if cur is None or (cur.draft_type, cur.teams) != (setup.draft_type, setup.teams):
                errs["teams"] = f"{n} pick(s) are already logged. Start a new draft before changing the format or teams."
            elif n > setup.teams * setup.rounds:
                errs["rounds"] = f"{n} picks are already logged, which is more than {setup.rounds} rounds allows."
            else:
                for k, v in auction.setup_conflict(setup, state.draft_log).items():
                    errs.setdefault(k, v)
        if errs:
            raise HTTPException(400, {"errors": errs})
        state.setup, state.config.roster.teams = setup, setup.teams
        if not hasattr(state.source, "fetch_settings") and setup.my_slot:      # manual mode: your team id is your slot
            state.my_team_id = setup.my_slot
        state.sync_wake = True                       # let the live-sync loop retry right away
        _save()
        return _setup_payload()


@app.get("/draft/status")
def draft_status():
    if state.setup is None:
        raise HTTPException(404, "Save your draft setup first")
    return draft.status(state.setup, len(state.draft_log))


@app.post("/draft/reset")
def reset_draft():
    """Clear every logged pick (keeps your setup). For starting a fresh draft or mock."""
    with state.lock:
        cleared = len(state.draft_log)
        state.draft_log.clear()
        state.idem.clear()
        _save()
        return {"cleared": cleared}


@app.get("/projections/blend")
def get_blend():
    """The live blend between the own model (0) and the reference list (1)."""
    if state.blend is None:
        return {"available": False, "relevant": state.config.sport == "nba", "weight": None,
                "message": _blend_reason or "Live blending isn't set up. Run python build_own_model.py --blend 0.5, then restart the server."}
    return {"available": True, "weight": state.blend, "built_weight": _blend_src.get("built"), "own_label": "Own model", "reference_label": "Reference list"}


@app.put("/projections/blend")
def set_blend(body: BlendBody):
    if state.blend is None:
        raise HTTPException(400, {"message": _blend_reason or "Live blending isn't set up. Run python build_own_model.py --blend 0.5, then restart the server."})
    with state.lock:
        try:
            state.base_raw_df = live_blend.blend_frame(_blend_src["own"], _blend_src["ref"], body.weight, _blend_src["comps"])
        except ValueError as e:
            raise HTTPException(400, {"message": str(e)})
        state.blend = round(float(body.weight), 3)
        _apply_overrides()                                 # your what-if overrides stay on top of the new blend
        _save_blend()
        return {"available": True, "weight": state.blend}


@app.get("/players/overrides")
def list_overrides():
    return {"supported": state.config.sport == "nba", "overrides": _override_rows()}


@app.put("/players/override")
def set_override(body: OverrideBody):
    """What-if edit for one player: expected games and/or minutes per game (basketball). Everything is recomputed."""
    if state.config.sport != "nba":
        raise HTTPException(400, {"message": "Player overrides are available for basketball."})
    pid = _find_player(body.player)
    if body.games is None and body.mpg is None:
        raise HTTPException(400, {"message": "Give games, mpg, or both."})
    if body.games is not None and not (1 <= body.games <= 82):
        raise HTTPException(400, {"message": "Games must be between 1 and 82."})
    if body.mpg is not None and not (1 <= body.mpg <= 48):
        raise HTTPException(400, {"message": "Minutes per game must be between 1 and 48."})
    with state.lock:
        state.overrides[pid] = {"games": body.games, "mpg": body.mpg, "note": (body.note or "").strip()[:200] or None}
        _apply_overrides()
        _save_overrides()
        return {"saved": state.names.get(pid), "overrides": _override_rows()}


@app.delete("/players/override/{player_id}")
def delete_override(player_id: str):
    if player_id not in state.overrides:
        raise HTTPException(404, {"message": "That player has no override."})
    with state.lock:
        if player_id not in state.overrides:
            raise HTTPException(404, {"message": "That player has no override."})
        del state.overrides[player_id]
        _apply_overrides()
        _save_overrides()
        return {"removed": player_id, "overrides": _override_rows()}


@app.delete("/players/overrides")
def clear_overrides():
    with state.lock:
        n = len(state.overrides)
        state.overrides.clear()
        _apply_overrides()
        _save_overrides()
        return {"cleared": n}


@app.get("/health/data")
def health_data():
    """Is the data usable? Lists errors (must fix), warnings (look at), what was auto-cleaned at startup,
    and whether every position can be filled from your player pool."""
    return validate.build_report(state.raw_df, state.config, state.history, state.cleaning)


@app.get("/league/config")
def get_config():
    return state.config


@app.put("/league/config")
def set_config(cfg: LeagueConfig):
    """Pick the format with scoring_type: 'roto' | 'h2h_categories' | 'points'.
    Valuations are derived on every read, so nothing else needs recomputing."""
    with state.lock:
        if cfg.sport != state.config.sport:
            raise HTTPException(400, f"This league is {state.config.sport}. Use PUT /league/sport to switch sports")
        if state.setup and cfg.roster.teams != state.setup.teams:
            raise HTTPException(400, f"Teams is set in Draft Setup ({state.setup.teams}); change it there")
        r_errs, r_warns = validate.check_roster(cfg)
        if r_errs:
            raise HTTPException(400, {"message": "That roster can't be used. " + " ".join(r_errs), "problems": r_errs})
        old_roster = state.config.roster
        changed = cfg.roster.slots != old_roster.slots
        if changed and state.draft_log:
            raise HTTPException(400, "Roster slots can't change after picks have been made. Start a new draft first.")
        problems = validate.check_config(state.raw_df, cfg)
        if problems:                                   # reject now instead of failing on the next request
            raise HTTPException(400, {"message": "Your projections can't support these settings", "problems": problems})
        old_default = len(old_roster.slot_list())
        state.config = cfg
        if changed and state.setup and state.setup.rounds == old_default:        # rounds followed the roster size: keep following it
            state.setup = dataclasses.replace(state.setup, rounds=len(cfg.roster.slot_list()))
        _save()
        return {"ok": True, "scoring_type": cfg.scoring_type, "warnings": r_warns,
                "default_rounds": len(cfg.roster.slot_list()), "slots": cfg.roster.slots}


@app.post("/league/config/import-yahoo")
def import_yahoo():
    """Pull format, roster slots and categories from your Yahoo league, then review the warnings."""
    from .connectors.yahoo import yahoo_to_league_config
    if not hasattr(state.source, "fetch_settings"):
        raise HTTPException(400, "DRAFT_PLATFORM is not yahoo")
    settings, cats = state.source.fetch_settings()
    state.config, warnings = yahoo_to_league_config(settings, cats, state.config)
    _save()
    return {"config": state.config, "warnings": warnings}


# ---------- draft ----------
@app.get("/players/available")
def available(limit: int = 50, exclude_injury_prone: bool | None = None, position: str | None = None,
              q: str | None = None, sort: str = "value"):
    """exclude_injury_prone: omit to follow the league config; true/false overrides it for this call.
    position: only players eligible there (e.g. SS). q: part of a name (accents and case ignored)."""
    if sort not in ("value", "rank"):
        raise HTTPException(400, {"message": "sort must be 'value' (value over replacement) or 'rank' (overall rank)."})
    df, _ = valuated()
    if _is_auction():
        df = _auction_columns(df)                 # adds bid and pre_bid columns
    return views.available_view(df, state.config, limit, exclude_injury_prone, position, q, sort)


@app.get("/players/search")
def search(q: str, stat: str | None = None, limit: int = 10):
    """Find a player by name, then ask for one stat. Examples:
      q='Eli Perez stats'          -> status 'choose_stat' (lists the stats you can ask for)
      q='Eli Perez', stat='OBP'    -> previous season, career and projected OBP
      q='Eli Perez ERA'            -> same, with the stat typed at the end of the query
      stat='all'                   -> every stat in one response
    status 'choose_player' means several players matched: pick one and use /players/{player_id}/stats."""
    df, _ = valuated()
    return stats.search_flow(df, state.raw_df, state.history, q, stat, limit, state.config.sport)


@app.get("/players/{player_id}/stats")
def player_stats(player_id: str, stat: str | None = None, kind: str | None = None):
    """Same as /players/search once you know the player_id. kind ('H' or 'P') only matters for two-way players."""
    df, _ = valuated()
    res = stats.stats_for_player(df, state.raw_df, state.history, player_id, stat, kind, state.config.sport)
    if res["status"] == "no_match":
        raise HTTPException(404, res["message"])
    return res


@app.get("/players/rankings")
def rankings(limit: int = 50, position: str | None = None, kind: str | None = None,
             include_drafted: bool = True):
    """The full board by projected_rank. Filter with position (e.g. SS) or kind (H or P).
    include_drafted=false shows only players still available."""
    df, _ = valuated()
    return views.rankings_view(df, limit, position, kind, include_drafted)


@app.get("/players/explain")
def explain(q: str):
    """Why is a player ranked where he is? e.g. /players/explain?q=tatum"""
    df, _ = valuated()
    return views.explain_player(df, q)


@app.get("/players/injury-prone")
def injury_prone(limit: int = 50, include_drafted: bool = False):
    """Every player flagged injury-prone (avg games missed >= threshold over the last 3 seasons),
    best value first. 'count' is the total number flagged, before the limit."""
    df, _ = valuated()
    return views.injury_prone_view(df, limit, include_drafted)


@app.post("/draft/pick")
def pick(req: PickRequest, idempotency_key: str | None = Header(default=None)):
    """Record a pick. Safe against simultaneous requests: the checks and the append happen under one lock, so a player can't be
    drafted twice, two picks can't share a pick number, and a finished draft can't overflow. A repeated Idempotency-Key (or
    request_id) returns the first result instead of drafting again, so a double click or a retry is harmless."""
    header_key = idempotency_key if isinstance(idempotency_key, str) else None      # called directly (not over HTTP) it is a placeholder
    key = (header_key or req.request_id or "").strip()[:80]
    with state.lock:
        if key and key in state.idem:
            return dict(state.idem[key])
        if (why := _manual_entry_blocked()):
            raise HTTPException(409, why)
        if state.setup and len(state.draft_log) >= state.setup.teams * state.setup.rounds:
            raise HTTPException(400, "The draft is complete")
        team_id = req.team_id
        if team_id is None:
            if state.setup is None or state.setup.draft_type == "auction":
                raise HTTPException(400, "team_id is required (or save a snake/linear draft setup first)")
            team_id = draft.slot_for_pick(state.setup.draft_type, state.setup.teams, len(state.draft_log) + 1)
        if team_id < 1 or team_id > state.config.roster.teams:
            raise HTTPException(400, f"team_id must be between 1 and {state.config.roster.teams}")
        if req.player_id not in set(state.raw_df["player_id"]):
            raise HTTPException(404, "player not found")
        if req.player_id in valuation.expand_twins(state.raw_df, state.drafted_ids()):
            raise HTTPException(400, "already drafted")
        if _is_auction():
            if req.price is None:
                raise HTTPException(400, "Enter the price the team paid (whole dollars).")
            if (why := auction.check_sale(state.setup, state.draft_log, team_id, req.price)):
                raise HTTPException(400, why)
        elif req.price is not None:
            raise HTTPException(400, "Prices only apply to auction drafts.")
        rec_ = {"pick_number": len(state.draft_log) + 1, "player_id": req.player_id,
                "team_id": team_id, "source": "manual"}
        if req.price is not None:
            rec_["price"] = req.price
        state.draft_log.append(rec_)
        if key:
            state.idem[key] = dict(rec_)
            while len(state.idem) > 500:
                state.idem.popitem(last=False)
        _save()
        return rec_


@app.post("/draft/undo")
def undo():
    with state.lock:
        if (why := _manual_entry_blocked()):
            raise HTTPException(409, why)
        if not state.draft_log:
            raise HTTPException(400, "nothing to undo")
        popped = state.draft_log.pop()
        state.idem.clear()
        _save()
        return popped


@app.get("/auction/state")
def auction_state():
    """Auction drafts: each team's money and open roster spots, the most each can bid, who nominates next, and how
    far prices are running above (+) or below (-) the pre-draft price list."""
    if not _is_auction():
        raise HTTPException(400, "This draft isn't an auction. Choose the Auction format in Draft setup.")
    return _auction_payload()


@app.get("/draft/saved")
def draft_saved():
    """Is the draft being saved to disk? Where, when it last saved, and what was restored at startup."""
    return {**state.saved, "file": str(persist.path_for(state.config.sport))}


@app.get("/draft/log")
def log():
    with state.lock:
        return list(state.draft_log)


@app.get("/draft/recent")
def recent_picks(limit: int = 15):
    """The latest picks, newest first, with player names (for the pick log on the page)."""
    out = []
    for p in reversed(state.draft_log[-max(1, min(limit, 100)):]):
        out.append({**p, "player": state.names.get(p["player_id"]) if p["player_id"] else None,
                    "mine": p["team_id"] == state.my_team_id})
    return {"picks": out, "total": len(state.draft_log)}


# ---------- team ----------
@app.get("/sync/status")
def sync_status():
    """Is the live connection working, when did it last hear from the platform, and what picks just happened?"""
    return {**state.sync, "my_team_id": state.my_team_id, "picks_logged": len(state.draft_log),
            "events": state.events[-12:][::-1]}


@app.get("/team/summary")
def team_summary(team_id: int | None = None):
    """Your roster by slot, your team score, and how it compares with the average team drafting from your slot."""
    sc = predraft_scorer()
    ids = state.team_player_ids(team_id or state.my_team_id)
    out = scoring.summary(sc, ids)
    out["picks"] = len(ids)
    s = state.setup
    if s is not None and s.draft_type == "auction":
        out["benchmark"] = {"status": "unavailable", "reason": "Pace comparison isn't available for auction drafts. Your budget is in the Draft status card."}
        return out
    if s is None or not s.my_slot:
        out["benchmark"] = {"status": "unavailable", "reason": "Save a snake or linear draft setup with your pick to see your pace."}
        return out
    key = (repr(state.config), s.draft_type, s.teams, s.rounds, state.data_version)
    res = _bench.get(key, lambda: scoring.simulate(sc, s.draft_type, s.teams, s.rounds))
    if res is None:
        out["benchmark"] = {"status": "calculating"}
    elif "error" in res:
        out["benchmark"] = {"status": "error", "reason": res["error"]}
    else:
        out["benchmark"] = scoring.compare(res, s.my_slot, out["score"], len(ids))
    return out


@app.get("/team/{team_id}/roster")
def roster(team_id: int):
    df, repl = valuated()
    team = df[df["player_id"].isin(state.team_player_ids(team_id))]
    return optimizer.optimize_roster(team, state.config.roster, repl, state.config.sport)


def _recommend(team_id, limit, exclude_injury_prone, use_lookahead: bool):
    df, repl = valuated()
    mine = state.team_player_ids(team_id or state.my_team_id)
    gaps = None
    if use_lookahead and (team_id is None or team_id == state.my_team_id):   # look-ahead is about YOUR picks
        gaps = lookahead.pick_gaps(state.setup, len(state.draft_log))
    rounds_total = state.setup.rounds if state.setup else None
    out, plan = rec.recommend(df, repl, state.config, mine, limit=limit, rounds_total=rounds_total,
                              exclude_injury_prone=exclude_injury_prone, gaps=gaps, with_plan=True)
    if _is_auction():                             # suggested price next to each recommendation
        cols = _auction_columns(df)
        bids = dict(zip(cols["player_id"], cols["bid"]))
        for r in out:
            b = bids.get(r["player_id"])
            r["bid"] = None if b is None or b != b else int(b)
        plan = {"status": "auction", "reason": "Look-ahead isn't used in auctions (anyone can be nominated at any time)."}
        try:
            wide = rec.recommend(df, repl, state.config, mine, limit=30, rounds_total=rounds_total,
                                 exclude_injury_prone=exclude_injury_prone)
            for r in wide:
                b = bids.get(r["player_id"])
                r["bid"] = None if b is None or b != b else int(b)
            payload = _auction_payload(cols)
            filled = len(state.draft_log) / max(state.setup.teams * state.setup.rounds, 1)
            ideas = auction.nominate_ideas(wide, payload["mine"], payload["nomination"], state.my_team_id, filled)
            plan.update(nominate=ideas, headline=ideas["headline"])
        except Exception as e:                    # advice is optional; never break the list
            plan["nominate_error"] = f"{type(e).__name__}: {e}"
    plan["adp_note"] = state.adp_note
    return out, plan


@app.get("/recommend")
def recommend(team_id: int | None = None, limit: int = 10, exclude_injury_prone: bool | None = None,
              lookahead: bool = True):
    """Marginal gain to your team under the active league format (roto z-sum, H2H category win-probability, or
    points), blended with global VORP early in the draft. In snake/linear drafts the top candidates are re-ranked
    by whether they'll last until your next turn (set lookahead=false for the plain ranking)."""
    return _recommend(team_id, limit, exclude_injury_prone, lookahead)[0]


@app.get("/recommend/plan")
def recommend_plan(team_id: int | None = None, limit: int = 10, exclude_injury_prone: bool | None = None,
                   lookahead: bool = True):
    """Same picks as /recommend plus the plan: a one-line headline, when your next turns are, and why."""
    picks, plan = _recommend(team_id, limit, exclude_injury_prone, lookahead)
    return {"plan": plan, "picks": picks}


@app.get("/health/adp")
def adp_check(limit: int = 10):
    """Compare this app's ranking with the ADP you loaded (data/<sport>/adp.csv). Use it to see how far the
    projections-based ranking is from where your league's drafters actually take players."""
    if not state.adp:
        return {"loaded": False, "note": state.adp_note or "No adp.csv found in this sport's data folder. "
                "Put a file there with columns player_id (or name) and adp, then restart."}
    return views.adp_report(predraft_board(), state.adp, max(1, min(limit, 50)), state.adp_note)

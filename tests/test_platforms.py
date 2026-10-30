"""Tests for the platform layer (Yahoo/ESPN/mock sync, matching, setup checks) and the draft board endpoints."""
import asyncio
import os
import tempfile
from pathlib import Path

import pandas as pd

from app import ingestion, sync, views
from app.config import LeagueConfig
from app.connectors import check_env
from app.connectors.base import RemotePick, SourceError, dense_index
from app.connectors.espn import EspnSource, clean_s2, clean_swid
from app.state import GlobalState
from app.valuation import calculate_player_valuations
from tests.test_math import _FakeRemote, fake_projections


class set_env:
    """Temporarily set (or with None, remove) environment variables."""
    def __init__(self, **kv): self.kv, self.old = kv, {}
    def __enter__(self):
        for k, v in self.kv.items():
            self.old[k] = os.environ.get(k)
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    def __exit__(self, *a):
        for k, v in self.old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# ---------- reading files ----------
def test_csv_reader_handles_bom_cp1252_and_header_spaces():
    with tempfile.TemporaryDirectory() as d:
        a, b, c = Path(d, "a.csv"), Path(d, "b.csv"), Path(d, "c.csv")
        a.write_bytes("\ufeffName, playerid\nJosé Ramírez,1\n".encode("utf-8"))      # Excel / FanGraphs: UTF-8 with BOM
        b.write_bytes("Name,playerid\nJosé Ramírez,1\n".encode("cp1252"))            # older Excel: Windows-1252
        c.write_bytes("Name,playerid\nJosé Ramírez,1\n".encode("utf-8"))             # plain UTF-8
        for f in (a, b, c):
            df = ingestion.read_csv_robust(f)
            assert list(df.columns) == ["Name", "playerid"] and df["Name"][0] == "José Ramírez", f.name


# ---------- name matching ----------
def _raw(rows):
    df = pd.DataFrame(rows)
    df["clean_name"] = df["name"].map(ingestion.clean_name)
    return df


def test_name_map_only_keeps_names_that_mean_one_human():
    raw = _raw([dict(player_id="1", name="José Ramírez", base_id="1"),
                dict(player_id="2", name="Sam Walker", base_id="2"),        # two different people, same name
                dict(player_id="3", name="Sam Walker", base_id="3"),
                dict(player_id="4", name="Shohei Ohtani", base_id="4"),     # two-way: same human, two rows
                dict(player_id="4_P", name="Shohei Ohtani", base_id="4"),
                dict(player_id="5", name="Ken Griffey Jr.", base_id="5")])
    m = ingestion.build_name_map(raw)
    assert m["jose ramirez"] == "1" and "sam walker" not in m and m["shohei ohtani"] == "4"
    assert m[ingestion.clean_name("Ken Griffey")] == "5"                      # suffix and punctuation ignored


def test_build_log_prefers_id_then_falls_back_to_name():
    remote = [RemotePick(1, "y1", "1", "Whoever"), RemotePick(2, "yZ", "2", "José Ramírez"),
              RemotePick(3, "yQ", "3", "Nobody Known"), RemotePick(4, "yR", "4")]
    log, unmapped = sync.build_log(remote, {"y1": "h1"}, int, {"jose ramirez": "h9"})
    assert [p["player_id"] for p in log] == ["h1", "h9", None, None] and unmapped == ["yQ", "yR"]
    log, _ = sync.build_log(remote, {"y1": "h1"}, int)                          # old call style still works
    assert [p["player_id"] for p in log] == ["h1", None, None, None]


def test_sync_looks_up_names_once_and_reports_matches():
    class Src(_FakeRemote):
        asked = []
        def resolve_names(self, ids):
            Src.asked.append(list(ids))
            return {"y2": "Hitter Two"}
    st = GlobalState()
    st.names = {"h1": "Hitter One", "h2": "Hitter Two"}
    st.name_map = {"hitter two": "h2"}
    src = Src([("y1", 1), ("y2", 2), ("y3", 3)])
    async def go():
        t = asyncio.create_task(sync.run(st, lambda: src, lambda: {"y1": "h1"}, 0.01, retry_delay=0.01))
        await asyncio.sleep(0.2)
        t.cancel()
    asyncio.run(go())
    assert [p["player_id"] for p in st.draft_log] == ["h1", "h2", None]
    assert Src.asked == [["y2", "y3"]]                                           # asked once, not every poll
    assert st.sync["name_matched"] == 1 and "matched by player name" in st.sync["message"]
    assert st.sync["unmatched"] == [{"id": "y3", "name": ""}] and st.sync["unmapped"] == ["y3"]


def test_sync_shows_friendly_source_errors_and_recovers_after_backoff():
    class Src(_FakeRemote):
        async def fetch_picks(self):
            if self.fail:
                raise SourceError("ESPN says access denied. Set ESPN_S2.")
            return await super().fetch_picks()
    st = GlobalState()
    src = Src([("y1", 1)]); src.fail = True
    async def go():
        t = asyncio.create_task(sync.run(st, lambda: src, lambda: {"y1": "h1"}, 0.01, retry_delay=0.01))
        await asyncio.sleep(0.1)
        assert not st.sync["connected"] and st.sync["last_error"] == "ESPN says access denied. Set ESPN_S2."   # no class name noise
        src.fail = False
        await asyncio.sleep(0.3)
        t.cancel()
    asyncio.run(go())
    assert st.sync["connected"] and len(st.draft_log) == 1


# (the ESPN connector's tests live in tests/test_espn.py)


def test_cookie_cleanup_and_dense_team_numbers():
    assert clean_swid("abcdef01-2345-6789-abcd-ef0123456789") == "{abcdef01-2345-6789-abcd-ef0123456789}"
    assert clean_swid(' "{ABCDEF01-2345-6789-ABCD-EF0123456789}" ') == "{ABCDEF01-2345-6789-ABCD-EF0123456789}"
    assert clean_swid("") == "" and clean_s2(" 'AEB%2Fxyz' ") == "AEB%2Fxyz"
    f = dense_index([1, 2, 4, 5])
    assert [f(1), f(2), f(4), f(5), f(9)] == [1, 2, 3, 4, 9]


# ---------- start-up checks ----------
def test_check_env_names_what_is_missing():
    keys = dict(DRAFT_PLATFORM=None, ESPN_LEAGUE_ID=None, ESPN_YEAR=None, ESPN_S2=None, ESPN_SWID=None, YAHOO_LEAGUE_KEY=None)
    with set_env(**keys):
        assert check_env("manual") == ([], []) and check_env("mock") == ([], [])
        assert "isn't one of" in check_env("fantasyland")[0][0]
        problems, _ = check_env("espn")
        assert any("ESPN_LEAGUE_ID" in p for p in problems) and any("ESPN_YEAR" in p for p in problems)
    with set_env(**{**keys, "ESPN_LEAGUE_ID": "123456", "ESPN_YEAR": "2027"}):
        problems, warnings = check_env("espn")
        assert not any("ESPN_LEAGUE_ID" in p or "ESPN_YEAR" in p for p in problems)
        assert any("private league" in w for w in warnings)                              # warning, not a blocker
    with set_env(**{**keys, "ESPN_LEAGUE_ID": "12x", "ESPN_YEAR": "27"}):
        assert len([p for p in check_env("espn")[0] if "ESPN_" in p]) == 2
    with set_env(**{**keys, "YAHOO_LEAGUE_KEY": "not-a-key"}):
        assert any("458.l.123456" in p for p in check_env("yahoo")[0])


# ---------- crosswalk and positions files ----------
def test_platform_map_never_raises_and_says_why():
    with tempfile.TemporaryDirectory() as d:
        orig = ingestion.DATA_DIR
        ingestion.DATA_DIR = Path(d)
        try:
            m, note = ingestion.load_platform_map("mlb", "espn")
            assert m == {} and "by player name" in note                                   # no file
            Path(d, "id_crosswalk.csv").write_text("player_id,yahoo_id\n1000,9001\n")
            m, note = ingestion.load_platform_map("mlb", "espn")
            assert m == {} and "espn_id" in note                                          # file, but no espn column
            m, note = ingestion.load_platform_map("mlb", "yahoo")
            assert m == {"9001": "1000"} and note is None
            Path(d, "id_crosswalk.csv").write_text("player_id,yahoo_id,espn_id\n1000,9001,31\n1001,9002,\n")
            assert ingestion.load_platform_map("mlb", "espn")[0] == {"31": "1000"}        # blank cells skipped
        finally:
            ingestion.DATA_DIR = orig


def test_platform_specific_positions_file_wins():
    with tempfile.TemporaryDirectory() as d:
        orig = ingestion.DATA_DIR
        ingestion.DATA_DIR = Path(d)
        try:
            Path(d, "positions.csv").write_text("player_id,eligible_positions\n1000,SS\n")
            Path(d, "positions_espn.csv").write_text("player_id,eligible_positions\n1000,\"SS,2B\"\n")
            assert ingestion._load_positions_override("mlb", "espn") == {"1000": ["SS", "2B"]}
            assert ingestion._load_positions_override("mlb", "yahoo") == {"1000": ["SS"]}   # falls back to the generic file
            assert ingestion._load_positions_override("mlb") == {"1000": ["SS"]}
        finally:
            ingestion.DATA_DIR = orig


# ---------- board filters ----------
def test_available_view_filters_by_position_and_name():
    raw = fake_projections()
    df, _ = calculate_player_valuations(raw, LeagueConfig(), set(), 2)
    cfg = LeagueConfig()
    ss = views.available_view(df, cfg, 500, position="ss")
    assert ss and all("SS" in p["eligible_positions"] for p in ss)
    named = views.available_view(df, cfg, 500, q="  hitter 1 ")
    assert named and all("hitter 1" in p["name"].lower() for p in named)
    both = views.available_view(df, cfg, 500, position="SS", q="Pitcher")
    assert both == []
    assert len(views.available_view(df, cfg, 5)) == 5                                    # defaults unchanged
    drafted, _ = calculate_player_valuations(raw, cfg, {"h1"}, 2)
    assert "h1" not in [p["player_id"] for p in views.available_view(drafted, cfg, 500)]


# ---------- the endpoints (need the project's data files; skipped quietly if there are none) ----------
def _have_data() -> bool:
    return Path(ingestion.DATA_DIR, "hitters.csv").exists()


def test_board_endpoints_and_the_live_connection_guard():
    if not _have_data():
        return
    from fastapi import HTTPException
    from app import main, persist
    from app.state import state
    with tempfile.TemporaryDirectory() as d:
        orig = persist.DATA_DIR
        persist.DATA_DIR = Path(d)
        try:
            main.load_sport("mlb")
            state.sync.update(platform="manual", connected=False)
            p = main.put_setup({"draft_type": "snake", "teams": 12, "my_slot": 5, "rounds": ""})
            assert p["format"] == "roto" and p["platform"] == "manual" and "SS" in p["positions"] and "DH" not in p["positions"]
            assert p["manual_entry_blocked"] is None
            top = main.available(limit=3, exclude_injury_prone=False, position=None, q=None)
            main.pick(main.PickRequest(player_id=top[0]["player_id"]))
            rp = main.recent_picks(limit=5)
            assert rp["total"] == 1 and rp["picks"][0]["player"] == top[0]["name"] and rp["picks"][0]["pick_number"] == 1
            assert main.available(limit=500, exclude_injury_prone=False, position=None, q=top[0]["name"]) == []   # drafted: not available
            # a working live connection: hand entry is refused (it would be overwritten at the next poll)
            state.sync.update(platform="espn", connected=True)
            assert "ESPN" in main._setup_payload()["manual_entry_blocked"]
            for call in (lambda: main.pick(main.PickRequest(player_id=top[1]["player_id"])), main.undo):
                try:
                    call()
                    assert False, "should be refused"
                except HTTPException as e:
                    assert e.status_code == 409
            state.sync.update(connected=False)                                            # connection dropped: hand entry is back
            assert main._setup_payload()["manual_entry_blocked"] is None
            main.pick(main.PickRequest(player_id=top[1]["player_id"]))
            assert len(state.draft_log) == 2
        finally:
            persist.DATA_DIR = orig
            state.sync.update(platform="manual", connected=False)
            main.load_sport("mlb")


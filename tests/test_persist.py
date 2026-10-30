"""Tests for saving/restoring the draft (app/persist.py) and the endpoints that use it."""
import asyncio
import json
import tempfile
from pathlib import Path

from app import persist
from app.config import LeagueConfig
from app.draft import DraftSetup
from app.state import GlobalState
from tests.test_math import fake_projections

RAW = fake_projections()


def make_state(**kw) -> GlobalState:
    st = GlobalState(config=LeagueConfig(), raw_df=RAW.copy())
    st.setup = DraftSetup("snake", 12, 5, 25, None)
    st.config.roster.teams = 12
    st.draft_log[:] = [{"pick_number": 1, "player_id": "h0", "team_id": 1, "source": "manual"},
                       {"pick_number": 2, "player_id": "h1", "team_id": 2, "source": "manual"},
                       {"pick_number": 3, "player_id": "p1", "team_id": 3, "source": "manual"}]
    for k, v in kw.items():
        setattr(st, k, v)
    return st


def fresh_state() -> GlobalState:
    return GlobalState(config=LeagueConfig(), raw_df=RAW.copy())


def test_round_trip_restores_setup_picks_and_settings():
    with tempfile.TemporaryDirectory() as d:
        a = make_state()
        a.config.category_weights = {"HR": 1.5}
        a.config.injury_prone_penalty = 0.1
        persist.save(a, directory=d)
        b = fresh_state()
        res = persist.restore(b, directory=d)
        assert res["restored"] and res["picks"] == 3 and res["setup"] and res["config"] and res["warning"] is None
        assert b.draft_log == a.draft_log and b.setup == a.setup and b.my_team_id == 5
        assert b.config == a.config and b.config.roster.teams == 12


def test_save_is_atomic_and_leaves_no_temp_files():
    with tempfile.TemporaryDirectory() as d:
        st = make_state()
        persist.save(st, directory=d)
        persist.save(st, directory=d)
        assert [p.name for p in Path(d).iterdir()] == ["draft_state_mlb.json"]
        assert json.loads(Path(d, "draft_state_mlb.json").read_text())["draft_log"][0]["player_id"] == "h0"


def test_failed_replace_keeps_old_file_and_cleans_up():
    with tempfile.TemporaryDirectory() as d:
        st = make_state()
        persist.save(st, directory=d)
        before = Path(d, "draft_state_mlb.json").read_text()
        st.draft_log.pop()
        orig = persist._replace
        persist._replace = lambda tmp, dest: (_ for _ in ()).throw(OSError("disk full"))
        try:
            msg = persist.try_save(st, directory=d)
        finally:
            persist._replace = orig
        assert msg and "disk full" in msg and "still running in memory" in msg and st.saved["error"] == msg
        assert Path(d, "draft_state_mlb.json").read_text() == before
        assert [p.name for p in Path(d).iterdir()] == ["draft_state_mlb.json"]       # temp file removed


def test_try_save_never_raises():
    with tempfile.TemporaryDirectory() as d:
        blocker = Path(d, "data")
        blocker.write_text("a file, not a folder")
        st = make_state()
        assert persist.try_save(st, directory=blocker) is not None
        assert persist.try_save(st, directory=d) is None and st.saved["error"] is None and st.saved["saved_at"]


def test_no_file_is_a_clean_start():
    with tempfile.TemporaryDirectory() as d:
        st = fresh_state()
        res = persist.restore(st, directory=d)
        assert not res["restored"] and res["warning"] is None and st.setup is None and st.draft_log == []


def test_corrupt_file_is_set_aside_not_deleted():
    with tempfile.TemporaryDirectory() as d:
        Path(d, "draft_state_mlb.json").write_text('{"version": 1, "sport": "mlb", "draft_l')    # cut off mid-write
        st = fresh_state()
        res = persist.restore(st, directory=d)
        names = sorted(p.name for p in Path(d).iterdir())
        assert not res["restored"] and "starting fresh" in res["warning"] and st.draft_log == []
        assert len(names) == 1 and ".bad-" in names[0]
        assert Path(d, names[0]).read_text().startswith('{"version"')               # contents kept


def test_wrong_sport_or_version_or_shape_is_set_aside():
    for bad in ({"version": 1, "sport": "nba", "draft_log": []}, {"version": 99, "sport": "mlb"}, [1, 2],
                {"version": 1, "sport": "mlb", "draft_log": "nope"}):
        with tempfile.TemporaryDirectory() as d:
            Path(d, "draft_state_mlb.json").write_text(json.dumps(bad))
            st = fresh_state()
            res = persist.restore(st, directory=d)
            assert not res["restored"] and res["warning"] and not Path(d, "draft_state_mlb.json").exists()


def _write(d, **over):
    st = make_state()
    persist.save(st, directory=d)
    p = Path(d, "draft_state_mlb.json")
    data = json.loads(p.read_text())
    data.update(over)
    p.write_text(json.dumps(data))


def test_bad_picks_are_skipped_but_setup_and_settings_survive():
    cases = {
        "unknown player": [{"pick_number": 1, "player_id": "zzz", "team_id": 1, "source": "manual"}],
        "twice": [{"pick_number": 1, "player_id": "h0", "team_id": 1, "source": "manual"},
                  {"pick_number": 2, "player_id": "h0", "team_id": 2, "source": "manual"}],
        "team range": [{"pick_number": 1, "player_id": "h0", "team_id": 40, "source": "manual"}],
        "order": [{"pick_number": 2, "player_id": "h0", "team_id": 1, "source": "manual"},
                  {"pick_number": 1, "player_id": "h1", "team_id": 2, "source": "manual"}],
        "no player": [{"pick_number": 1, "player_id": None, "team_id": 1, "source": "manual"}],
        "junk": ["hello"],
    }
    for name, log in cases.items():
        with tempfile.TemporaryDirectory() as d:
            _write(d, draft_log=log)
            st = fresh_state()
            res = persist.restore(st, directory=d)
            assert res["picks"] == 0 and st.draft_log == [] and res["setup"] and st.setup.teams == 12, name
            assert "weren't restored" in res["warning"], name
            assert any(".bad-" in p.name for p in Path(d).iterdir()), name          # copy kept for inspection


def test_too_many_picks_for_setup_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        _write(d, setup={"draft_type": "snake", "teams": 2, "my_slot": 1, "rounds": 1, "budget": None},
               draft_log=[{"pick_number": i + 1, "player_id": f"h{i}", "team_id": i % 2 + 1, "source": "manual"}
                          for i in range(3)])
        res = persist.restore(fresh_state(), directory=d)
        assert res["picks"] == 0 and "more than this draft allows" in res["warning"]


def test_live_pick_without_a_matched_player_is_kept():
    with tempfile.TemporaryDirectory() as d:
        st = make_state()
        st.draft_log.append({"pick_number": 4, "player_id": None, "team_id": 4, "source": "live"})
        persist.save(st, directory=d)
        b = fresh_state()
        assert persist.restore(b, directory=d)["picks"] == 4 and b.draft_log[3]["player_id"] is None


def test_invalid_saved_setup_and_settings_fall_back_to_defaults():
    with tempfile.TemporaryDirectory() as d:
        _write(d, setup={"draft_type": "snake", "teams": 99, "my_slot": 5, "rounds": 25, "budget": None})
        st = fresh_state()
        res = persist.restore(st, directory=d)
        assert st.setup is None and not res["setup"] and "wasn't valid" in res["warning"]
    with tempfile.TemporaryDirectory() as d:
        st0 = make_state()
        data = persist.snapshot(st0)
        data["config"]["categories"] = {"H": ["HR", "NOPE"], "P": ["W"]}
        Path(d, "draft_state_mlb.json").write_text(json.dumps(data))
        st = fresh_state()
        res = persist.restore(st, directory=d)
        assert not res["config"] and "defaults are in place" in res["warning"] and st.config == LeagueConfig()
        assert res["picks"] == 3                                                    # the rest is still restored
    with tempfile.TemporaryDirectory() as d:
        data = persist.snapshot(make_state())
        data["config"] = {"roster": "garbage"}
        Path(d, "draft_state_mlb.json").write_text(json.dumps(data))
        res = persist.restore(fresh_state(), directory=d)
        assert not res["config"] and res["picks"] == 3


def test_mock_mode_saves_and_restores_settings_but_not_picks():
    with tempfile.TemporaryDirectory() as d:
        st = make_state()
        persist.save(st, include_picks=False, directory=d)
        assert json.loads(Path(d, "draft_state_mlb.json").read_text())["draft_log"] == []
        persist.save(st, directory=d)
        b = fresh_state()
        res = persist.restore(b, restore_picks=False, directory=d)
        assert res["picks"] == 0 and b.draft_log == [] and b.setup is not None


def test_yahoo_mode_keeps_my_team_from_the_platform():
    with tempfile.TemporaryDirectory() as d:
        persist.save(make_state(), directory=d)
        b = fresh_state()
        b.my_team_id = 9
        persist.restore(b, set_my_team=False, directory=d)
        assert b.my_team_id == 9 and b.setup.my_slot == 5


def test_stale_temp_files_are_cleaned_and_other_sports_are_reported():
    with tempfile.TemporaryDirectory() as d:
        Path(d, "draft_state_mlb.json.abc123.tmp").write_text("half")
        nfl = make_state()
        nfl.config = LeagueConfig.for_sport("nfl")
        persist.save(nfl, directory=d)
        persist.save(GlobalState(config=LeagueConfig.for_sport("nba"), raw_df=RAW), directory=d)   # empty: not reported
        persist.restore(fresh_state(), directory=d)
        assert not Path(d, "draft_state_mlb.json.abc123.tmp").exists()
        assert persist.other_saved_drafts("mlb", directory=d) == ["nfl"]


def test_numpy_values_in_picks_are_saved_as_plain_json():
    import numpy as np
    with tempfile.TemporaryDirectory() as d:
        st = make_state()
        st.draft_log.append({"pick_number": np.int64(4), "player_id": "h2", "team_id": np.int64(4), "source": "live"})
        persist.save(st, directory=d)
        assert json.loads(Path(d, "draft_state_mlb.json").read_text())["draft_log"][3]["team_id"] == 4


def test_sync_calls_on_change_only_when_the_log_changes():
    from app import sync
    from tests.test_math import _FakeRemote
    st = GlobalState()
    st.names = {"h1": "One", "h2": "Two"}
    remote = _FakeRemote([("y1", 1)])
    calls = []
    def boom():
        calls.append(1)
        if len(calls) == 2:
            raise OSError("disk")                         # a failing save must not stop the sync
    async def go():
        t = asyncio.create_task(sync.run(st, lambda: remote, lambda: {"y1": "h1", "y2": "h2"}, 0.01,
                                         retry_delay=0.01, on_change=boom))
        await asyncio.sleep(0.15)
        assert len(calls) == 1                            # many polls, one change
        remote.picks.append(("y2", 2)); await asyncio.sleep(0.1)
        assert len(calls) == 2 and len(st.draft_log) == 2 and st.sync["connected"]
        t.cancel()
    asyncio.run(go())


# ---------- the endpoints (needs the project's data files; skipped quietly if there are none) ----------
def _have_data() -> bool:
    return Path(persist.DATA_DIR, "hitters.csv").exists()


def test_endpoints_save_and_a_restart_brings_everything_back():
    if not _have_data():
        return
    from fastapi import HTTPException
    from app import main
    from app.state import state
    with tempfile.TemporaryDirectory() as d:
        orig = persist.DATA_DIR
        persist.DATA_DIR = Path(d)
        try:
            main.load_sport("mlb")
            state.sync["platform"] = "manual"
            main.put_setup({"draft_type": "snake", "teams": 10, "my_slot": 4, "rounds": ""})
            f = Path(d, "draft_state_mlb.json")
            assert f.exists() and json.loads(f.read_text())["setup"]["teams"] == 10
            ids = list(state.raw_df["player_id"][:3])
            main.pick(main.PickRequest(player_id=ids[0]))
            main.pick(main.PickRequest(player_id=ids[1], team_id=2))
            main.pick(main.PickRequest(player_id=ids[2]))
            assert [p["player_id"] for p in json.loads(f.read_text())["draft_log"]] == ids
            main.undo()
            assert len(json.loads(f.read_text())["draft_log"]) == 2
            assert main.draft_saved()["error"] is None and main.get_setup()["saved"]["saved_at"]

            # "crash": the process forgets everything, then starts again
            saved_log = [dict(p) for p in state.draft_log]
            main.load_sport("mlb")
            assert state.setup is None and state.draft_log == []
            main._restore("manual")
            assert state.draft_log == saved_log and state.setup.teams == 10 and state.my_team_id == 4
            assert state.config.roster.teams == 10
            assert main.draft_status()["picks_made"] == 2 and main.draft_saved()["picks_restored"] == 2

            assert main.reset_draft()["cleared"] == 2
            assert json.loads(f.read_text())["draft_log"] == [] and json.loads(f.read_text())["setup"]["teams"] == 10

            main.pick(main.PickRequest(player_id=ids[0]))
            if Path(orig, "nba", "players.csv").exists():
                main.set_sport({"sport": "nba"})
                assert not f.exists() and Path(d, "draft_state_nba.json").exists()
                assert state.setup is None and state.draft_log == []
            try:
                main.put_setup({"draft_type": "snake", "teams": "abc", "my_slot": 1})
            except HTTPException as e:
                assert e.status_code == 400
        finally:
            persist.DATA_DIR = orig
            main.load_sport("mlb")


"""The live blend: own model <-> reference list, in memory, on top of which what-if overrides still apply."""
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import make_sample_sport
from app import ingestion, persist
from app.live_blend import blend_frame

CAT = ["GP", "MIN", "PTS", "REB"]


def _frame(names, scale):
    return pd.DataFrame({"name": names, "player_id": ["n:" + n.lower() for n in names], "eligible_positions": [["PG"]] * len(names),
                         "GP": [60.0 * scale] * len(names), "MIN": [2000.0 * scale] * len(names), "PTS": [1500.0 * scale] * len(names),
                         "REB": [400.0 * scale] * len(names)})


def test_blend_frame_is_the_midpoint_zero_is_own_one_is_reference_and_labels_stay():
    own, ref = _frame(["Ann", "Bob", "Cy"], 1.0), _frame(["Ann", "Bob"], 2.0)
    half = blend_frame(own, ref, 0.5, CAT)
    assert half.loc[0, "PTS"] == pytest.approx(2250) and half.loc[2, "PTS"] == 1500          # Cy is not in the reference: unchanged
    assert blend_frame(own, ref, 0.0, CAT)[CAT].equals(own[CAT].astype(float))
    assert blend_frame(own, ref, 1.0, CAT).loc[1, "REB"] == pytest.approx(800)
    assert list(half["eligible_positions"][0]) == ["PG"] and list(half["name"]) == ["Ann", "Bob", "Cy"]
    assert own["PTS"].iloc[0] == 1500                                                        # the inputs are never modified


def test_blend_frame_matches_names_ignoring_accents_and_rejects_bad_weights():
    own, ref = _frame(["Nikola Jokić"], 1.0), _frame(["Nikola Jokic"], 3.0)
    assert blend_frame(own, ref, 1.0, CAT).loc[0, "PTS"] == pytest.approx(4500)
    for bad in (-0.1, 1.1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            blend_frame(own, ref, bad, CAT)


@contextmanager
def api(tmp, monkeypatch, with_blend=True, multi=False):
    for k, v in {"SPORT": "nba", "LEAGUE_PRESET": "", "DRAFT_PLATFORM": "manual", "MY_TEAM_ID": "1", "MULTI_SESSION": "1" if multi else ""}.items():
        monkeypatch.setenv(k, v)
    old_i, old_p = ingestion.DATA_DIR, persist.DATA_DIR
    ingestion.DATA_DIR = persist.DATA_DIR = Path(tmp)
    try:
        nba = Path(tmp) / "nba"
        make_sample_sport.generate("nba", nba, n=300)
        if with_blend:                                                     # own = the sample; reference = the same players, 40% stronger
            own = pd.read_csv(nba / "players.csv")
            own.to_csv(nba / "players_own.csv", index=False)
            ref = own.copy()
            for c in ("PTS", "REB", "AST", "STL", "BLK", "3PM", "FGM", "FGA", "FTM", "FTA"):
                ref[c] = ref[c] * np.where(np.arange(len(ref)) % 2 == 0, 1.4, 0.8)       # stronger for even rows, weaker for odd rows
            ref.to_csv(nba / "players_third_party.csv", index=False)
            (nba / "blend.json").write_text(json.dumps({"weight": 0.0}), encoding="utf-8")
        from app.main import app
        with TestClient(app) as c:
            yield c
    finally:
        ingestion.DATA_DIR, persist.DATA_DIR = old_i, old_p


def _board(c):
    rows = c.get("/players/rankings?limit=300").json()
    rows = rows if isinstance(rows, list) else next(v for v in rows.values() if isinstance(v, list))
    return {r["name"]: r for r in rows}


def test_moving_the_slider_changes_the_board_and_returning_to_zero_restores_it(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as c:
        info = c.get("/projections/blend").json()
        assert info["available"] and info["weight"] == 0.0
        base = {n: r["projected_rank"] for n, r in _board(c).items()}
        assert c.put("/projections/blend", json={"weight": 1.0}).status_code == 200
        moved = {n: r["projected_rank"] for n, r in _board(c).items()}
        assert moved != base and sum(moved[n] != base[n] for n in base) > 50
        assert c.put("/projections/blend", json={"weight": 0.0}).status_code == 200
        assert {n: r["projected_rank"] for n, r in _board(c).items()} == base


def test_blend_and_player_overrides_stack_and_each_can_be_undone(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as c:
        star = next(iter(_board(c)))
        c.put("/players/override", json={"player": star, "games": 30})
        c.put("/projections/blend", json={"weight": 0.5})
        assert c.get("/players/explain", params={"q": star}).json()["projected_games"] == 30     # the override survived the blend change
        c.delete("/players/overrides")
        assert c.get("/players/explain", params={"q": star}).json()["projected_games"] > 30
        assert c.get("/projections/blend").json()["weight"] == 0.5                            # and the blend survived clearing overrides


def test_a_blend_change_flows_into_my_team_score(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as c:
        c.put("/draft/setup", json={"draft_type": "snake", "teams": 12, "my_slot": 1})
        first = next(iter(_board(c)))
        c.post("/draft/pick", json={"player_id": _board(c)[first]["player_id"], "team_id": 1})
        s0 = c.get("/team/summary").json()["score"]
        c.put("/projections/blend", json={"weight": 1.0})
        assert c.get("/team/summary").json()["score"] != pytest.approx(s0)


def test_bad_weights_and_missing_files_get_clear_messages(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as c:
        for bad in (-0.2, 1.5):
            r = c.put("/projections/blend", json={"weight": bad})
            assert r.status_code == 400 and "0 to 1" in str(r.json())
        assert c.put("/projections/blend", json={"weight": "abc"}).status_code == 422
        assert c.get("/projections/blend").json()["weight"] == 0.0                              # nothing half-applied
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch, with_blend=False) as c:
        assert c.get("/projections/blend").json()["available"] is False
        r = c.put("/projections/blend", json={"weight": 0.5})
        assert r.status_code == 400 and "build_own_model" in str(r.json())


def test_the_chosen_weight_survives_a_restart_in_single_user_mode(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        with api(d, monkeypatch) as c:
            c.put("/projections/blend", json={"weight": 0.35})
            ranks = {n: r["projected_rank"] for n, r in _board(c).items()}
        with api(d, monkeypatch) as c2:
            assert c2.get("/projections/blend").json()["weight"] == pytest.approx(0.35)
            assert {n: r["projected_rank"] for n, r in _board(c2).items()} == ranks


def test_hosted_sessions_have_their_own_blend_and_write_nothing(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch, multi=True) as a:
        from app.main import app
        b = TestClient(app)
        a.get("/"); b.get("/")
        a.put("/projections/blend", json={"weight": 1.0})
        assert a.get("/projections/blend").json()["weight"] == 1.0 and b.get("/projections/blend").json()["weight"] == 0.0
        assert {n: r["projected_rank"] for n, r in _board(a).items()} != {n: r["projected_rank"] for n, r in _board(b).items()}
        assert not list(Path(d).glob("blend_*.json"))


def test_a_stale_own_file_switches_the_blend_off_instead_of_breaking(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        for k, v in {"SPORT": "nba", "LEAGUE_PRESET": "", "DRAFT_PLATFORM": "manual", "MULTI_SESSION": ""}.items():
            monkeypatch.setenv(k, v)
        old_i, old_p = ingestion.DATA_DIR, persist.DATA_DIR
        ingestion.DATA_DIR = persist.DATA_DIR = Path(d)
        try:
            nba = Path(d) / "nba"
            make_sample_sport.generate("nba", nba, n=300)
            own = pd.read_csv(nba / "players.csv")
            own.head(250).to_csv(nba / "players_own.csv", index=False)                          # a different player set than players.csv
            own.to_csv(nba / "players_third_party.csv", index=False)
            (nba / "blend.json").write_text('{"weight": 0.2}')
            from app.main import app
            with TestClient(app) as c:
                assert c.get("/projections/blend").json()["available"] is False
                assert c.get("/health/data").json()["ok"] is True
        finally:
            ingestion.DATA_DIR, persist.DATA_DIR = old_i, old_p


def test_when_the_blend_is_unavailable_the_reason_names_the_missing_files(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch, with_blend=False) as c:
        b = c.get("/projections/blend").json()
        assert b["available"] is False and b["relevant"] is True
        for name in ("blend.json", "players_own.csv", "players_third_party.csv"):
            assert name in b["message"]
        assert "build_own_model.py --blend 0.5" in b["message"]
        assert name in str(c.put("/projections/blend", json={"weight": 0.5}).json())


def test_a_stale_own_file_explains_itself(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        for k, v in {"SPORT": "nba", "LEAGUE_PRESET": "", "DRAFT_PLATFORM": "manual", "MULTI_SESSION": ""}.items():
            monkeypatch.setenv(k, v)
        old_i, old_p = ingestion.DATA_DIR, persist.DATA_DIR
        ingestion.DATA_DIR = persist.DATA_DIR = Path(d)
        try:
            nba = Path(d) / "nba"
            make_sample_sport.generate("nba", nba, n=300)
            own = pd.read_csv(nba / "players.csv")
            own.head(250).to_csv(nba / "players_own.csv", index=False)
            own.to_csv(nba / "players_third_party.csv", index=False)
            (nba / "blend.json").write_text('{"weight": 0.2}')
            from app.main import app
            with TestClient(app) as c:
                msg = c.get("/projections/blend").json()["message"]
                assert "doesn't match players.csv" in msg and "Rebuild" in msg
        finally:
            ingestion.DATA_DIR, persist.DATA_DIR = old_i, old_p

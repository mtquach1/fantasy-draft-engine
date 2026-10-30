"""Live per-player overrides: an edit must flow through every value, rank and team score, and be undoable and persistent."""
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import make_sample_sport
from app import ingestion, persist


@contextmanager
def api(tmp, monkeypatch):
    for k, v in {"SPORT": "nba", "LEAGUE_PRESET": "", "DRAFT_PLATFORM": "manual", "MY_TEAM_ID": "1"}.items():
        monkeypatch.setenv(k, v)
    old_i, old_p = ingestion.DATA_DIR, persist.DATA_DIR
    ingestion.DATA_DIR = persist.DATA_DIR = Path(tmp)
    try:
        make_sample_sport.generate("nba", Path(tmp) / "nba", n=300) if not (Path(tmp) / "nba" / "players.csv").exists() else None
        from app.main import app
        with TestClient(app) as c:
            yield c
    finally:
        ingestion.DATA_DIR, persist.DATA_DIR = old_i, old_p


def _board(c):
    rows = c.get("/players/rankings?limit=300").json()
    rows = rows if isinstance(rows, list) else next(v for v in rows.values() if isinstance(v, list))
    return {r["name"]: r for r in rows}


def _setup_and_draft(c, name):
    assert c.put("/draft/setup", json={"draft_type": "snake", "teams": 12, "my_slot": 1}).status_code == 200
    pid = _board(c)[name]["player_id"]
    assert c.post("/draft/pick", json={"player_id": pid, "team_id": 1}).status_code == 200
    return pid


def test_cutting_games_lowers_value_and_my_team_score_and_clearing_restores_both(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as c:
        star = next(iter(_board(c)))                                   # the board's best player
        pid = _setup_and_draft(c, star)
        before_board, before_score = _board(c), c.get("/team/summary").json()["score"]
        r = c.put("/players/override", json={"player": star, "games": 30})
        assert r.status_code == 200, r.text
        after_score = c.get("/team/summary").json()["score"]
        assert after_score < before_score                              # my roster is worth less: the score moved with the edit
        assert c.get("/players/explain", params={"q": star}).json()["projected_games"] == 30
        assert c.delete(f"/players/override/{pid}").status_code == 200
        assert c.get("/team/summary").json()["score"] == pytest.approx(before_score)
        assert _board(c)[star]["projected_rank"] == before_board[star]["projected_rank"]


def test_the_pace_benchmark_and_scorer_are_rebuilt_not_served_stale(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as c:
        star = next(iter(_board(c)))
        _setup_and_draft(c, star)
        from app import main
        v0 = main.state.data_version
        sc0 = main.predraft_scorer()
        c.put("/players/override", json={"player": star, "mpg": 20})
        assert main.state.data_version > v0 and main.predraft_scorer() is not sc0
        # the star plays far fewer minutes, so the fixed pre-draft ruler must now score him lower
        assert main.predraft_board().set_index("name").loc[star, "score"] < sc0.pre.set_index("name").loc[star, "score"]


def test_bad_overrides_are_refused_with_clear_messages(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as c:
        star = next(iter(_board(c)))
        cases = [({"player": star, "games": 0}, 400, "between 1 and 82"), ({"player": star, "games": 83}, 400, "between 1 and 82"),
                 ({"player": star, "mpg": 60}, 400, "between 1 and 48"), ({"player": star}, 400, "games, mpg"),
                 ({"player": "Nobody Atall", "games": 40}, 404, "No player matches"), ({"player": "", "games": 40}, 400, "Say which player"),
                 ({"player": "Young", "games": 40}, 400, "matches")]
        for body, code, msg in cases:
            r = c.put("/players/override", json=body)
            assert r.status_code == code and msg in str(r.json()), (body, r.text)
        assert c.get("/players/overrides").json()["overrides"] == []        # nothing half-applied


def test_overrides_survive_a_restart_and_clear_removes_the_file(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        with api(d, monkeypatch) as c:
            star = next(iter(_board(c)))
            c.put("/players/override", json={"player": star, "games": 45, "note": "knee"})
            cut_rank = _board(c)[star]["projected_rank"]
            assert (Path(d) / "overrides_nba.json").exists()
        with api(d, monkeypatch) as c2:                                    # a fresh start reads the saved file
            rows = c2.get("/players/overrides").json()["overrides"]
            assert len(rows) == 1 and rows[0]["games"] == 45 and rows[0]["note"] == "knee"
            assert _board(c2)[star]["projected_rank"] == cut_rank
            assert c2.delete("/players/overrides").json()["cleared"] == 1
            assert not (Path(d) / "overrides_nba.json").exists()

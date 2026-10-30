"""The 'sort by rank' option, the always-fresh page, and the slow-request log."""
import tempfile
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import make_sample_sport
from app import ingestion, persist


@contextmanager
def api(tmp, monkeypatch):
    for k, v in {"SPORT": "nba", "LEAGUE_PRESET": "", "DRAFT_PLATFORM": "manual", "MULTI_SESSION": ""}.items():
        monkeypatch.setenv(k, v)
    old_i, old_p = ingestion.DATA_DIR, persist.DATA_DIR
    ingestion.DATA_DIR = persist.DATA_DIR = Path(tmp)
    try:
        make_sample_sport.generate("nba", Path(tmp) / "nba", n=300)
        from app.main import app
        with TestClient(app) as c:
            yield c
    finally:
        ingestion.DATA_DIR, persist.DATA_DIR = old_i, old_p


def test_sorting_by_rank_orders_by_overall_rank_and_value_stays_the_default(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as c:
        by_rank = c.get("/players/available?limit=60&sort=rank").json()
        ranks = [p["projected_rank"] for p in by_rank]
        assert ranks == sorted(ranks) and ranks[0] == 1                        # best overall player first, strictly in rank order
        # Before any pick the two orders coincide (rank comes from value). Once the draft moves, replacement levels shift
        # and value over replacement reorders the board while the overall rank stays fixed.
        c.put("/draft/setup", json={"draft_type": "snake", "teams": 12, "my_slot": 1})
        for p in c.get("/players/available?limit=40&sort=rank").json()[:30]:
            c.post("/draft/pick", json={"player_id": p["player_id"]})
        by_value = c.get("/players/available?limit=60").json()
        by_rank = c.get("/players/available?limit=60&sort=rank").json()
        vorps = [p["vorp"] for p in by_value]
        assert vorps == sorted(vorps, reverse=True)                           # the default is unchanged: value over replacement
        assert c.get("/players/available?limit=60&sort=value").json() == by_value
        ranks = [p["projected_rank"] for p in by_rank]
        assert ranks == sorted(ranks) and ranks[0] == 31                      # 30 are gone: the best left is overall #31
        assert [p["player_id"] for p in by_rank] != [p["player_id"] for p in by_value]      # the two orders now genuinely differ


def test_rank_sort_respects_drafted_players_position_filter_and_search(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as c:
        c.put("/draft/setup", json={"draft_type": "snake", "teams": 12, "my_slot": 1})
        first = c.get("/players/available?limit=1&sort=rank").json()[0]
        c.post("/draft/pick", json={"player_id": first["player_id"], "team_id": 1})
        again = c.get("/players/available?limit=5&sort=rank").json()
        assert first["player_id"] not in [p["player_id"] for p in again]       # drafted players leave the list in either order
        assert [p["projected_rank"] for p in again] == sorted(p["projected_rank"] for p in again)
        centers = c.get("/players/available?limit=20&sort=rank&position=C").json()
        assert centers and all("C" in p["eligible_positions"] for p in centers)
        assert [p["projected_rank"] for p in centers] == sorted(p["projected_rank"] for p in centers)
        name = again[2]["name"]
        hit = c.get("/players/available", params={"q": name.split()[0], "sort": "rank"}).json()
        assert name in [p["name"] for p in hit]


def test_a_bad_sort_value_is_refused_with_a_message(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as c:
        r = c.get("/players/available?sort=alphabet")
        assert r.status_code == 400 and "sort must be" in str(r.json())


def test_the_page_is_never_served_from_a_stale_browser_cache(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as c:
        r = c.get("/")
        assert r.status_code == 200 and "no-cache" in r.headers["cache-control"] and 'id="sortBy"' in r.text and 'id="blendBox"' in r.text


def test_slow_requests_are_named_in_the_server_window(monkeypatch, capsys):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as c:
        monkeypatch.setenv("SLOW_REQUEST_SECONDS", "0")
        c.get("/health/data")
        assert "[slow] GET /health/data took" in capsys.readouterr().out
        monkeypatch.setenv("SLOW_REQUEST_SECONDS", "9999")
        c.get("/health/data")
        assert "[slow]" not in capsys.readouterr().out

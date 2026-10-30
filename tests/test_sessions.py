"""Per-browser sessions (hosted mode): drafts must be private, nothing may touch the disk, and single-user mode must be unchanged."""
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import make_sample_sport
from app import ingestion, persist


@contextmanager
def host(tmp, monkeypatch, multi=True, max_sessions=100):
    env = {"SPORT": "nba", "LEAGUE_PRESET": "", "DRAFT_PLATFORM": "manual", "MY_TEAM_ID": "1",
           "MULTI_SESSION": "1" if multi else "", "MAX_SESSIONS": str(max_sessions)}
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    old_i, old_p = ingestion.DATA_DIR, persist.DATA_DIR
    ingestion.DATA_DIR = persist.DATA_DIR = Path(tmp)
    try:
        make_sample_sport.generate("nba", Path(tmp) / "nba", n=300)
        make_sample_sport.generate("nfl", Path(tmp) / "nfl", n=300)
        from app.main import app
        with TestClient(app) as first:
            yield first, lambda: TestClient(app)               # more browsers: new clients have their own cookie jar
    finally:
        ingestion.DATA_DIR, persist.DATA_DIR = old_i, old_p


def _players(c, n=3):
    rows = c.get("/players/rankings?limit=300").json()
    rows = rows if isinstance(rows, list) else next(v for v in rows.values() if isinstance(v, list))
    return rows[:n]


def test_two_browsers_have_separate_drafts_setups_and_resets(monkeypatch):
    with tempfile.TemporaryDirectory() as d, host(d, monkeypatch) as (a, new_client):
        b = new_client()
        assert a.put("/draft/setup", json={"draft_type": "snake", "teams": 12, "my_slot": 3}).status_code == 200
        assert b.put("/draft/setup", json={"draft_type": "snake", "teams": 8, "my_slot": 2}).status_code == 200
        pid = _players(a)[0]["player_id"]
        assert a.post("/draft/pick", json={"player_id": pid, "team_id": 1}).status_code == 200
        assert len(a.get("/draft/log").json()) == 1 and b.get("/draft/log").json() == []
        assert a.get("/draft/setup").json()["setup"]["teams"] == 12 and b.get("/draft/setup").json()["setup"]["teams"] == 8
        assert b.post("/draft/pick", json={"player_id": pid, "team_id": 1}).status_code == 200      # B may take the same player: B's pool is B's own
        assert a.post("/draft/reset").json()["cleared"] == 1
        assert len(b.get("/draft/log").json()) == 1                                                # A's reset did not touch B
        assert a.get("/session").json()["active_sessions"] == 2 and a.get("/session").json()["mode"] == "multi"


def test_a_player_override_in_one_session_does_not_change_anyone_elses_board(monkeypatch):
    with tempfile.TemporaryDirectory() as d, host(d, monkeypatch) as (a, new_client):
        b = new_client()
        star = _players(a)[0]["name"]
        before = {r["name"]: r["projected_rank"] for r in _players(b, 300)}
        assert a.put("/players/override", json={"player": star, "games": 20}).status_code == 200
        assert {r["name"]: r["projected_rank"] for r in _players(a, 300)}[star] > before[star]
        assert {r["name"]: r["projected_rank"] for r in _players(b, 300)} == before
        assert b.get("/players/overrides").json()["overrides"] == []


def test_switching_sport_in_one_session_leaves_the_other_alone(monkeypatch):
    with tempfile.TemporaryDirectory() as d, host(d, monkeypatch) as (a, new_client):
        b = new_client()
        r = a.put("/league/sport", json={"sport": "nfl"})
        assert r.status_code == 200, r.text
        assert a.get("/league/config").json()["sport"] == "nfl" and b.get("/league/config").json()["sport"] == "nba"


def test_hosted_sessions_never_write_to_disk(monkeypatch):
    with tempfile.TemporaryDirectory() as d, host(d, monkeypatch) as (a, new_client):
        a.put("/draft/setup", json={"draft_type": "snake", "teams": 12, "my_slot": 1})
        a.post("/draft/pick", json={"player_id": _players(a)[0]["player_id"], "team_id": 1})
        a.put("/players/override", json={"player": _players(a)[1]["name"], "games": 30})
        assert not list(Path(d).glob("draft_state_*")) and not list(Path(d).glob("overrides_*"))


def test_a_forged_or_unknown_cookie_just_gets_a_fresh_session(monkeypatch):
    with tempfile.TemporaryDirectory() as d, host(d, monkeypatch) as (a, new_client):
        b = new_client()
        b.cookies.set("draft_session", "not-a-real-session-id-" + "x" * 300)
        r = b.get("/draft/log")
        assert r.status_code == 200 and r.json() == [] and "draft_session" in r.headers.get("set-cookie", "")


def test_the_least_recently_used_session_is_dropped_when_the_host_is_full(monkeypatch):
    with tempfile.TemporaryDirectory() as d, host(d, monkeypatch, max_sessions=2) as (a, new_client):
        b, c = new_client(), new_client()
        a.put("/draft/setup", json={"draft_type": "snake", "teams": 12, "my_slot": 3})
        b.get("/draft/log")
        c.get("/draft/log")                                    # three browsers, room for two: A (least recent) is evicted
        assert a.get("/session").json()["active_sessions"] == 2
        assert a.get("/draft/setup").json()["setup"] is None   # A starts over with a fresh, empty draft (and a new cookie)


def test_simultaneous_drafting_in_two_sessions_does_not_mix_picks(monkeypatch):
    with tempfile.TemporaryDirectory() as d, host(d, monkeypatch) as (a, new_client):
        b = new_client()
        for c in (a, b):
            assert c.put("/draft/setup", json={"draft_type": "snake", "teams": 12, "my_slot": 1}).status_code == 200
        ids = [p["player_id"] for p in _players(a, 40)]
        errors = []

        def draft(client, mine):
            for k, pid in enumerate(mine):
                r = client.post("/draft/pick", json={"player_id": pid, "team_id": (k % 12) + 1})
                if r.status_code != 200:
                    errors.append(r.text)

        ta = threading.Thread(target=draft, args=(a, ids[:20])); tb = threading.Thread(target=draft, args=(b, ids[20:]))
        ta.start(); tb.start(); ta.join(); tb.join()
        assert not errors
        assert [p["player_id"] for p in a.get("/draft/log").json()] == ids[:20]
        assert [p["player_id"] for p in b.get("/draft/log").json()] == ids[20:]


def test_single_user_mode_is_unchanged_one_draft_and_no_cookie(monkeypatch):
    with tempfile.TemporaryDirectory() as d, host(d, monkeypatch, multi=False) as (a, new_client):
        b = new_client()
        info = a.get("/session").json()
        assert info["mode"] == "single" and info["saved_to_disk"] is True
        r = a.put("/draft/setup", json={"draft_type": "snake", "teams": 12, "my_slot": 3})
        assert r.status_code == 200 and "set-cookie" not in r.headers
        assert b.get("/draft/setup").json()["setup"]["teams"] == 12          # everyone shares the one draft, as before
        assert list(Path(d).glob("draft_state_nba.json"))                    # and it is saved to disk


def test_health_checks_and_docs_do_not_create_sessions_but_the_page_does(monkeypatch):
    with tempfile.TemporaryDirectory() as d, host(d, monkeypatch) as (a, new_client):
        probe = new_client()
        for _ in range(5):                                       # a host's health check, every few seconds, with no cookie
            assert probe.get("/health/data").status_code == 200
        probe.get("/docs"); probe.get("/openapi.json")
        assert a.get("/session").json()["active_sessions"] == 0
        r = probe.get("/")                                       # opening the page is what starts a session
        assert r.status_code == 200 and "draft_session=" in r.headers.get("set-cookie", "")
        assert probe.get("/session").json()["active_sessions"] == 1

"""The event endpoints against a running app on sample NBA data: values really move, retries change nothing, one batch is one recompute,
and hosted sessions stay private and never write a database file."""
import tempfile
import threading
from pathlib import Path

import pytest

from app.own_model import norm_name
from pipeline.ingest import ingest_raw_dir
from tests.pipeline_helpers import api, board, br_frame, explain, write_br


def ev(i, typ="games_out", player=None, value=10, ts="2026-10-28T12:00:00Z", **kw):
    return {"event_id": f"t{i}", "type": typ, "player": player, "value": value, "ts": ts, **kw}


def names(c, n=3):
    return [r["name"] for r in board(c)[:n]]


def team_of(main, name):
    df = main.state.raw_df
    return df.loc[df["name"] == name, "team"].iloc[0]


def post(c, events):
    r = c.post("/events", json={"events": events})
    assert r.status_code == 200, r.text
    return r.json()


def test_games_out_lowers_projected_games_and_value_and_clearing_restores_both(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        star = names(c, 1)[0]; base = explain(c, star)
        r = post(c, [ev(1, player=star, value=30)])
        assert r["accepted"] == ["t1"] and r["recomputes"] == 1
        hurt = explain(c, star)
        assert hurt["projected_games"] == base["projected_games"] - 30 and hurt["score"] < base["score"]
        assert c.delete("/events").json() == {"cleared": 1}
        again = explain(c, star)
        assert again["projected_games"] == base["projected_games"] and again["score"] == pytest.approx(base["score"])


def test_a_retried_batch_changes_nothing_and_does_not_recompute(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        star = names(c, 1)[0]
        batch = [ev(1, player=star, value=20)]
        first = post(c, batch); v = main.state.data_version
        second = post(c, batch)
        assert second["accepted"] == [] and second["duplicates"] == ["t1"] and second["recomputes"] == 0
        assert main.state.data_version == v == first["data_version"]
        assert c.get("/events").json()["count"] == 1


def test_a_batch_of_ten_events_is_one_recompute(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        ns = names(c, 10); v0 = main.state.data_version
        r = post(c, [ev(i, player=n, value=5 + i, ts=f"2026-10-28T12:00:{i:02d}Z") for i, n in enumerate(ns)])
        assert len(r["accepted"]) == 10 and r["recomputes"] == 1
        assert main.state.data_version == v0 + 1                        # exactly one new version for ten events


def test_a_late_event_with_an_older_timestamp_cannot_overwrite_newer_news(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        star = names(c, 1)[0]; base = explain(c, star)["projected_games"]
        post(c, [ev(1, player=star, value=40, ts="2026-10-28T12:00:00Z")])
        post(c, [ev(2, player=star, value=5, ts="2026-10-27T12:00:00Z")])      # arrives later, happened earlier
        assert explain(c, star)["projected_games"] == base - 40
        assert c.get("/events/effective").json()["players"][0]["games"] == {"kind": "out", "n": 40}


def test_partial_rejection_applies_the_good_events_and_explains_each_bad_one(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        a, b = names(c, 2); base = explain(c, a)["projected_games"]
        main.state.names["dup-pid"] = b                                         # a second player with the same name -> ambiguous
        try:
            r = post(c, [ev(1, player=a, value=10),
                         ev(2, player="Nobody Atall", value=3),                       # unknown
                         ev(3, player=b, value=3),                                    # ambiguous
                         ev(4, "mpg", player=a, value=99),                            # bad value
                         {"type": "mpg", "player": a, "value": 30},                   # missing id
                         ev(7, player=a, value=2.5)])                                 # fractional games
        finally:
            main.state.names.pop("dup-pid", None)
        assert r["accepted"] == ["t1"] and r["recomputes"] == 1
        reasons = {x["event_id"]: x["reason"] for x in r["rejected"]}
        assert "No player matches" in reasons["t2"] and "matches 2 players" in reasons["t3"] and "between 1 and 48" in reasons["t4"]
        assert "event_id is required" in reasons["#5"] and "whole number" in reasons["t7"]
        assert explain(c, a)["projected_games"] == base - 10                    # the good one landed


def test_an_item_that_is_not_an_object_refuses_the_whole_batch_and_stores_nothing(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        star = names(c, 1)[0]; v = main.state.data_version
        r = c.post("/events", json={"events": [ev(1, player=star, value=10), "not an object"]})
        assert r.status_code == 422 and "valid dictionary" in r.text
        assert c.get("/events").json()["count"] == 0 and main.state.data_version == v


def test_batch_size_limits_are_enforced(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        star = names(c, 1)[0]
        assert c.post("/events", json={"events": []}).status_code == 400
        too_many = [ev(i, "mpg", player=star, value=30) for i in range(501)]
        assert c.post("/events", json={"events": too_many}).status_code == 400
        assert c.get("/events").json()["count"] == 0                            # a refused batch stores nothing


def test_events_listing_limit_is_validated(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        assert c.get("/events", params={"limit": 0}).status_code == 400
        assert c.get("/events", params={"limit": 501}).status_code == 400


def test_a_manual_override_beats_an_event_and_removing_it_lets_the_event_show_again(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        star = names(c, 1)[0]; base = explain(c, star)["projected_games"]
        pid = board(c)[0]["player_id"]
        post(c, [ev(1, player=star, value=10)])
        assert c.put("/players/override", json={"player": star, "games": 30}).status_code == 200
        assert explain(c, star)["projected_games"] == 30                        # the manual what-if wins
        assert c.delete(f"/players/override/{pid}").status_code == 200
        assert explain(c, star)["projected_games"] == base - 10                 # the event was still there underneath


def test_a_team_event_relabels_the_player_and_the_history_answers_by_date(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        star = names(c, 1)[0]; before = team_of(main, star)
        post(c, [ev(1, "team", player=star, value="PHI", ts="2026-10-30T00:00:00Z", source="news")])
        assert team_of(main, star) == "PHI"
        h = c.get("/pipeline/team-history", params={"player": star, "on": "2026-10-29T00:00:00Z"}).json()
        assert h["team_on"] is None and h["history"][0]["team"] == "PHI"       # before the trade there is no recorded fact
        h = c.get("/pipeline/team-history", params={"player": star, "on": "2026-10-31T00:00:00Z"}).json()
        assert h["team_on"] == "PHI" and h["name"] == star
        c.delete("/events")
        assert team_of(main, star) == before                                    # clearing events undoes the label too
        assert c.get("/pipeline/team-history", params={"player": star}).json()["history"] == []


def test_team_history_rejects_unknown_players_and_bad_dates(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        star = names(c, 1)[0]
        assert c.get("/pipeline/team-history", params={"player": "Nobody Atall"}).status_code == 400
        assert c.get("/pipeline/team-history", params={"player": star, "on": "last tuesday"}).status_code == 400


def test_a_basketball_reference_id_resolves_through_the_crosswalk(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        star = names(c, 1)[0]; base = explain(c, star)["projected_games"]
        main._store().link_id("br", "zzz01", norm_name(star))
        r = post(c, [ev(1, player="br:zzz01", value=12), ev(2, player="br:unlinked", value=12)])
        assert r["accepted"] == ["t1"] and "No player is linked" in r["rejected"][0]["reason"]
        assert explain(c, star)["projected_games"] == base - 12


def test_snapshots_stored_by_ingest_are_listed_by_the_app(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        raw = tmp_path / "raw"
        for y in (2025, 2026):
            write_br(raw / f"per_game_{y}.csv", br_frame(shift=y - 2025))
        ingest_raw_dir(main._store(), raw)
        snaps = c.get("/pipeline/snapshots").json()["snapshots"]
        assert sorted((s["kind"], s["current"]) for s in snaps) == [("per_game_2025", True), ("per_game_2026", True)]


def test_events_survive_a_restart_in_single_user_mode(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        star = names(c, 1)[0]; base = explain(c, star)["projected_games"]
        post(c, [ev(1, player=star, value=25)])
    assert (tmp_path / "pipeline.sqlite").exists()
    with api(tmp_path, monkeypatch) as (c, main):                               # a new process, same folder
        assert explain(c, star)["projected_games"] == base - 25
        assert c.get("/events").json()["count"] == 1


def test_hosted_sessions_are_private_and_write_no_database_file(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch, multi=True) as (a, main):
        from fastapi.testclient import TestClient
        b = TestClient(main.app)
        star = names(a, 1)[0]; base = explain(a, star)["projected_games"]
        post(a, [ev(1, player=star, value=30)])
        assert a.get("/events").json()["count"] == 1
        assert b.get("/events").json()["count"] == 0                           # the other browser sees nothing
        assert explain(b, star)["projected_games"] == base                     # and its projections are untouched
        assert explain(a, star)["projected_games"] == base - 30
    assert not (tmp_path / "pipeline.sqlite").exists()


def test_the_event_endpoints_refuse_other_sports(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch, sport="nfl") as (c, main):
        r = c.post("/events", json={"events": [ev(1, player="x")]})
        assert r.status_code == 400 and "basketball" in r.text


def test_many_threads_posting_the_same_events_store_each_exactly_once(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        ns = names(c, 5); v0 = main.state.data_version
        batch = [ev(i, player=n, value=10) for i, n in enumerate(ns)]
        out, barrier = [], threading.Barrier(8)

        def go():
            barrier.wait()
            out.append(c.post("/events", json={"events": batch}).json())

        ts = [threading.Thread(target=go) for _ in range(8)]
        [t.start() for t in ts]; [t.join() for t in ts]
        assert len(out) == 8
        assert sum(len(r["accepted"]) for r in out) == 5 and sum(len(r["duplicates"]) for r in out) == 35
        assert sorted(e for r in out for e in r["accepted"]) == sorted(e["event_id"] for e in batch)
        assert c.get("/events").json()["count"] == 5
        assert main.state.data_version == v0 + sum(r["recomputes"] for r in out)    # a version only moves when something was recomputed

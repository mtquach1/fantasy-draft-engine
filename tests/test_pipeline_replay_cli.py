"""Replay (burst grouping, speed, idempotent second run) and the `python -m pipeline` commands, with a temporary database."""
import json
from pathlib import Path

import httpx
import pytest

from pipeline.__main__ import main
from pipeline.events import parse_event
from pipeline.replay import bursts, load_events, replay
from tests.pipeline_helpers import api, board, br_frame, explain, write_br

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "pipeline" / "examples" / "injury_news.jsonl"


def e(i, ts, player="x", typ="mpg", value=30):
    return {"event_id": f"r{i}", "type": typ, "player": player, "value": value, "ts": ts}


def write_feed(path, events, header="# demo\n"):
    Path(path).write_text(header + "\n".join(json.dumps(x) for x in events) + "\n", encoding="utf-8")


# ------------------------------------------------------------------------------------------------ load_events
def test_load_events_skips_blank_and_comment_lines_and_a_bom(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_bytes(("\ufeff# header\n\n" + json.dumps(e(1, "2026-10-28T12:00:00Z")) + "\n   \n  # indented comment\n" + json.dumps(e(2, "2026-10-28T12:00:05Z")) + "\n").encode("utf-8"))
    assert [x["event_id"] for x in load_events(p)] == ["r1", "r2"]


@pytest.mark.parametrize("body,msg", [
    ('{"event_id": "a"}\n{not json}\n', "line 2: not valid JSON"),
    ('# c\n\n[1, 2]\n', "line 3: each line must be a JSON object"),
    ('"just a string"\n', "line 1: each line must be a JSON object"),
])
def test_load_events_errors_name_the_line(tmp_path, body, msg):
    p = tmp_path / "f.jsonl"; p.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=msg):
        load_events(p)


def test_a_missing_feed_file_raises_a_file_error(tmp_path):
    with pytest.raises(OSError):
        load_events(tmp_path / "nope.jsonl")


def test_the_example_feed_is_valid_and_forms_two_bursts():
    events = load_events(EXAMPLE)
    assert len(events) == 5
    assert [parse_event(x).type for x in events] == ["games_out", "mpg", "team", "games_out", "clear"]
    assert [len(g) for g in bursts(events)] == [3, 2]


# ------------------------------------------------------------------------------------------------ bursts and replay
def test_events_within_the_window_of_a_bursts_first_event_share_a_burst():
    evs = [e(1, "2026-10-28T12:00:00Z"), e(2, "2026-10-28T12:00:00.5Z"), e(3, "2026-10-28T12:00:01Z"), e(4, "2026-10-28T12:00:01.001Z")]
    assert [[x["event_id"] for x in g] for g in bursts(evs, window=1.0)] == [["r1", "r2", "r3"], ["r4"]]      # exactly 1.0s still counts
    assert len(bursts(evs, window=0)) == 4 and len(bursts(evs, window=60)) == 1


def test_bursts_sort_by_time_regardless_of_file_order_and_handle_empty_input():
    evs = [e(3, "2026-10-28T12:10:00Z"), e(1, "2026-10-28T12:00:00Z"), e(2, "2026-10-28T12:00:00.2Z")]
    assert [[x["event_id"] for x in g] for g in bursts(evs)] == [["r1", "r2"], ["r3"]]
    assert bursts([]) == []


def test_a_bad_timestamp_in_a_feed_is_a_clear_error():
    with pytest.raises(ValueError, match="ISO-8601"):
        bursts([e(1, "yesterday-ish")])


def test_replay_sends_one_request_per_burst_and_totals_the_answers():
    sent = []

    def post(batch):
        sent.append([x["event_id"] for x in batch])
        return {"accepted": [x["event_id"] for x in batch][:-1], "duplicates": [batch[-1]["event_id"]], "rejected": [], "recomputes": 1}

    evs = [e(1, "2026-10-28T12:00:00Z"), e(2, "2026-10-28T12:00:00.5Z"), e(3, "2026-10-28T12:05:00Z")]
    total = replay(evs, post)
    assert sent == [["r1", "r2"], ["r3"]]
    assert total == {"bursts": 2, "events": 3, "accepted": 1, "duplicates": 2, "rejected": 0, "recomputes": 2}


def test_speed_scales_the_pauses_between_bursts_and_zero_means_no_waiting():
    evs = [e(1, "2026-10-28T12:00:00Z"), e(2, "2026-10-28T12:01:00Z"), e(3, "2026-10-28T12:03:00Z")]
    ok = lambda b: {"accepted": [x["event_id"] for x in b], "duplicates": [], "rejected": [], "recomputes": 1}   # noqa: E731
    naps = []
    replay(evs, ok, speed=60, sleep=naps.append)
    assert naps == [1.0, 2.0]                                                   # 60s -> 1s, 120s -> 2s
    naps.clear(); replay(evs, ok, speed=0, sleep=naps.append)
    assert naps == []


def test_on_burst_is_called_with_the_burst_number_group_and_response():
    seen = []
    replay([e(1, "2026-10-28T12:00:00Z"), e(2, "2026-10-28T13:00:00Z")], lambda b: {"accepted": ["x"], "duplicates": [], "rejected": [], "recomputes": 1},
           on_burst=lambda n, g, r: seen.append((n, len(g), r["recomputes"])))
    assert seen == [(1, 1, 1), (2, 1, 1)]


def test_replay_against_the_real_app_recomputes_once_per_burst_and_a_second_run_is_all_duplicates(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main_mod):
        a, b, d = [r["name"] for r in board(c)[:3]]
        base = explain(c, a)["projected_games"]
        events = [e(1, "2026-10-28T12:00:00Z", a, "games_out", 20), e(2, "2026-10-28T12:00:00.5Z", b, "mpg", 28),
                  e(3, "2026-10-28T12:10:00Z", d, "games_out", 5)]
        post = lambda batch: c.post("/events", json={"events": batch}).json()   # noqa: E731
        v0 = main_mod.state.data_version
        first = replay(events, post)
        assert first["bursts"] == 2 and first["accepted"] == 3 and first["recomputes"] == 2
        assert main_mod.state.data_version == v0 + 2                            # three events, two recomputes
        assert explain(c, a)["projected_games"] == base - 20
        second = replay(events, post)
        assert second["accepted"] == 0 and second["duplicates"] == 3 and second["recomputes"] == 0
        assert main_mod.state.data_version == v0 + 2                            # nothing moved the second time
        assert c.get("/events").json()["count"] == 3


# ------------------------------------------------------------------------------------------------ the command line
@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "p.sqlite")


def run(db, *args):
    return main(["--db", db, *args])


def test_snapshots_on_an_empty_database_says_what_to_do(db, capsys):
    assert run(db, "snapshots") == 0 and "python -m pipeline ingest" in capsys.readouterr().out


def test_ingest_lists_each_season_and_a_second_run_says_unchanged(db, tmp_path, capsys):
    raw = tmp_path / "raw"
    for y in (2025, 2026):
        write_br(raw / f"per_game_{y}.csv", br_frame(shift=y - 2025))
    assert run(db, "ingest", "--raw", str(raw)) == 0
    out = capsys.readouterr().out
    assert "per_game_2025" in out and "per_game_2026" in out and out.count("stored") == 2 and "snapshot #1" in out
    assert run(db, "ingest", "--raw", str(raw)) == 0
    assert capsys.readouterr().out.count("unchanged") == 2


def test_ingest_prints_the_error_and_exits_1_when_a_table_is_rejected(db, tmp_path, capsys):
    raw = tmp_path / "raw"
    write_br(raw / "per_game_2026.csv", br_frame().drop(columns=["STL"]))
    assert run(db, "ingest", "--raw", str(raw)) == 1
    out = capsys.readouterr().out
    assert "rejected" in out and "ERROR" in out and "STL" in out


def test_ingest_in_an_empty_folder_exits_1_with_a_message(db, tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert run(db, "ingest", "--raw", str(tmp_path / "empty")) == 1 and "No per_game" in capsys.readouterr().out


def test_snapshots_marks_the_current_one_and_rollback_plus_materialize_restore_the_old_file(db, tmp_path, capsys):
    raw, out = tmp_path / "raw", tmp_path / "out"
    p = raw / "per_game_2026.csv"
    old = write_br(p, br_frame()); run(db, "ingest", "--raw", str(raw))
    write_br(p, br_frame(shift=3)); run(db, "ingest", "--raw", str(raw)); capsys.readouterr()
    run(db, "snapshots"); listing = capsys.readouterr().out
    assert "  #1" in listing and "* #2" in listing and "* = current" in listing
    assert run(db, "rollback", "per_game_2026", "1") == 0 and "snapshot #1 is now current" in capsys.readouterr().out
    assert run(db, "materialize", "--out", str(out)) == 0
    assert (out / "per_game_2026.csv").read_bytes().decode("utf-8") == old


def test_rollback_to_something_that_does_not_exist_exits_1_and_changes_nothing(db, tmp_path, capsys):
    raw = tmp_path / "raw"; write_br(raw / "per_game_2026.csv", br_frame()); run(db, "ingest", "--raw", str(raw)); capsys.readouterr()
    assert run(db, "rollback", "per_game_2026", "99") == 1 and "no snapshot #99" in capsys.readouterr().out.lower()
    assert run(db, "rollback", "per_game_2025", "1") == 1 and "not 'per_game_2025'" in capsys.readouterr().out
    run(db, "snapshots"); assert "* #1" in capsys.readouterr().out


def test_materialize_with_nothing_stored_says_so(db, tmp_path, capsys):
    assert run(db, "materialize", "--out", str(tmp_path / "o")) == 0 and "Nothing to write" in capsys.readouterr().out


def test_events_and_team_on_an_empty_database(db, capsys):
    assert run(db, "events") == 0 and "No events yet" in capsys.readouterr().out
    assert run(db, "team", "Nobody") == 1 and "No team history" in capsys.readouterr().out


def test_team_prints_the_history_that_ingest_recorded_matching_names_case_insensitively(db, tmp_path, capsys):
    raw = tmp_path / "raw"; write_br(raw / "per_game_2026.csv", br_frame()); run(db, "ingest", "--raw", str(raw)); capsys.readouterr()
    assert run(db, "team", "Player 000") == 0
    out = capsys.readouterr().out
    assert "2025-10-01" in out and "AAA" in out and "br_per_game" in out


def test_replay_with_a_missing_or_broken_feed_exits_1_and_names_the_problem(db, tmp_path, capsys):
    assert run(db, "replay", str(tmp_path / "nope.jsonl")) == 1 and "nope.jsonl" in capsys.readouterr().out
    bad = tmp_path / "bad.jsonl"; bad.write_text('{"ok": 1}\n{oops}\n', encoding="utf-8")
    assert run(db, "replay", str(bad)) == 1 and "line 2" in capsys.readouterr().out


def test_replay_with_nothing_listening_says_to_start_the_server(db, tmp_path, capsys):
    feed = tmp_path / "f.jsonl"; write_feed(feed, [e(1, "2026-10-28T12:00:00Z")])
    assert run(db, "replay", str(feed), "--url", "http://127.0.0.1:1") == 1
    assert "Is the server running" in capsys.readouterr().out


def test_replay_command_end_to_end_against_the_app(db, monkeypatch, tmp_path, capsys):
    with api(tmp_path, monkeypatch) as (c, main_mod):
        a, b = [r["name"] for r in board(c)[:2]]
        base = explain(c, a)["projected_games"]
        feed = tmp_path / "feed.jsonl"
        write_feed(feed, [e(1, "2026-10-28T12:00:00Z", a, "games_out", 15), e(2, "2026-10-28T12:00:00.4Z", b, "mpg", 25), e(3, "2026-10-28T12:30:00Z", a, "clear", None)])
        monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: c.post("/events", json=json))   # route the CLI's request to the test app
        assert run(db, "replay", str(feed), "--url", "http://app.test") == 0
        out = capsys.readouterr().out
        assert "burst 1: 2 event(s) -> accepted 2" in out and "burst 2: 1 event(s) -> accepted 1" in out
        assert "done: 3 events in 2 burst(s); 2 recompute(s) instead of 3." in out
        assert explain(c, a)["projected_games"] == base                          # the later 'clear' undid the earlier injury
        assert run(db, "replay", str(feed), "--url", "http://app.test") == 0
        assert "duplicates 2" in capsys.readouterr().out


def test_replay_shows_why_an_event_was_rejected(db, monkeypatch, tmp_path, capsys):
    with api(tmp_path, monkeypatch) as (c, main_mod):
        feed = tmp_path / "feed.jsonl"
        write_feed(feed, [e(1, "2026-10-28T12:00:00Z", "Nobody Atall", "mpg", 30)])
        monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: c.post("/events", json=json))
        assert run(db, "replay", str(feed), "--url", "http://app.test") == 0
        assert "rejected r1: No player matches" in capsys.readouterr().out


def test_a_server_error_stops_the_replay_with_the_status_code(db, monkeypatch, tmp_path):
    feed = tmp_path / "f.jsonl"; write_feed(feed, [e(1, "2026-10-28T12:00:00Z")])
    monkeypatch.setattr(httpx, "post", lambda url, json=None, timeout=None: httpx.Response(500, text="boom"))
    with pytest.raises(SystemExit, match="500"):
        run(db, "replay", str(feed), "--url", "http://app.test")

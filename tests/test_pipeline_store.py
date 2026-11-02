"""The SQLite store: idempotent events, durable and private stores, versioned snapshots with rollback, effective-dated team history."""
import tempfile
import threading
from pathlib import Path

import pytest

from pipeline.events import Event, normalize_ts, parse_event
from pipeline.store import Store


def ev(i=1, typ="mpg", value=30.0, ts="2026-10-28T12:00:00Z"):
    return parse_event({"event_id": f"e{i}", "type": typ, "player": "x", "value": value, "ts": ts})


def test_an_event_is_stored_once_however_many_times_it_is_sent():
    s = Store(":memory:")
    assert s.add_event(ev(1), "n:a", "A") is True
    assert s.add_event(ev(1), "n:a", "A") is False
    assert s.add_event(ev(1, value=44.0), "n:a", "A") is False        # same id, different payload: still the first one wins
    assert s.event_count() == 1 and s.events()[0].value == 30.0


def test_events_come_back_ordered_by_time_then_arrival_with_their_value_types_intact():
    s = Store(":memory:")
    s.add_event(ev(1, "team", "PHI", "2026-10-02T00:00:00Z"), "n:a")
    s.add_event(ev(2, "games_out", 7, "2026-10-01T00:00:00Z"), "n:a")
    s.add_event(ev(3, "clear", None, "2026-10-01T00:00:00Z"), "n:a")
    got = s.events()
    assert [e.event_id for e in got] == ["e2", "e3", "e1"]
    assert [e.value for e in got] == [7, None, "PHI"] and isinstance(got[0].value, int)


def test_a_file_store_survives_closing_and_reopening_but_memory_stores_are_private():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "p.sqlite"
        s = Store(path)
        s.add_event(ev(1), "n:a")
        s.add_snapshot("per_game_2026", "src", "a,b\n1,2\n", 1, {"ok": True})
        s.close()
        again = Store(path)
        assert again.event_count() == 1 and again.current_snapshot("per_game_2026")["rows"] == 1
        again.close()
    one, two = Store(":memory:"), Store(":memory:")
    one.add_event(ev(1), "n:a")
    assert two.event_count() == 0


def test_twenty_threads_sending_the_same_event_store_it_exactly_once():
    s = Store(":memory:")
    wins, barrier = [], threading.Barrier(20)

    def go():
        barrier.wait()
        wins.append(s.add_event(ev(1), "n:a"))

    ts = [threading.Thread(target=go) for _ in range(20)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert wins.count(True) == 1 and s.event_count() == 1


def test_fifty_threads_storing_different_events_lose_none_and_get_unique_sequence_numbers():
    s = Store(":memory:")
    barrier = threading.Barrier(50)

    def go(i):
        barrier.wait()
        s.add_event(ev(i), "n:a")

    ts = [threading.Thread(target=go, args=(i,)) for i in range(50)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    seqs = [e.seq for e in s.events()]
    assert len(seqs) == 50 and len(set(seqs)) == 50


def test_clearing_events_also_removes_team_facts_that_came_from_them_but_keeps_source_facts():
    s = Store(":memory:")
    s.add_team_fact("n:a", "TOR", "2025-10-01T00:00:00Z", "br_per_game")
    s.add_event(ev(1, "team", "PHI"), "n:a")
    s.add_team_fact("n:a", "PHI", "2026-10-28T12:00:00.000000Z", "feed", "e1")
    assert s.clear_events() == 1 and s.event_count() == 0
    assert [h["team"] for h in s.team_history("n:a")] == ["TOR"]


def test_snapshots_are_versioned_deduplicated_and_can_be_rolled_back():
    s = Store(":memory:")
    a, created_a = s.add_snapshot("per_game_2026", "file", "x\n1\n", 1, {"ok": True})
    same, created_same = s.add_snapshot("per_game_2026", "file", "x\n1\n", 1, {"ok": True})
    b, created_b = s.add_snapshot("per_game_2026", "file", "x\n2\n", 1, {"ok": True, "warnings": ["w"]})
    assert (created_a, created_same, created_b) == (True, False, True) and same == a and b != a
    assert s.current_snapshot("per_game_2026")["id"] == b and s.current_snapshot("per_game_2026")["quality"]["warnings"] == ["w"]
    s.set_current("per_game_2026", a)
    assert s.current_snapshot("per_game_2026")["id"] == a and s.snapshot_csv(a) == "x\n1\n" and s.snapshot_csv(b) == "x\n2\n"
    listing = {x["id"]: x["current"] for x in s.snapshots()}
    assert listing == {a: True, b: False}
    # going back to content that was stored before still works (and re-adding it later is recognised as the current one)
    assert s.add_snapshot("per_game_2026", "file", "x\n1\n", 1, {})[1] is False


def test_rollback_refuses_unknown_ids_and_the_wrong_kind_and_changes_nothing():
    s = Store(":memory:")
    a, _ = s.add_snapshot("per_game_2026", "f", "x\n", 1, {})
    other, _ = s.add_snapshot("per_game_2025", "f", "y\n", 1, {})
    with pytest.raises(ValueError, match="no snapshot #99"):
        s.set_current("per_game_2026", 99)
    with pytest.raises(ValueError, match="not 'per_game_2026'"):
        s.set_current("per_game_2026", other)
    assert s.current_snapshot("per_game_2026")["id"] == a
    with pytest.raises(ValueError):
        s.snapshot_csv(99)


def test_team_history_is_effective_dated_so_a_trade_is_a_new_fact_not_an_overwrite():
    s = Store(":memory:")
    s.add_team_fact("n:brown", "BOS", "2025-10-01T00:00:00.000000Z", "br_per_game")
    s.add_team_fact("n:brown", "PHI", "2026-07-15T00:00:00.000000Z", "feed")
    assert s.team_on("n:brown", "2025-09-30T00:00:00.000000Z") is None             # before any fact
    assert s.team_on("n:brown", "2026-01-01T00:00:00.000000Z") == "BOS"
    assert s.team_on("n:brown", "2026-07-15T00:00:00.000000Z") == "PHI"            # a fact applies from its exact moment
    assert s.team_on("n:brown", "2027-01-01T00:00:00.000000Z") == "PHI"
    assert s.team_on("n:nobody", "2027-01-01T00:00:00.000000Z") is None
    assert [h["team"] for h in s.team_history("n:brown")] == ["BOS", "PHI"]         # the old fact is kept
    assert s.add_team_fact("n:brown", "PHI", "2026-07-15T00:00:00.000000Z", "feed") is False   # no duplicates


def test_the_crosswalk_links_ids_and_relinks_without_duplicating():
    s = Store(":memory:")
    s.link_id("br", "tatumja01", "jayson tatum")
    assert s.key_for("br", "tatumja01") == "jayson tatum" and s.key_for("br", "nobody01") is None and s.key_for("nba_api", "tatumja01") is None
    s.link_id("br", "tatumja01", "jayson tatum jr")
    assert s.key_for("br", "tatumja01") == "jayson tatum jr"

"""Events: validation (every way an event can be wrong) and the fold that turns a log into adjustments."""
import random
from datetime import datetime, timezone

import pytest

from pipeline.events import Event, StoredEvent, fold, normalize_ts, parse_event

OK = {"event_id": "e1", "type": "games_out", "player": "Jayson Tatum", "value": 20, "ts": "2026-10-28T14:30:00Z"}


def test_each_event_type_parses_and_normalizes_its_value():
    assert parse_event(OK).value == 20
    assert parse_event({**OK, "type": "games", "value": 55.0}).value == 55 and isinstance(parse_event({**OK, "type": "games", "value": 55.0}).value, int)
    assert parse_event({**OK, "type": "mpg", "value": 31.456}).value == 31.46
    assert parse_event({**OK, "type": "team", "value": "  PHI "}).value == "PHI"
    assert parse_event({**OK, "type": "clear", "value": "ignored"}).value is None


def test_timestamps_in_any_iso_form_normalize_to_the_same_sortable_text():
    forms = ["2026-10-28T14:30:00Z", "2026-10-28T14:30:00+00:00", "2026-10-28T10:30:00-04:00", "2026-10-28T14:30:00", "2026-10-28 14:30:00Z"]
    assert len({normalize_ts(f) for f in forms}) == 1
    assert normalize_ts("2026-10-28T14:30:00Z") < normalize_ts("2026-10-28T14:30:01Z") < normalize_ts("2027-01-01T00:00:00Z")
    fixed = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    assert normalize_ts(None, now=fixed) == normalize_ts("", now=fixed) == "2026-01-02T03:04:05.000000Z"


@pytest.mark.parametrize("patch,message", [
    ({"event_id": None}, "event_id is required"), ({"event_id": ""}, "event_id is required"), ({"event_id": "has space"}, "event_id is required"),
    ({"event_id": "x" * 81}, "event_id is required"), ({"type": "trade"}, "type must be one of"), ({"type": None}, "type must be one of"),
    ({"player": ""}, "player is required"), ({"player": "   "}, "player is required"), ({"player": "x" * 81}, "player is required"),
    ({"player": 7}, "player is required"),
    ({"value": -1}, "0 to 82"), ({"value": 83}, "0 to 82"), ({"value": 1.5}, "whole number"), ({"value": "5"}, "must be a number"),
    ({"value": True}, "must be a number"), ({"value": None}, "must be a number"), ({"value": float("nan")}, "finite"),
    ({"type": "games", "value": 0}, "1 to 82"), ({"type": "games", "value": 83}, "1 to 82"),
    ({"type": "mpg", "value": 0.5}, "between 1 and 48"), ({"type": "mpg", "value": 49}, "between 1 and 48"), ({"type": "mpg", "value": "30"}, "must be a number"),
    ({"type": "team", "value": ""}, "team's name"), ({"type": "team", "value": 5}, "team's name"), ({"type": "team", "value": "x" * 31}, "team's name"),
    ({"ts": "yesterday"}, "ISO-8601"), ({"ts": 12345}, "text timestamp"), ({"source": "s" * 41}, "source must be"), ({"note": "n" * 201}, "note must be"),
])
def test_every_kind_of_bad_event_is_refused_with_a_message_that_says_what_to_fix(patch, message):
    with pytest.raises(ValueError) as e:
        parse_event({**OK, **patch})
    assert message in str(e.value)


@pytest.mark.parametrize("raw", [None, "text", 5, [], [OK]])
def test_something_that_is_not_an_object_is_refused(raw):
    with pytest.raises(ValueError, match="must be an object"):
        parse_event(raw)


def S(seq, ts, typ, pid, value):
    return StoredEvent(seq, f"e{seq}", normalize_ts(ts), typ, pid, value)


def test_fold_last_write_wins_per_field_and_games_fields_replace_each_other():
    log = [S(1, "2026-10-01T00:00:00Z", "games_out", "a", 10), S(2, "2026-10-02T00:00:00Z", "mpg", "a", 30.0),
           S(3, "2026-10-03T00:00:00Z", "team", "a", "PHI"), S(4, "2026-10-04T00:00:00Z", "games", "a", 55)]
    assert fold(log) == {"a": {"games": {"kind": "abs", "n": 55}, "mpg": 30.0, "team": "PHI"}}
    log.append(S(5, "2026-10-05T00:00:00Z", "games_out", "a", 5))
    assert fold(log)["a"]["games"] == {"kind": "out", "n": 5}


def test_a_late_arriving_older_event_cannot_overwrite_newer_news():
    newer_first = [S(1, "2026-10-28T12:00:00Z", "games_out", "a", 30), S(2, "2026-10-28T09:00:00Z", "games_out", "a", 5)]   # arrived second, happened earlier
    assert fold(newer_first)["a"]["games"]["n"] == 30


def test_equal_timestamps_fall_back_to_arrival_order():
    log = [S(1, "2026-10-28T12:00:00Z", "mpg", "a", 20.0), S(2, "2026-10-28T12:00:00Z", "mpg", "a", 25.0)]
    assert fold(log)["a"]["mpg"] == 25.0


def test_clear_removes_what_came_before_it_but_not_what_comes_after():
    log = [S(1, "2026-10-01T00:00:00Z", "games_out", "a", 10), S(2, "2026-10-02T00:00:00Z", "clear", "a", None),
           S(3, "2026-10-03T00:00:00Z", "mpg", "a", 28.0), S(4, "2026-10-01T00:00:00Z", "games_out", "b", 3)]
    assert fold(log) == {"a": {"mpg": 28.0}, "b": {"games": {"kind": "out", "n": 3}}}
    assert fold(log[:2]) == {}


def test_fold_is_deterministic_whatever_order_the_log_arrives_in():
    base = [S(i, f"2026-10-{1 + i % 9:02d}T00:00:00Z", random.choice(["games_out", "mpg", "team", "games"]), random.choice("abc"), 20) for i in range(1, 60)]
    base = [e._replace(value={"games_out": 20, "games": 20, "mpg": 20.0, "team": "PHI"}[e.type]) for e in base]
    want = fold(base)
    for seed in range(5):
        shuffled = base[:]
        random.Random(seed).shuffle(shuffled)
        assert fold(shuffled) == want


def test_fold_of_nothing_is_nothing():
    assert fold([]) == {}

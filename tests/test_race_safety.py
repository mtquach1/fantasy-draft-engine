"""Concurrency: simultaneous requests must never corrupt the draft. Every test fires real threads at the real API."""
import random
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import make_sample_sport
from app import ingestion, persist


@contextmanager
def api(tmp, monkeypatch, multi=False):
    for k, v in {"SPORT": "nba", "LEAGUE_PRESET": "", "DRAFT_PLATFORM": "manual", "MULTI_SESSION": "1" if multi else ""}.items():
        monkeypatch.setenv(k, v)
    old_i, old_p = ingestion.DATA_DIR, persist.DATA_DIR
    ingestion.DATA_DIR = persist.DATA_DIR = Path(tmp)
    try:
        make_sample_sport.generate("nba", Path(tmp) / "nba", n=300)
        from app.main import app
        with TestClient(app) as c:
            yield c, app
    finally:
        ingestion.DATA_DIR, persist.DATA_DIR = old_i, old_p


def _widen_race_window(monkeypatch, ms=2):
    """Pause briefly between a request's checks and its write. With the lock this changes nothing; without it, races become near-certain,
    so these tests fail reliably if the lock is ever removed instead of passing by luck."""
    import time
    from app import main, valuation
    real_blocked, real_expand = main._manual_entry_blocked, valuation.expand_twins

    def slow_blocked():
        r = real_blocked()
        time.sleep(ms / 1000)
        return r

    def slow_expand(*a, **k):
        r = real_expand(*a, **k)
        time.sleep(ms / 1000)
        return r

    monkeypatch.setattr(main, "_manual_entry_blocked", slow_blocked)
    monkeypatch.setattr(valuation, "expand_twins", slow_expand)


class _SlowLog(list):
    """A pick log whose length check pauses: widens the window between 'is there anything to undo?' and the undo itself."""
    def __len__(self):
        import time
        n = super().__len__()
        time.sleep(0.002)                                  # the answer goes stale while other requests run
        return n


def _slow_log(monkeypatch):
    from app import main
    monkeypatch.setattr(main.state, "draft_log", _SlowLog(main.state.draft_log))


def _ids(c, n):
    rows = c.get(f"/players/available?limit={n}&sort=rank").json()
    return [r["player_id"] for r in rows]


def _setup(c, teams=12, rounds=None):
    body = {"draft_type": "snake", "teams": teams, "my_slot": 1}
    if rounds:
        body["rounds"] = rounds
    assert c.put("/draft/setup", json=body).status_code == 200


def _fire(app, jobs):
    """Run every (method, url, kwargs) in its own thread, released together by a barrier. Returns the responses."""
    out, barrier = [None] * len(jobs), threading.Barrier(len(jobs))

    def run(i, job):
        client = TestClient(app)                       # its own client: its own connection and cookie jar
        barrier.wait()
        out[i] = client.request(job[0], job[1], **job[2])

    ts = [threading.Thread(target=run, args=(i, j)) for i, j in enumerate(jobs)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return out


def test_twenty_simultaneous_picks_of_one_player_let_exactly_one_through(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as (c, app):
        _setup(c)
        for pid in _ids(c, 25):                         # 25 separate races, each with 20 contenders
            res = _fire(app, [("POST", "/draft/pick", {"json": {"player_id": pid}})] * 20)
            codes = sorted(r.status_code for r in res)
            assert codes.count(200) == 1 and codes.count(400) == 19, codes
        log = c.get("/draft/log").json()
        assert len(log) == 25 and len({p["player_id"] for p in log}) == 25            # nobody drafted twice


def test_simultaneous_picks_of_different_players_get_unique_consecutive_pick_numbers(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as (c, app):
        _setup(c)
        ids = _ids(c, 24)
        res = _fire(app, [("POST", "/draft/pick", {"json": {"player_id": p}}) for p in ids])
        assert all(r.status_code == 200 for r in res)
        log = c.get("/draft/log").json()
        assert [p["pick_number"] for p in log] == list(range(1, 25))                # no two picks share a number, none skipped
        assert {p["player_id"] for p in log} == set(ids)
        from app import draft
        assert [p["team_id"] for p in log] == [draft.slot_for_pick("snake", 12, n) for n in range(1, 25)]   # snake order held


def test_a_finished_draft_cannot_overflow_under_load(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as (c, app):
        _widen_race_window(monkeypatch)
        _setup(c, teams=2, rounds=1)                    # exactly two picks exist
        res = _fire(app, [("POST", "/draft/pick", {"json": {"player_id": p}}) for p in _ids(c, 12)])
        codes = sorted(r.status_code for r in res)
        assert codes.count(200) == 2 and codes.count(400) == 10
        assert len(c.get("/draft/log").json()) == 2


def test_the_same_idempotency_key_drafts_once_and_every_caller_gets_the_same_answer(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as (c, app):
        _widen_race_window(monkeypatch)
        _setup(c)
        pid = _ids(c, 1)[0]
        res = _fire(app, [("POST", "/draft/pick", {"json": {"player_id": pid}, "headers": {"Idempotency-Key": "click-123"}})] * 15)
        assert all(r.status_code == 200 for r in res)
        assert len({str(r.json()) for r in res}) == 1                                 # identical response for every retry
        assert len(c.get("/draft/log").json()) == 1
        # a genuinely new click (new key) on the same player is refused as usual
        again = c.post("/draft/pick", json={"player_id": pid}, headers={"Idempotency-Key": "click-456"})
        assert again.status_code == 400 and "already drafted" in again.text
        # the key may also travel in the body
        other = _ids(c, 2)[0]
        r1 = c.post("/draft/pick", json={"player_id": other, "request_id": "body-key"})
        r2 = c.post("/draft/pick", json={"player_id": other, "request_id": "body-key"})
        assert r1.status_code == r2.status_code == 200 and r1.json() == r2.json() and len(c.get("/draft/log").json()) == 2


def test_picks_without_a_key_behave_exactly_as_before(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as (c, app):
        _setup(c)
        pid = _ids(c, 1)[0]
        assert c.post("/draft/pick", json={"player_id": pid}).status_code == 200
        assert c.post("/draft/pick", json={"player_id": pid}).status_code == 400


def test_an_undo_storm_undoes_each_pick_exactly_once(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as (c, app):
        _widen_race_window(monkeypatch)
        _slow_log(monkeypatch)
        _setup(c)
        for p in _ids(c, 3):
            c.post("/draft/pick", json={"player_id": p})
        res = _fire(app, [("POST", "/draft/undo", {})] * 10)
        codes = sorted(r.status_code for r in res)
        assert codes.count(200) == 3 and codes.count(400) == 7
        assert c.get("/draft/log").json() == []
        assert len({r.json()["player_id"] for r in res if r.status_code == 200}) == 3           # three different picks were popped


def test_undo_clears_remembered_keys_so_a_deliberate_redo_works(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as (c, app):
        _setup(c)
        pid = _ids(c, 1)[0]
        k = {"Idempotency-Key": "same-key"}
        assert c.post("/draft/pick", json={"player_id": pid}, headers=k).status_code == 200
        assert c.post("/draft/undo").status_code == 200
        r = c.post("/draft/pick", json={"player_id": pid}, headers=k)             # same key after an undo is a new decision
        assert r.status_code == 200 and len(c.get("/draft/log").json()) == 1


def test_a_mixed_storm_of_picks_undos_resets_and_reads_never_errors_or_corrupts(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as (c, app):
        _widen_race_window(monkeypatch)
        _slow_log(monkeypatch)
        _setup(c)
        ids = _ids(c, 60)
        rng = random.Random(7)
        jobs = []
        for _ in range(160):
            r = rng.random()
            if r < 0.55:
                jobs.append(("POST", "/draft/pick", {"json": {"player_id": rng.choice(ids)}}))
            elif r < 0.70:
                jobs.append(("POST", "/draft/undo", {}))
            elif r < 0.74:
                jobs.append(("POST", "/draft/reset", {}))
            elif r < 0.85:
                jobs.append(("GET", "/draft/log", {}))
            elif r < 0.93:
                jobs.append(("GET", "/players/available?limit=20", {}))
            else:
                jobs.append(("GET", "/team/summary", {}))
        res = _fire(app, jobs)
        assert all(r.status_code in (200, 400, 404, 409) for r in res), sorted({r.status_code for r in res})
        log = c.get("/draft/log").json()
        assert [p["pick_number"] for p in log] == list(range(1, len(log) + 1))                # consecutive from 1
        assert len({p["player_id"] for p in log}) == len(log)                                  # no player twice
        assert c.get("/team/summary").status_code == 200 and c.get("/players/available").status_code == 200


def test_overrides_and_reads_can_happen_at_the_same_time(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch) as (c, app):
        names = [r["name"] for r in c.get("/players/available?limit=8&sort=rank").json()]
        jobs = []
        for i, n in enumerate(names):
            jobs += [("PUT", "/players/override", {"json": {"player": n, "games": 30 + i}}), ("GET", "/players/rankings?limit=10", {}),
                     ("GET", "/players/overrides", {})]
        jobs += [("DELETE", "/players/overrides", {})] * 2 + [("GET", "/players/available?limit=10", {})] * 4
        res = _fire(app, jobs)
        assert all(r.status_code in (200, 404) for r in res), sorted({r.status_code for r in res})
        assert c.get("/players/rankings?limit=5").status_code == 200


def test_each_hosted_session_has_its_own_lock_and_one_busy_draft_never_stalls_another(monkeypatch):
    with tempfile.TemporaryDirectory() as d, api(d, monkeypatch, multi=True) as (a, app):
        from app.state import sessions
        b = TestClient(app)
        a.get("/"); b.get("/")
        locks = [item[0].lock for item in sessions._items.values()]
        assert len(locks) == 2 and locks[0] is not locks[1]
        holder_ready, release = threading.Event(), threading.Event()

        def hold_first():
            with locks[0]:
                holder_ready.set()
                release.wait(10)

        t = threading.Thread(target=hold_first)
        t.start()
        holder_ready.wait(5)
        done = []
        w = threading.Thread(target=lambda: done.append(b.get("/draft/log").status_code))   # the other browser, while the first draft is locked
        w.start()
        w.join(5)
        release.set()
        t.join()
        assert done == [200]                                                                  # it was not blocked

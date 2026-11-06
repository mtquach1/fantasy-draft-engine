"""research/load_test.py against a fake in-memory server (httpx.MockTransport): the maths, the safety refusals, and that it can tell a server
that handles retries correctly from one that does not."""
import importlib.util
import json
import threading
import uuid
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("load_test", ROOT / "research" / "load_test.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class FakeServer:
    """Per-visitor drafts keyed by a cookie, like the hosted app. complete_at: picks after which the draft is 'complete'."""
    def __init__(self, mode="multi", complete_at=6, dup_bug=False, fail_recommend=False, net_error_recommend=False, session_status=200, down=False):
        self.mode, self.complete_at, self.dup_bug, self.fail_recommend = mode, complete_at, dup_bug, fail_recommend
        self.net_error_recommend, self.session_status, self.down = net_error_recommend, session_status, down
        self.drafts, self.idem, self.writes, self.auth_seen, self.lock = {}, {}, 0, set(), threading.Lock()

    def __call__(self, req: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("refused", request=req)
        self.auth_seen.add(req.headers.get("authorization"))
        path, m = req.url.path, req.method
        if path == "/session":
            return httpx.Response(self.session_status, json={"mode": self.mode})
        if m in ("PUT", "POST"):
            self.writes += 1
        sid = req.cookies.get("sid") if hasattr(req, "cookies") else None
        sid = dict(c.split("=", 1) for c in req.headers.get("cookie", "").split("; ") if "=" in c).get("sid")
        headers = {}
        with self.lock:
            if sid is None:
                sid = uuid.uuid4().hex; headers["set-cookie"] = f"sid={sid}; Path=/"
            log = self.drafts.setdefault(sid, [])
            if path == "/draft/setup":
                return httpx.Response(200, json={}, headers=headers)
            if path == "/players/available":
                taken = {p["player_id"] for p in log}
                return httpx.Response(200, json=[{"player_id": f"p{i}"} for i in range(200) if f"p{i}" not in taken][:50], headers=headers)
            if path == "/recommend/plan":
                if self.net_error_recommend:
                    raise httpx.ReadTimeout("slow", request=req)
                return httpx.Response(500 if self.fail_recommend else 200, json={"plan": {}, "picks": []}, headers=headers)
            if path == "/draft/log":
                return httpx.Response(200, json=list(log), headers=headers)
            if path == "/draft/reset":
                log.clear(); return httpx.Response(200, json={}, headers=headers)
            if path == "/draft/pick":
                key = req.headers.get("idempotency-key")
                if key in self.idem and not self.dup_bug:
                    return httpx.Response(200, json=self.idem[key], headers=headers)
                if len(log) >= self.complete_at:
                    return httpx.Response(400, json={"detail": "The draft is complete"}, headers=headers)
                rec = {"pick_number": len(log) + 1, "player_id": json.loads(req.content)["player_id"]}
                log.append(rec); self.idem[key] = rec
                return httpx.Response(200, json=rec, headers=headers)
        return httpx.Response(404, json={})


def run(*args, server=None, extra=()):
    mod = load(); out = []
    code = mod.main(["--url", "http://fake", "--duration", "0.4", "--pace", "0", "--users", "2", *args, *extra], transport=httpx.MockTransport(server or FakeServer()), out=out.append)
    return code, "\n".join(out), mod


# ------------------------------------------------------------------------------------------------ maths
def test_the_report_has_exact_percentiles_error_counts_and_rates():
    mod = load(); rec = mod.Recorder()
    for ms in range(1, 101):
        rec.add("a", ms / 1000, ok=(ms != 100))
    rec.add("b", 0.5, ok=True)
    rep = rec.report(wall_seconds=10)
    a = rep["endpoints"]["a"]
    assert (a["count"], a["errors"], a["p50_ms"], a["p95_ms"], a["p99_ms"], a["max_ms"]) == (100, 1, 50.0, 95.0, 99.0, 100.0)
    assert rep["total_requests"] == 101 and rep["total_errors"] == 1 and rep["requests_per_second"] == 10.1 and rep["error_rate"] == round(1 / 101, 4)


def test_an_empty_report_does_not_divide_by_zero():
    rep = load().Recorder().report(wall_seconds=0)
    assert rep["total_requests"] == 0 and rep["requests_per_second"] is None and rep["error_rate"] is None


# ------------------------------------------------------------------------------------------------ safety
def test_it_refuses_a_single_user_server_and_sends_nothing_that_could_overwrite_a_draft():
    srv = FakeServer(mode="single")
    code, out, _ = run(server=srv)
    assert code == 2 and "single-user mode" in out and "overwrite" in out and srv.writes == 0


def test_the_override_flag_allows_a_disposable_single_user_server():
    srv = FakeServer(mode="single")
    code, out, _ = run(server=srv, extra=["--allow-reset-draft"])
    assert code == 0 and srv.writes > 0


@pytest.mark.parametrize("args,needle", [(["--users", "0"], "between 1 and 50"), (["--users", "51"], "between 1 and 50"),
                                         (["--duration", "0"], "above 0"), (["--pace", "-1"], "cannot be negative")])
def test_silly_settings_are_refused_before_any_request(args, needle):
    srv = FakeServer(); mod = load(); out = []
    code = mod.main(["--url", "http://fake", *args], transport=httpx.MockTransport(srv), out=out.append)
    assert code == 2 and needle in "\n".join(out) and srv.writes == 0


def test_the_url_is_required():
    with pytest.raises(SystemExit):
        load().main([])


def test_an_unreachable_or_wrong_server_gets_a_clear_message():
    code, out, _ = run(server=FakeServer(down=True))
    assert code == 2 and "Could not reach" in out
    code, out, _ = run(server=FakeServer(session_status=404))
    assert code == 2 and "is this the draft app" in out


# ------------------------------------------------------------------------------------------------ behaviour
def test_a_healthy_server_gets_a_clean_report_and_every_step_is_exercised(tmp_path):
    srv = FakeServer(); j = tmp_path / "r.json"
    code, out, _ = run(server=srv, extra=["--json", str(j)])
    rep = json.loads(j.read_text(encoding="utf-8"))
    assert code == 0 and rep["mode"] == "multi" and rep["users"] == 2 and "first_request_seconds" in rep
    assert {"draft_setup", "players_available", "recommend_plan", "draft_pick", "draft_pick_retry", "draft_reset", "draft_log"} <= set(rep["endpoints"])
    assert rep["total_errors"] == 0 and rep["idempotent_retries"] > 0 and rep["idempotency_violations"] == 0
    assert rep["log_checks"] > 0 and rep["log_violations"] == 0
    assert "Correctness under load" in out and "not a capacity guarantee" in out and "First request" in out


def test_each_virtual_user_has_its_own_draft():
    srv = FakeServer(complete_at=1000)
    run(server=srv, extra=["--users", "3"])
    assert len(srv.drafts) == 3


def test_a_server_that_double_drafts_on_a_retry_is_caught():
    code, out, _ = run(server=FakeServer(dup_bug=True, complete_at=1000))
    assert code == 1
    rep_line = next(ln for ln in out.splitlines() if ln.startswith("Correctness"))
    assert " 0 not answered" not in rep_line and " 0 with the wrong number" not in rep_line


def test_server_errors_and_network_errors_are_counted_not_fatal():
    code, out, _ = run(server=FakeServer(fail_recommend=True))
    row = next(ln for ln in out.splitlines() if ln.startswith("recommend_plan")).split()
    assert code == 0 and row[1] == row[2] and int(row[1]) > 0                       # every recommend call failed, and was counted as failed
    code, out, _ = run(server=FakeServer(net_error_recommend=True))
    row = next(ln for ln in out.splitlines() if ln.startswith("recommend_plan")).split()
    assert code == 0 and row[1] == row[2] and "failed (" in out


def test_an_expected_400_when_the_draft_completes_is_not_an_error():
    code, out, _ = run(server=FakeServer(complete_at=3))
    pick = next(ln for ln in out.splitlines() if ln.startswith("draft_pick ")).split()
    assert pick[2] == "0"                                                          # the "draft is complete" answers did not count as failures


def test_the_token_is_sent_when_given():
    srv = FakeServer()
    run(server=srv, extra=["--token", "abc123"])
    assert srv.auth_seen == {"Bearer abc123"}

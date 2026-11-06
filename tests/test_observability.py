"""app/observability.py: request ids, structured logs and metrics. The logs must not leak query strings or cookies; the metrics must stay bounded."""
import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import observability as obs
from tests.pipeline_helpers import api


@pytest.fixture(autouse=True)
def fresh_metrics(monkeypatch):
    obs.METRICS.reset()
    for k in ("REQUEST_LOG", "LOG_FORMAT", "METRICS_TOKEN"):
        monkeypatch.delenv(k, raising=False)


def mini():
    app = FastAPI()

    @app.get("/items/{item_id}")
    def item(item_id: int):
        return {"id": item_id, "rid": obs.current_request_id()}

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    @app.get("/metrics")
    def m():
        return {}

    app.add_middleware(obs.RequestObservability)
    return TestClient(app, raise_server_exceptions=False)


# ------------------------------------------------------------------------------------------------ helpers
def test_percentile_is_nearest_rank():
    v = list(range(1, 11))
    assert [obs.percentile(v, q) for q in (0, 10, 50, 95, 100)] == [1, 1, 5, 10, 10]
    assert obs.percentile([], 50) is None and obs.percentile([7], 99) == 7
    assert obs.percentile([5, 1, 3], 50) == 3                                   # unsorted input is fine


@pytest.mark.parametrize("incoming", [None, "", "has space", "tab\there", "x" * 65, "semi;colon", "new\nline", "emoji-\u2603"])
def test_a_bad_request_id_is_replaced_by_a_fresh_one(incoming):
    a, b = obs.clean_request_id(incoming), obs.clean_request_id(incoming)
    assert a != b and len(a) == 16 and all(c in "0123456789abcdef" for c in a)


def test_a_sane_request_id_is_kept():
    assert obs.clean_request_id("trace-12.ab_C") == "trace-12.ab_C" and obs.clean_request_id("a" * 64) == "a" * 64


# ------------------------------------------------------------------------------------------------ metrics
def test_the_histogram_is_cumulative_and_counts_match():
    obs.METRICS.observe("GET", "/r", 200, 0.003); obs.METRICS.observe("GET", "/r", 200, 0.2); obs.METRICS.observe("GET", "/r", 500, 3.0)
    text = obs.METRICS.prometheus()
    line = lambda le: next(ln for ln in text.splitlines() if f'route="/r",le="{le}"' in ln).rsplit(" ", 1)[1]    # noqa: E731
    assert [line(le) for le in ("0.005", "0.25", "2.5", "5.0", "+Inf")] == ["1", "2", "2", "3", "3"]
    assert 'http_request_duration_seconds_count{route="/r"} 3' in text and 'http_request_duration_seconds_sum{route="/r"} 3.203000' in text
    assert 'http_requests_total{method="GET",route="/r",status="200"} 2' in text and 'status="500"} 1' in text
    assert "# TYPE http_requests_total counter" in text and "# TYPE http_request_duration_seconds histogram" in text and text.endswith("\n")


def test_label_values_are_escaped():
    obs.METRICS.observe("GET", 'a"b\\c\nd', 200, 0.01)
    assert 'route="a\\"b\\\\c\\nd"' in obs.METRICS.prometheus()


def test_summary_percentiles_and_error_counts():
    for ms in range(1, 101):
        obs.METRICS.observe("GET", "/x", 200, ms / 1000)
    obs.METRICS.observe("GET", "/x", 503, 0.5)
    r = obs.METRICS.summary()["routes"]["/x"]
    assert r["count"] == 101 and r["errors_5xx"] == 1 and r["p50_ms"] == 51.0 and r["p95_ms"] == 96.0 and r["p99_ms"] == 100.0 and r["max_ms"] == 500.0


def test_memory_stays_bounded_while_counts_keep_counting():
    for i in range(obs.RESERVOIR + 500):
        obs.METRICS.observe("GET", "/y", 200, 0.001)
    r = obs.METRICS.summary()["routes"]["/y"]
    assert r["count"] == obs.RESERVOIR + 500 and r["percentiles_from_last"] == obs.RESERVOIR


def test_concurrent_updates_lose_nothing():
    def work():
        for _ in range(500):
            obs.METRICS.observe("GET", "/z", 200, 0.002)
    ts = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert obs.METRICS.summary()["routes"]["/z"]["count"] == 4000 and obs.METRICS.summary()["total_requests"] == 4000


def test_in_flight_returns_to_zero_even_after_an_error():
    c = mini(); c.get("/items/1"); c.get("/boom")
    assert obs.METRICS.summary()["in_flight"] == 0


# ------------------------------------------------------------------------------------------------ the middleware
def test_every_response_carries_a_request_id_and_the_handler_can_see_it():
    c = mini()
    r = c.get("/items/5")
    assert r.headers["x-request-id"] == r.json()["rid"] and len(r.headers["x-request-id"]) == 16
    assert c.get("/items/5", headers={"X-Request-ID": "caller-1"}).headers["x-request-id"] == "caller-1"
    assert c.get("/items/5", headers={"X-Request-ID": "bad id"}).headers["x-request-id"] != "bad id"


def test_metrics_are_labelled_by_route_template_not_by_url():
    c = mini()
    for i in (1, 2, 3):
        c.get(f"/items/{i}")
    s = obs.METRICS.summary()["routes"]
    assert s["/items/{item_id}"]["count"] == 3 and "/items/1" not in s


def test_unknown_urls_all_land_in_one_bucket_so_a_scanner_cannot_grow_the_metrics():
    c = mini()
    for i in range(50):
        assert c.get(f"/wp-admin/{i}").status_code == 404
    s = obs.METRICS.summary()
    assert s["routes"]["unmatched"]["count"] == 50 and len(s["routes"]) == 1


def test_an_exception_is_counted_as_a_500_and_still_returned_as_one():
    c = mini()
    assert c.get("/boom").status_code == 500
    assert 'http_requests_total{method="GET",route="/boom",status="500"} 1' in obs.METRICS.prometheus()


# ------------------------------------------------------------------------------------------------ the request log
def test_nothing_is_logged_unless_asked(capsys):
    mini().get("/items/1")
    assert capsys.readouterr().out == ""


def test_json_log_lines_have_the_expected_fields_and_no_secrets(monkeypatch, capsys):
    monkeypatch.setenv("REQUEST_LOG", "1"); monkeypatch.setenv("LOG_FORMAT", "json")
    c = mini()
    r = c.get("/items/5?secret=hunter2&q=private+name", headers={"Cookie": "draft_session=sessionvalue123", "Authorization": "Bearer tok"})
    out = capsys.readouterr().out
    assert "hunter2" not in out and "private" not in out and "sessionvalue123" not in out and "Bearer" not in out
    rec = json.loads(out.strip())
    assert rec["event"] == "request" and rec["level"] == "info" and rec["method"] == "GET" and rec["route"] == "/items/{item_id}"
    assert rec["status"] == 200 and rec["request_id"] == r.headers["x-request-id"] and rec["duration_ms"] >= 0 and rec["path"] is None and rec["ts"].endswith("+00:00")


def test_unmatched_urls_log_their_path_and_errors_log_at_error_level(monkeypatch, capsys):
    monkeypatch.setenv("REQUEST_LOG", "1"); monkeypatch.setenv("LOG_FORMAT", "json")
    c = mini(); c.get("/nope?x=1"); c.get("/boom")
    a, b = [json.loads(x) for x in capsys.readouterr().out.strip().splitlines()]
    assert a["route"] == "unmatched" and a["path"] == "/nope" and a["status"] == 404 and "x=1" not in json.dumps(a)
    assert b["level"] == "error" and b["status"] == 500 and b["error"] == "RuntimeError"


def test_text_format_and_the_metrics_endpoint_is_not_logged(monkeypatch, capsys):
    monkeypatch.setenv("REQUEST_LOG", "1")
    c = mini(); c.get("/items/2"); c.get("/metrics")
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and "event=request" in lines[0] and "route=/items/{item_id}" in lines[0] and "status=200" in lines[0]
    assert obs.METRICS.summary()["routes"]["/metrics"]["count"] == 1               # still counted, just not logged


# ------------------------------------------------------------------------------------------------ the token
def test_the_token_check(monkeypatch):
    assert obs.check_token(None) is True                                           # no token configured: open
    monkeypatch.setenv("METRICS_TOKEN", "s3cret")
    assert obs.check_token(None) is False and obs.check_token("") is False and obs.check_token("Bearer wrong") is False
    assert obs.check_token("Basic s3cret") is False and obs.check_token("s3cret") is False
    assert obs.check_token("Bearer s3cret") is True and obs.check_token("bearer  s3cret ") is True


# ------------------------------------------------------------------------------------------------ the real app
def test_the_real_app_serves_metrics_and_counts_requests_exactly(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        def count():
            return obs.METRICS.summary()["routes"].get("/session", {}).get("count", 0)
        before = count()
        for _ in range(3):
            assert c.get("/session").headers["x-request-id"]
        assert count() == before + 3
        r = c.get("/metrics")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain") and 'route="/session"' in r.text
        j = c.get("/metrics/summary").json()
        assert j["routes"]["/session"]["p95_ms"] is not None and j["total_requests"] >= 4


def test_with_a_token_set_the_real_app_refuses_anonymous_scrapes(monkeypatch, tmp_path):
    monkeypatch.setenv("METRICS_TOKEN", "tok-123")
    with api(tmp_path, monkeypatch) as (c, main):
        assert c.get("/metrics").status_code == 401 and c.get("/metrics/summary").status_code == 401
        assert c.get("/metrics", headers={"Authorization": "Bearer nope"}).status_code == 401
        ok = c.get("/metrics/summary", headers={"Authorization": "Bearer tok-123"})
        assert ok.status_code == 200 and "routes" in ok.json()
        assert c.get("/session").status_code == 200                                # the rest of the app is not affected by the token


def test_scraping_metrics_does_not_use_up_a_hosted_session(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch, multi=True) as (c, main):
        for _ in range(5):
            assert c.get("/metrics").status_code == 200
        assert c.get("/session").json()["active_sessions"] == 0

"""Request IDs, structured request logs, and in-process metrics. Standard library only.

* Every response carries an `X-Request-ID` header. A valid id sent by the caller is kept (so a request can be followed across systems);
  anything else is replaced. The id is also available to code through `current_request_id()`.
* One log line per request when REQUEST_LOG=1 (JSON when LOG_FORMAT=json, otherwise key=value text). It holds the method, the ROUTE (e.g.
  /players/{player_id}/stats, never the raw URL's query string), status, duration, and the request id: no cookies, no bodies, no query values.
* Metrics are always collected (a few dictionary updates per request) and served at GET /metrics (Prometheus text format) and
  GET /metrics/summary (JSON with p50/p95/p99 from the most recent requests). Routes unknown to the app are all counted as "unmatched" so a
  scanner hitting random URLs cannot grow the metrics without bound. If METRICS_TOKEN is set, both endpoints require `Authorization: Bearer <token>`.

Metrics are per process and kept in memory: they reset on restart, and with several workers each one has its own. Honest limit: percentiles
come from the last RESERVOIR requests per route, not from all time."""
from __future__ import annotations

import contextvars
import hmac
import json
import os
import re
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)       # seconds
RESERVOIR = 2000
_ID_OK = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_print_lock = threading.Lock()


def current_request_id() -> str | None:
    return _request_id.get()


def clean_request_id(incoming: str | None) -> str:
    """Keep a sane caller-supplied id; otherwise make one. (Never trust header content: it goes into logs.)"""
    return incoming if incoming and _ID_OK.match(incoming) else uuid.uuid4().hex[:16]


def percentile(values, q: float) -> float | None:
    """Nearest-rank percentile (q in 0..100) of a list: the smallest value with at least q% of the values at or below it."""
    if not values:
        return None
    s = sorted(values)
    rank = max(1, -(-len(s) * q // 100))                       # ceil(len * q / 100)
    return s[int(rank) - 1]


class Metrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        with self.lock:
            self.started = time.time()
            self.counts: dict[tuple, int] = {}
            self.hist: dict[str, dict] = {}
            self.recent: dict[str, deque] = {}
            self.in_flight = 0

    def begin(self):
        with self.lock:
            self.in_flight += 1

    def end(self):
        with self.lock:
            self.in_flight = max(0, self.in_flight - 1)

    def observe(self, method: str, route: str, status: int, seconds: float):
        with self.lock:
            k = (method, route, status)
            self.counts[k] = self.counts.get(k, 0) + 1
            h = self.hist.setdefault(route, {"buckets": [0] * len(BUCKETS), "sum": 0.0, "count": 0})
            for i, edge in enumerate(BUCKETS):
                if seconds <= edge:
                    h["buckets"][i] += 1
            h["sum"] += seconds; h["count"] += 1
            self.recent.setdefault(route, deque(maxlen=RESERVOIR)).append(seconds)

    # ---- output ----
    def prometheus(self) -> str:
        with self.lock:
            out = ["# HELP http_requests_total Requests handled, by method, route and status.", "# TYPE http_requests_total counter"]
            for (m, r, s), n in sorted(self.counts.items()):
                out.append(f'http_requests_total{{method="{_esc(m)}",route="{_esc(r)}",status="{s}"}} {n}')
            out += ["# HELP http_request_duration_seconds Request duration by route.", "# TYPE http_request_duration_seconds histogram"]
            for r, h in sorted(self.hist.items()):
                for edge, n in zip(BUCKETS, h["buckets"]):
                    out.append(f'http_request_duration_seconds_bucket{{route="{_esc(r)}",le="{edge}"}} {n}')
                out.append(f'http_request_duration_seconds_bucket{{route="{_esc(r)}",le="+Inf"}} {h["count"]}')
                out.append(f'http_request_duration_seconds_sum{{route="{_esc(r)}"}} {h["sum"]:.6f}')
                out.append(f'http_request_duration_seconds_count{{route="{_esc(r)}"}} {h["count"]}')
            out += ["# TYPE http_requests_in_flight gauge", f"http_requests_in_flight {self.in_flight}",
                    "# TYPE process_uptime_seconds gauge", f"process_uptime_seconds {time.time() - self.started:.1f}"]
        return "\n".join(out) + "\n"

    def summary(self) -> dict:
        with self.lock:
            routes = {}
            for r, h in self.hist.items():
                rec = list(self.recent.get(r, ()))
                errors = sum(n for (m, rr, s), n in self.counts.items() if rr == r and s >= 500)
                routes[r] = {"count": h["count"], "errors_5xx": errors, "mean_ms": round(1000 * h["sum"] / h["count"], 2),
                             "p50_ms": _ms(percentile(rec, 50)), "p95_ms": _ms(percentile(rec, 95)), "p99_ms": _ms(percentile(rec, 99)),
                             "max_ms": _ms(max(rec) if rec else None), "percentiles_from_last": len(rec)}
            return {"uptime_seconds": round(time.time() - self.started, 1), "in_flight": self.in_flight,
                    "total_requests": sum(self.counts.values()), "routes": dict(sorted(routes.items()))}


def _ms(x):
    return None if x is None else round(1000 * x, 2)


def _esc(v: str) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


METRICS = Metrics()


# ---- logging ----
def log_event(event: str, level: str = "info", **fields) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "level": level, "event": event, **fields}
    line = json.dumps(rec, ensure_ascii=False, default=str) if os.getenv("LOG_FORMAT", "text").lower() == "json" \
        else " ".join(f"{k}={v}" for k, v in rec.items())
    with _print_lock:
        print(line, file=sys.stdout, flush=True)


def check_token(authorization: str | None) -> bool:
    """True when no token is configured, or the Authorization header carries it (constant-time comparison)."""
    token = os.getenv("METRICS_TOKEN", "")
    if not token:
        return True
    got = (authorization or "")
    return got.lower().startswith("bearer ") and hmac.compare_digest(got[7:].strip().encode(), token.encode())


class RequestObservability:
    """Pure ASGI middleware, the outermost one: request id, timing, metrics and the optional request log."""
    QUIET = ("/metrics",)                                         # scraping the metrics is not itself worth logging

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers") or [])
        rid = clean_request_id(headers.get(b"x-request-id", b"").decode("latin-1") or None)
        token = _request_id.set(rid)
        status = {"code": 500}

        async def send_with_id(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message["headers"] = [(k, v) for k, v in (message.get("headers") or []) if k.lower() != b"x-request-id"] + [(b"x-request-id", rid.encode())]
            await send(message)

        METRICS.begin()
        t0 = time.perf_counter()
        error = None
        try:
            await self.app(scope, receive, send_with_id)
        except BaseException as e:                                  # the server turns this into a 500; we still count and log it
            error = type(e).__name__
            raise
        finally:
            took = time.perf_counter() - t0
            METRICS.end()
            route = getattr(scope.get("route"), "path", None) or "unmatched"
            code = 500 if error else status["code"]
            METRICS.observe(scope.get("method", "?"), route, code, took)
            path = scope.get("path", "")
            if os.getenv("REQUEST_LOG", "") not in ("", "0") and not path.startswith(self.QUIET):
                log_event("request", "error" if code >= 500 else "info", request_id=rid, method=scope.get("method"), route=route,
                          path=path if route == "unmatched" else None, status=code, duration_ms=round(1000 * took, 2), **({"error": error} if error else {}))
            _request_id.reset(token)

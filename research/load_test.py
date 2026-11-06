"""Load test for a HOSTED copy of the app (one with private sessions per visitor, MULTI_SESSION=1), e.g. your Render site.

    python research/load_test.py --url https://your-app.onrender.com [--users 5] [--duration 20] [--pace 0.2] [--json results.json]

Each virtual user is a separate browser (own cookie, so own private draft). It saves a draft setup, then loops: list the best available players,
ask for a recommendation, draft the top player with an Idempotency-Key, and every few picks send the SAME pick again with the SAME key to
check the retry returns the original result and does not draft twice. When a draft completes it resets and goes on. At the end of each
draft it checks that the server's draft log has exactly as many picks as the user made.

Safety: a single-user server (your own copy) keeps ONE draft, and this test would overwrite it. The script asks the server (GET /session) and
refuses unless it is hosted mode, or you pass --allow-reset-draft. It also caps users at 50 and never runs without --url.

Read the numbers honestly: they describe THAT server (free hosting shares a small CPU), from YOUR network, at THAT moment. The first request is
reported separately because a sleeping free-tier host takes a while to wake. Only run it against a server you own."""
import argparse
import asyncio
import json
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import httpx                                     # noqa: E402

from app.observability import percentile         # noqa: E402

MAX_USERS = 50
RETRY_EVERY = 4                                  # every Nth pick is sent a second time with the same key


class Recorder:
    def __init__(self):
        self.samples = defaultdict(list)          # label -> seconds
        self.errors = defaultdict(int)            # label -> count of failures (5xx, network errors, unexpected 4xx)
        self.retries = self.retry_violations = self.log_checks = self.log_violations = 0

    def add(self, label, seconds, ok):
        self.samples[label].append(seconds)
        if not ok:
            self.errors[label] += 1

    def report(self, wall_seconds):
        rows = {}
        for label, v in self.samples.items():
            rows[label] = {"count": len(v), "errors": self.errors[label], "p50_ms": round(1000 * percentile(v, 50), 1),
                           "p95_ms": round(1000 * percentile(v, 95), 1), "p99_ms": round(1000 * percentile(v, 99), 1), "max_ms": round(1000 * max(v), 1)}
        total = sum(r["count"] for r in rows.values()); errs = sum(r["errors"] for r in rows.values())
        return {"endpoints": dict(sorted(rows.items())), "total_requests": total, "total_errors": errs, "wall_seconds": round(wall_seconds, 2),
                "requests_per_second": round(total / wall_seconds, 2) if wall_seconds > 0 else None,
                "error_rate": round(errs / total, 4) if total else None,
                "idempotent_retries": self.retries, "idempotency_violations": self.retry_violations,
                "log_checks": self.log_checks, "log_violations": self.log_violations}


async def timed(client, rec, label, method, url, expect=(), **kw):
    """One request, timed. Failures are 5xx, network errors, and 4xx that the caller did not expect. Returns the response or None."""
    t0 = time.perf_counter()
    try:
        r = await client.request(method, url, **kw)
    except httpx.HTTPError:
        rec.add(label, time.perf_counter() - t0, False)
        return None
    rec.add(label, time.perf_counter() - t0, r.status_code < 400 or (r.status_code < 500 and r.status_code in expect))
    return r


async def virtual_user(base, rec, stop_at, pace, transport, headers):
    async with httpx.AsyncClient(base_url=base, transport=transport, timeout=60.0, follow_redirects=True, headers=headers) as c:
        await timed(c, rec, "draft_setup", "PUT", "/draft/setup", json={"draft_type": "snake", "teams": 12, "my_slot": 3})
        made = 0
        while time.monotonic() < stop_at:
            r = await timed(c, rec, "players_available", "GET", "/players/available", params={"limit": 50, "sort": "rank"})
            players = r.json() if r is not None and r.status_code == 200 else []
            if not players:
                await asyncio.sleep(pace); continue
            await timed(c, rec, "recommend_plan", "GET", "/recommend/plan", params={"limit": 5})
            key = uuid.uuid4().hex
            pick = {"player_id": players[0]["player_id"]}
            r = await timed(c, rec, "draft_pick", "POST", "/draft/pick", expect=(400,), json=pick, headers={"Idempotency-Key": key})
            if r is not None and r.status_code == 200:
                made += 1
                if made % RETRY_EVERY == 0:
                    again = await timed(c, rec, "draft_pick_retry", "POST", "/draft/pick", json=pick, headers={"Idempotency-Key": key})
                    rec.retries += 1
                    if again is None or again.status_code != 200 or again.json() != r.json():
                        rec.retry_violations += 1
            elif r is not None and r.status_code == 400 and "complete" in r.text:
                await check_log(c, rec, made); made = 0
                await timed(c, rec, "draft_reset", "POST", "/draft/reset")
            await asyncio.sleep(pace)
        await check_log(c, rec, made)


async def check_log(c, rec, made):
    r = await timed(c, rec, "draft_log", "GET", "/draft/log")
    if r is not None and r.status_code == 200:
        rec.log_checks += 1
        if len(r.json()) != made:
            rec.log_violations += 1


async def run(base, users, duration, pace, allow_reset, transport=None, token=None, out=print):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx.AsyncClient(base_url=base, transport=transport, timeout=120.0, follow_redirects=True, headers=headers) as c:
        t0 = time.perf_counter()
        try:
            first = await c.get("/session")
        except httpx.HTTPError as e:
            out(f"Could not reach {base} ({type(e).__name__}). Check the address, and that the server is awake."); return None, 2
        cold = time.perf_counter() - t0
        if first.status_code != 200:
            out(f"{base}/session answered {first.status_code}: is this the draft app?"); return None, 2
        mode = first.json().get("mode")
    out(f"First request (includes any wake-up of a sleeping host): {cold:.2f} s. Server mode: {mode}.")
    if mode != "multi" and not allow_reset:
        out("This server is in single-user mode: it keeps ONE draft and this test would overwrite it. Run the test against a hosted copy "
            "(MULTI_SESSION=1), or add --allow-reset-draft if the draft on that server is disposable.")
        return None, 2
    rec = Recorder()
    out(f"Running {users} virtual user(s) for {duration:g} s ...")
    t0 = time.perf_counter(); stop_at = time.monotonic() + duration
    await asyncio.gather(*[virtual_user(base, rec, stop_at, pace, transport, headers) for _ in range(users)])
    rep = rec.report(time.perf_counter() - t0)
    rep["first_request_seconds"] = round(cold, 3); rep["mode"] = mode; rep["users"] = users
    return rep, 0


def print_report(rep, out=print):
    out(f"\n{'endpoint':<20}{'count':>7}{'errors':>8}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'max ms':>9}")
    for label, r in rep["endpoints"].items():
        out(f"{label:<20}{r['count']:>7}{r['errors']:>8}{r['p50_ms']:>9}{r['p95_ms']:>9}{r['p99_ms']:>9}{r['max_ms']:>9}")
    out(f"\n{rep['total_requests']} requests in {rep['wall_seconds']} s = {rep['requests_per_second']} requests/s; "
        f"{rep['total_errors']} failed ({(rep['error_rate'] or 0):.1%}).")
    out(f"Correctness under load: {rep['idempotent_retries']} duplicate-key retries sent, {rep['idempotency_violations']} not answered with the original result; "
        f"{rep['log_checks']} end-of-draft log checks, {rep['log_violations']} with the wrong number of picks.")
    out("These numbers describe this server, your network and this moment. They are not a capacity guarantee.")


def main(argv=None, transport=None, out=print) -> int:
    ap = argparse.ArgumentParser(description="Load test for a hosted copy of the draft app.")
    ap.add_argument("--url", required=True); ap.add_argument("--users", type=int, default=5); ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--pace", type=float, default=0.2, help="seconds each user waits between rounds"); ap.add_argument("--json")
    ap.add_argument("--allow-reset-draft", action="store_true"); ap.add_argument("--token", help="METRICS_TOKEN-style bearer token if the server needs one")
    a = ap.parse_args(argv)
    if not 1 <= a.users <= MAX_USERS:
        out(f"--users must be between 1 and {MAX_USERS}."); return 2
    if a.duration <= 0 or a.pace < 0:
        out("--duration must be above 0 and --pace cannot be negative."); return 2
    rep, code = asyncio.run(run(a.url.rstrip("/"), a.users, a.duration, a.pace, a.allow_reset_draft, transport, a.token, out))
    if rep is None:
        return code
    print_report(rep, out)
    if a.json:
        Path(a.json).write_text(json.dumps(rep, indent=1), encoding="utf-8"); out(f"Saved {a.json}")
    return 1 if rep["idempotency_violations"] or rep["log_violations"] else 0


if __name__ == "__main__":
    sys.exit(main())

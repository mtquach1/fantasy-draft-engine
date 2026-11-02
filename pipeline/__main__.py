"""Command line:  python -m pipeline <command>

  ingest [--raw data/nba/raw]        store each per_game_<year>.csv as a versioned snapshot (quality gate first)
  snapshots                          list stored snapshots; the current one for each table is marked
  rollback KIND ID                   make an older snapshot current again (then run 'materialize')
  materialize [--out data/nba/raw]   write the current snapshots back to files for build_own_model.py
  events [--limit 20]                show the latest events the app has received
  team "Player Name"                 a player's team history (effective-dated)
  replay FILE [--url ...]            send a recorded event feed to a running app, in bursts

The database defaults to data/pipeline.sqlite, the same file the app uses in single-user mode."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _store(args):
    from .store import Store
    return Store(args.db or ROOT / "data" / "pipeline.sqlite")


def cmd_ingest(args) -> int:
    from .ingest import ingest_raw_dir
    from .sources import SourceError
    store = _store(args)
    try:
        results = ingest_raw_dir(store, args.raw or ROOT / "data" / "nba" / "raw")
    except SourceError as e:
        print(e)
        return 1
    bad = 0
    for r in results:
        line = f"{r['kind']:16s} {r['status']:10s}"
        if r.get("snapshot_id"):
            line += f" snapshot #{r['snapshot_id']}"
        print(line)
        for w in r.get("quality", {}).get("warnings", []):
            print(f"    warning: {w}")
        for e in r.get("quality", {}).get("errors", []) + ([r["error"]] if r.get("error") else []):
            print(f"    ERROR: {e}")
        bad += r["status"] in ("rejected", "failed")
    return 1 if bad else 0


def cmd_snapshots(args) -> int:
    snaps = _store(args).snapshots()
    if not snaps:
        print("No snapshots yet. Run: python -m pipeline ingest")
        return 0
    for s in snaps:
        print(f"{'*' if s['current'] else ' '} #{s['id']:<3d} {s['kind']:18s} {s['rows']:>5d} rows  {s['created_at'][:19]}  {s['sha256'][:10]}")
    print("* = current")
    return 0


def cmd_rollback(args) -> int:
    try:
        _store(args).set_current(args.kind, args.id)
    except ValueError as e:
        print(e)
        return 1
    print(f"{args.kind}: snapshot #{args.id} is now current. Run 'python -m pipeline materialize' to rewrite the files.")
    return 0


def cmd_materialize(args) -> int:
    from .ingest import materialize
    written = materialize(_store(args), args.out or ROOT / "data" / "nba" / "raw")
    for w in written:
        print("wrote", w)
    if not written:
        print("Nothing to write: no snapshots yet.")
    return 0


def cmd_events(args) -> int:
    rows = _store(args).recent_events(args.limit)
    if not rows:
        print("No events yet.")
    for r in rows:
        print(f"#{r['seq']:<4d} {r['ts'][:19]}  {r['type']:9s} {str(r['player_name'] or r['player_id'])[:24]:24s} {r['value']!s:8s} {r['source'] or ''}")
    return 0


def cmd_team(args) -> int:
    from .ingest import app_player_id
    hist = _store(args).team_history(app_player_id(args.player))
    if not hist:
        print("No team history for that player (names are matched exactly, in lowercase).")
        return 1
    for h in hist:
        print(f"{h['effective_from'][:10]}  {h['team']:6s}  {h['source']}")
    return 0


def cmd_replay(args) -> int:
    import httpx
    from .replay import load_events, replay
    try:
        events = load_events(args.file)
    except (OSError, ValueError) as e:
        print(f"{args.file}: {e}")
        return 1
    url = args.url.rstrip("/") + "/events"

    def post(batch):
        r = httpx.post(url, json={"events": batch}, timeout=30.0)
        if r.status_code != 200:
            raise SystemExit(f"The app answered {r.status_code}: {r.text[:300]}")
        return r.json()

    def show(n, group, resp):
        print(f"burst {n}: {len(group)} event(s) -> accepted {len(resp['accepted'])}, duplicates {len(resp['duplicates'])}, "
              f"rejected {len(resp['rejected'])}, recomputed {resp['recomputes']}x")
        for rej in resp["rejected"]:
            print(f"    rejected {rej['event_id']}: {rej['reason']}")

    try:
        total = replay(events, post, window=args.window, speed=args.speed, on_burst=show)
    except httpx.HTTPError as e:
        print(f"Couldn't reach the app at {url} ({e}). Is the server running?")
        return 1
    print(f"done: {total['events']} events in {total['bursts']} burst(s); {total['recomputes']} recompute(s) instead of {total['accepted']}.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pipeline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", help="path of the SQLite database (default data/pipeline.sqlite)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("ingest"); p.add_argument("--raw"); p.set_defaults(fn=cmd_ingest)
    sub.add_parser("snapshots").set_defaults(fn=cmd_snapshots)
    p = sub.add_parser("rollback"); p.add_argument("kind"); p.add_argument("id", type=int); p.set_defaults(fn=cmd_rollback)
    p = sub.add_parser("materialize"); p.add_argument("--out"); p.set_defaults(fn=cmd_materialize)
    p = sub.add_parser("events"); p.add_argument("--limit", type=int, default=20); p.set_defaults(fn=cmd_events)
    p = sub.add_parser("team"); p.add_argument("player"); p.set_defaults(fn=cmd_team)
    p = sub.add_parser("replay"); p.add_argument("file"); p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--window", type=float, default=1.0, help="seconds of recorded time that count as one burst")
    p.add_argument("--speed", type=float, default=0.0, help="0 = send at once; 60 = one recorded minute per second")
    p.set_defaults(fn=cmd_replay)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())

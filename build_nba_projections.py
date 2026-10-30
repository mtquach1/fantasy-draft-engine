#!/usr/bin/env python3
"""Build data/nba/players.csv (and friends) from free Basketball-Reference exports plus your Yahoo-style top-N ranking.

Put these in data/nba/raw/ :
  per_game_2026.csv   Basketball-Reference > the 2025-26 season > Per Game table > Share & Export > Get table as CSV
  per_game_2025.csv   (optional but recommended) the season before
  per_game_2024.csv   (optional but recommended) the one before that
  rankings.txt        your top-200 list, pasted as is (lines like "1. 1. Nikola Jokić")
Optional:
  positions.csv       name,pos     (positions for rookies, if you don't use  python yahoo_check.py positions )
  injuries.csv        name,games_out   (stars expected to miss the start of this season)

Then:  python build_nba_projections.py
The files are named by the year the season ENDS (2025-26 = 2026). Newest first is worked out from the file names.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app import nba_build  # noqa: E402
from app.ingestion import data_dir  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build NBA projections")
    ap.add_argument("--raw-dir", help="folder holding the input files (default: data/nba/raw)")
    args = ap.parse_args(argv)
    out_dir = data_dir("nba")
    raw = Path(args.raw_dir) if args.raw_dir else out_dir / "raw"
    found = sorted(((int(m.group(1)), p) for p in raw.glob("per_game_*.csv") if (m := re.fullmatch(r"per_game_(\d{4})\.csv", p.name))),
                   reverse=True)[:3]
    rk = raw / "rankings.txt"
    problems = []
    if not found:
        problems.append(f"No per_game_YYYY.csv file in {raw} (for example per_game_2026.csv).")
    if not rk.exists():
        problems.append(f"No rankings.txt in {raw}: paste your top-200 list into it.")
    if problems:
        print("\n".join(problems))
        return 1
    seasons = [p for _, p in found]
    print("Using seasons:", ", ".join(f"{y} ({p.name})" for y, p in found))
    if len(found) < 3:
        print(f"Note: only {len(found)} season file(s). Two or three give much better numbers for injured stars.")
    existing = out_dir / "players.csv"
    if existing.exists():
        shutil.copy(existing, out_dir / "players.previous.csv")
    yahoo_pos = out_dir / "positions_yahoo.csv"
    try:
        report = nba_build.build(seasons, rk.read_text(encoding="utf-8-sig"), found[0][0], out_dir,
                                 positions_file=raw / "positions.csv", yahoo_positions_file=yahoo_pos if yahoo_pos.exists() else None,
                                 injuries_file=raw / "injuries.csv")
    except nba_build.BuildError as e:
        print(f"Problem: {e}")
        return 1
    print()
    print(nba_build.format_report(report))
    print(f"Wrote players.csv, history.csv, player_history.csv, adp.csv and build_report.txt in {out_dir}")
    print("Next: restart the server, then open http://127.0.0.1:8000/health/data and http://127.0.0.1:8000/health/adp")
    return 0


if __name__ == "__main__":
    sys.exit(main())

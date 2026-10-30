#!/usr/bin/env python3
"""Build data/nba/players.csv from Basketball-Reference's Simple Projection System (per-36 rates).

The SPS file has per-36 rates only: no minutes, games, position or team. Those come from last season's Per Game export.

Put in data/nba/raw/ :
  sps_projections.csv   the SPS table saved as CSV exactly as exported (two header rows are fine)
  per_game_2026.csv     last season's Per Game table (needed: minutes, games, position, team)
  per_game_2025.csv, per_game_2024.csv   strongly recommended: games and minutes are estimated from up to 3 seasons
                        (weights 3/2/1), pulled toward a baseline for the player's role, and they feed the injury-prone flag
Optional: positions.csv (name,pos)   rankings.txt (becomes adp.csv)
          injuries.csv (name + games_out to subtract games, and/or games_expected to set the games directly)

Run:  python build_sps_projections.py
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app import nba_build  # noqa: E402
from app.ingestion import clean_name, data_dir, parse_positions  # noqa: E402

# Games played barely repeat from year to year (injuries are mostly bad luck), so the estimate leans on a baseline
# for the player's role and only partly on his own history: SHRINK is the share given to that history.
SEASON_WEIGHTS = (3.0, 2.0, 1.0)          # newest season first
SHRINK = 0.30
MIN_CLIP = (8.0, 38.0)          # projected minutes per game


def baseline_games(mpg: float) -> float:
    return 68.0 if mpg >= 28 else 64.0 if mpg >= 20 else 55.0


def history_estimate(rows: list) -> tuple[float, float, int]:
    """rows: newest season first, each (games, minutes_per_game) or None if he did not play that season.
    Returns (recency-weighted games, minutes per game weighted by games played and recency, seasons used).
    A season missing between two he played counts as 0 games; before his first season it is ignored."""
    first = max((i for i, r in enumerate(rows) if r), default=-1)
    w = SEASON_WEIGHTS[:len(rows)]
    seen = [(w[i], rows[i][0] if rows[i] else 0.0, rows[i][1] if rows[i] else 0.0) for i in range(first + 1)]
    games = sum(wi * g for wi, g, _ in seen) / sum(wi for wi, _, _ in seen)
    mins_w = sum(wi * g for wi, g, _ in seen)
    mpg = sum(wi * g * m for wi, g, m in seen) / mins_w if mins_w else seen[0][2]
    return games, mpg, first + 1


def read_overrides(path) -> dict:
    """Optional injuries.csv: name + games_out (subtract) and/or games_expected (use this number of games)."""
    if not path or not Path(path).exists():
        return {}
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "name" not in df.columns:
        raise nba_build.BuildError(f"{Path(path).name}: needs a 'name' column (plus games_out and/or games_expected).")
    out = {}
    for r in df.to_dict("records"):
        out[clean_name(r["name"])] = {k: float(r[k]) for k in ("games_out", "games_expected") if k in r and pd.notna(r[k])}
    return out
RENAME = {"3P": "3PM", "TRB": "REB", "TOV": "TO", "FG": "FGM", "FT": "FTM"}
RATES = ["PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]


def read_sps(path) -> pd.DataFrame:
    try:
        raw = pd.read_csv(path, header=None, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = pd.read_csv(path, header=None, dtype=str, encoding="cp1252")
    hdr = next((i for i in range(min(5, len(raw))) if "Player" in raw.iloc[i].astype(str).tolist()), None)
    if hdr is None:
        raise nba_build.BuildError(f"{Path(path).name}: no header row with a 'Player' column.")
    df = raw.iloc[hdr + 1:].copy()
    df.columns = [str(c).strip() for c in raw.iloc[hdr]]
    df = df.loc[:, ~df.columns.duplicated()].rename(columns=RENAME)
    missing = [c for c in ["Player"] + RATES if c not in df.columns]
    if missing:
        raise nba_build.BuildError(f"{Path(path).name}: missing column(s) {', '.join(missing)}.")
    df = df[df["Player"].notna() & (df["Player"] != "Player")].copy()
    for c in RATES:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=RATES)
    df["clean"] = df["Player"].map(clean_name)
    return df.drop_duplicates("clean").reset_index(drop=True)


def build(sps_path, per_game_files, out_dir, last_season, positions_file=None, injuries_file=None, rankings_text=None) -> dict:
    if not per_game_files:
        raise nba_build.BuildError("Need at least last season's per_game_YYYY.csv (for minutes, games, position, team).")
    sps = read_sps(sps_path)
    seasons = [nba_build.read_season(f) for f in per_game_files]          # newest first
    last = seasons[0].set_index("clean")
    extra = {}
    pp = nba_build._read_optional(positions_file, ["name", "pos"])
    if pp is not None:
        extra = {clean_name(r.name): set(parse_positions(r.pos, "nba")) for r in pp.itertuples()}
    over = read_overrides(injuries_file)
    per_season = [{k: (float(g), float(m)) for k, g, m in zip(se["clean"], se["GP"], se["MIN"])} for se in seasons]

    rows, no_data, no_pos = [], [], []
    for r in sps.to_dict("records"):
        key = r["clean"]
        if key not in last.index:
            no_data.append(r["Player"]); continue
        lp = last.loc[key]
        pos = extra.get(key) or set(parse_positions(lp["Pos"], "nba"))
        if not pos:
            no_pos.append(r["Player"]); continue
        hist_games, hist_mpg, n_seasons = history_estimate([ps.get(key) for ps in per_season])
        mpg = float(np.clip(hist_mpg, *MIN_CLIP))
        games = float(np.clip(SHRINK * hist_games + (1 - SHRINK) * baseline_games(mpg), 20, 82))
        o, is_inj = over.get(key, {}), False
        if "games_expected" in o:
            games = float(np.clip(o["games_expected"], 1, 82))
        if "games_out" in o:
            games = max(games - o["games_out"], 10.0)
            is_inj = o["games_out"] >= 10
        tot = {c: float(r[c]) * mpg / 36.0 * games for c in RATES}
        rows.append({"Name": r["Player"], "Team": lp["team"], "Pos": ",".join(sorted(pos, key=["PG", "SG", "SF", "PF", "C"].index)),
                     "playerid": "n:" + key, "GP": round(games), "MIN": round(mpg * games, 1),
                     **{c: round(v, 1) for c, v in tot.items()}, "is_injured": is_inj, "note": f"SPS per-36 x {mpg:.1f} mpg x {games:.0f} g ({n_seasons} season(s) of history)"})
    if not rows:
        raise nba_build.BuildError("No SPS player matched the Per Game file. Are the seasons right (names must match)?")
    out = pd.DataFrame(rows)
    out["playerid"] = out["playerid"] + out.groupby("playerid").cumcount().map(lambda i: "" if i == 0 else f"_{i + 1}")
    out = out.sort_values("PTS", ascending=False).reset_index(drop=True)

    hist, phist = [], []
    for si, s in enumerate(seasons):
        for _, r in s.iterrows():
            pid, season = "n:" + r["clean"], last_season - si
            hist.append({"player_id": pid, "season": season, "games_played": int(r["GP"])})
            phist.append({"Season": season, "playerid": pid, "GP": int(r["GP"]),
                          **{c: round(float(r[c]) * float(r["GP"]), 1) for c in nba_build.STATS}})
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "players.csv", index=False, encoding="utf-8")
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False, encoding="utf-8")
    pd.DataFrame(phist).to_csv(out_dir / "player_history.csv", index=False, encoding="utf-8")
    if rankings_text:
        rk = nba_build.parse_rankings(rankings_text)
        pd.DataFrame({"name": [n for _, n in rk], "adp": [i for i, _ in rk]}).to_csv(out_dir / "adp.csv", index=False, encoding="utf-8")
    rep = {"players": len(out), "no_last_season_data": no_data, "needs_position": no_pos}
    lines = [f"Built {len(out)} players from the SPS file."]
    if len(seasons) < 3:
        lines.append(f"NOTE: only {len(seasons)} Per Game season(s) supplied. Add per_game_2025.csv and per_game_2024.csv "
                     "for a steadier games and minutes estimate (a star with one injury-shortened season is the case that needs it).")
    if no_data:
        lines.append(f"{len(no_data)} SPS players are not in your last-season Per Game file (left out): " + ", ".join(no_data[:15]) + (" ..." if len(no_data) > 15 else ""))
    if no_pos:
        lines.append("NEED A POSITION (left out): " + ", ".join(no_pos) + "  -> add to positions.csv (name,pos) and rerun.")
    (out_dir / "build_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    rep["text"] = "\n".join(lines)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir")
    a = ap.parse_args(argv)
    out_dir = data_dir("nba")
    raw = Path(a.raw_dir) if a.raw_dir else out_dir / "raw"
    found = sorted(((int(m.group(1)), p) for p in raw.glob("per_game_*.csv") if (m := re.fullmatch(r"per_game_(\d{4})\.csv", p.name))), reverse=True)[:3]
    sps = raw / "sps_projections.csv"
    problems = [m for ok, m in ((sps.exists(), f"Missing {sps}"), (bool(found), f"No per_game_YYYY.csv in {raw}")) if not ok]
    if problems:
        print("\n".join(problems)); return 1
    existing = out_dir / "players.csv"
    if existing.exists():
        shutil.copy(existing, out_dir / "players.previous.csv")
    rk = raw / "rankings.txt"
    try:
        rep = build(sps, [p for _, p in found], out_dir, found[0][0], raw / "positions.csv", raw / "injuries.csv",
                    rk.read_text(encoding="utf-8-sig") if rk.exists() else None)
    except nba_build.BuildError as e:
        print(f"Problem: {e}"); return 1
    print(rep["text"]); print(f"Wrote players.csv, history.csv, player_history.csv in {out_dir}. Restart the server, then open /health/data")
    return 0


if __name__ == "__main__":
    sys.exit(main())

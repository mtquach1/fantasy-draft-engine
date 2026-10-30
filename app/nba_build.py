"""Build basketball projections that don't have the two classic holes: injured stars and rookies.

Inputs (Basketball-Reference per-game CSVs for up to 3 seasons, newest first, plus your Yahoo-style top-N ranking):

  INJURED STARS  Last season alone would bury a star who missed most of it. So rates come from up to three seasons,
                 weighted by recency AND by games played (a 20-game year counts for little), and the games you're
                 projected to play come from average availability, not from last year's games.
  ROOKIES        A player with no NBA data takes the average per-game line of the players ranked next to him in the
  & THIN DATA    ranking (same position when possible). A player with only a little data is blended part-way. Every such
                 row is flagged in the `note` column and listed in build_report.txt: these are ESTIMATES that lean on the
                 ranking, so the valuation can only be as good as that ranking for them.

Outputs in the sport's data folder: players.csv (season TOTALS), history.csv (for the injury-prone flag), player_history.csv
(for stat search), adp.csv (from the ranking order) and build_report.txt (what was estimated, and what needs your attention).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .ingestion import clean_name, dedupe_traded, parse_positions, read_csv_robust
from .sports import get_sport

STATS = ["MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]   # per-game columns we keep
RECENCY = [1.0, 0.6, 0.3]          # newest season first
AVAIL_PRIOR = 68.0                 # games a typical rotation player is expected to play
ROOKIE_GAMES = 65.0
FULL_TRUST_GAMES = 60              # raw games over the seasons at which a player's own numbers are fully trusted
COMP_WINDOW = 24                   # look this many ranks either side for comparable players
N_COMPS = 8


class BuildError(Exception):
    """A problem with the input files, worded for a person."""


# ------------------------------------------------------------------ reading
def parse_rankings(text: str) -> list[tuple[int, str]]:
    """'1. 1. Nikola Jokić' / '1. Nikola Jokić' / 'Nikola Jokić' -> [(rank, name)], in the order given."""
    out = []
    for line in text.splitlines():
        line = line.strip().lstrip("\ufeff")
        if not line:
            continue
        m = re.match(r"^(?:\d+[.)]\s*)+(.+)$", line)
        name = (m.group(1) if m else line).strip().strip(",")
        if name and not re.fullmatch(r"[\d.\s]+", name) and not name.lower().startswith(("top ", "rank")):
            out.append((len(out) + 1, name))
    return out


def read_season(path) -> pd.DataFrame:
    """One Basketball-Reference (or similar) export -> one row per player: name, clean, team, pos, GP and PER-GAME stats."""
    sp = get_sport("nba")
    df = read_csv_robust(path).rename(columns=sp.groups[0].rename)
    df = dedupe_traded(df)
    for need in ("name", "GP", "PTS"):
        if need not in df.columns:
            raise BuildError(f"{Path(path).name}: no '{need}' column found. Export the 'Per Game' table from "
                             f"Basketball-Reference as CSV (columns Player, Pos, G, MP, FG, FGA, 3P, FT, FTA, TRB, AST, STL, BLK, TOV, PTS).")
    missing = [c for c in STATS if c not in df.columns and c != "MIN"]
    if missing:
        raise BuildError(f"{Path(path).name}: missing column(s) {', '.join(missing)}. Export the full Per Game table.")
    df = df[df["name"].notna()].copy()
    df["GP"] = pd.to_numeric(df["GP"], errors="coerce")
    df = df[df["GP"] > 0].copy()
    for c in STATS:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df.columns else 0.0
    if df["PTS"].max() >= 60:                               # a TOTALS table: turn it into per-game
        for c in STATS:
            df[c] = df[c] / df["GP"]
    df[STATS] = df[STATS].fillna(0.0)
    df["clean"] = df["name"].map(clean_name)
    df["team"] = df["team"] if "team" in df.columns else ""
    df["Pos"] = df["Pos"] if "Pos" in df.columns else ""
    warn = []
    dup = df.duplicated("clean", keep=False)
    if dup.any():                                           # two different players with one name: keep the one with more minutes
        warn = sorted(set(df.loc[dup, "name"]))
        df = df.sort_values(["clean", "MIN"], ascending=[True, False]).drop_duplicates("clean").copy()
    df.attrs["dup_names"] = warn
    return df.reset_index(drop=True)


def _read_optional(path, cols) -> pd.DataFrame | None:
    if path is None or not Path(path).exists():
        return None
    df = read_csv_robust(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if not set(cols) <= set(df.columns):
        raise BuildError(f"{Path(path).name} needs the columns: {', '.join(cols)}")
    return df


# ------------------------------------------------------------------ the model
def _weighted_rates(rows: list[tuple[int, pd.Series]]) -> tuple[np.ndarray, float]:
    """rows = [(season_index, row)] -> (per-game rates weighted by recency x games, total weight)."""
    w = np.array([RECENCY[min(i, len(RECENCY) - 1)] * float(r["GP"]) for i, r in rows])
    mat = np.array([[float(r[c]) for c in STATS] for _, r in rows])
    return (w[:, None] * mat).sum(0) / w.sum(), float(w.sum())


def comparable_rates(rank: int, positions: set, trusted: pd.DataFrame) -> tuple[np.ndarray, list[str]] | None:
    """Average per-game line of players ranked near `rank` (same position first). `trusted` has rank, pos(set), rates, name."""
    if trusted.empty:
        return None
    near = trusted.assign(d=(trusted["rank"] - rank).abs()).sort_values(["d", "rank"])
    near = near[near["d"] <= COMP_WINDOW] if (near["d"] <= COMP_WINDOW).sum() >= 4 else near.head(N_COMPS)
    if positions:
        same = near[near["pos"].map(lambda p: bool(p & positions))]
        pick = same.head(N_COMPS) if len(same) >= 4 else pd.concat([same, near.drop(same.index)]).head(N_COMPS)
    else:
        pick = near.head(N_COMPS)
    if pick.empty:
        return None
    return np.mean(np.vstack(pick["rates"].to_list()), axis=0), list(pick["name"])


def build(season_files: list, rankings_text: str, last_season: int, out_dir, positions_file=None, yahoo_positions_file=None,
          injuries_file=None) -> dict:
    """Write the projection files and return a report dict (also written to build_report.txt)."""
    if not season_files:
        raise BuildError("Give at least one per-game CSV (newest season first).")
    ranking = parse_rankings(rankings_text)
    if not ranking:
        raise BuildError("The rankings text had no player names.")
    seasons = [read_season(f) for f in season_files]
    dup_names = sorted({n for s in seasons for n in s.attrs.get("dup_names", [])})
    rank_of = {clean_name(n): r for r, n in ranking}
    display = {clean_name(n): n for _, n in ranking}

    extra_pos: dict[str, set] = {}
    yp = _read_optional(yahoo_positions_file, ["player_id", "eligible_positions"])
    if yp is not None:                                      # Yahoo's own eligibility (player_id is 'n:<clean name>')
        for r in yp.itertuples():
            extra_pos[str(r.player_id).replace("n:", "", 1)] = set(parse_positions(r.eligible_positions, "nba"))
    pp = _read_optional(positions_file, ["name", "pos"])
    if pp is not None:
        for r in pp.itertuples():
            extra_pos.setdefault(clean_name(r.name), set(parse_positions(r.pos, "nba")))

    # ---- everyone who has data
    players: dict[str, dict] = {}
    for si, df in enumerate(seasons):
        for _, r in df.iterrows():
            players.setdefault(r["clean"], {"name": r["name"], "rows": []})["rows"].append((si, r))
    for key, p in players.items():
        p["rows"].sort(key=lambda x: x[0])
        p["own"], p["w"] = _weighted_rates(p["rows"])
        p["raw_games"] = sum(float(r["GP"]) for _, r in p["rows"])
        p["avg_gp"] = float(np.mean([float(r["GP"]) for _, r in p["rows"]]))
        newest = p["rows"][0][1]
        p["team"] = newest["team"]
        p["pos"] = extra_pos.get(key) or set(parse_positions(newest["Pos"], "nba"))
    # ---- ranked players with trustworthy data = the comparison pool
    trusted = pd.DataFrame([{"name": p["name"], "rank": rank_of[k], "pos": p["pos"], "rates": p["own"]}
                            for k, p in players.items() if k in rank_of and p["raw_games"] >= FULL_TRUST_GAMES])
    # ---- ranked players with NO data (rookies, new arrivals, spelling differences)
    for key, rank in rank_of.items():
        if key not in players:
            players[key] = {"name": display[key], "rows": [], "own": None, "w": 0.0, "raw_games": 0.0, "avg_gp": None,
                            "team": "", "pos": extra_pos.get(key, set())}

    inj = _read_optional(injuries_file, ["name", "games_out"])
    out_games = {clean_name(r.name): float(r.games_out) for r in inj.itertuples()} if inj is not None else {}

    rows, comp_notes, thin, no_pos, star_fixes = [], [], [], [], []
    for key, p in players.items():
        rank = rank_of.get(key)
        w_comp, comps, note = 0.0, None, "stats"
        if rank is not None and p["raw_games"] < FULL_TRUST_GAMES:
            comps = comparable_rates(rank, p["pos"], trusted)
            if comps is not None:
                w_comp = 1.0 if p["own"] is None else float(np.clip((FULL_TRUST_GAMES - p["raw_games"]) / FULL_TRUST_GAMES, 0, 1))
        if p["own"] is None and comps is None:
            continue                                        # no data and nothing to compare with: leave out
        rates = comps[0] if p["own"] is None else (p["own"] if comps is None else (1 - w_comp) * p["own"] + w_comp * comps[0])
        if p["own"] is None:
            note = "rookie/new: estimated from comparable ranked players (no NBA stats)"
            comp_notes.append((p["name"], rank, comps[1][:4]))
            games = ROOKIE_GAMES
        else:
            games = float(np.clip(0.5 * p["avg_gp"] + 0.5 * AVAIL_PRIOR, 20, 82))
            if w_comp > 0:
                note = f"thin data ({int(p['raw_games'])} games): {round(w_comp * 100)}% blended with comparable ranked players"
                thin.append((p["name"], rank, int(p["raw_games"]), round(w_comp, 2)))
            newest_gp = float(p["rows"][0][1]["GP"]) if p["rows"][0][0] == 0 else None
            if rank is not None and rank <= 100 and newest_gp is not None and newest_gp < 50 and len(p["rows"]) > 1:
                star_fixes.append((p["name"], rank, int(newest_gp), round(p["avg_gp"]), round(games)))
        is_injured = False
        if key in out_games:
            games = max(games - out_games[key], 10.0)
            is_injured = out_games[key] >= 10
        pos = sorted(p["pos"], key=["PG", "SG", "SF", "PF", "C"].index)
        if not pos:
            no_pos.append(p["name"])
            continue                                        # can't be placed in a lineup without a position
        tot = {c: float(rates[i]) * games for i, c in enumerate(STATS)}
        rows.append({"Name": p["name"], "Team": p["team"], "Pos": ",".join(pos), "playerid": "n:" + key, "GP": round(games),
                     **{c: round(v, 1) for c, v in tot.items()}, "is_injured": is_injured, "note": note})

    out = pd.DataFrame(rows)
    # same-name players that share an id must get unique ids (matches the app's own id rule)
    out["playerid"] = out["playerid"] + out.groupby("playerid").cumcount().map(lambda i: "" if i == 0 else f"_{i + 1}")
    out = out.sort_values("PTS", ascending=False).reset_index(drop=True)

    hist, phist = [], []
    for key, p in players.items():
        pid = "n:" + key
        for si, r in p["rows"]:
            season = last_season - si
            hist.append({"player_id": pid, "season": season, "games_played": int(r["GP"])})
            phist.append({"Season": season, "playerid": pid, "GP": int(r["GP"]),
                          **{c: round(float(r[c]) * float(r["GP"]), 1) for c in STATS}})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "players.csv", index=False, encoding="utf-8")
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False, encoding="utf-8")
    pd.DataFrame(phist).to_csv(out_dir / "player_history.csv", index=False, encoding="utf-8")
    pd.DataFrame({"name": [n for _, n in ranking], "adp": [r for r, _ in ranking]}).to_csv(out_dir / "adp.csv", index=False, encoding="utf-8")

    report = {"players": len(out), "seasons": len(seasons), "ranked": len(ranking),
              "ranked_in_file": int(out["Name"].map(clean_name).isin(rank_of).sum()),
              "rookies_estimated": comp_notes, "thin_data": thin, "injured_star_fixes": star_fixes,
              "needs_position": no_pos, "duplicate_names": dup_names}
    (out_dir / "build_report.txt").write_text(format_report(report), encoding="utf-8")
    return report


def format_report(r: dict) -> str:
    L = [f"Built {r['players']} players from {r['seasons']} season file(s); {r['ranked_in_file']} of your {r['ranked']} ranked players are in the file.", ""]
    if r["injured_star_fixes"]:
        L.append("INJURED / SHORT-SEASON STARS (ranked top 100, played under 50 games last season). Their rates use all seasons, and "
                 "their projected games use average availability:")
        L += [f"   #{rk} {n}: last season {last} games, average {avg}, projected {proj}" for n, rk, last, avg, proj in r["injured_star_fixes"]]
        L.append("   If one of them is hurt for the start of this season, list him in injuries.csv (name,games_out).")
        L.append("")
    if r["rookies_estimated"]:
        L.append("ROOKIES / NEW PLAYERS with no NBA stats (estimated from comparable players near them in the ranking):")
        L += [f"   #{rk} {n}  ~ like {', '.join(c)}" for n, rk, c in r["rookies_estimated"]]
        L.append("")
    if r["thin_data"]:
        L.append("THIN DATA (blended with comparable ranked players):")
        L += [f"   #{rk} {n}: {g} games of data, {int(w * 100)}% blended" for n, rk, g, w in r["thin_data"]]
        L.append("")
    if r["needs_position"]:
        L.append("NEED A POSITION (left out until you give one): " + ", ".join(r["needs_position"]))
        L.append("   Run: python yahoo_check.py positions   (pulls Yahoo's positions), or add them to positions.csv (name,pos), then rebuild.")
        L.append("")
    if r["duplicate_names"]:
        L.append("Two players share a name (kept the one with more minutes): " + ", ".join(r["duplicate_names"]))
        L.append("")
    return "\n".join(L).rstrip() + "\n"


"""Generates FAKE but realistic-looking NBA or NFL projections so you can try those sports end to end.
    python make_sample_sport.py nba     -> data/nba/players.csv, player_history.csv, history.csv
    python make_sample_sport.py nfl     -> data/nfl/players.csv, player_history.csv, history.csv
Replace them with real projections (same column names) when you have them."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIRST = ["Alex", "Ben", "Carlos", "Derek", "Eli", "Frank", "Gabe", "Hector", "Ian", "Jake", "Kyle", "Luis", "Marcus",
         "Nate", "Omar", "Pete", "Quinn", "Ryan", "Sam", "Tyler", "Victor", "Will", "Xavier", "Zane"]
LAST = ["Alvarez", "Brooks", "Castillo", "Diaz", "Ellis", "Foster", "Garcia", "Hayes", "Ibarra", "Jensen", "Kim", "Lopez",
        "Morales", "Nolan", "Ortiz", "Perez", "Quintero", "Reyes", "Santos", "Turner", "Usher", "Vega", "Walker", "Young"]
NBA_TEAMS = ["BOS", "NYK", "MIL", "DEN", "PHX", "LAL", "GSW", "MIA", "DAL", "PHI", "MIN", "OKC"]
NFL_TEAMS = ["KC", "BUF", "SF", "PHI", "DAL", "BAL", "MIA", "DET", "CIN", "GB", "NYJ", "LAR"]


def _names(rng, n):
    combos = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(combos)
    return combos[:n]


def nba(rng, n: int = 200) -> pd.DataFrame:
    pos_pool = ["PG", "SG", "SF", "PF", "C"] * (n // 5)
    # per-game profile multipliers by position: (reb, ast, stl, blk, 3pm, fg%, ft%)
    prof = {"PG": (0.6, 1.9, 1.3, 0.5, 1.4, .450, .840), "SG": (0.8, 1.0, 1.1, 0.6, 1.3, .455, .830),
            "SF": (1.0, 0.9, 1.0, 0.8, 1.0, .465, .790), "PF": (1.3, 0.8, 0.9, 1.1, 0.7, .490, .760),
            "C": (1.8, 0.7, 0.7, 2.0, 0.3, .560, .690)}
    other = {"PG": "SG", "SG": "SF", "SF": "PF", "PF": "C", "C": "PF"}
    rows = []
    for i, (name, pos) in enumerate(zip(_names(rng, n), rng.permutation(pos_pool)[:n])):
        t = rng.normal(0, 1)
        reb_m, ast_m, stl_m, blk_m, tpm_m, fgp, ftp = prof[pos]
        gp = int(np.clip(rng.normal(68 + 3 * t, 9), 35, 82))
        mpg = float(np.clip(27 + 6 * t + rng.normal(0, 2), 12, 38))
        ppg = max(4.0, 14 + 6.5 * t + rng.normal(0, 1.5))
        fga_pg, fta_pg = ppg * 0.78, ppg * 0.26
        rows.append({
            "Name": name, "Team": NBA_TEAMS[i % 12], "Pos": pos if rng.random() > 0.35 else f"{pos}/{other[pos]}",
            "playerid": str(2000 + i), "GP": gp, "MIN": round(mpg * gp),
            "PTS": round(ppg * gp), "REB": round(max(1.5, 5.0 * reb_m + 1.2 * t) * gp),
            "AST": round(max(0.6, 3.0 * ast_m + 0.9 * t) * gp), "STL": round(max(0.3, 0.9 * stl_m + 0.15 * t) * gp),
            "BLK": round(max(0.1, 0.55 * blk_m + 0.12 * t) * gp), "3PM": round(max(0.0, 1.1 * tpm_m + 0.35 * t) * gp),
            "TO": round(max(0.6, 1.6 + 0.35 * t) * gp),
            "FGA": round(fga_pg * gp), "FGM": round(fga_pg * gp * float(np.clip(fgp + rng.normal(0, .02), .38, .68))),
            "FTA": round(fta_pg * gp), "FTM": round(fta_pg * gp * float(np.clip(ftp + rng.normal(0, .05), .5, .93))),
            "is_injured": False, "games_pct": 1.0})
    return pd.DataFrame(rows)


def nfl(rng) -> pd.DataFrame:
    rows, n = [], 0
    spec = [("QB", 36), ("RB", 80), ("WR", 110), ("TE", 45), ("K", 20), ("DST", 32)]
    names = iter(_names(rng, 300))
    for pos, count in spec:
        for j in range(count):
            t = rng.normal(0, 1) - j * 0.012                      # a bit of depth decay
            g = int(np.clip(rng.normal(16 - (0.6 if t < -1 else 0), 1.2), 8, 17))
            r = {"Name": next(names) if pos != "DST" else f"{NFL_TEAMS[j % 12]} Defense {j}", "Team": NFL_TEAMS[j % 12],
                 "Pos": pos, "playerid": str(3000 + n), "G": g, "is_injured": False, "games_pct": 1.0}
            n += 1
            if pos == "QB":
                att = max(150, 520 + 60 * t) * g / 17
                r.update(pass_att=att, pass_cmp=att * np.clip(.64 + .02 * t, .55, .72), pass_yds=att * (6.9 + .35 * t),
                         pass_tds=max(4, (24 + 6 * t) * g / 17), ints=max(2, (11 - 1.5 * t) * g / 17),
                         rush_att=50 + 25 * abs(t), rush_yds=(50 + 25 * abs(t)) * 4.6, rush_tds=max(0, 2.5 + t), fumbles=4)
            elif pos == "RB":
                ra = max(40, 190 + 85 * t) * g / 17
                rec = max(5, 32 + 18 * t)
                r.update(rush_att=ra, rush_yds=ra * (4.2 + .25 * t), rush_tds=max(0, (7 + 3.5 * t) * g / 17), rec=rec,
                         rec_yds=rec * 7.6, rec_tds=max(0, 2 + t), fumbles=2)
            elif pos == "WR":
                rec = max(15, 62 + 24 * t) * g / 17
                r.update(rec=rec, rec_yds=rec * (12.6 + .8 * t), rec_tds=max(0, (6 + 3 * t) * g / 17), rush_att=3, rush_yds=15, fumbles=1)
            elif pos == "TE":
                rec = max(10, 42 + 20 * t) * g / 17
                r.update(rec=rec, rec_yds=rec * (10.8 + .5 * t), rec_tds=max(0, (4.5 + 2.5 * t) * g / 17), fumbles=1)
            elif pos == "K":
                r.update(fgm=max(12, 26 + 4 * t) * g / 17, xpm=max(15, 36 + 8 * t) * g / 17)
            else:
                r.update(sacks=max(15, 42 + 9 * t), def_int=max(4, 12 + 3 * t), fum_rec=max(3, 8 + 2 * t),
                         def_td=max(0, 2 + t), safeties=max(0.0, .6 + .3 * t), G=17)
            rows.append(r)
    df = pd.DataFrame(rows)
    num = [c for c in df.columns if c not in ("Name", "Team", "Pos", "playerid", "is_injured", "games_pct")]
    df[num] = df[num].fillna(0).round(0).astype(int)
    return df


def history(rng, players: pd.DataFrame, games_col: str, season_games: int, seasons=(2024, 2025, 2026)):
    """Past seasons scaled from each projection, plus a games-played file for the injury-prone flag."""
    num = [c for c in players.columns if c not in ("Name", "Team", "Pos", "playerid", "is_injured", "games_pct")]
    full = players[games_col].max()
    stats, games = [], []
    for _, p in players.iterrows():
        fragile = rng.random() < 0.12
        for season in seasons:
            if fragile and rng.random() < 0.08:
                continue                                               # missed the whole season: no row
            g = int(rng.integers(int(season_games * .3), int(season_games * .75))) if fragile else int(
                np.clip(rng.normal(season_games * .9, season_games * .08), 1, season_games))
            f = g / max(p[games_col], 1) * rng.uniform(.85, 1.15)
            row = {"Season": season, "playerid": p["playerid"]}
            for c in num:
                row[c] = int(round(p[c] * f * (1 if c == games_col else rng.uniform(.8, 1.2))))
            row[games_col] = g
            for made, tried in (("FGM", "FGA"), ("FTM", "FTA"), ("pass_cmp", "pass_att")):
                if made in row:
                    row[made] = min(row[made], row[tried])
            stats.append(row)
            games.append({"player_id": p["playerid"], "season": season, "games_played": g})
    return pd.DataFrame(stats), pd.DataFrame(games)


def generate(sport: str, out_dir: Path, seed: int = 11, n: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    players = nba(rng, n or 200) if sport == "nba" else nfl(rng)
    games_col, season_games = ("GP", 82) if sport == "nba" else ("G", 17)
    stats, games = history(rng, players, games_col, season_games)
    out_dir.mkdir(parents=True, exist_ok=True)
    players.to_csv(out_dir / "players.csv", index=False)
    stats.to_csv(out_dir / "player_history.csv", index=False)
    games.to_csv(out_dir / "history.csv", index=False)
    return {"players": len(players), "history_rows": len(stats)}


if __name__ == "__main__":
    sport = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    if sport not in ("nba", "nfl"):
        sys.exit("Usage: python make_sample_sport.py nba|nfl")
    info = generate(sport, Path(__file__).parent / "data" / sport)
    print(f"Wrote {info['players']} players and {info['history_rows']} past-season rows to data/{sport}/")




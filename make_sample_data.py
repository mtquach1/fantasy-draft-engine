
"""Generates FAKE but realistic-looking projections so you can test the server end to end.
Run from the project root:  python make_sample_data.py
Writes data/hitters.csv and data/pitchers.csv in the same column format as a FanGraphs export
(plus a Pos column, which FanGraphs exports don't have). Replace them with real exports later."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

FIRST = ["Alex", "Ben", "Carlos", "Derek", "Eli", "Frank", "Gabe", "Hector", "Ian", "Jake", "Kyle", "Luis",
         "Marcus", "Nate", "Omar", "Pete", "Quinn", "Ryan", "Sam", "Tyler", "Victor", "Will", "Xavier", "Zane"]
LAST = ["Alvarez", "Brooks", "Castillo", "Diaz", "Ellis", "Foster", "Garcia", "Hayes", "Ibarra", "Jensen",
        "Kim", "Lopez", "Morales", "Nolan", "Ortiz", "Perez", "Quintero", "Reyes", "Santos", "Turner",
        "Usher", "Vega", "Walker", "Young", "Zimmer"]
TEAMS = ["NYY", "BOS", "LAD", "SFG", "HOU", "ATL", "CHC", "SEA", "TEX", "TOR", "PHI", "SDP"]


def names(n):
    combos = [f"{f} {l}" for f in FIRST for l in LAST]
    rng.shuffle(combos)
    return combos[:n]


# ---------------- hitters ----------------
N_H = 300
POS = ["C"] * 30 + ["1B"] * 45 + ["2B"] * 40 + ["3B"] * 42 + ["SS"] * 40 + ["OF"] * 103
rng.shuffle(POS)
MULTI = {"2B": "SS", "SS": "2B", "1B": "3B", "3B": "1B", "OF": "1B"}      # some players qualify at two spots
rows = []
for i, (name, pos) in enumerate(zip(names(N_H), POS)):
    talent = rng.normal(0, 1)                                              # one number drives overall quality
    ab = int(np.clip(rng.normal(480 + 40 * talent, 90), 120, 640))
    bb = int(np.clip(rng.normal(45 + 10 * talent, 15), 8, 110))
    hbp = int(rng.integers(0, 12))
    avg = float(np.clip(rng.normal(0.253 + 0.012 * talent, 0.015), 0.200, 0.320))
    h = int(ab * avg)
    hr = int(np.clip(rng.normal(18 + 9 * talent, 7), 2, 55))
    d = int(np.clip(rng.normal(24, 7), 5, 50))
    t = int(rng.integers(0, 8))
    sb = int(np.clip(rng.exponential(8) + (2 if pos in ("SS", "2B", "OF") else 0), 0, 55))
    posstr = pos if (pos == "C" or rng.random() > 0.15) else f"{pos}/{MULTI[pos]}"
    rows.append({
        "Name": name, "Team": TEAMS[i % len(TEAMS)], "playerid": str(1000 + i), "Pos": posstr,
        "PA": ab + bb + hbp, "AB": ab, "H": h, "1B": max(h - hr - d - t, 0), "2B": d, "3B": t, "HR": hr,
        "R": int(np.clip(rng.normal(70 + 15 * talent, 15), 20, 135)),
        "RBI": int(np.clip(rng.normal(68 + 16 * talent, 17), 15, 140)),
        "BB": bb, "SO": int(np.clip(rng.normal(110, 28), 40, 220)), "HBP": hbp, "SB": sb,
        "is_injured": False, "games_pct": 1.0,
    })
hit = pd.DataFrame(rows)
inj = hit.sample(6, random_state=1).index                                  # a few injured hitters
hit.loc[inj, "is_injured"] = True
hit.loc[inj, "games_pct"] = [0.75, 0.6, 0.85, 0.5, 0.7, 0.9]
hit.to_csv(DATA / "hitters.csv", index=False)

# ---------------- pitchers ----------------
N_SP, N_RP = 140, 110
rows = []
for i, name in enumerate(names(N_SP + N_RP)):
    sp = i < N_SP
    talent = rng.normal(0, 1)
    if sp:
        ip = float(np.clip(rng.normal(150 + 20 * talent, 30), 60, 215)); gs = int(ip / 5.8); g = gs
        era = float(np.clip(rng.normal(3.9 - 0.45 * talent, 0.5), 2.4, 5.6))
        k9 = float(np.clip(rng.normal(8.8 + 0.9 * talent, 1.1), 6, 12.5))
        w, sv, qs = int(ip / 11.5), 0, int(gs * 0.5)
    else:
        ip = float(np.clip(rng.normal(62, 12), 35, 85)); gs, g = 0, int(ip / 1.0)
        era = float(np.clip(rng.normal(3.7 - 0.4 * talent, 0.6), 2.0, 5.5))
        k9 = float(np.clip(rng.normal(10.2 + 0.9 * talent, 1.3), 7, 14))
        w, sv, qs = int(rng.integers(2, 7)), int(np.clip(rng.normal(8 + 8 * talent, 10), 0, 42)), 0
    ip = round(ip, 1)
    whip = float(np.clip(rng.normal(1.25 - 0.08 * talent, 0.09), 0.9, 1.6))
    bb = int(ip * rng.normal(0.32, 0.06)); hits = int(max(ip * whip - bb, 0))
    rows.append({
        "Name": name, "Team": TEAMS[(i + 5) % len(TEAMS)], "playerid": str(5000 + i), "G": g, "GS": gs,
        "IP": ip, "W": w, "L": int(ip / 14), "SV": sv, "QS": qs, "SO": int(ip * k9 / 9), "H": hits,
        "ER": int(ip * era / 9), "BB": bb,
    })
pd.DataFrame(rows).to_csv(DATA / "pitchers.csv", index=False)
print(f"Wrote {len(hit)} hitters and {len(rows)} pitchers to {DATA}")

# ---------------- injury history (hitters only) ----------------
# data/history.csv lets the injury-prone flag work: ~12% of hitters are fragile (miss 25%+ of games).
hist = []
for pid in hit["playerid"]:
    fragile = rng.random() < 0.12
    for season in (2024, 2025, 2026):
        if fragile and rng.random() < 0.08:
            continue                                   # whole season missed -> no row (counts as 100% missed)
        g = int(rng.integers(60, 125)) if fragile else int(np.clip(rng.normal(148, 12), 100, 162))
        hist.append({"player_id": pid, "season": season, "games_played": g})
pd.DataFrame(hist).to_csv(DATA / "history.csv", index=False)
print(f"Wrote {len(hist)} history rows to {DATA / 'history.csv'}")

# ---------------- past-season stats for the player search (FAKE, generated around each projection) ----------------
# data/hitter_history.csv and data/pitcher_history.csv. IP uses baseball notation (180.1 = 180 1/3).
hit_by_id = hit.set_index("playerid")
hrows = []
for r in hist:                                          # same player-seasons as history.csv, so games line up
    p, g = hit_by_id.loc[r["player_id"]], r["games_played"]
    f = g / 148 * rng.uniform(0.9, 1.1)
    u = lambda: rng.uniform(0.75, 1.25)
    ab = max(int(p["AB"] * f), 20)
    h = int(ab * float(np.clip(p["H"] / p["AB"] + rng.normal(0, 0.02), 0.19, 0.34)))
    hr = min(int(p["HR"] * f * u()), h)
    d = min(int(p["2B"] * f * u()), h - hr)
    t = min(int(p["3B"] * f * u()), h - hr - d)
    bb, hbp, sf = int(p["BB"] * f * u()), int(p["HBP"] * f), int(rng.integers(0, 7))
    hrows.append({"Season": r["season"], "playerid": r["player_id"], "G": g, "PA": ab + bb + hbp + sf, "AB": ab,
                  "H": h, "1B": h - d - t - hr, "2B": d, "3B": t, "HR": hr, "R": int(p["R"] * f * u()),
                  "RBI": int(p["RBI"] * f * u()), "BB": bb, "SO": int(p["SO"] * f * u()), "HBP": hbp, "SF": sf,
                  "SB": int(p["SB"] * f * u())})
pd.DataFrame(hrows).to_csv(DATA / "hitter_history.csv", index=False)

pit = pd.read_csv(DATA / "pitchers.csv", dtype={"playerid": str})
prow = []
for _, p in pit.iterrows():
    for season in (2024, 2025, 2026):
        if season == 2024 and rng.random() < 0.1:
            continue                                    # rookies: no 2024 line
        f = rng.uniform(0.75, 1.1)
        ip_true = float(p["IP"]) * f
        whole, thirds = int(ip_true), int(round((ip_true - int(ip_true)) * 3))
        if thirds == 3:
            whole, thirds = whole + 1, 0
        prow.append({"Season": season, "playerid": p["playerid"], "G": max(int(p["G"] * f), 1),
                     "GS": int(p["GS"] * f), "IP": whole + thirds / 10, "W": int(p["W"] * f),
                     "L": int(p["L"] * f), "SV": int(p["SV"] * f), "HLD": int(rng.integers(0, 12)) if p["GS"] == 0 else 0,
                     "QS": int(p["QS"] * f), "SO": int(p["SO"] * f * rng.uniform(0.9, 1.1)),
                     "H": int(p["H"] * f * rng.uniform(0.9, 1.1)), "ER": int(p["ER"] * f * rng.uniform(0.8, 1.25)),
                     "BB": int(p["BB"] * f * rng.uniform(0.85, 1.15)),
                     "HR": int(ip_true / 9 * rng.normal(1.1, 0.3))})
pd.DataFrame(prow).to_csv(DATA / "pitcher_history.csv", index=False)
print(f"Wrote {len(hrows)} hitter-season and {len(prow)} pitcher-season stat rows (hitter_history.csv, pitcher_history.csv)")




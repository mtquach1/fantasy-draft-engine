"""Did the model's projected risers and fallers actually rise and fall?   (the honest test behind any 'breakout' claim)

Uses only data that existed before the drafted season: the model sees per-game files up to the second-to-last season, projects the last
one, and we compare its projected CHANGE in fantasy points per game (projection minus last season's rate) with the REAL change.
Reported overall and by age group, plus the 20 players the model expected to rise the most.

What this does NOT cover: rookies and players with no history (the model ignores pre-debut seasons) and role changes from trades or injuries
to teammates. Those enter the app only through data/nba/overrides.csv, which is hand-entered judgment. Do not describe override
entries as something the model found.

Run from the project root:  python research/growth_check.py     (needs three per-game files in data/nba/raw)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import ingestion, own_model as om      # noqa: E402

OUT_CSV = ROOT / "research" / "growth_check_players.csv"
MIN_MINS = 800           # only players with a real sample in the last history season


def boot_corr(x, y, n=3000, seed=0):
    rng = np.random.default_rng(seed); r = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            r.append(np.corrcoef(x[i], y[i])[0, 1])
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def main() -> int:
    files = om.find_seasons(ingestion.DATA_DIR / "nba" / "raw"); yrs = sorted(files)
    if len(yrs) < 3:
        print("Need three per-game seasons in data/nba/raw."); return 1
    a_y, b_y, c_y = yrs[-3:]
    S = {y: om.load_season(files[y]) for y in (a_y, b_y, c_y)}
    p = om.Params(); beta, _ = om.fit_age_from_transition(S, [a_y, b_y], p)
    proj = om.project_next(S, [b_y, a_y], p, beta)
    last, act = S[b_y], S[c_y]
    w = om.W
    fp_tot = sum(wt * proj[col] for col, wt in w.items())
    proj_fpg = (fp_tot / proj["GP"]).rename(index=lambda i: i)
    df = pd.DataFrame({"name": proj["Name"], "age": last["Age"].reindex(proj.index) + 1, "proj_fpg": proj_fpg})
    df["last_fpg"] = last["fpg"].reindex(df.index)
    df["last_min"] = last["mins"].reindex(df.index)
    df["act_fpg"] = act["fpg"].reindex(df.index)
    df["act_min"] = act["mins"].reindex(df.index)
    df = df[(df["last_min"] >= MIN_MINS) & (df["act_min"] >= MIN_MINS)].dropna()
    if len(df) < 30:
        print(f"Only {len(df)} players with enough minutes in both seasons: too few to say anything."); return 1
    df["proj_chg"] = df["proj_fpg"] - df["last_fpg"]
    df["act_chg"] = df["act_fpg"] - df["last_fpg"]
    x, y = df["proj_chg"].to_numpy(), df["act_chg"].to_numpy()
    lo, hi = boot_corr(x, y)
    print(f"Season drafted for: {c_y}. Players with >= {MIN_MINS} minutes in both {b_y} and {c_y}: {len(df)}")
    print(f"Correlation between projected change and real change in fantasy points per game: {np.corrcoef(x, y)[0, 1]:+.2f} (95% CI {lo:+.2f} to {hi:+.2f})")
    print(f"Share of players where the model got the direction right: {(np.sign(x) == np.sign(y)).mean():.0%}  (a coin flip would be about 50%)")
    print(f"Average real change when the model expected a rise: {y[x > 0].mean():+.1f}   when it expected a fall: {y[x < 0].mean():+.1f} (fantasy points per game)\n")
    df["group"] = pd.cut(df["age"], [0, 23.5, 27.5, 99], labels=["age <= 23", "age 24-27", "age 28+"])
    print(f"{'group':10s} {'n':>4s} {'projected change':>17s} {'real change':>12s} {'correlation':>12s}")
    for g, d in df.groupby("group", observed=True):
        c = np.corrcoef(d["proj_chg"], d["act_chg"])[0, 1] if len(d) > 4 else float("nan")
        print(f"{str(g):10s} {len(d):4d} {d['proj_chg'].mean():+17.1f} {d['act_chg'].mean():+12.1f} {c:+12.2f}")
    top = df.sort_values("proj_chg", ascending=False).head(20)
    print(f"\nThe 20 biggest projected risers: model said {top['proj_chg'].mean():+.1f} on average, they really changed by {top['act_chg'].mean():+.1f}; "
          f"{(top['act_chg'] > 0).sum()} of 20 actually rose.")
    out = df.sort_values("proj_chg", ascending=False)[["name", "age", "last_fpg", "proj_fpg", "act_fpg", "proj_chg", "act_chg"]].round(1)
    out.to_csv(OUT_CSV, index=False)
    print("\nPlayer-level table written to research/growth_check_players.csv (derived from your files: it is git-ignored, do not publish it).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

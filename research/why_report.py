"""Where the app's team differs from an autodraft-style team: a stat-by-stat breakdown of the two starting lineups.

Reads research/draft_backtest_results.csv (written by research/draft_backtest.py, which scores every team with the REAL stats of the
season it drafted for). For each paired draft (same seed, same slot, same opponents) it compares the lineup the app drafted with the
lineup 'follow_adp' drafted. 'follow_adp' stands in for autodraft: it always takes the best player left by last season's per-game
production. (It is NOT ESPN's actual autodraft, whose rules we cannot see.)

Each stat gets: the average difference in the starters' season totals, what that is worth in fantasy points under the ESPN default
weights, and a 95% bootstrap interval over paired drafts. Fantasy points = PTS + 2*FGM - FGA + FTM - FTA + 3PM + REB + 2*AST + 4*STL + 4*BLK - 2*TO,
so the contributions add up to the lineup's drafted-starter points (empty starting slots are filled from the waiver pool in the score
itself, so the two totals can differ a little).

Run from the project root:  python research/why_report.py [other=follow_adp] [mine=bot] [results.csv]
Writes research/WHY_REPORT.md (aggregate numbers only, no player data from the source files)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "research" / "draft_backtest_results.csv"
OUT = ROOT / "research" / "WHY_REPORT.md"

# (label, {stat: weight}) -- groups of the ESPN default points formula
GROUPS = [
    ("Scoring volume (PTS + 3PM)", {"PTS": 1, "3PM": 1}),
    ("Shooting efficiency (2*FGM - FGA + FTM - FTA)", {"FGM": 2, "FGA": -1, "FTM": 1, "FTA": -1}),
    ("Rebounding", {"REB": 1}),
    ("Assists", {"AST": 2}),
    ("Steals + blocks", {"STL": 4, "BLK": 4}),
    ("Turnovers (fewer is better)", {"TO": -2}),
]


def boot(d, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    csv_path, out_path = CSV, OUT
    paths = [a for a in argv if a.lower().endswith(".csv")]
    if paths:                                                   # e.g. research/draft_backtest_results_merged.csv
        csv_path = Path(paths[0])
        out_path = csv_path.with_name("WHY_REPORT_" + csv_path.stem.replace("draft_backtest_results", "").strip("_") + ".md")
    argv = [a for a in argv if not a.lower().endswith(".csv")]
    OTHER = argv[0] if len(argv) > 0 else "follow_adp"
    MINE = argv[1] if len(argv) > 1 else "bot"
    if not csv_path.exists():
        print(f"{csv_path} not found. Run research/draft_backtest.py first."); return 1
    df = pd.read_csv(csv_path)
    st = [c for c in df.columns if c.startswith("st_")]
    if not st:
        print("The results file has no lineup-stat columns (st_*). It was made by an older draft_backtest.py: run that again."); return 1
    pv = df.pivot_table(index=["seed", "slot"], columns="who", values=["score"] + st)
    if ("score", OTHER) not in pv or ("score", MINE) not in pv:
        print(f"Need both '{MINE}' and '{OTHER}' in the results. Found: {sorted(df['who'].unique())}"); return 1
    n = len(pv)
    lines = [f"# '{MINE}' versus '{OTHER}': where the difference comes from", "",
             f"{n} paired drafts (same seed, slot and opponents), scored by the real stats of the drafted season, starting lineup only. "
             f"'{OTHER}' = best player left by last season's per-game production (a stand-in for autodraft, not ESPN's actual algorithm).", ""]
    d_score = (pv[("score", MINE)] - pv[("score", OTHER)]).values
    lo, hi = boot(d_score)
    lines += [f"**Season score difference: {d_score.mean():+.0f} points (95% CI {lo:+.0f} to {hi:+.0f}); ahead in {(d_score > 0).mean():.0%} of drafts.**", "",
              "| Source of the difference | Fantasy points | 95% CI |", "|---|---:|---:|"]
    total = np.zeros(n)
    for label, w in GROUPS:
        if not all(("st_" + k, MINE) in pv for k in w):
            continue
        d = sum(wt * (pv[("st_" + k, MINE)] - pv[("st_" + k, OTHER)]).values for k, wt in w.items())
        total += d
        lo, hi = boot(d)
        lines.append(f"| {label} | {d.mean():+.0f} | {lo:+.0f} to {hi:+.0f} |")
    lo, hi = boot(total)
    lines += [f"| **Sum of the above (drafted starters)** | **{total.mean():+.0f}** | {lo:+.0f} to {hi:+.0f} |", ""]
    lines += ["Availability and role (per starter, real season):", "", "| | " + MINE + " | " + OTHER + " | difference (95% CI) |", "|---|---:|---:|---:|"]
    for label, col, per in (("Games played", "st_GP", None), ("Minutes per game", "st_MIN", "st_GP"), ("Fantasy points per game", None, "st_GP")):
        if per is None and col:
            a, b = pv[(col, MINE)] / pv[("st_n", MINE)], pv[(col, OTHER)] / pv[("st_n", OTHER)]
        elif col:
            a, b = pv[(col, MINE)] / pv[(per, MINE)], pv[(col, OTHER)] / pv[(per, OTHER)]
        else:
            fp = lambda who: sum(wt * pv[("st_" + k, who)] for lbl, w in GROUPS for k, wt in w.items()) / pv[("st_GP", who)]   # noqa: E731
            a, b = fp(MINE), fp(OTHER)
        d = (a - b).values; lo, hi = boot(d)
        lines.append(f"| {label} | {a.mean():.1f} | {b.mean():.1f} | {d.mean():+.1f} ({lo:+.1f} to {hi:+.1f}) |")
    lines += ["", "How to read it: a row whose interval excludes zero is a place where the app's lineup is reliably different. Rows whose interval "
              "includes zero are not evidence of anything. One season, simulated opponents, hindsight scoring: describe it that way.", ""]
    text = "\n".join(lines)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

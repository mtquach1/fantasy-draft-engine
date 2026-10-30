"""Out-of-sample backtest of app/own_model.py.  Run from the project root:  python research/evaluate_backtest.py
Fits the age curve on the 2024->2025 transition only, then predicts the 2025-26 season (history 2024+2025) for the top 150
players by 2024-25 fantasy points who played in 2025-26, and compares against two naive baselines."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import own_model as om                      # noqa: E402
from app.ingestion import data_dir                   # noqa: E402

files = om.find_seasons(Path(data_dir()) / "nba" / "raw")
S = {y: om.load_season(p) for y, p in files.items()}
yrs = sorted(S)
assert len(yrs) >= 3, "need three per-game seasons in data/nba/raw"
a, b, c = yrs[-3], yrs[-2], yrs[-1]                  # fit transition a->b, test target c with history a+b
p = om.Params()
beta, n_fit = om.fit_age_from_transition(S, [a, b], p)
proj = om.project_next(S, [b, a], p, beta)
last = S[b]
pool = (last["fpg"] * last["G"]).sort_values(ascending=False).index[:150]
y = S[c].reindex(pool)
ok = y["G"].notna() & (y["G"] >= 1)
idx = ok[ok].index.intersection(proj.index)
act = (y["fpg"] * y["G"]).loc[idx]
cands = {"own model": proj["fpts"].loc[idx], "repeat last season total": (last["fpg"] * last["G"]).loc[idx],
         "last per-game rate x 68 games": (last["fpg"] * 68).loc[idx]}
print(f"Age curve fitted on {a}->{b} ({n_fit} players). Test: season ending {c}, n={len(idx)} players from the top-150 pool.\n")
print(f"{'method':32s} {'MAE':>6s} {'RMSE':>6s} {'rank rho':>9s}")
for k, v in cands.items():
    print(f"{k:32s} {np.abs(v - act).mean():6.0f} {np.sqrt(((v - act) ** 2).mean()):6.0f} {spearmanr(v, act)[0]:9.3f}")
rng = np.random.default_rng(0)
print()
for k in ["repeat last season total", "last per-game rate x 68 games"]:
    d = (np.abs(cands[k] - act) - np.abs(cands["own model"] - act)).values
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    print(f"own-model advantage vs {k}: {d.mean():.0f} fantasy pts, 95% CI [{np.percentile(bs, 2.5):.0f}, {np.percentile(bs, 97.5):.0f}]")

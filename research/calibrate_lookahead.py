"""Calibrate the look-ahead's model of the other drafters against real draft logs.   (see app/calibration.py for the method)

Needs: one or more draft logs the app saved (data/draft_state_nba.json: a mock draft or a real league's draft), and a board: the ranking the
other drafters are assumed to follow.
  default      the app's OWN board, exactly what the look-ahead uses (data/nba/adp.csv if you have one, otherwise the app's projected_rank)
  --board X    a CSV in rank order with a Name column, e.g. data/nba/players_third_party.csv, to measure how far drafters stray from a public list
Calibrating against the app's own board is the one that matters for the look-ahead: it includes both the drafters' randomness and the gap
between their ranking and yours.

    python research/calibrate_lookahead.py [log.json ...] [--board board.csv] [--keep-mine]

Prints (1) how noisy the real drafters are compared with the model's assumption (2.0 + 0.12 * rank), with a bootstrap interval, and (2) a
reliability table: when the model says "70% likely to last until my next turn", how often did the player really last? It changes nothing
by itself: if the result says the constants are off, you edit NOISE_BASE and NOISE_SLOPE in app/lookahead.py (and the pace benchmark's copy).
With two or more logs it also checks the fitted model on a log it was NOT fitted on."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                   # noqa: E402
import pandas as pd                                  # noqa: E402

from app import calibration as cal, ingestion, persist, validate   # noqa: E402
from app.config import LeagueConfig                 # noqa: E402
from app.lookahead import rank_key                  # noqa: E402
from app.valuation import add_ranks, calculate_player_valuations   # noqa: E402


def _bootstrap_diff(per_state_a, per_state_b, boot=2000, seed=0):
    d = per_state_a - per_state_b; rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(boot)]
    return float(d.mean()), (float(np.percentile(bs, 5)), float(np.percentile(bs, 95)))


def board_names(board_arg, teams):
    """(names in rank order, a sentence saying which board it is). Raises ValueError with a message to show."""
    if board_arg:
        p = Path(board_arg)
        if not p.exists():
            raise ValueError(f"{p} was not found. Pass a CSV in rank order with a Name column.")
        try:
            return pd.read_csv(p, encoding="utf-8-sig")["Name"].tolist(), f"Board: {p.name}, in file order."
        except (KeyError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
            raise ValueError(f"{p} is not a usable board ({type(e).__name__}): it needs a Name column, rows in rank order.")
    if not (ingestion.DATA_DIR / "nba" / "players.csv").exists():
        raise ValueError("No data/nba/players.csv to build the app's board from. Build it (python build_own_model.py) or pass --board.")
    raw, _ = validate.clean_projections(ingestion.load_projections("nba"), "nba")
    cfg = LeagueConfig.for_sport("nba", "espn_default"); cfg.roster.teams = teams
    df, _ = calculate_player_valuations(raw, cfg, frozenset())
    df = add_ranks(df, df)
    adp, note = ingestion.load_adp("nba", raw)
    if adp:
        df["adp"] = df["player_id"].map(adp)
    df = df.assign(_key=rank_key(df)).sort_values(["_key", "name"], kind="stable")
    return df["name"].tolist(), "Board: the app's own ranking" + (" (blended with your adp.csv)" if adp else " (projected_rank)") + ", i.e. what the look-ahead really uses."


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate the look-ahead's opponent model against real draft logs.")
    ap.add_argument("logs", nargs="*"); ap.add_argument("--board"); ap.add_argument("--keep-mine", action="store_true")
    ap.add_argument("--runs", type=int, default=400)
    a = ap.parse_args(argv)
    paths = [Path(p) for p in a.logs] or [persist.DATA_DIR / "draft_state_nba.json"]
    logs = []
    for p in paths:
        try:
            logs.append((p, cal.load_picks(p)))
        except ValueError as e:
            print(e); return 1
    try:
        names, board_note = board_names(a.board, logs[0][1]["teams"])
    except ValueError as e:
        print(e); return 1
    print(board_note)
    ranks = cal.board_from_names(names)
    fit_states, surv, per_log = [], [], []
    for p, d in logs:
        skip = () if a.keep_mine or not d["my_slot"] else (d["my_slot"],)
        st = cal.decision_states(d["picks"], ranks, skip)
        sv = cal.survival_states(d["picks"], ranks)
        known = sum(1 for q in d["picks"] if q["key"] in ranks)
        print(f"{p.name}: {len(d['picks'])} picks, {known} found on the board, {len(st)} used for the fit" + (f" (slot {d['my_slot']}, yours, left out)" if skip else "")
              + f", {len(sv)} decision points for the reliability check")
        per_log.append((st, sv)); fit_states += st; surv += sv
    if len(fit_states) < 30:
        print(f"\nOnly {len(fit_states)} usable picks: too few to fit anything. Use more or longer drafts (a full 12-team, 13-round draft has 156)."); return 1
    print(f"\nFitting on {len(fit_states)} picks over {len(cal.GRID_BASE) * len(cal.GRID_SLOPE)} noise settings ...", flush=True)
    L, combos = cal.loglik_matrix(fit_states, runs=a.runs)
    f = cal.fit(L, combos)
    print(f"\n1. HOW NOISY ARE THE OTHER DRAFTERS?   (noise sd in places of rank; the model's rank + noise decides who is taken)")
    print(f"   {'at board rank':>14} " + " ".join(f"{r:>12}" for r in f["best_sd_at"]))
    print(f"   {'model now':>14} " + " ".join(f"{v:>12.1f}" for v in f["current_sd_at"].values()))
    print(f"   {'best fit':>14} " + " ".join(f"{v:>12.1f}" for v in f["best_sd_at"].values()))
    print(f"   {'90% interval':>14} " + " ".join(f"{lo:>5.1f}-{hi:<5.1f} " for lo, hi in f["sd_at"].values()))
    lo, hi = f["gain_ci"]
    print(f"   Best fit = {f['best'][0]} + {f['best'][1]} * rank. Versus the model's current {CURRENT_TXT}: {f['gain_per_pick']:+.3f} log-likelihood per pick "
          f"(90% interval {lo:+.3f} to {hi:+.3f}).")
    inside = all(f['sd_at'][r][0] <= f['current_sd_at'][r] <= f['sd_at'][r][1] for r in f['sd_at'])
    print("   " + ("The current setting sits inside the interval at every rank: this data does not show it is wrong." if inside else
                   "The current setting falls OUTSIDE the interval at some rank: this data says the noise assumption is off."))
    print(f"\n2. WHEN THE MODEL SAYS 'X% LIKELY TO LAST', HOW OFTEN DID THE PLAYER LAST?   ({sum(len(s['survived']) for s in surv)} player-moments, {len(surv)} decision points)")
    r_cur, r_fit = cal.reliability(surv, *cal.CURRENT, a.runs), cal.reliability(surv, *f["best"], a.runs)
    print(f"   {'model said':>12} {'players':>8} {'avg said':>9} {'really lasted':>14}   (current model)")
    for row in r_cur["table"]:
        print(f"   {row['lo']:>5.0%}-{row['hi']:<5.0%} {row['n']:>8} {row['predicted']:>9.0%} {row['observed']:>14.0%}")
    d, (dlo, dhi) = _bootstrap_diff(r_cur["per_state"], r_fit["per_state"])
    print(f"   Brier score (lower is better): current {r_cur['brier']:.4f}, best fit {r_fit['brier']:.4f}. Current minus fitted: {d:+.4f} (90% interval {dlo:+.4f} to {dhi:+.4f}).")
    print("   Both numbers are in-sample: the fit saw these drafts. Treat any gain as an upper bound.")
    if len(logs) >= 2:
        print("\n3. DOES THE FIT HOLD ON A DRAFT IT DID NOT SEE?  (leave one log out)")
        cur_b, fit_b, cnt = [], [], 0
        for i, (st, sv) in enumerate(per_log):
            rest = [s for j, (q, _) in enumerate(per_log) if j != i for s in q]
            if len(rest) < 30 or not sv:
                continue
            Lr, cb = cal.loglik_matrix(rest, runs=a.runs)
            best = cb[int(np.argmax(Lr.sum(0)))]
            cur_b.append(cal.reliability(sv, *cal.CURRENT, a.runs)["per_state"]); fit_b.append(cal.reliability(sv, *best, a.runs)["per_state"]); cnt += 1
        if cnt:
            d, (dlo, dhi) = _bootstrap_diff(np.concatenate(cur_b), np.concatenate(fit_b))
            print(f"   Brier current minus fitted-on-the-others: {d:+.4f} (90% interval {dlo:+.4f} to {dhi:+.4f}). Positive = the refit helps on unseen drafts.")
    print("\nWhat this can and cannot tell you: it measures whether the noise assumption matches these drafters. It does not model positional "
          "need or team strategy, and a handful of drafts is a handful of drafters.")
    return 0


CURRENT_TXT = f"{cal.CURRENT[0]} + {cal.CURRENT[1]} * rank"

if __name__ == "__main__":
    sys.exit(main())

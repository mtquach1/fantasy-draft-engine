"""Calibrating the look-ahead against real draft logs.

The look-ahead (app/lookahead.py) assumes the other drafters follow a shared ranking with noise: each pick takes the player whose
rank + noise is smallest, with noise standard deviation  base + slope * rank  (today 2.0 + 0.12 * rank, a guess). Two questions:

  1. FIT. Which (base, slope) makes the picks in real logs most likely?  (`fit`)
     For each observed pick we know who was still available and which of them the drafter took, as a position in the board order
     (0 = the best player left). The noise model predicts a distribution over that position (by simulation); we score the real picks
     by that distribution and keep the parameters with the highest total log-likelihood.
  2. RELIABILITY. When the model says "X% likely to still be there at my next turn", is he there about X% of the time?  (`survival_states`,
     `reliability`)  Every pick in a log, by any team, is a decision point: the player took one of the players left, and we know
     which of the others lasted until that team's next turn.

Honest limits: a log of one draft is ~100 to 150 picks by drafters who are not independent, so the fit has wide uncertainty (reported as a
bootstrap interval). Picks by the person who used the app are excluded from the fit (they follow the app, not the field). The board is whatever
ranking you pass in; if the real drafters used a different one, the "noise" absorbs that difference."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .own_model import norm_name

CURRENT = (2.0, 0.12)
GRID_BASE = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
GRID_SLOPE = (0.02, 0.05, 0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40)
BINS = (0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0001)


# ------------------------------------------------------------------------------------------------ reading
def board_from_names(names) -> dict:
    """Name order -> {normalized name: rank starting at 1}. A name that appears twice keeps its first place."""
    out = {}
    for n in names:
        out.setdefault(norm_name(n), len(out) + 1)
    return out


def load_picks(path) -> dict:
    """Read a draft file the app saved. -> {"teams", "my_slot", "picks": [{"n", "team", "key"}]}, key None for a pick with no readable player.
    Raises ValueError with a message to show."""
    p = Path(path)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{p} was not found.")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"{p} is not a valid draft file ({e}).")
    if not isinstance(d, dict) or not isinstance(d.get("draft_log"), list) or not d["draft_log"]:
        raise ValueError(f"{p} has no picks (no draft_log list).")
    if d.get("sport", "nba") != "nba":
        raise ValueError(f"{p} is a {d.get('sport')} draft: calibration here is for basketball.")
    setup = d.get("setup") or {}
    picks = []
    for i, x in enumerate(d["draft_log"], 1):
        if not isinstance(x, dict) or not isinstance(x.get("team_id"), int) or isinstance(x.get("team_id"), bool):
            raise ValueError(f"{p}: pick #{i} has no whole-number team_id.")
        pid = x.get("player_id")
        key = norm_name(pid[2:]) if isinstance(pid, str) and pid.startswith("n:") and pid[2:].strip() else None
        n = x.get("pick_number")
        picks.append({"n": n if isinstance(n, int) and not isinstance(n, bool) else i, "team": x["team_id"], "key": key})
    picks.sort(key=lambda q: q["n"])
    teams = setup.get("teams") if isinstance(setup.get("teams"), int) else max(q["team"] for q in picks)
    return {"teams": teams, "my_slot": setup.get("my_slot") if isinstance(setup.get("my_slot"), int) else None, "picks": picks}


# ------------------------------------------------------------------------------------------------ part 1: fitting the noise
def decision_states(picks, ranks, skip_teams=()) -> list:
    """One state per observed pick of a board player: the ranks still available (ascending) and the position the drafter took (0 = best left).
    Players picked by teams in `skip_teams` are still removed from the pool; their picks just don't count as observations."""
    taken, out = set(), []
    for p in sorted(picks, key=lambda q: q["n"]):
        k = p["key"]
        if k is None or k not in ranks or k in taken:
            continue
        if p["team"] not in skip_teams:
            avail = np.array(sorted(r for kk, r in ranks.items() if kk not in taken), dtype=float)
            out.append({"n": p["n"], "team": p["team"], "avail": avail, "obs": int(np.searchsorted(avail, ranks[k]))})
        taken.add(k)
    return out


def _z(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape)


def loglik_matrix(states, grid_base=GRID_BASE, grid_slope=GRID_SLOPE, runs=400, seed=7):
    """L[s, c]: log-probability of state s's observed pick under parameter combo c. Same random draws for every combo (common random numbers),
    so differences between combos are not simulation noise. Returns (L, combos)."""
    combos = [(a, b) for a in grid_base for b in grid_slope]
    L = np.zeros((len(states), len(combos)))
    for s, st in enumerate(states):
        r = st["avail"]; z = _z((runs, len(r)), seed + s)
        for c, (a, b) in enumerate(combos):
            counts = np.bincount(np.argmin(r[None, :] + (a + b * r[None, :]) * z, axis=1), minlength=len(r))
            L[s, c] = np.log((counts[st["obs"]] + 0.1) / (runs + 0.1 * len(r)))
    return L, combos


def fit(L, combos, boot=1000, seed=0) -> dict:
    """Best combo by total log-likelihood, how it compares with CURRENT, and a bootstrap over picks for how sure that is."""
    tot = L.sum(0); best = int(np.argmax(tot))
    cur = combos.index(CURRENT) if CURRENT in combos else None
    rng = np.random.default_rng(seed); n = L.shape[0]
    picks = [int(np.argmax(L[rng.integers(0, n, n)].sum(0))) for _ in range(boot)]
    ranks = (5, 20, 50, 100)
    sd = np.array([[combos[i][0] + combos[i][1] * r for r in ranks] for i in picks])
    out = {"n": n, "best": combos[best], "best_ll": float(tot[best]), "sd_at": {r: (float(np.percentile(sd[:, k], 5)), float(np.percentile(sd[:, k], 95)))
                                                                                  for k, r in enumerate(ranks)},
           "best_sd_at": {r: combos[best][0] + combos[best][1] * r for r in ranks}}
    if cur is not None:
        d = L[:, best] - L[:, cur]
        bs = [d[rng.integers(0, n, n)].mean() for _ in range(boot)]
        out.update(current=CURRENT, current_ll=float(tot[cur]), gain_per_pick=float(d.mean()), gain_ci=(float(np.percentile(bs, 5)), float(np.percentile(bs, 95))))
        out["current_sd_at"] = {r: CURRENT[0] + CURRENT[1] * r for r in ranks}
    return out


# ------------------------------------------------------------------------------------------------ part 2: is "X% likely to last" right?
def survival_states(picks, ranks, top_m=12) -> list:
    """Every pick that has a later pick by the same team is a decision: that team took one player and passed on the rest, and g other picks
    came before its next turn. For the top_m best players left (excluding the one it took): did each last through those g picks?"""
    ps = sorted(picks, key=lambda q: q["n"])
    nxt = {}
    last_idx = {}
    for i in range(len(ps) - 1, -1, -1):
        nxt[i] = last_idx.get(ps[i]["team"]); last_idx[ps[i]["team"]] = i
    taken, out = set(), []
    for i, p in enumerate(ps):
        k = p["key"]
        j = nxt[i]
        if j is not None and k is not None and k in ranks and k not in taken:
            between = [q["key"] for q in ps[i + 1:j]]
            if all(b is not None for b in between):                              # an unreadable pick in between: we can't tell who lasted
                pool = sorted((r, kk) for kk, r in ranks.items() if kk not in taken and kk != k)
                if len(pool) > top_m + 1:
                    gone = set(between)
                    out.append({"n": p["n"], "team": p["team"], "g": j - i - 1, "pool": np.array([r for r, _ in pool], dtype=float),
                                "survived": np.array([kk not in gone for _, kk in pool[:top_m]], dtype=bool)})
        if k is not None and k in ranks:
            taken.add(k)
    return out


def predict_survival(st, a, b, runs=400, seed=11) -> np.ndarray:
    """The model's chance that each of the top_m candidates is still there after st['g'] other picks."""
    r = st["pool"]; m = len(st["survived"])
    z = _z((runs, len(r)), seed + st["n"])
    keys = r[None, :] + (a + b * r[None, :]) * z
    pos = np.argsort(np.argsort(keys, axis=1, kind="stable"), axis=1)[:, :m]     # place each candidate would be taken (0 = first)
    return (pos >= st["g"]).mean(0)


def reliability(states, a, b, runs=400, bins=BINS) -> dict:
    """Brier score and a table of predicted vs observed, over all candidates of all states."""
    preds = [predict_survival(s, a, b, runs) for s in states]
    P = np.concatenate(preds); Y = np.concatenate([s["survived"] for s in states]).astype(float)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (P >= lo) & (P < hi)
        if m.any():
            rows.append({"lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()), "predicted": float(P[m].mean()), "observed": float(Y[m].mean())})
    return {"brier": float(((P - Y) ** 2).mean()), "n": int(len(P)), "table": rows,
            "per_state": np.array([((pr - s["survived"]) ** 2).mean() for pr, s in zip(preds, states)])}

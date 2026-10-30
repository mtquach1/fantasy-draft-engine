"""'Who should I pick?': the gain to YOUR final roster under the active league format.

Two ideas keep this honest:
 1. An empty starting slot is not worth zero (zero = a league-average player). It will be filled from the waiver
    pool, so it is valued at that slot's replacement level. Drafting a player is worth what he adds over that.
 2. Every starting slot must be filled by the end. Once you have only as many picks left as open starting slots,
    any pick that cannot fill one is a wasted pick and is ranked below every pick that can."""
from collections import Counter

import numpy as np
import pandas as pd

from . import lookahead
from .config import LeagueConfig, ScoringType
from .optimizer import is_eligible, open_starters, optimize_roster, slot_level
from .sports import get_sport
from .valuation import team_value


def _int(v):
    return None if v is None or pd.isna(v) else int(v)


def _ghosts(pool: pd.DataFrame, cfg: LeagueConfig, repl: dict) -> dict:
    """For each starting slot type: the waiver-level player who would fill it (the available player whose score is
    closest to that slot's replacement level). Used only as a stand-in for an empty slot."""
    sp, out = get_sport(cfg.sport), {}
    for slot, n in cfg.roster.slots.items():
        if slot == "BN" or not n:
            continue
        lvl = slot_level(slot, sp, repl)
        elig = pool[pool["eligible_positions"].map(lambda p, s=slot: is_eligible(p, s, cfg.sport))]
        if elig.empty or lvl is None:
            continue
        row = elig.loc[[(elig["score"] - lvl).abs().idxmin()]].copy()
        row["player_id"] = f"_waiver_{slot}"
        out[slot] = row
    return out


def _lineup_value(df, team_df, cfg, repl, ghosts, score_of=None) -> float:
    """Value of the best lineup from these players, with every starting slot they leave empty filled by waiver level.
    `score_of` (player_id -> score) is an optional shortcut: in points and roto leagues the lineup value is just the sum of
    the starters' scores, so no tables need to be built. Head-to-head categories take the full path."""
    res = optimize_roster(team_df, cfg.roster, repl, cfg.sport)
    placed = [a for a in res["assignments"] if a["slot"] != "BN"]
    ids = [a["player_id"] for a in placed]
    left = [s for s in cfg.roster.slot_list() if s != "BN"]
    for a in placed:
        left.remove(a["slot"])
    if score_of is not None and cfg.scoring_type in (ScoringType.POINTS, ScoringType.ROTO):
        return float(sum(score_of[i] for i in ids) + sum(float(ghosts[s]["score"].iloc[0]) for s in left if s in ghosts))
    parts = [df[df["player_id"].isin(ids)]] + [ghosts[s] for s in left if s in ghosts]
    return team_value(pd.concat(parts), cfg)


def _shortfall(team_df, cfg, rounds_total: int) -> int:
    """Starting slots that can no longer be filled: open slots minus the picks still available to fill them."""
    left = max(0, rounds_total - len(team_df))
    return max(0, len(open_starters(team_df, cfg.roster, cfg.sport)) - left)


def recommend(df: pd.DataFrame, repl: dict, cfg: LeagueConfig, my_ids: list[str],
              top_k: int = 40, limit: int = 10, exclude_injury_prone: bool | None = None,
              gaps: dict | None = None, with_plan: bool = False, rounds_total: int | None = None):
    """Ranked picks. With `gaps` (from lookahead.pick_gaps) the top candidates are re-ranked by the look-ahead
    (will he last until your next turn?). `with_plan=True` returns (rows, plan) instead of just rows."""
    rounds_total = rounds_total or len(cfg.roster.slot_list())
    mine = df[df["player_id"].isin(my_ids)]
    pool = df[df["available"]]
    ghosts = _ghosts(df[df["available"] & ~df["avoid"]], cfg, repl)
    score_of = dict(zip(df["player_id"], df["score"].astype(float)))
    value = lambda t: _lineup_value(df, t, cfg, repl, ghosts, score_of)
    base = value(mine)
    if cfg.avoid_injury_prone if exclude_injury_prone is None else exclude_injury_prone:
        pool = pool[~pool["injury_prone"]]
    if gaps is not None and gaps["k1"] > 0 and not pool.empty:
        # Your pick is still some picks away: only consider players who can realistically last until then
        try:
            surv = pool["player_id"].map(lookahead.survival_by_id(df, gaps)).fillna(0.0)
            ok = pool[surv >= 0.15]
            pool = ok if len(ok) >= 12 else pool.assign(_p=surv).nlargest(12, "_p").drop(columns="_p")
        except Exception:
            pass
    open_now = open_starters(mine, cfg.roster, cfg.sport)
    cands = pool.nlargest(top_k, "vorp")
    for slot in sorted(set(open_now)):                      # always consider the best players for every slot you still need
        elig = pool[pool["eligible_positions"].map(lambda p, s=slot: is_eligible(p, s, cfg.sport))]
        cands = pd.concat([cands, elig.nlargest(12, "vorp")])
    cands = cands[~cands.index.duplicated()]

    left_before = max(0, rounds_total - len(mine))
    pen_before = max(0, len(open_now) - left_before)
    rows = []
    for _, c in cands.iterrows():
        team = pd.concat([mine, c.to_frame().T])
        open_after = open_starters(team, cfg.roster, cfg.sport)
        fills = sorted((Counter(open_now) - Counter(open_after)).elements())
        rows.append({"player_id": c["player_id"], "name": c["name"], "positions": c["eligible_positions"],
                     "projected_rank": _int(c.get("projected_rank")), "pos_rank": c.get("pos_rank"),
                     "vorp": float(c["vorp"]), "lineup_gain": float(value(team) - base), "fills": fills,
                     "injury_prone": bool(c["injury_prone"]), "is_injured": bool(c["is_injured"]),
                     "games_pct": float(c["games_pct"]), "primary_pos": c.get("primary_pos"),
                     "wasted": max(0, len(open_after) - max(0, left_before - 1)) > pen_before})
    if not rows:
        return ([], {"status": "off", "reason": "No players left."}) if with_plan else []
    gains = np.array([r["lineup_gain"] for r in rows])
    pos_fill = [r["lineup_gain"] for r in rows if r["fills"] and r["lineup_gain"] > 0]
    unit = float(np.median(pos_fill)) if pos_fill else 0.1 * float(np.abs(gains).max() or 1.0)
    big = 3.0 * float(np.abs(gains).max() or 1.0)           # larger than any real gain: a wasted pick can never win
    spare = left_before - len(open_now)
    for r in rows:
        if r["wasted"]:
            r["team_gain"] = -abs(r["lineup_gain"]) - unit
            need = " / ".join(sorted(set(open_now))) or "nothing"
            r["why"] = (f"Wasted pick: you still need {need} and have only {left_before} "
                        f"pick{'s' if left_before != 1 else ''} left")
        else:
            r["team_gain"] = r["lineup_gain"]
            if not r["fills"] and r["lineup_gain"] <= 1e-9:
                # bench-only: prefer cover at positions where you are thin, not a fifth player at one position
                same = sum(1 for p in mine["primary_pos"] if p == r["primary_pos"])
                r["team_gain"] = 0.05 * max(r["vorp"], 0.0) / (1 + same) ** 2
            if r["fills"]:
                r["why"] = "Fills your open " + " / ".join(r["fills"]) + (" slots" if len(r["fills"]) > 1 else " slot")
            elif r["lineup_gain"] > 1e-9:
                r["why"] = "Upgrade: better than a starter you already have"
            else:
                r["why"] = (f"Bench depth only ({max(spare, 0)} spare pick{'s' if spare != 1 else ''} "
                            "before every starting slot must be filled)")
        r["score"] = r["team_gain"]
        r["open_slots"], r["picks_left"] = sorted(open_now), left_before
    plain = sorted(rows, key=lambda r: (r["wasted"], -r["team_gain"], -r["vorp"]))
    plan = {"status": "off", "reason": "Look-ahead needs a snake or linear draft setup with your pick."}
    if gaps is not None:
        try:
            value_of = lambda t: value(t) - big * _shortfall(t, cfg, rounds_total)
            plain, plan = lookahead.rerank(df, plain, mine, gaps, 1.0, value_of)
        except Exception as e:                       # a look-ahead problem must never take the recommendations down
            plan = {"status": "error", "reason": f"Look-ahead unavailable ({type(e).__name__}: {e})"}
    plain = plain[:limit]
    return (plain, plan) if with_plan else plain

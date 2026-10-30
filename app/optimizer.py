
"""Assign a team's drafted players to roster slots with the Hungarian algorithm."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .config import RosterConfig
from .sports import get_sport

BIG = 1e9


def is_eligible(positions: list[str], slot: str, sport: str = "mlb") -> bool:
    if slot == "BN":
        return True
    rule = get_sport(sport).rule(slot)
    return True if not rule.accepts else bool(set(positions) & set(rule.accepts))


def slot_value(player: pd.Series, slot: str, repl: dict[str, float], sport: str = "mlb") -> float:
    """Value in a slot = score minus THAT slot's replacement level (bench is 0), so a catcher in UTIL isn't credited
    with catcher-level value. Pooled slots (UTIL, FLEX) use their own level; multi-position flex slots (CI, G, F)
    take the best of the positions they accept."""
    if slot == "BN":
        return 0.0
    rule = get_sport(sport).rule(slot)
    if rule.pool or len(rule.accepts) <= 1:
        return float(player["score"] - repl.get(slot, repl.get(rule.accepts[0], 0.0) if rule.accepts else 0.0))
    pos = set(player["eligible_positions"])
    return max(player["score"] - repl.get(x, 0.0) for x in rule.accepts if x in pos)


def optimize_roster(team_df: pd.DataFrame, roster: RosterConfig, repl: dict[str, float], sport: str = "mlb") -> dict:
    slots = roster.slot_list()
    n, m = len(team_df), len(slots)
    if n == 0:
        return {"assignments": [], "unassigned": [], "total_value": 0.0}
    cost = np.full((n, m), BIG)
    for i, (_, p) in enumerate(team_df.iterrows()):
        for j, s in enumerate(slots):
            if is_eligible(p["eligible_positions"], s, sport):
                cost[i, j] = -slot_value(p, s, repl, sport)
    rows, cols = linear_sum_assignment(cost)
    assigned, unassigned = [], set(range(n))
    for r, c in zip(rows, cols):
        if cost[r, c] >= BIG:                  # forced ineligible placement -> drop
            continue
        unassigned.discard(r)
        assigned.append({"player_id": team_df.iloc[r]["player_id"], "name": team_df.iloc[r].get("name"),
                         "slot": slots[c], "value": float(-cost[r, c])})
    return {"assignments": assigned,
            "unassigned": [team_df.iloc[i]["player_id"] for i in unassigned],
            "total_value": sum(a["value"] for a in assigned if a["slot"] != "BN")}





def open_starters(team_df: pd.DataFrame, roster: RosterConfig, sport: str = "mlb") -> list[str]:
    """Starting slots (bench excluded) that this set of players cannot fill, using the best possible matching of
    players to slots by eligibility alone. A guard in UTIL leaves SG open only if no other arrangement fills it."""
    starters = [s for s in roster.slot_list() if s != "BN"]
    if not starters or len(team_df) == 0:
        return list(starters)
    m = np.zeros((len(team_df), len(starters)))
    for i, pos in enumerate(team_df["eligible_positions"]):
        for j, slot in enumerate(starters):
            if is_eligible(pos, slot, sport):
                m[i, j] = 1.0
    rows, cols = linear_sum_assignment(-m)
    filled = {int(c) for r, c in zip(rows, cols) if m[r, c] > 0}
    return [slot for j, slot in enumerate(starters) if j not in filled]


def slot_level(slot: str, roster_sport, repl: dict):
    """Replacement-level score for a starting slot (None if unknown). Pooled/single-position slots use their own
    level; multi-position flex slots (G, F, CI ...) use the best level among the positions they accept."""
    rule = roster_sport.rule(slot)
    if rule.pool or len(rule.accepts) <= 1:
        return repl.get(slot, repl.get(rule.accepts[0]) if rule.accepts else None)
    return max((repl[x] for x in rule.accepts if x in repl), default=None)

"""Auction drafts: team budgets, suggested bids, who nominates next, and checks on a recorded sale.

Everything is derived from the pick log (each pick may carry a `price`), the same way the rest of the engine is.

Money model (the usual one): every team must keep $1 for each roster spot it still has to fill. Dollars above
that minimum are the "free" money. A player's suggested bid is $1 plus a share of the league's free money that is
proportional to his VORP (value over replacement, positive part only) among the players still available.
Because it uses the remaining money and the remaining pool, inflation and deflation are built in: if teams
overspend early, the bids on everyone left go DOWN; if they underspend, bids go UP.
"""
from __future__ import annotations

import pandas as pd

MIN_BID = 1


def _valid_team(t, teams: int) -> bool:
    return isinstance(t, int) and not isinstance(t, bool) and 1 <= t <= teams


def team_budgets(setup, log: list[dict]) -> dict:
    """Per-team money and roster spots from the pick log.
    A pick with no price (a live feed that doesn't report prices) counts as the $1 minimum and is reported in
    `unpriced`, so the page can say the budgets are approximate."""
    spent = [0] * setup.teams
    count = [0] * setup.teams
    unpriced = 0
    for p in log:
        t = p.get("team_id")
        if not _valid_team(t, setup.teams):
            continue                                   # a live pick whose team couldn't be worked out
        count[t - 1] += 1
        price = p.get("price")
        if price is None:
            unpriced += 1
            price = MIN_BID
        spent[t - 1] += int(price)
    teams = []
    for i in range(setup.teams):
        remaining = setup.budget - spent[i]
        open_spots = max(setup.rounds - count[i], 0)
        # must leave $1 for every OTHER open spot
        max_bid = max(remaining - (open_spots - 1), 0) if open_spots else 0
        teams.append({"team_id": i + 1, "spent": spent[i], "remaining": remaining, "players": count[i],
                      "open_spots": open_spots, "max_bid": max_bid})
    return {"teams": teams, "unpriced": unpriced}


def free_money(teams: list[dict]) -> int:
    """League-wide dollars above the $1-per-open-spot minimum (never negative)."""
    return max(sum(t["remaining"] for t in teams) - sum(t["open_spots"] for t in teams) * MIN_BID, 0)


def bid_values(df: pd.DataFrame, free: float) -> pd.Series:
    """Suggested bid for every AVAILABLE player (NaN for drafted ones), as whole dollars, minimum $1.
    Players you've said you'd avoid (injury-prone with the avoid switch on) don't soak up any of the money."""
    avail = df["available"]
    pos = df["vorp"].clip(lower=0.0).where(avail, 0.0)
    if "avoid" in df.columns:
        pos = pos.where(~df["avoid"], 0.0)
    total = float(pos.sum())
    raw = MIN_BID + (pos * (free / total) if total > 0 else pos * 0.0)      # no positive value left: everyone is $1
    return raw.round().clip(lower=MIN_BID).where(avail).astype(float)


def inflation(free_now: float, pre_bids: pd.Series, available: pd.Series) -> float | None:
    """How much the market has moved against the pre-draft price list: money left above the minimum, divided by
    the pre-draft (price - $1) of everyone still available. +0.20 means prices are running 20% hot."""
    value_left = float((pre_bids.where(available, 0.0) - MIN_BID).clip(lower=0.0).sum())
    if value_left <= 0:
        return None
    return free_now / value_left - 1.0


def nomination(setup, log: list[dict]) -> dict:
    """Who nominates next. Assumes teams nominate in order 1..N, one nomination per sale, skipping any team whose
    roster is full. (The pick log doesn't record who nominated, so a league that uses a different order, or that
    skips someone, will drift: the page says so.)"""
    n = setup.teams
    counts = [0] * n
    ptr = 0

    def step(ptr: int) -> int:
        for _ in range(n):
            if counts[ptr] < setup.rounds:
                return ptr
            ptr = (ptr + 1) % n
        return -1                                      # every roster is full

    for p in log:
        cur = step(ptr)
        if cur < 0:
            break
        t = p.get("team_id")
        if _valid_team(t, n):
            counts[t - 1] += 1
        ptr = (cur + 1) % n
    nxt = step(ptr)
    if nxt < 0:
        return {"next_team": None, "until_mine": None}
    out = {"next_team": nxt + 1, "until_mine": None}
    if setup.my_slot:
        mine, seen, p2 = setup.my_slot - 1, 0, nxt
        for _ in range(n):
            if p2 == mine:
                out["until_mine"] = seen if counts[mine] < setup.rounds else None
                break
            if counts[p2] < setup.rounds:
                seen += 1
            p2 = (p2 + 1) % n
    return out


def summary(setup, log: list[dict], my_team_id: int, df: pd.DataFrame | None = None,
            pre_bids: pd.Series | None = None) -> dict:
    """Everything the page needs about the auction in one object."""
    b = team_budgets(setup, log)
    free = free_money(b["teams"])
    out = {"budget": setup.budget, "teams": b["teams"], "unpriced": b["unpriced"], "free_money": free,
           "my_team_id": my_team_id, "mine": next((t for t in b["teams"] if t["team_id"] == my_team_id), None),
           "nomination": nomination(setup, log), "inflation": None}
    if df is not None and pre_bids is not None:
        out["inflation"] = inflation(free, pre_bids, df["available"])
    return out


def check_sale(setup, log: list[dict], team_id: int, price) -> str | None:
    """None if recording this sale is legal, otherwise a message for a person."""
    if isinstance(price, bool) or not isinstance(price, int):
        return "Enter the price as a whole number of dollars, like 23."
    if price < MIN_BID:
        return f"The price can't be lower than ${MIN_BID}."
    info = team_budgets(setup, log)["teams"][team_id - 1]
    if info["open_spots"] == 0:
        return f"Team {team_id}'s roster is already full ({setup.rounds} players)."
    if price > info["max_bid"]:
        return (f"Team {team_id} can bid at most ${info['max_bid']}: it has ${info['remaining']} left and must keep "
                f"$1 for each of its other {info['open_spots'] - 1} open roster spot(s).")
    return None


def setup_conflict(setup, log: list[dict]) -> dict[str, str]:
    """Would this (new) setup contradict the sales already logged? Used when budget or rounds change mid-draft."""
    if setup.draft_type != "auction" or not log:
        return {}
    errs: dict[str, str] = {}
    for t in team_budgets(setup, log)["teams"]:
        if t["players"] > setup.rounds:
            errs["rounds"] = f"Team {t['team_id']} already has {t['players']} players, more than {setup.rounds} rounds allows."
            break
    for t in team_budgets(setup, log)["teams"]:
        need = t["spent"] + t["open_spots"] * MIN_BID
        if need > setup.budget:
            errs["budget"] = (f"Team {t['team_id']} has spent ${t['spent']} and needs $1 for each of its {t['open_spots']} "
                              f"open spots, so the budget can't be lower than ${need}.")
            break
    return errs



def nominate_ideas(rows: list[dict], mine: dict | None, nomination_info: dict, my_team_id: int,
                   filled: float, n: int = 3) -> dict:
    """Advice on whom to put up for bid. `rows` are candidate players with `name`, `bid` (what he's worth now)
    and `team_gain` (what he adds to YOUR roster). Two kinds of nomination:
      drain   a pricey player who does little for you. Rivals spend their money on him and have less for the players
              you want. Best early, while everyone still has money.
      targets players you want that you can afford. Best late, when rivals' money is running out and they can't push
              the price up.
    `filled` is the share of all roster spots already sold (0..1). Heuristic advice, not a guarantee."""
    max_bid = (mine or {}).get("max_bid") or 0
    live = [r for r in rows if r.get("bid") is not None and r["bid"] == r["bid"]]
    afford = sorted((r for r in live if r["bid"] <= max_bid), key=lambda r: r["team_gain"], reverse=True)
    targets = afford[:n]
    ids = {r["player_id"] for r in targets}
    gains = sorted(r["team_gain"] for r in live)
    median_gain = gains[len(gains) // 2] if gains else 0.0
    drain = sorted((r for r in live if r["player_id"] not in ids and r["team_gain"] <= median_gain and r["bid"] > 1),
                   key=lambda r: r["bid"], reverse=True)[:n]
    my_turn = nomination_info.get("next_team") == my_team_id
    slim = lambda r: {"player_id": r["player_id"], "name": r["name"], "bid": int(r["bid"]), "team_gain": r["team_gain"]}
    early = filled < 0.5
    first, second = (drain, targets) if early else (targets, drain)
    names = lambda lst: ", ".join(f"{r['name']} (${int(r['bid'])})" for r in lst)
    parts = []
    if first:
        parts.append(("Nominate to drain rivals' money: " if early else "Nominate a player you want: ") + names(first) + ".")
    if second:
        parts.append(("Players you most want (nominate them when you are ready to pay): " if early else "To drain rivals instead: ") + names(second) + ".")
    lead = "It's your turn to nominate. " if my_turn else (
        f"Team {nomination_info['next_team']} nominates next. " if nomination_info.get("next_team") else "")
    return {"my_turn": my_turn, "stage": "early" if early else "late", "targets": [slim(r) for r in targets],
            "drain": [slim(r) for r in drain], "headline": (lead + " ".join(parts)).strip()}


"""Draft setup (format, teams, your pick), validation, and pick-order math. Nothing here is hard-coded to a league."""
from __future__ import annotations

from dataclasses import dataclass

TYPES = [
    {"value": "snake", "label": "Snake", "description": "Pick order reverses every round (1 to N, then N to 1)."},
    {"value": "linear", "label": "Linear", "description": "The same pick order every round (1 to N each round)."},
    {"value": "auction", "label": "Auction", "description": "Teams nominate players and bid from a fixed budget."},
]
LIMITS = {"min_teams": 2, "max_teams": 30, "max_rounds": 60, "max_budget": 2000}


@dataclass
class DraftSetup:
    draft_type: str
    teams: int
    my_slot: int | None
    rounds: int
    budget: int | None


def _int(v):
    """Whole numbers only: '12', 12 and 12.0 pass; '', 'abc', '-3', '12.5', True do not."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if v.is_integer() else None
    s = str(v).strip()
    return int(s) if s.isdigit() else None


def _blank(v) -> bool:
    return v is None or str(v).strip() == ""


def validate_setup(raw: dict, default_rounds: int) -> tuple[DraftSetup | None, dict[str, str]]:
    """Returns (setup, {}) or (None, {field: friendly message})."""
    errs: dict[str, str] = {}
    L = LIMITS
    kind = raw.get("draft_type")
    if kind not in {t["value"] for t in TYPES}:
        errs["draft_type"] = "Choose a draft format."
    auction = kind == "auction"

    teams = None
    if _blank(raw.get("teams")):
        errs["teams"] = "Enter how many teams are in your league."
    elif (n := _int(raw["teams"])) is None:
        errs["teams"] = "Use a whole number, like 12."
    elif not L["min_teams"] <= n <= L["max_teams"]:
        errs["teams"] = f"Teams must be between {L['min_teams']} and {L['max_teams']}."
    else:
        teams = n

    slot = None
    if _blank(raw.get("my_slot")):
        if not auction:
            errs["my_slot"] = "Enter your draft pick, your spot in the first round."
    elif (n := _int(raw["my_slot"])) is None:
        errs["my_slot"] = "Use a whole number, like 5."
    elif n < 1:
        errs["my_slot"] = "Your pick can't be lower than 1."
    elif teams is not None and n > teams:
        errs["my_slot"] = f"Your pick can't be higher than {teams}: a {teams}-team league has picks 1 to {teams}."
    elif teams is not None:
        slot = n

    rounds = default_rounds
    if not _blank(raw.get("rounds")):
        n = _int(raw["rounds"])
        if n is None or not 1 <= n <= L["max_rounds"]:
            errs["rounds"] = f"Rounds must be a whole number from 1 to {L['max_rounds']}."
        else:
            rounds = n

    budget = None
    if auction:
        if _blank(raw.get("budget")):
            errs["budget"] = "Enter each team's auction budget, like 260."
        elif (n := _int(raw["budget"])) is None or not 1 <= n <= L["max_budget"]:
            errs["budget"] = f"Budget must be a whole number from 1 to {L['max_budget']}."
        elif "rounds" not in errs and n < rounds:
            errs["budget"] = f"Budget must cover at least $1 for each of the {rounds} roster spots."
        else:
            budget = n
    if errs:
        return None, errs
    return DraftSetup(kind, teams, slot, rounds, budget), {}


def slot_for_pick(draft_type: str, teams: int, pick: int) -> int:
    """Which draft slot (1..teams) makes overall pick number `pick` (1-based)."""
    rnd, i = (pick - 1) // teams + 1, (pick - 1) % teams + 1
    return teams - i + 1 if draft_type == "snake" and rnd % 2 == 0 else i


def status(setup: DraftSetup, picks_made: int) -> dict:
    total = setup.teams * setup.rounds
    base = {"mode": setup.draft_type, "picks_made": picks_made, "total_picks": total,
            "complete": picks_made >= total, "teams": setup.teams, "my_slot": setup.my_slot}
    if setup.draft_type == "auction":
        return {**base, "budget": setup.budget}
    cur = picks_made + 1
    mine = [n for n in range(1, total + 1) if slot_for_pick(setup.draft_type, setup.teams, n) == setup.my_slot]
    upcoming = [n for n in mine if n >= cur]
    out = {**base, "my_pick_numbers": mine, "my_picks_made": len(mine) - len(upcoming),
           "my_picks_remaining": len(upcoming)}
    if base["complete"]:
        return out
    rnd, pos = (cur - 1) // setup.teams + 1, (cur - 1) % setup.teams + 1
    first = (rnd - 1) * setup.teams + 1
    order = [slot_for_pick(setup.draft_type, setup.teams, first + k) for k in range(setup.teams)]
    nxt = upcoming[0] if upcoming else None
    return {**out, "current_pick": cur, "round": rnd, "pick_in_round": pos, "round_order": order,
            "on_clock_slot": order[pos - 1], "you_are_on_the_clock": nxt == cur,
            "my_next_pick": nxt, "picks_until_my_turn": None if nxt is None else nxt - cur,
            "my_next_round": None if nxt is None else (nxt - 1) // setup.teams + 1}




"""Turn a platform's pick list (ESPN, Yahoo) into the draft-log file the app saves, so a finished league draft can feed
research/calibrate_lookahead.py and research/mock_rank.py. The picks are read-only facts about a draft that already happened."""
from __future__ import annotations

import math

DRAFT_TYPES = {"SNAKE": "snake", "LINEAR": "linear", "AUCTION": "auction"}


def export_log(picks, team_index, teams: int, my_team: int | None = None, draft_type: str | None = None, sport: str = "nba",
               source: str = "espn") -> dict:
    """picks: RemotePick-like objects (pick_number, platform_team_id, player_name). team_index maps the platform's team id to 1..teams.
    Team numbers in the file are the PLATFORM's team order, not draft slots (the platform's order is what the pick list gives us).
    A pick whose player name could not be resolved is kept with player_id None so pick numbers stay intact."""
    log = []
    for p in sorted(picks, key=lambda x: x.pick_number):
        name = (getattr(p, "player_name", "") or "").strip()
        log.append({"pick_number": int(p.pick_number), "player_id": ("n:" + name.lower()) if name else None,
                    "team_id": int(team_index(p.platform_team_id)), "source": source})
    return {"version": 1, "sport": sport, "exported_from": source,
            "setup": {"draft_type": DRAFT_TYPES.get(str(draft_type or "").upper(), "snake"), "teams": int(teams), "my_slot": my_team,
                      "rounds": math.ceil(len(log) / teams) if teams and log else None, "budget": None},
            "draft_log": log}


def unresolved(doc: dict) -> int:
    return sum(1 for p in doc["draft_log"] if p["player_id"] is None)

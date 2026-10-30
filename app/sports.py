
"""Everything that differs between sports lives here: positions, roster slots (and which can flex), stat
categories, projection/history file layouts and Yahoo mappings. The engine (valuation, optimizer, scoring,
draft, sync) reads these profiles and contains no sport-specific constants.
To add a sport: write one Sport(...) below and register it in SPORTS."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SlotRule:
    accepts: tuple = ()           # positions that can fill the slot; () = any player (bench)
    pool: bool = False            # valued against ONE pooled replacement level (UTIL/FLEX), not per-position levels
    spread: str = "even"          # how its league-wide demand is split over `accepts`: "even" | "proportional"


@dataclass
class Group:
    """A pool of players that are scored against each other (MLB: hitters and pitchers; NBA/NFL: everyone)."""
    kind: str                     # short code stored in the 'kind' column
    label: str                    # plural, used in messages ("hitters", "players")
    noun: str                     # adjective for stats ("hitting", "pitching", "player")
    positions: tuple
    cats: dict                    # category registry: name -> {kind: count|rate, ...}
    default_cats: tuple
    components: tuple             # numeric projection columns the cleaner coerces / the stat search reads
    derived: dict                 # derived column -> raw columns it needs (for validation messages)
    file: str                     # projections file inside the sport's data folder
    history_file: str             # past-season stats file
    rename: dict                  # raw column -> internal column
    history_extra_rename: dict = field(default_factory=dict)
    default_games: int | None = None      # games in a season, for the games-missed (injury-prone) calculation
    prep: object = None                   # adds derived columns before scoring
    default_positions: object = None      # fallback positions when a file has no Pos column


@dataclass
class Sport:
    key: str
    label: str
    groups: tuple
    slot_rules: dict
    default_slots: dict
    default_points: dict
    default_scoring: str          # "roto" | "points"
    default_teams: int
    default_il: int
    data_subdir: str
    aliases: dict
    presets: dict = field(default_factory=dict)
    yahoo_roster: dict = field(default_factory=dict)      # Yahoo roster position -> our slot
    yahoo_il: tuple = ("IL", "IL+", "DL", "DL+")
    yahoo_stats: dict = field(default_factory=dict)       # Yahoo position_type ('*' = any) -> (kind, {display name: category})
    templates: dict = field(default_factory=dict)         # preset name -> {scoring, slots, il}: a whole league setup, not just point weights

    @property
    def valid_positions(self) -> set:
        return {p for g in self.groups for p in g.positions}

    @property
    def has_categories(self) -> bool:
        return any(g.cats for g in self.groups)

    def group(self, kind: str) -> Group:
        return next(g for g in self.groups if g.kind == kind)

    def rule(self, slot: str) -> SlotRule:
        return self.slot_rules.get(slot, SlotRule((slot,)))

    def group_starters(self, slots: dict, kind: str) -> int:
        """Starting slots (per team) that only this group's players can fill."""
        pos = set(self.group(kind).positions)
        return sum(n for s, n in slots.items() if (r := self.slot_rules.get(s)) and r.accepts and set(r.accepts) <= pos)

    def points(self, preset: str | None = None) -> dict:
        return dict(self.presets.get(preset or "", self.default_points))


def _ded(*positions) -> dict:
    return {p: SlotRule((p,)) for p in positions}


# ================================ MLB ================================
HITTER_POS = ("C", "1B", "2B", "3B", "SS", "OF", "DH")
PITCHER_POS = ("SP", "RP")
HIT_CATS = {   # kind='count': sum the column.  kind='rate': (num/den) weighted by den.  invert: lower is better.
    "R": dict(kind="count", col="R"), "HR": dict(kind="count", col="HR"),
    "RBI": dict(kind="count", col="RBI"), "SB": dict(kind="count", col="SB"),
    "H": dict(kind="count", col="H"), "BB": dict(kind="count", col="BB"),
    "SO": dict(kind="count", col="SO", invert=True),
    "AVG": dict(kind="rate", num="H", den="AB"),
    "OBP": dict(kind="rate", num="OBP_num", den="PA"),
}
PIT_CATS = {
    "W": dict(kind="count", col="W"), "SV": dict(kind="count", col="SV"),
    "K": dict(kind="count", col="K"), "QS": dict(kind="count", col="QS"),
    "HLD": dict(kind="count", col="HLD"), "IP": dict(kind="count", col="IP"),
    "L": dict(kind="count", col="L", invert=True),
    "ERA": dict(kind="rate", num="ER", den="IP", invert=True),
    "WHIP": dict(kind="rate", num="WHIP_num", den="IP", invert=True),
}   # OPS is intentionally unsupported: it can't be cleanly volume-weighted from components here.


def _mlb_hit_prep(d):
    if "HBP" not in d:
        d["HBP"] = 0.0
    d["OBP_num"] = d["H"] + d["BB"] + d["HBP"]
    return d


def _mlb_pit_prep(d):
    d["WHIP_num"] = d["H_allowed"] + d["BB_allowed"]
    return d


def _mlb_pitcher_roles(p):
    if "GS" in p and "G" in p:
        return list(np.where(p["GS"] / p["G"].replace(0, np.nan) > 0.5, "SP", "RP"))
    return list(np.where(p["IP"] > 100, "SP", "RP"))


MLB = Sport(
    key="mlb", label="Baseball (MLB)", data_subdir="", default_teams=12, default_il=2,
    default_scoring="roto", default_slots={"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "INF": 0, "CI": 1, "MI": 1,
                                           "OF": 4, "UTIL": 2, "SP": 5, "RP": 2, "P": 1, "BN": 4},
    default_points={"singles": 1, "doubles": 2, "triples": 3, "HR": 4, "R": 1, "RBI": 1, "SB": 2, "BB": 1, "SO": -1,
                    "IP": 3, "W": 5, "L": -3, "SV": 5, "K": 1, "ER": -2, "H_allowed": -1, "BB_allowed": -1},
    aliases={"LF": "OF", "CF": "OF", "RF": "OF", "UTIL": "DH", "P": "SP"},
    groups=(
        Group("H", "hitters", "hitting", HITTER_POS, HIT_CATS, ("R", "HR", "RBI", "SB", "AVG"),
              ("G", "PA", "AB", "H", "singles", "doubles", "triples", "HR", "R", "RBI", "SB", "BB", "SO", "HBP", "SF"),
              {"OBP_num": ["H", "BB"]}, "hitters.csv", "hitter_history.csv",
              {"playerid": "player_id", "Name": "name", "Team": "team", "1B": "singles", "2B": "doubles", "3B": "triples"},
              default_games=162, prep=_mlb_hit_prep),
        Group("P", "pitchers", "pitching", PITCHER_POS, PIT_CATS, ("W", "SV", "K", "ERA", "WHIP"),
              ("G", "GS", "IP", "W", "L", "SV", "HLD", "QS", "K", "H_allowed", "BB_allowed", "ER", "HR_allowed"),
              {"WHIP_num": ["H_allowed", "BB_allowed"]}, "pitchers.csv", "pitcher_history.csv",
              {"playerid": "player_id", "Name": "name", "Team": "team", "SO": "K", "H": "H_allowed", "BB": "BB_allowed"},
              history_extra_rename={"HR": "HR_allowed"}, prep=_mlb_pit_prep, default_positions=_mlb_pitcher_roles),
    ),
    slot_rules={**_ded("C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"),
                "CI": SlotRule(("1B", "3B")), "MI": SlotRule(("2B", "SS")), "INF": SlotRule(("1B", "2B", "3B", "SS")),
                "UTIL": SlotRule(HITTER_POS, pool=True, spread="proportional"),
                "P": SlotRule(PITCHER_POS, pool=True, spread="even"), "BN": SlotRule(())},
    yahoo_roster={"C": "C", "1B": "1B", "2B": "2B", "3B": "3B", "SS": "SS", "IF": "INF", "CI": "CI", "MI": "MI",
                  "OF": "OF", "UTIL": "UTIL", "SP": "SP", "RP": "RP", "P": "P", "BN": "BN"},
    yahoo_stats={"B": ("H", {"R": "R", "HR": "HR", "RBI": "RBI", "SB": "SB", "AVG": "AVG", "OBP": "OBP",
                             "H": "H", "BB": "BB", "K": "SO"}),
                 "P": ("P", {"W": "W", "SV": "SV", "K": "K", "QS": "QS", "ERA": "ERA", "WHIP": "WHIP",
                             "HLD": "HLD", "L": "L", "IP": "IP"})},
)

# ================================ NBA ================================
NBA_POS = ("PG", "SG", "SF", "PF", "C")
NBA_CATS = {
    "PTS": dict(kind="count", col="PTS"), "REB": dict(kind="count", col="REB"), "AST": dict(kind="count", col="AST"),
    "STL": dict(kind="count", col="STL"), "BLK": dict(kind="count", col="BLK"), "3PM": dict(kind="count", col="3PM"),
    "TO": dict(kind="count", col="TO", invert=True), "FGM": dict(kind="count", col="FGM"),
    "FTM": dict(kind="count", col="FTM"),
    "FG%": dict(kind="rate", num="FGM", den="FGA"),       # volume-weighted: a 100% shooter on 1 shot adds ~nothing
    "FT%": dict(kind="rate", num="FTM", den="FTA"),
}
NBA = Sport(
    key="nba", label="Basketball (NBA)", data_subdir="nba", default_teams=12, default_il=3,
    default_scoring="roto", default_slots={"PG": 1, "SG": 1, "G": 1, "SF": 1, "PF": 1, "F": 1, "C": 2, "UTIL": 2, "BN": 3},
    default_points={"PTS": 1, "REB": 1.2, "AST": 1.5, "STL": 3, "BLK": 3, "TO": -1},
    aliases={}, presets={"standard": {"PTS": 1, "REB": 1.2, "AST": 1.5, "STL": 3, "BLK": 3, "TO": -1},
                         "with_threes": {"PTS": 1, "REB": 1.2, "AST": 1.5, "STL": 3, "BLK": 3, "TO": -1, "3PM": 0.5},
                         # ESPN's default points league (stat weights as listed on ESPN's settings page)
                         "espn_default": {"PTS": 1, "FGM": 2, "FGA": -1, "FTM": 1, "FTA": -1, "3PM": 1, "REB": 1, "AST": 2, "STL": 4, "BLK": 4, "TO": -2}},
    templates={"espn_default": dict(scoring="points", il=3,
                                    slots={"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UTIL": 3, "BN": 3})},
    groups=(Group("X", "players", "player", NBA_POS, NBA_CATS,
                  ("FG%", "FT%", "3PM", "PTS", "REB", "AST", "STL", "BLK", "TO"),
                  ("GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"), {},
                  "players.csv", "player_history.csv",
                  {"playerid": "player_id", "Name": "name", "Team": "team", "TOV": "TO", "TRB": "REB", "3P": "3PM", "G": "GP",
                   # Basketball-Reference / common site exports
                   "Player": "name", "PLAYER": "name", "NAME": "name", "Tm": "team", "TM": "team", "TEAM": "team",
                   "POS": "Pos", "Position": "Pos", "MP": "MIN", "MPG": "MIN", "FG": "FGM", "FT": "FTM",
                   "3PTM": "3PM", "3PM/G": "3PM", "ST": "STL", "TOs": "TO"},
                  default_games=82),),
    slot_rules={**_ded("PG", "SG", "SF", "PF", "C"), "G": SlotRule(("PG", "SG")), "F": SlotRule(("SF", "PF")),
                "UTIL": SlotRule(NBA_POS, pool=True, spread="proportional"), "BN": SlotRule(())},
    yahoo_roster={"PG": "PG", "SG": "SG", "G": "G", "SF": "SF", "PF": "PF", "F": "F", "C": "C", "UTIL": "UTIL", "BN": "BN"},
    yahoo_stats={"*": ("X", {"FG%": "FG%", "FT%": "FT%", "3PTM": "3PM", "PTS": "PTS", "REB": "REB", "AST": "AST",
                             "ST": "STL", "BLK": "BLK", "TO": "TO", "FGM": "FGM", "FTM": "FTM"})},
)

# ================================ NFL ================================
NFL_POS = ("QB", "RB", "WR", "TE", "K", "DST")


def _nfl_points(rec: float) -> dict:
    return {"pass_yds": 0.04, "pass_tds": 4, "ints": -2, "rush_yds": 0.1, "rush_tds": 6, "rec": rec, "rec_yds": 0.1,
            "rec_tds": 6, "fumbles": -2, "fgm": 3, "xpm": 1, "sacks": 1, "def_int": 2, "fum_rec": 2, "def_td": 6,
            "safeties": 2}


NFL = Sport(
    key="nfl", label="Football (NFL)", data_subdir="nfl", default_teams=12, default_il=1,
    default_scoring="points", default_slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "W/R": 0, "SUPERFLEX": 0,
                                             "K": 1, "DST": 1, "BN": 6},
    default_points=_nfl_points(0.5), aliases={"DEF": "DST", "D/ST": "DST", "PK": "K", "D": "DST"},
    presets={"standard": _nfl_points(0.0), "half_ppr": _nfl_points(0.5), "ppr": _nfl_points(1.0)},
    groups=(Group("X", "players", "player", NFL_POS, {}, (),
                  ("G", "pass_att", "pass_cmp", "pass_yds", "pass_tds", "ints", "rush_att", "rush_yds", "rush_tds", "rec",
                   "rec_yds", "rec_tds", "fumbles", "fgm", "xpm", "sacks", "def_int", "fum_rec", "def_td", "safeties"), {},
                  "players.csv", "player_history.csv", {"playerid": "player_id", "Name": "name", "Team": "team"},
                  default_games=17),),
    slot_rules={**_ded("QB", "RB", "WR", "TE", "K", "DST"),
                "FLEX": SlotRule(("RB", "WR", "TE"), pool=True, spread="proportional"),
                "W/R": SlotRule(("RB", "WR"), pool=True, spread="proportional"),
                "SUPERFLEX": SlotRule(("QB", "RB", "WR", "TE"), pool=True, spread="proportional"),
                "BN": SlotRule(())},
    yahoo_roster={"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "W/R/T": "FLEX", "W/R": "W/R", "Q/W/R/T": "SUPERFLEX",
                  "K": "K", "DEF": "DST", "BN": "BN"},
    yahoo_il=("IR", "IR+"),
)

SPORTS = {s.key: s for s in (MLB, NBA, NFL)}
RATE_COLUMNS = {"AVG", "OBP", "OPS", "ERA", "WHIP", "FG%", "FT%"}     # already rates: never scaled by games played


def board_positions(key: str) -> list[str]:
    """Positions for the page's filter dropdown, in a natural order (hitters before pitchers; no DH)."""
    sp = get_sport(key)
    return [p for g in sp.groups for p in g.positions if p != "DH"]


def get_sport(key: str) -> Sport:
    try:
        return SPORTS[key]
    except KeyError:
        raise ValueError(f"Unknown sport '{key}'. Available: {', '.join(SPORTS)}") from None





"""League settings only. No math, no network."""
from dataclasses import dataclass, field
from enum import Enum

from . import sports
from .sports import HITTER_POS, PITCHER_POS    # noqa: F401  (kept for older imports)


class ScoringType(str, Enum):
    ROTO = "roto"                      # season-long category standings
    H2H_CATEGORIES = "h2h_categories"  # weekly category matchups
    POINTS = "points"                  # points (season-long or H2H points)


@dataclass
class RosterConfig:
    teams: int = 12
    slots: dict[str, int] = field(default_factory=lambda: dict(sports.MLB.default_slots))   # slot name -> count per team
    il_slots: int = 2

    def slot_list(self) -> list[str]:
        """Flat list of slot names for ONE team, e.g. ['C','1B',...,'OF','OF',...,'BN']."""
        return [s for s, n in self.slots.items() for _ in range(n)]


@dataclass
class LeagueConfig:
    sport: str = "mlb"
    scoring_type: ScoringType = ScoringType.ROTO
    roster: RosterConfig = field(default_factory=RosterConfig)

    # Used by ROTO and H2H_CATEGORIES (ignored for POINTS): group kind -> category names, as defined in sports.py.
    categories: dict[str, list[str]] = field(default_factory=lambda: {g.kind: list(g.default_cats) for g in sports.MLB.groups})
    category_weights: dict[str, float] = field(default_factory=dict)  # strategy pivots / punting; default 1.0

    # Used by POINTS only. Keys are internal projection column names.
    point_weights: dict[str, float] = field(default_factory=lambda: dict(sports.MLB.default_points))

    # H2H categories only: higher = category outcomes are noisier week to week (heuristic, tune it)
    h2h_noise: float = 1.0

    # Injury-prone flag: avg share of games missed over the last 3 seasons >= threshold.
    injury_prone_threshold: float = 0.25
    injury_prone_min_seasons: int = 2      # need at least this many seasons of history to flag
    avoid_injury_prone: bool = False       # standing switch: hide flagged players from /recommend and
                                           # /players/available, and ignore them when setting replacement levels
    injury_prone_penalty: float = 0.0      # 0 = just flag; e.g. 0.15 shaves 15% off a flagged player's positive score

    il_penalty_open_slot: float = 0.92
    il_penalty_no_slot: float = 0.70

    @classmethod
    def for_sport(cls, key: str, preset: str | None = None) -> "LeagueConfig":
        """A league with that sport's standard roster, categories and scoring."""
        sp = sports.get_sport(key)
        if preset and preset not in sp.presets:
            raise ValueError(f"Unknown scoring preset '{preset}' for {sp.label}. Available: {', '.join(sp.presets) or 'none'}")
        tpl = sp.templates.get(preset or "")        # a template also sets the scoring type, roster slots and IL spots
        return cls(sport=key, scoring_type=ScoringType(tpl["scoring"] if tpl else sp.default_scoring),
                   roster=RosterConfig(teams=sp.default_teams, slots=dict(tpl["slots"] if tpl else sp.default_slots),
                                       il_slots=tpl["il"] if tpl else sp.default_il),
                   categories={g.kind: list(g.default_cats) for g in sp.groups if g.cats},
                   point_weights=sp.points(preset))




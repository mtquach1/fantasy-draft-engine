"""The ESPN default NBA points league: weights, roster, and that a whole board can be valued with it."""
import tempfile
from pathlib import Path

import pytest

import make_sample_sport
from app import ingestion, validate
from app.config import LeagueConfig, ScoringType
from app.valuation import calculate_player_valuations


def test_template_sets_points_scoring_and_the_espn_roster():
    cfg = LeagueConfig.for_sport("nba", "espn_default")
    assert cfg.scoring_type == ScoringType.POINTS
    assert cfg.roster.slots == {"PG": 1, "SG": 1, "SF": 1, "PF": 1, "C": 1, "G": 1, "F": 1, "UTIL": 3, "BN": 3}
    assert len(cfg.roster.slot_list()) == 13 and cfg.roster.il_slots == 3
    assert cfg.point_weights == {"PTS": 1, "FGM": 2, "FGA": -1, "FTM": 1, "FTA": -1, "3PM": 1, "REB": 1, "AST": 2,
                                 "STL": 4, "BLK": 4, "TO": -2}


def test_standard_nba_league_is_unchanged_and_unknown_presets_are_refused():
    assert LeagueConfig.for_sport("nba").scoring_type == ScoringType.ROTO
    assert LeagueConfig.for_sport("nfl", "ppr").scoring_type == ScoringType.POINTS
    with pytest.raises(ValueError):
        LeagueConfig.for_sport("nba", "no_such_league")


def test_weights_reproduce_a_published_season_total():
    """Giannis, 67 games: the projected season totals must score about 3856, the total ESPN lists for him."""
    w = LeagueConfig.for_sport("nba", "espn_default").point_weights
    totals = {"PTS": 2150.7, "FGM": 807.42, "FGA": 1293.95, "FTM": 502.35, "FTA": 772.85, "3PM": 33.5, "REB": 763.8,
              "AST": 422.1, "STL": 73.7, "BLK": 53.6, "TO": 247.9}
    pts = sum(w[k] * v for k, v in totals.items())
    assert abs(pts - 3856) / 3856 < 0.005


def test_a_board_can_be_valued_with_the_espn_league():
    old = ingestion.DATA_DIR
    with tempfile.TemporaryDirectory() as d:
        ingestion.DATA_DIR = Path(d)
        try:
            make_sample_sport.generate("nba", Path(d) / "nba", n=300)
            raw, _ = validate.clean_projections(ingestion.load_projections("nba"), "nba")
        finally:
            ingestion.DATA_DIR = old
    cfg = LeagueConfig.for_sport("nba", "espn_default")
    cfg.roster.teams = 12
    val, levels = calculate_player_valuations(raw, cfg, frozenset())
    assert len(val) == len(raw) and val["vorp"].notna().all()
    assert val["vorp"].max() > 0


def test_a_lone_point_guard_is_shown_at_pg_not_parked_in_util():
    from app.scoring import TeamScorer, summary
    from app.valuation import calculate_player_valuations
    old = ingestion.DATA_DIR
    with tempfile.TemporaryDirectory() as d:
        ingestion.DATA_DIR = Path(d)
        try:
            make_sample_sport.generate("nba", Path(d) / "nba", n=300)
            raw, _ = validate.clean_projections(ingestion.load_projections("nba"), "nba")
        finally:
            ingestion.DATA_DIR = old
    cfg = LeagueConfig.for_sport("nba", "espn_default")
    cfg.roster.teams = 12
    val, repl = calculate_player_valuations(raw, cfg, frozenset())
    sc = TeamScorer(val, repl, cfg)
    pg = val[val["eligible_positions"].apply(lambda e: list(e) == ["PG"])].sort_values("score", ascending=False).iloc[0]
    s = summary(sc, [pg["player_id"]])
    placed = {r["slot"]: r["name"] for r in s["roster"] if r["name"]}
    assert placed == {"PG": pg["name"]}                       # in his own slot, nowhere else
    assert s["score"] == summary(sc, [pg["player_id"]])["score"]

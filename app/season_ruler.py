"""Score teams with what a season REALLY produced (basketball): the same lineup assignment and slot values the app uses, but fed with
actual stats instead of projections. Used by the backtests and by research/mock_rank.py.

It works for a finished season and for a partial one: the valuation is relative to the players in the same table, so a table of
7-game totals is ranked against itself. (A partial season is noisy: see the notes in research/mock_rank.py.)"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from . import ingestion, own_model as om, validate
from .scoring import TeamScorer
from .valuation import calculate_player_valuations

# per-game column in the loaded season -> projection column the app's valuation expects
_STATS = {"PTS": "PTS", "REB": "TRB", "AST": "AST", "STL": "STL", "BLK": "BLK", "3PM": "TPM", "TO": "TOV",
          "FGM": "FGM", "FGA": "FGA", "FTM": "FTM", "FTA": "FTA"}


def actual_frame(season: pd.DataFrame, min_minutes: float = 100) -> pd.DataFrame:
    """A season table (as om.load_season returns it) in the shape the app loads projections: season TOTALS, one row per player.
    Players under `min_minutes` total minutes are left out: they are waiver-wire players, not draftable ones."""
    act = season[season["mins"] >= min_minutes]
    return pd.DataFrame({
        "Name": act["Player"].values, "Team": act["Tm"].values, "Pos": act["Pos"].astype(str).str.replace("-", ",").values,
        "playerid": ["n:" + str(n).lower() for n in act["Player"]], "GP": act["G"].values, "MIN": act["mins"].values,
        **{k: (act[v] * act["G"]).values for k, v in _STATS.items()},
        "is_injured": False, "note": "actual"})


def build_ruler(season: pd.DataFrame, cfg, min_minutes: float = 100):
    """Returns (scorer, id_by_name_key, valuated_frame). scorer.score(rows) values a set of players on the real stats; id_by_name_key maps
    om.norm_name(player name) -> the scorer's player_id, so callers can look players up by name."""
    frame = actual_frame(season, min_minutes)
    tmp = Path(tempfile.mkdtemp()); (tmp / "nba").mkdir()
    frame.to_csv(tmp / "nba" / "players.csv", index=False)
    old = ingestion.DATA_DIR
    ingestion.DATA_DIR = tmp
    try:
        raw, _ = validate.clean_projections(ingestion.load_projections("nba"), "nba")
    finally:
        ingestion.DATA_DIR = old
    val, repl = calculate_player_valuations(raw, cfg, frozenset())
    scorer = TeamScorer(val, repl, cfg)
    ids = {om.norm_name(n): q for n, q in zip(val["name"], val["player_id"])}
    return scorer, ids, val

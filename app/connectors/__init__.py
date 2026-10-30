import importlib.util
import os
import re
from pathlib import Path

from .base import DraftSource, SourceError

PLATFORMS = ("manual", "yahoo", "espn", "mock")
DATA = Path(__file__).resolve().parents[2] / "data"


def get_source(sport: str = "mlb") -> DraftSource:
    platform = os.getenv("DRAFT_PLATFORM", "manual").lower()
    if platform == "yahoo":
        from .yahoo import YahooSource
        return YahooSource()
    if platform == "espn":
        from .espn import EspnSource
        return EspnSource(sport)
    from .manual import ManualSource
    return ManualSource()


def _has(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def check_env(platform: str) -> tuple[list[str], list[str]]:
    """(problems, warnings) for the chosen DRAFT_PLATFORM, worded for a person. Problems stop the live sync from
    starting; warnings don't. Cheap local checks only: nothing here touches the network."""
    problems, warnings = [], []
    if platform not in PLATFORMS:
        return [f"DRAFT_PLATFORM='{platform}' isn't one of: {', '.join(PLATFORMS)}."], []
    if platform == "yahoo":
        if not os.getenv("YAHOO_LEAGUE_KEY", "").strip():
            problems.append("Set YAHOO_LEAGUE_KEY in .env (looks like 458.l.123456).")
        elif not re.fullmatch(r"\d+\.l\.\d+", os.getenv("YAHOO_LEAGUE_KEY", "").strip()):
            problems.append("YAHOO_LEAGUE_KEY should look like 458.l.123456 (game key, '.l.', league id).")
        if not (DATA / "oauth2.json").exists():
            problems.append("Missing data/oauth2.json (README, 'Yahoo setup').")
        if not (_has("yahoo_oauth") and _has("yahoo_fantasy_api")):
            problems.append("Run: pip install yahoo_oauth yahoo_fantasy_api")
    elif platform == "espn":
        if not os.getenv("ESPN_LEAGUE_ID", "").strip().isdigit():
            problems.append("Set ESPN_LEAGUE_ID in .env to the number in your league's ESPN URL.")
        if not re.fullmatch(r"\d{4}", os.getenv("ESPN_YEAR", "").strip()):
            problems.append("Set ESPN_YEAR in .env to the season year, like 2027.")
        if not (os.getenv("ESPN_S2", "").strip() and os.getenv("ESPN_SWID", "").strip()):
            warnings.append("ESPN_S2 and ESPN_SWID are empty. That's fine for a public league; a private league needs them.")
    return problems, warnings


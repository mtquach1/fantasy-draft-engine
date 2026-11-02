"""Source adapters. Each one turns an outside data source into a pandas table, behind one small interface, so adding a source means
writing one class. Two ship here:
  BRPerGameCsv  reads a Basketball-Reference per-game table you saved as CSV (the path the project uses today)
  NbaApiSource  reads stats.nba.com through the free `nba_api` package. EXPERIMENTAL: written against the documented response shape and
                tested only with a hand-made sample, never against the live service (which also blocks many cloud networks)."""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class SourceError(Exception):
    """A source could not be read, or returned something that isn't the expected shape."""


class Source(ABC):
    name: str = ""
    kind: str = ""                      # the snapshot kind: one table per kind, versioned over time

    @abstractmethod
    def fetch_text(self) -> str:
        """The table as CSV text (what gets stored in a snapshot)."""

    def fetch(self) -> pd.DataFrame:
        return pd.read_csv(io.StringIO(self.fetch_text()))


class BRPerGameCsv(Source):
    name = "basketball-reference per-game csv"

    def __init__(self, path, season: int):
        self.path, self.season = Path(path), int(season)
        self.kind = f"per_game_{self.season}"

    def fetch_text(self) -> str:
        try:
            raw = self.path.read_bytes()
        except FileNotFoundError:
            raise SourceError(f"{self.path} does not exist.")
        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise SourceError(f"{self.path.name} isn't UTF-8. Save it from Notepad as UTF-8 (not Excel) so names like Jokić survive.")
        return text.replace("\r\n", "\n")


class NbaApiSource(Source):
    name = "nba_api league dashboard (experimental)"

    def __init__(self, season: str, fetcher=None):
        """season like '2025-26'. fetcher(season) -> the response dict; the default one calls the live service."""
        self.season, self.fetcher = season, fetcher or self._live_fetch
        self.kind = f"nba_api_leaguedash_{season}"

    @staticmethod
    def _live_fetch(season: str) -> dict:
        try:
            from nba_api.stats.endpoints import LeagueDashPlayerStats
            return LeagueDashPlayerStats(season=season, per_mode_detailed="PerGame", timeout=30).get_dict()
        except Exception as e:                           # missing package, blocked network, changed endpoint ...
            raise SourceError(f"Could not fetch {season} from stats.nba.com ({type(e).__name__}: {e}). "
                              "The service blocks many cloud networks; run this from your own computer.")

    def fetch_text(self) -> str:
        payload = self.fetcher(self.season)
        try:
            sets = payload["resultSets"]
            block = next(s for s in sets if s.get("name") == "LeagueDashPlayerStats") if isinstance(sets, list) else sets
            frame = pd.DataFrame(block["rowSet"], columns=block["headers"])
        except (KeyError, TypeError, StopIteration, ValueError) as e:
            raise SourceError(f"The response isn't the expected shape ({type(e).__name__}: {e}). The endpoint may have changed.")
        return frame.to_csv(index=False)

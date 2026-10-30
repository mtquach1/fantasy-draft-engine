"""ESPN Fantasy connector: reads your league's picks LIVE, straight from ESPN's (unofficial) fantasy web API.

Why not the `espn-api` package? Its draft loader (checked in its source) skips the pick list until ESPN marks the draft
COMPLETE, so it shows nothing during a live draft, and it has no cheap refresh. This connector asks ESPN for just the draft
view (`mDraftDetail`) on every poll, using the same address and the same two cookies that package uses.

SETUP:
  .env:  DRAFT_PLATFORM=espn, ESPN_LEAGUE_ID=<number in your league's URL>, ESPN_YEAR=<season END year, e.g. 2027>
  Private leagues also need two cookies from a logged-in browser: ESPN_S2 and ESPN_SWID (README, 'ESPN setup').
  Players are matched to your projections by name (or id_crosswalk.csv with an espn_id column).

CAVEATS: ESPN's API is unofficial and can change without notice. Written from the espn-api package's source and ESPN's
published JSON field names; it has NOT been run against a live ESPN draft. Always keep manual mode as the fallback.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

import httpx

from .base import DraftSource, RemotePick, SourceError, dense_index, parse_cost

BASE = "https://lm-api-reads.fantasy.espn.com/apis/v3/games/"
SPORT_CODE = {"mlb": "flb", "nba": "fba", "nfl": "ffl"}
NAME_RELOAD_SECONDS = 120          # re-download the player-name list at most this often when an id is missing
_GUID = re.compile(r"^[0-9A-Fa-f]{8}-([0-9A-Fa-f]{4}-){3}[0-9A-Fa-f]{12}$")


def clean_swid(raw: str) -> str:
    """ESPN's SWID cookie looks like {XXXXXXXX-XXXX-...}. People often paste it without the braces or with quotes."""
    s = (raw or "").strip().strip("'\"")
    return "{" + s + "}" if _GUID.match(s) else s


def clean_s2(raw: str) -> str:
    return (raw or "").strip().strip("'\"")


class EspnSource(DraftSource):
    def __init__(self, sport: str = "nba", getter=None, league_id: int | None = None, year: int | None = None):
        if sport not in SPORT_CODE:
            raise SourceError(f"ESPN isn't supported for sport '{sport}' yet.")
        self.sport = sport
        self.s2 = clean_s2(os.getenv("ESPN_S2", ""))
        self.swid = clean_swid(os.getenv("ESPN_SWID", ""))
        self.cookies = {"espn_s2": self.s2, "SWID": self.swid} if self.s2 and self.swid else None
        if getter is None:
            try:
                self.league_id = int(league_id if league_id is not None else os.environ["ESPN_LEAGUE_ID"])
                self.year = int(year if year is not None else os.environ["ESPN_YEAR"])
            except (KeyError, ValueError):
                raise SourceError("Set ESPN_LEAGUE_ID and ESPN_YEAR (numbers) in .env.")
        else:
            self.league_id, self.year = league_id or 0, year or 0
        self._get = getter or self._http_get
        self._names: dict[str, str] | None = None
        self._names_at = 0.0
        data = self._get(True, {"view": ["mTeam", "mSettings"]})
        self.teams = data.get("teams") or []
        self.settings = data.get("settings") or {}
        nums = []
        for t in self.teams:
            try:
                nums.append(int(t["id"]))
            except (KeyError, TypeError, ValueError):
                pass
        if not nums:
            raise SourceError("ESPN answered, but the league has no teams in it. Check ESPN_LEAGUE_ID and ESPN_YEAR "
                              "(the year is the season's END year, like 2027).")
        self._dense = dense_index(nums)

    # ---- talking to ESPN ----
    def _http_get(self, league: bool, params=None, headers=None, extend: str = ""):
        url = f"{BASE}{SPORT_CODE[self.sport]}/seasons/{self.year}" + (f"/segments/0/leagues/{self.league_id}" if league else "") + extend
        try:
            r = httpx.get(url, params=params, headers={"Accept": "application/json", **(headers or {})},
                          cookies=self.cookies, timeout=15.0, follow_redirects=True)
        except httpx.TimeoutException:
            raise SourceError("ESPN took too long to answer. It will retry on its own.")
        except httpx.HTTPError as e:
            raise SourceError(f"Couldn't reach ESPN ({type(e).__name__}). Check your internet connection.")
        if r.status_code in (401, 403):
            raise SourceError("ESPN says access denied. If this league is private, set ESPN_S2 and ESPN_SWID in .env from your "
                              "browser cookies (README, 'ESPN setup'). The cookies expire, so copy them again if they stopped working.")
        if r.status_code == 404:
            raise SourceError(f"ESPN can't find league {self.league_id} for {self.year}. Check ESPN_LEAGUE_ID and ESPN_YEAR "
                              "in .env (the year is the season's END year, like 2027).")
        if r.status_code != 200:
            raise SourceError(f"ESPN answered with error {r.status_code}. It will retry on its own.")
        try:
            data = r.json()
        except ValueError:
            raise SourceError("ESPN sent a page that isn't data (often a login page). For a private league, check ESPN_S2 and ESPN_SWID.")
        return data[0] if isinstance(data, list) and data else data

    # ---- league facts (for the checker and for people) ----
    def info(self) -> dict:
        ds = self.settings.get("draftSettings") or {}
        return {"name": self.settings.get("name"), "teams": len(self.teams), "size": self.settings.get("size"),
                "draft_type": str(ds.get("type", "")).upper() or None}

    def draft_detail(self) -> dict:
        return (self._get(True, {"view": "mDraftDetail"}) or {}).get("draftDetail") or {}

    # ---- names ----
    def _load_names(self) -> None:
        players = self._get(False, {"view": "players_wl"}, {"x-fantasy-filter": json.dumps({"filterActive": {"value": True}})}, "/players")
        names = {}
        for p in players if isinstance(players, list) else []:
            if isinstance(p, dict) and p.get("id") is not None and p.get("fullName"):
                names[str(p["id"])] = str(p["fullName"])
        self._names, self._names_at = names, time.time()

    def resolve_names(self, platform_player_ids: list[str]) -> dict[str, str]:
        ids = [str(i) for i in platform_player_ids]
        stale = self._names is None or (any(i not in self._names for i in ids) and time.time() - self._names_at > NAME_RELOAD_SECONDS)
        if stale:
            try:
                self._load_names()
            except SourceError:
                self._names_at = time.time()                # don't hammer ESPN; try again in a couple of minutes
                if self._names is None:
                    return {}
        return {i: self._names[i] for i in ids if self._names and i in self._names}

    # ---- live draft ----
    def _fetch_sync(self) -> list[RemotePick]:
        teams = len(self.teams)
        raw = []
        for p in self.draft_detail().get("picks") or []:
            pid, tid = p.get("playerId"), p.get("teamId")
            if pid in (None, "", 0, -1) or tid in (None, ""):
                continue                                     # a slot nobody has picked yet
            try:
                overall = (int(p["overallPickNumber"]) if p.get("overallPickNumber")
                           else (int(p["roundId"]) - 1) * teams + int(p["roundPickNumber"]))
            except (KeyError, TypeError, ValueError):
                continue
            raw.append((overall, str(pid), str(tid), parse_cost(p.get("bidAmount"))))
        names = self.resolve_names([r[1] for r in raw]) if raw else {}
        return sorted((RemotePick(o, pid, tid, names.get(pid, ""), cost) for o, pid, tid, cost in raw), key=lambda x: x.pick_number)

    async def fetch_picks(self) -> list[RemotePick]:
        return await asyncio.to_thread(self._fetch_sync)    # blocking HTTP: keep it off the event loop

    def team_index(self, platform_team_id: str) -> int:
        return self._dense(int(platform_team_id))

    def my_team_id(self) -> int | None:
        """The team whose owner list contains your SWID cookie."""
        swid = self.swid.lower()
        if not swid:
            return None
        for t in self.teams:
            for o in t.get("owners") or []:
                oid = (o.get("id") if isinstance(o, dict) else o) or ""
                if str(oid).strip().lower() == swid:
                    return self.team_index(str(t["id"]))
        return None

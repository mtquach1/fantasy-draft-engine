

"""Yahoo Fantasy connector (OAuth2) + settings importer.

SETUP (one time):
 1. Create an app at https://developer.yahoo.com/apps/ with Fantasy Sports read access.
 2. Save data/oauth2.json:  {"consumer_key": "...", "consumer_secret": "..."}
 3. First run prompts you to open a URL and paste a verification code; the token is then cached
    in that same file and refreshed automatically.
 4. Find your league key (e.g. '458.l.123456'): yfa.Game(oauth,'mlb').league_ids(year=2027)
    (the MLB game key changes every season, so re-check each year).

CAVEAT: I wrote this against the yahoo_fantasy_api / yahoo_oauth docs from memory and could not
test it against a live league. Run it against a MOCK draft first and print raw responses.
"""
import asyncio
import os
from pathlib import Path

from ..config import LeagueConfig, RosterConfig, ScoringType
from ..ingestion import parse_positions
from ..sports import get_sport
from .base import DraftSource, RemotePick, SourceError, dense_index, parse_cost

OAUTH_FILE = Path(__file__).resolve().parents[2] / "data" / "oauth2.json"


class YahooSource(DraftSource):
    def __init__(self, league_key: str | None = None, oauth_file: Path = OAUTH_FILE):
        try:
            import yahoo_fantasy_api as yfa
            from yahoo_oauth import OAuth2
        except ModuleNotFoundError:
            raise SourceError("The Yahoo packages aren't installed. Run: pip install yahoo_oauth yahoo_fantasy_api")
        self.league_key = (league_key or os.getenv("YAHOO_LEAGUE_KEY", "")).strip()
        if not self.league_key:
            raise SourceError("Set YAHOO_LEAGUE_KEY in .env (looks like 458.l.123456; the first number changes every season).")
        if not Path(oauth_file).exists():
            raise SourceError(f"Missing {Path(oauth_file).name} in the data folder. See README, 'Yahoo setup'.")
        try:
            self.oauth = OAuth2(None, None, from_file=str(oauth_file))
            self.league = yfa.League(self.oauth, self.league_key)
        except Exception as e:
            raise SourceError(f"Yahoo sign-in or league lookup failed ({type(e).__name__}: {e}). "
                              "Check the league key and re-authorize if your token expired.") from e
        self._my_team_key = None
        self._dense = None

    # ---- live draft ----
    def _ensure_dense(self) -> None:
        """Team numbers can have gaps; map them to 1..N once (a network call, so it runs in the worker thread)."""
        if self._dense is None:
            try:
                self._dense = dense_index([int(k.split(".t.")[-1]) for k in self.league.teams()])
            except Exception:
                self._dense = lambda n: int(n)

    def _fetch_sync(self) -> list[RemotePick]:
        if not self.oauth.token_is_valid():
            self.oauth.refresh_access_token()
        self._ensure_dense()
        picks = []
        for p in self.league.draft_results():
            # unpicked slots may be missing player_id
            if p.get("player_id") in (None, "", 0):
                continue
            picks.append(RemotePick(int(p["pick"]), str(p["player_id"]), str(p["team_key"]),
                                    cost=parse_cost(p.get("cost"))))      # auction drafts only; unverified key name
        return sorted(picks, key=lambda x: x.pick_number)

    async def fetch_picks(self) -> list[RemotePick]:
        return await asyncio.to_thread(self._fetch_sync)   # yahoo libs are blocking

    def team_index(self, platform_team_id: str) -> int:
        n = int(platform_team_id.split(".t.")[-1])         # '458.l.123456.t.3' -> 3
        return self._dense(n) if self._dense else n

    def my_team_id(self) -> int | None:
        try:
            self._ensure_dense()
            return self.team_index(self.league.team_key())
        except Exception:
            return None

    def resolve_names(self, platform_player_ids: list[str]) -> dict[str, str]:
        """Draft results carry ids only; ask Yahoo for the names (player_details: verified to return
        {'player_id': '3983', 'name': {'full': ...}}). One unknown id makes the library raise for the whole batch,
        so on failure we retry id by id and keep whatever resolves."""
        def grab(ids, out):
            for d in self.league.player_details([int(i) for i in ids]):
                name = d.get("name")
                full = name.get("full") if isinstance(name, dict) else name
                if full and d.get("player_id") is not None:
                    out[str(d["player_id"])] = str(full)
        out: dict[str, str] = {}
        try:
            grab(platform_player_ids, out)
        except Exception:
            for i in platform_player_ids:
                try:
                    grab([i], out)
                except Exception:
                    pass
        return out

    # ---- league settings ----
    def fetch_settings(self) -> tuple[dict, list[dict]]:
        """(settings, stat categories). NOTE (verified in the library's source): league.settings() deliberately leaves
        out 'roster_positions', so the roster comes from league.positions() -> {'PG': {'count': 1, ...}, 'BN': {'count': 3}}.
        Without this step every slot would silently import as zero."""
        settings = dict(self.league.settings())
        try:
            pos = self.league.positions()
        except Exception as e:
            raise SourceError(f"Couldn't read the roster positions from Yahoo ({type(e).__name__}: {e}).") from e
        settings["roster_positions"] = [{"position": k, "count": int(float(v.get("count", 0) or 0))}
                                        for k, v in pos.items() if isinstance(v, dict)]
        return settings, self.league.stat_categories()

    def fetch_eligibility(self, sport: str = "mlb") -> dict[str, list[str]]:
        """Yahoo player_id -> eligible positions, using free agents (pre-draft everyone is one).
        TODO verify: free_agents(pos) may be paginated/limited; if you get too few players,
        export eligibility from Yahoo's player list into data/positions.csv instead."""
        out = {}
        for pos in sorted(get_sport(sport).valid_positions - {"DH"}):
            for p in self.league.free_agents(pos):
                out[str(p["player_id"])] = parse_positions(",".join(p.get("eligible_positions", [pos])), sport)
        return out


def yahoo_to_league_config(settings: dict, stat_cats: list[dict],
                           base: LeagueConfig | None = None) -> tuple[LeagueConfig, list[str]]:
    """Build a LeagueConfig from Yahoo settings for the config's sport (mappings live in sports.py).
    Returns (config, warnings). Anything Yahoo has that this engine can't model is reported, not silently dropped."""
    cfg = base or LeagueConfig()
    sp = get_sport(cfg.sport)
    warnings: list[str] = []

    # --- format ---
    st = str(settings.get("scoring_type", "")).lower()
    if st == "roto":
        cfg.scoring_type = ScoringType.ROTO
    elif "point" in st:
        cfg.scoring_type = ScoringType.POINTS
    elif st.startswith("head"):
        cfg.scoring_type = ScoringType.H2H_CATEGORIES
    else:
        warnings.append(f"Unrecognized Yahoo scoring_type '{st}'; left format as {cfg.scoring_type.value}")
    if cfg.scoring_type != ScoringType.POINTS and not sp.has_categories:
        cfg.scoring_type = ScoringType.POINTS
        warnings.append(f"{sp.label} supports points scoring only here; switched the format to points")

    # --- roster ---
    slots = {s: 0 for s in sp.default_slots}
    il = 0
    for item in settings.get("roster_positions", []):
        pos, n = item["position"].upper(), int(item.get("count", 0))
        mapped = sp.yahoo_roster.get(pos) or sp.yahoo_roster.get(item["position"])
        if mapped:
            slots[mapped] = slots.get(mapped, 0) + n
        elif pos in sp.yahoo_il:
            il += n
        elif pos in {"LF", "CF", "RF"} and "OF" in slots:
            slots["OF"] += n
            warnings.append(f"Roster slot {pos} treated as generic OF")
        elif pos != "NA":
            warnings.append(f"Unmodeled roster slot '{item['position']}' x{n}")
    cfg.roster = RosterConfig(teams=int(settings.get("num_teams", cfg.roster.teams)), slots=slots, il_slots=il)

    # --- categories (category formats) ---
    if cfg.scoring_type != ScoringType.POINTS:
        found: dict[str, list[str]] = {}
        for s in stat_cats:
            name, ptype = s.get("display_name"), s.get("position_type")
            kind, table = sp.yahoo_stats.get(ptype) or sp.yahoo_stats.get("*") or (None, {})
            if kind and name in table:
                found.setdefault(kind, []).append(table[name])
            elif name not in ("H/AB", "IP/GS"):          # display-only stats
                warnings.append(f"Unsupported {ptype} category '{name}' (skipped)")
        cfg.categories = {**cfg.categories, **found}
    else:
        warnings.append("Points leagues: set point_weights manually (Yahoo returns stat_ids; map them "
                        "to column names yourself, or pick a preset, then PUT /league/config).")
    return cfg, warnings





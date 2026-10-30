#!/usr/bin/env python3
"""Yahoo connection checker: run this BEFORE trusting the live sync, and keep it for your mock drafts.

  python yahoo_check.py setup            one-time: saves your Yahoo app key and secret to data/oauth2.json
  python yahoo_check.py leagues          signs in and lists your leagues (and their league keys) for the sport in .env
  python yahoo_check.py check [KEY]      tests every step the app uses and prints PASS / WARN / FAIL for each
  python yahoo_check.py watch [KEY]      prints each pick as it happens (use it during a Yahoo MOCK draft)
  python yahoo_check.py positions [KEY]  saves Yahoo's position eligibility for every undrafted player (rookies included)
                                         to data/<sport>/positions_yahoo.csv. Run it BEFORE the draft starts.

KEY looks like 466.l.123456; leave it off to use YAHOO_LEAGUE_KEY from .env. Every run also saves what it printed to
yahoo_check_output.txt: paste that file's contents if you want help. Secrets and tokens are never printed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

OAUTH_FILE = ROOT / "data" / "oauth2.json"
OUT_FILE = ROOT / "yahoo_check_output.txt"
LINES: list[str] = []


def say(msg: str = "") -> None:
    print(msg)
    LINES.append(msg)


def result(status: str, title: str, detail: str = "") -> str:
    say(f"[{status}] {title}" + (f": {detail}" if detail else ""))
    return status


def save_output() -> None:
    try:
        OUT_FILE.write_text("\n".join(LINES) + "\n", encoding="utf-8")
        print(f"\n(Saved to {OUT_FILE.name})")
    except OSError:
        pass


def explain(err) -> str | None:
    """A plain-English next step for the Yahoo errors people actually hit."""
    t = str(err).lower()
    if "not authorized to perform this action" in t:
        return ("Yahoo signed you in but refuses fantasy data: the Yahoo APP doesn't have the 'Fantasy Sports' permission. "
                "Open https://developer.yahoo.com/apps/ , open your app and make sure API Permissions has Fantasy Sports ticked "
                "(Read is enough). If you can't change it, create a NEW app with it ticked. Then run  python yahoo_check.py setup  "
                "again (it clears the old sign-in) and  python yahoo_check.py leagues .")
    if "invalid_grant" in t or "invalid_client" in t or "unauthorized" in t or "401" in t:
        return ("Yahoo rejected the sign-in. Check the Client ID/Secret and that the redirect URI saved by setup is exactly the one on "
                "developer.yahoo.com. Then run  python yahoo_check.py setup  again and paste the NEWEST code.")
    if "token" in t and ("expired" in t or "refresh" in t):
        return "The saved sign-in expired and couldn't refresh. Run  python yahoo_check.py setup  then  leagues  again."
    return None


# ---------------------------------------------------------------- setup
def setup_credentials(ask=input, path: Path = OAUTH_FILE) -> None:
    say("Create your Yahoo app first: https://developer.yahoo.com/apps/  ->  Create an App")
    say("  Application Type: Installed Application (or Web Application); Permissions: Fantasy Sports, Read.")
    say("  Redirect URI: if Yahoo lets you, use  oob . If it insists on https, use  https://localhost:8080")
    key = ask("Paste the Client ID (Consumer Key): ").strip()
    secret = ask("Paste the Client Secret (Consumer Secret): ").strip()
    redirect = ask("Redirect URI you registered [oob]: ").strip() or "oob"
    if not key or not secret:
        result("FAIL", "Setup", "the key and secret can't be empty. Nothing was saved.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"consumer_key": key, "consumer_secret": secret, "callback_uri": redirect}, indent=2),
                    encoding="utf-8")
    result("PASS", "Setup", f"saved {path.name}. Next: python yahoo_check.py leagues")
    say("The first sign-in opens a browser page. Approve it. Yahoo then shows a code (for 'oob'), or sends your browser to "
        "a page that fails to load whose address contains  code=XXXX . Copy that code and paste it into this window.")


# ---------------------------------------------------------------- connection
def sport_from_env() -> str:
    return os.getenv("SPORT", "nba").strip().lower() or "nba"


def connect_oauth(path: Path = OAUTH_FILE):
    try:
        import yahoo_fantasy_api  # noqa: F401
        from yahoo_oauth import OAuth2
    except ModuleNotFoundError:
        result("FAIL", "Packages", "run: pip install yahoo_oauth yahoo_fantasy_api")
        return None
    result("PASS", "Packages installed")
    if not path.exists():
        result("FAIL", "Credentials file", f"{path.name} not found. Run: python yahoo_check.py setup")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        result("FAIL", "Credentials file", f"{path.name} isn't valid JSON ({e}). Run setup again.")
        return None
    if not data.get("consumer_key") or not data.get("consumer_secret"):
        result("FAIL", "Credentials file", "consumer_key / consumer_secret missing. Run setup again.")
        return None
    result("PASS", "Credentials file", f"found ({'token saved' if data.get('access_token') else 'no token yet: sign-in will start'})")
    import logging
    if data.get("access_token"):                    # normal runs: hide the library's debug chatter
        logging.getLogger("yahoo_oauth").setLevel(logging.WARNING)
    else:
        say("First sign-in: a browser page opens (or copy the long address printed below into your browser). Click Agree.")
        say("Then copy the code Yahoo shows (or the part after  code=  in the address of the page that fails to load) and paste it "
            "next to 'Enter verifier'.")
    try:
        oauth = OAuth2(None, None, from_file=str(path))
        if not oauth.token_is_valid():
            oauth.refresh_access_token()
    except Exception as e:
        result("FAIL", "Yahoo sign-in", f"{type(e).__name__}: {e}")
        say("   Hint: " + (explain(e) or "if the code was rejected, run it again and paste the NEWEST code. If it keeps failing, "
            "check that the redirect URI in oauth2.json (callback_uri) matches the one registered on developer.yahoo.com exactly."))
        return None
    result("PASS", "Yahoo sign-in")
    return oauth


def list_leagues(oauth, sport: str) -> list[str]:
    import yahoo_fantasy_api as yfa
    try:
        gm = yfa.Game(oauth, sport)
        ids = list(gm.league_ids())
    except Exception as e:
        result("FAIL", f"Listing your {sport.upper()} leagues", f"{type(e).__name__}: {str(e)[:300]}")
        if explain(e):
            say("   What to do: " + explain(e))
        return []
    try:                                           # the season code that starts every league key (it changes each season)
        gid = gm.game_id()
        result("PASS", f"Yahoo's {sport.upper()} season code", str(gid))
        say(f"   A league key is  {gid}.l.<league number>.  The league number is the last number in the league's web address,")
        say(f"   e.g. .../nba/12345 -> {gid}.l.12345  (use this if your mock draft isn't in the list below)")
    except Exception:
        pass
    if not ids:
        result("WARN", f"Your {sport.upper()} leagues", "none found. Mock drafts often aren't listed. Are you signed in as the "
               "right Yahoo account, and is SPORT in .env right? If so, build the key from the season code above.")
        return []
    result("PASS", f"Your {sport.upper()} leagues", f"{len(ids)} found")
    for k in ids:
        say(f"   {k}")
    say("   Newest season is usually last. Put the one you want in .env as  YAHOO_LEAGUE_KEY=...")
    return ids


# ---------------------------------------------------------------- the checks
def _name_matcher(sport: str):
    """-> (clean_name -> player_id map, note). Uses the same name rules as the live sync."""
    try:
        from app import ingestion, validate
        raw, _ = validate.clean_projections(ingestion.load_projections(sport), sport)
        return ingestion.build_name_map(raw), None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


def check_league(src, sport: str) -> dict:
    """Run every step the live app depends on against `src` (a YahooSource). Returns {status counts}."""
    from app.config import LeagueConfig
    from app.connectors.yahoo import yahoo_to_league_config
    from app.ingestion import clean_name
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    def rec(status, *a):
        counts[result(status, *a)] += 1

    # 1. settings
    settings = cats = None
    try:
        settings, cats = src.fetch_settings()
        shown = {k: settings.get(k) for k in ("name", "draft_status", "scoring_type", "num_teams", "is_auction_draft")
                 if k in settings}
        rec("PASS", "League settings", json.dumps(shown))
        if str(settings.get("draft_status", "")).lower() == "postdraft":
            rec("WARN", "Draft status", "this league's draft is already finished: nothing will stream live")
    except Exception as e:
        rec("FAIL", "League settings", f"{type(e).__name__}: {e}")

    # 2. settings -> this app's config
    if settings is not None:
        try:
            cfg, warns = yahoo_to_league_config(settings, cats or [], LeagueConfig.for_sport(sport))
            rec("PASS", "Settings imported", f"format={cfg.scoring_type.value}, teams={cfg.roster.teams}, "
                f"slots={ {k: v for k, v in cfg.roster.slots.items() if v} }, IL={cfg.roster.il_slots}")
            if cfg.categories:
                say(f"      categories: {cfg.categories}")
            for w in warns:
                rec("WARN", "Import note", w)
        except Exception as e:
            rec("FAIL", "Settings imported", f"{type(e).__name__}: {e}")

    # 3. which team is mine
    try:
        me = src.my_team_id()
        if me is None:
            rec("FAIL", "Your team number", "couldn't find your team in this league (wrong league key, or not signed in as a "
                "member). The app would assume MY_TEAM_ID from .env.")
        else:
            rec("PASS", "Your team number", f"team {me}")
    except Exception as e:
        rec("FAIL", "Your team number", f"{type(e).__name__}: {e}")

    # 4. the pick list
    picks = []
    try:
        picks = src._fetch_sync()
        if not picks:
            rec("PASS", "Pick list", "0 picks so far (normal before the draft starts)")
        else:
            nums = [p.pick_number for p in picks]
            rec("PASS", "Pick list", f"{len(picks)} pick(s); latest is #{max(nums)}")
            for p in picks[:3]:
                say(f"      #{p.pick_number}: team {src.team_index(p.platform_team_id)} took Yahoo player {p.platform_player_id}"
                    + (f" for ${p.cost}" if p.cost else ""))
            if nums != list(range(1, len(nums) + 1)):
                rec("WARN", "Pick numbers", f"not 1..{len(nums)} without gaps: {nums[:12]}...")
    except Exception as e:
        rec("FAIL", "Pick list", f"{type(e).__name__}: {e}")

    # 5. names for those ids (or, before the draft, for some free agents)
    names: dict[str, str] = {}
    try:
        if picks:
            names = src.resolve_names([str(p.platform_player_id) for p in picks[:10]])
            label = f"{len(names)} of {min(len(picks), 10)} drafted player(s) named"
        else:
            fa = src.league.free_agents("C") or src.league.free_agents("PG")
            names = {str(p["player_id"]): p["name"] for p in fa[:15] if p.get("name")}
            label = f"{len(names)} sample player(s) named (from the free-agent list, since nobody is drafted yet)"
        rec("PASS" if names else "WARN", "Player names", label if names else
            "Yahoo returned no names, so picks can only be matched through id_crosswalk.csv")
    except Exception as e:
        rec("WARN", "Player names", f"{type(e).__name__}: {e}")

    # 6. do those names match YOUR projections file?
    if names:
        nm, err = _name_matcher(sport)
        if nm is None:
            rec("WARN", "Matching to your projections", f"couldn't load them ({err}). Add data/{sport}/players.csv first.")
        else:
            hit = [n for n in names.values() if clean_name(n) in nm]
            miss = [n for n in names.values() if clean_name(n) not in nm]
            pct = round(100 * len(hit) / len(names))
            rec("PASS" if pct >= 80 else "WARN", "Matching to your projections",
                f"{len(hit)} of {len(names)} matched by name ({pct}%)")
            if miss:
                say(f"      not matched: {', '.join(miss[:8])}")
                say("      (rookies missing from the file, spelling differences, or two players sharing a name. Fix the name in "
                    "your file, or add id_crosswalk.csv)")
    say("")
    say(f"Summary: {counts['PASS']} passed, {counts['WARN']} warning(s), {counts['FAIL']} failed.")
    return counts


# ---------------------------------------------------------------- positions
def export_positions(src, sport: str, path=None) -> dict:
    """Ask Yahoo which positions every undrafted player can play (rookies included) and save them as positions_yahoo.csv.
    player_id is 'n:<clean name>', the same id the app builds from names. Names shared by two players are skipped."""
    import csv
    from app.ingestion import clean_name, data_dir, parse_positions
    from app.sports import get_sport
    found: dict[str, set] = {}
    shown: dict[str, set] = {}                    # clean name -> yahoo ids, to spot two players with one name
    for pos in sorted(get_sport(sport).valid_positions):
        for p in src.league.free_agents(pos):
            name = p.get("name")
            if not name:
                continue
            k = clean_name(name)
            shown.setdefault(k, set()).add(str(p.get("player_id")))
            found.setdefault(k, set()).update(parse_positions(",".join(p.get("eligible_positions", [pos])), sport))
    ambiguous = sorted(k for k, ids in shown.items() if len(ids) > 1)
    rows = [("n:" + k, ",".join(sorted(v))) for k, v in sorted(found.items()) if v and k not in ambiguous]
    path = Path(path) if path else data_dir(sport) / "positions_yahoo.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["player_id", "eligible_positions"])
        w.writerows(rows)
    if not rows:
        result("FAIL", "Positions", "Yahoo returned no players. Run this BEFORE the draft starts (drafted players aren't free agents).")
    else:
        result("PASS", "Positions", f"saved {len(rows)} players to {path}")
        if ambiguous:
            result("WARN", "Positions", f"{len(ambiguous)} name(s) belong to two players and were skipped: {', '.join(ambiguous[:6])}")
    return {"saved": len(rows), "ambiguous": ambiguous, "path": str(path)}


# ---------------------------------------------------------------- live watch
def watch(src, sport: str, poll: float = 3.0, max_seconds: float | None = None, sleep=time.sleep, clock=time.time) -> int:
    """Print every new pick as it arrives. Returns the number of picks seen."""
    from app.ingestion import clean_name
    nm, _ = _name_matcher(sport)
    me = None
    try:
        me = src.my_team_id()
    except Exception:
        pass
    say(f"Watching the draft every {poll:g}s (Ctrl+C to stop). You are team {me if me else '?'}.")
    seen: dict[int, tuple] = {}
    start = clock()
    while max_seconds is None or clock() - start < max_seconds:
        try:
            picks = src._fetch_sync()
        except Exception as e:
            say(f"[{time.strftime('%H:%M:%S')}] couldn't read picks ({type(e).__name__}: {e}); retrying")
            sleep(poll)
            continue
        if len(picks) < len(seen):
            say(f"[{time.strftime('%H:%M:%S')}] pick count dropped {len(seen)} -> {len(picks)}: a pick was undone or corrected")
            seen = {p.pick_number: (p.platform_player_id, p.platform_team_id) for p in picks}
        new = [p for p in picks if seen.get(p.pick_number) != (p.platform_player_id, p.platform_team_id)]
        if new:
            names = src.resolve_names([str(p.platform_player_id) for p in new]) or {}
            for p in new:
                seen[p.pick_number] = (p.platform_player_id, p.platform_team_id)
                team = src.team_index(p.platform_team_id)
                nm_txt = names.get(str(p.platform_player_id), f"Yahoo player {p.platform_player_id}")
                tag = "" if nm is None else ("  [matches your file]" if clean_name(nm_txt) in nm else "  [NOT in your file by name]")
                say(f"[{time.strftime('%H:%M:%S')}] pick #{p.pick_number}: team {team} took {nm_txt}"
                    + (f" for ${p.cost}" if p.cost else "") + tag + ("   <-- YOU" if me and team == me else ""))
        sleep(poll)
    return len(seen)


# ---------------------------------------------------------------- command line
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Yahoo connection checker")
    ap.add_argument("command", choices=["setup", "leagues", "check", "watch", "positions"])
    ap.add_argument("league_key", nargs="?", help="e.g. 466.l.123456 (default: YAHOO_LEAGUE_KEY in .env)")
    ap.add_argument("--sport", help="nba | mlb | nfl (default: SPORT in .env)")
    ap.add_argument("--poll", type=float, default=3.0, help="seconds between checks in watch mode")
    args = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ModuleNotFoundError:
        pass
    sport = (args.sport or sport_from_env()).lower()
    try:
        if args.command == "setup":
            setup_credentials()
            return 0
        oauth = connect_oauth()
        if oauth is None:
            return 1
        if args.command == "leagues":
            return 0 if list_leagues(oauth, sport) else 1
        key = (args.league_key or os.getenv("YAHOO_LEAGUE_KEY", "")).strip()
        if not key:
            result("FAIL", "League key", "give it on the command line or set YAHOO_LEAGUE_KEY in .env (run 'leagues' to find it)")
            return 1
        from app.connectors.base import SourceError
        from app.connectors.yahoo import YahooSource
        try:
            src = YahooSource(key)
        except SourceError as e:
            result("FAIL", "Connecting to the league", str(e)[:300])
            if explain(e):
                say("   What to do: " + explain(e))
            return 1
        result("PASS", "Connecting to the league", key)
        if args.command == "check":
            return 1 if check_league(src, sport)["FAIL"] else 0
        if args.command == "positions":
            return 0 if export_positions(src, sport)["saved"] else 1
        watch(src, sport, max(args.poll, 2.0))
        return 0
    except KeyboardInterrupt:
        say("\nStopped.")
        return 0
    finally:
        save_output()


if __name__ == "__main__":
    sys.exit(main())

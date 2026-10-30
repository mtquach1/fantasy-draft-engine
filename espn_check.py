#!/usr/bin/env python3
"""ESPN connection checker: run this BEFORE trusting the live sync.

  python espn_check.py check     tests every step the app uses and prints PASS / WARN / FAIL for each
  python espn_check.py watch     prints each pick as it happens (use it while a draft is running)

Reads ESPN_LEAGUE_ID, ESPN_YEAR, ESPN_S2, ESPN_SWID and SPORT from your .env file (README: 'ESPN setup').
Every run saves what it printed to espn_check_output.txt: paste it if you want help. Cookie values are never printed.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import yahoo_check as common  # noqa: E402  (shared printing, name matching and the live pick watcher)

common.OUT_FILE = ROOT / "espn_check_output.txt"
result, say = common.result, common.say


def check_env_values(env=os.environ) -> int:
    """Shape-check the .env values without printing them. Returns the number of problems."""
    bad = 0
    lid, year = env.get("ESPN_LEAGUE_ID", "").strip(), env.get("ESPN_YEAR", "").strip()
    if not lid.isdigit():
        result("FAIL", "ESPN_LEAGUE_ID", "must be the number after leagueId= in your league's web address")
        bad += 1
    else:
        result("PASS", "ESPN_LEAGUE_ID", "looks right")
    if not re.fullmatch(r"\d{4}", year):
        result("FAIL", "ESPN_YEAR", "must be the season's END year, like 2027 for the 2026-27 basketball season")
        bad += 1
    else:
        result("PASS", "ESPN_YEAR", year)
    s2, swid = env.get("ESPN_S2", "").strip(), env.get("ESPN_SWID", "").strip()
    if not s2 and not swid:
        result("WARN", "Cookies", "ESPN_S2 and ESPN_SWID are empty: fine for a PUBLIC league, but a private league will say 'access denied'")
    elif not (s2 and swid):
        result("FAIL", "Cookies", "you need BOTH ESPN_S2 and ESPN_SWID (or neither)")
        bad += 1
    else:
        from app.connectors.espn import clean_s2, clean_swid
        okswid = bool(re.fullmatch(r"\{[0-9A-Fa-f-]{36}\}", clean_swid(swid)))
        if not okswid:
            result("FAIL", "ESPN_SWID", "should look like {XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX}")
            bad += 1
        elif len(clean_s2(s2)) < 40:
            result("WARN", "ESPN_S2", "looks too short: it is a very long string. Re-copy the whole cookie value")
        else:
            result("PASS", "Cookies", "both present and the right shape")
    return bad


def check_league(src, sport: str) -> dict:
    """Run every step the live app depends on against `src` (an EspnSource). Returns {status: count}."""
    from app.ingestion import clean_name
    counts = {"PASS": 0, "WARN": 0, "FAIL": 0}

    def rec(status, *a):
        counts[result(status, *a)] += 1

    info = src.info()
    rec("PASS", "League", f"{info.get('name')!r}: {info['teams']} teams, draft type {info.get('draft_type') or 'unknown'}")
    if info.get("size") and info["size"] != info["teams"]:
        rec("WARN", "Team count", f"settings say {info['size']} teams but {info['teams']} were listed")

    try:
        me = src.my_team_id()
        if me:
            rec("PASS", "Your team number", f"team {me}")
        elif not src.swid:
            rec("WARN", "Your team number", "unknown: set ESPN_SWID so the app knows which team is yours (or set MY_TEAM_ID in .env)")
        else:
            rec("WARN", "Your team number", "your SWID isn't an owner of any team in this league. Are the cookies from the right ESPN account?")
    except Exception as e:
        rec("FAIL", "Your team number", f"{type(e).__name__}: {e}")

    picks = []
    try:
        dd = src.draft_detail()
        picks = src._fetch_sync()
        state = "finished" if dd.get("drafted") else "in progress" if dd.get("inProgress") else "not started"
        rec("PASS", "Draft", f"{state}; {len(picks)} pick(s) made")
        if dd.get("drafted"):
            rec("WARN", "Draft status", "this league's draft is already finished: nothing will stream live")
        nums = [p.pick_number for p in picks]
        if nums and nums != list(range(1, len(nums) + 1)):
            rec("WARN", "Pick numbers", f"not 1..{len(nums)} without gaps: {nums[:12]}")
        for p in picks[:3]:
            say(f"      #{p.pick_number}: team {src.team_index(p.platform_team_id)} took ESPN player {p.platform_player_id}"
                + (f" ({p.player_name})" if p.player_name else "") + (f" for ${p.cost}" if p.cost else ""))
    except Exception as e:
        rec("FAIL", "Draft", f"{type(e).__name__}: {e}")

    nm, err = common._name_matcher(sport)
    try:
        if src._names is None:
            src._load_names()
        pool = list(src._names.values())
    except Exception as e:
        pool = []
        rec("WARN", "Player names", f"couldn't download ESPN's player list ({type(e).__name__}: {e}); picks can only match through id_crosswalk.csv")
    if pool:
        if picks:
            named = sum(1 for p in picks if p.player_name)
            rec("PASS" if named == len(picks) else "WARN", "Player names on picks", f"{named} of {len(picks)} picks have a name")
        rec("PASS", "Player names", f"ESPN lists {len(pool)} active players")
        if nm is None:
            rec("WARN", "Matching to your projections", f"couldn't load them ({err}). Add data/{sport}/players.csv first.")
        else:
            miss = [n for n in pool if clean_name(n) not in nm]
            pct = round(100 * (len(pool) - len(miss)) / len(pool))
            rec("PASS" if pct >= 60 else "WARN", "Your file covers ESPN's players", f"{pct}% of ESPN's active players are in your projections by name")
            say("      (a low number here is normal if your file only has the top ~200-500 players; what matters is that the "
                "players being DRAFTED match. Use  watch  to see that live.)")
    say("")
    say(f"Summary: {counts['PASS']} passed, {counts['WARN']} warning(s), {counts['FAIL']} failed.")
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ESPN connection checker")
    ap.add_argument("command", choices=["check", "watch"])
    ap.add_argument("--sport", help="nba | mlb | nfl (default: SPORT in .env)")
    ap.add_argument("--poll", type=float, default=3.0, help="seconds between checks in watch mode")
    args = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ModuleNotFoundError:
        pass
    sport = (args.sport or common.sport_from_env()).lower()
    try:
        if check_env_values():
            return 1
        from app.connectors.base import SourceError
        from app.connectors.espn import EspnSource
        try:
            src = EspnSource(sport)
        except SourceError as e:
            result("FAIL", "Connecting to ESPN", str(e))
            return 1
        result("PASS", "Connecting to ESPN")
        if args.command == "check":
            return 1 if check_league(src, sport)["FAIL"] else 0
        common.watch(src, sport, max(args.poll, 2.0))
        return 0
    except KeyboardInterrupt:
        say("\nStopped.")
        return 0
    finally:
        common.save_output()


if __name__ == "__main__":
    sys.exit(main())

"""Where would each team in a recorded mock draft rank, judged by what the players really produced (a finished season or the season so far)?

Reads the draft log the app saved (data/draft_state_nba.json) and a Basketball-Reference per-game table, builds the scoring ruler from that
table (app/season_ruler.py), and scores every team's picks with the app's own lineup assignment under the ESPN default roster.

Fairness: teams in a snake draft can have different numbers of picks when the log stops mid-round, so by default only each team's first N
picks are compared, N being the smallest pick count of any team. Use --picks N to choose N, or --all to use every pick (unequal).
A drafted player with under 100 minutes in the table counts as nothing (the slot is filled at waiver level), so injuries, rookies who have
not played yet, and early-season benchings all hurt. It is ONE draft, and early in a season a noisy one: an illustration, not evidence.

Run from the project root:
    python research/mock_rank.py [draft.json] [season.csv] [--picks N] [--all]
Defaults: data/draft_state_nba.json and the newest file in data/nba/inseason/."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SPORT", "nba"); os.environ.setdefault("LEAGUE_PRESET", "espn_default"); os.environ.setdefault("DRAFT_PLATFORM", "manual")

import pandas as pd                                    # noqa: E402

from app import ingestion, persist, own_model as om    # noqa: E402
from app.config import LeagueConfig                    # noqa: E402
from app.season_ruler import build_ruler               # noqa: E402

MIN_MINUTES = 100


def load_draft(path):
    """-> (teams, my_slot or None, {team_id: [player names in pick order]}, unmatched_count). Raises ValueError with a message to show."""
    p = Path(path)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"{p} was not found. It is the file the app saves during a draft (data/draft_state_nba.json); pass its path as the first argument.")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"{p} is not a valid draft file ({e}).")
    if not isinstance(d, dict) or not isinstance(d.get("draft_log"), list):
        raise ValueError(f"{p} has no draft_log list: is it a saved draft?")
    if d.get("sport", "nba") != "nba":
        raise ValueError(f"{p} is a {d.get('sport')} draft: this script is basketball only.")
    log = d["draft_log"]
    if not log:
        raise ValueError(f"{p} has no picks in it.")
    setup = d.get("setup") or {}
    teams = setup.get("teams")
    if not isinstance(teams, int) or isinstance(teams, bool) or not 2 <= teams <= 30:
        try:
            teams = max(int(x["team_id"]) for x in log)
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"{p}: cannot tell how many teams were in the draft.")
    my_slot = setup.get("my_slot") if isinstance(setup.get("my_slot"), int) else None
    picks, unmatched = {t: [] for t in range(1, teams + 1)}, 0
    for i, x in enumerate(sorted(log, key=lambda q: q.get("pick_number", 0)), 1):
        if not isinstance(x, dict) or not isinstance(x.get("team_id"), int) or isinstance(x.get("team_id"), bool) or x["team_id"] not in picks:
            raise ValueError(f"pick #{i}: team_id must be a whole number from 1 to {teams} (got {x.get('team_id') if isinstance(x, dict) else x!r}).")
        pid = x.get("player_id")
        if isinstance(pid, str) and pid.startswith("n:") and pid[2:].strip():
            picks[x["team_id"]].append(pid[2:])
        else:
            unmatched += 1                           # a live-sync pick the app could not match to a player
    return teams, my_slot, picks, unmatched


def rank_teams(picks, scorer, ids, k):
    """One row per team: its first k picks scored on the real stats. Rank 1 = best."""
    rows = []
    for t, names in picks.items():
        use = names[:k] if k else names
        found = [om.norm_name(n) for n in use if om.norm_name(n) in ids]
        missing = [n for n in use if om.norm_name(n) not in ids]
        rows.append(dict(team=t, score=float(scorer.score([scorer.pos[ids[f]] for f in found])), counted=len(use), missing=missing))
    order = sorted(rows, key=lambda r: -r["score"])
    for rank, r in enumerate(order, 1):
        r["rank"] = rank
    return order


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rank the teams of a recorded mock draft by real stats.")
    ap.add_argument("draft", nargs="?"); ap.add_argument("season", nargs="?")
    ap.add_argument("--picks", type=int); ap.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    draft = Path(a.draft) if a.draft else persist.DATA_DIR / "draft_state_nba.json"
    season = Path(a.season) if a.season else None
    if season is None:
        found = om.find_seasons(ingestion.DATA_DIR / "nba" / "inseason")
        if not found:
            print("No season file given and none in data/nba/inseason. Save the Basketball-Reference per-game table there as per_game_<year>.csv "
                  "(or pass its path as the second argument)."); return 1
        season = found[max(found)]
    if not season.exists():
        print(f"{season} was not found."); return 1
    if a.picks is not None and a.picks < 1:
        print("--picks must be 1 or more."); return 1
    try:
        teams, my_slot, picks, unmatched = load_draft(draft)
        tbl = om.load_season(season)
    except ValueError as e:
        print(e); return 1
    except (KeyError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as e:
        print(f"{season} does not look like a Basketball-Reference per-game table ({type(e).__name__}: {e})."); return 1
    counts = {t: len(v) for t, v in picks.items()}
    if min(counts.values()) == 0:
        print("Team(s) " + ", ".join(str(t) for t, n in counts.items() if n == 0) + " have no picks in this draft, so teams cannot be compared."); return 1
    kmin = min(counts.values())
    if a.all:
        k, note = 0, "every pick (teams may have different numbers of picks)" if len(set(counts.values())) > 1 else f"all {kmin} picks"
    else:
        k = min(a.picks or kmin, kmin)
        note = f"each team's first {k} picks" + (f" (you asked for {a.picks}; the shortest team has {kmin})" if a.picks and a.picks > kmin else "")
    cfg = LeagueConfig.for_sport("nba", "espn_default"); cfg.roster.teams = teams
    scorer, ids, _ = build_ruler(tbl, cfg, MIN_MINUTES)
    order = rank_teams(picks, scorer, ids, k)
    games = tbl[tbl["mins"] >= MIN_MINUTES]["G"].median()
    print(f"Mock draft ({teams} teams, {sum(counts.values())} picks) scored with the real stats in {season.name}")
    print(f"Compared: {note}. Players with {MIN_MINUTES}+ minutes have played {games:.0f} games (median).\n")
    print(f"{'rank':>4}  {'team':<8} {'lineup points':>14}  {'picks':>5}  not in the table (under {MIN_MINUTES} min)")
    for r in order:
        tag = "  <- you" if r["team"] == my_slot else ""
        miss = ", ".join(r["missing"][:3]) + (f" +{len(r['missing']) - 3} more" if len(r["missing"]) > 3 else "")
        print(f"{r['rank']:>4}  team {r['team']:<3} {r['score']:>14.0f}  {r['counted']:>5}  {miss}{tag}")
    if my_slot in picks:
        mine = next(r for r in order if r["team"] == my_slot)
        print(f"\nYour team (slot {my_slot}) ranks {mine['rank']} of {teams}.")
    if unmatched:
        print(f"({unmatched} pick(s) in the log had no player the script could read and were skipped.)")
    if games < 40:
        print(f"\nPartial season: about {games:.0f} games per player. Early results move a lot from week to week, rookies who have not "
              "played count as nothing, and this is one draft. Rerun with a fresher file as the season goes on.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

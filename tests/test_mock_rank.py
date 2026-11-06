"""research/mock_rank.py and app/season_ruler.py: rank the teams of a recorded draft by real stats. Fake season, fake draft, temp folders only."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import ingestion, own_model as om, persist
from app.config import LeagueConfig
from app.season_ruler import actual_frame, build_ruler

ROOT = Path(__file__).resolve().parents[1]
POS = ["PG", "SG", "SF", "PF", "C", "PG-SG", "SF-PF", "PF-C", "SG-SF"]


def _load():
    spec = importlib.util.spec_from_file_location("mock_rank", ROOT / "research" / "mock_rank.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def fake_season(path, n=240, games=7, seed=4, rookie_low_minutes=()):
    """n players with a strict talent order (Player000 best). Everyone plays enough minutes to be draftable except `rookie_low_minutes`."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        t = 3.0 - 2.4 * i / n                                         # descending talent
        mpg = float(np.clip(14 + 8 * t, 12, 38)); pts = 0.55 * mpg * (0.5 + 0.12 * t)
        g = games if i not in rookie_low_minutes else 1
        rows.append({"Player": f"Player{i:03d}", "Player-additional": f"pl{i:03d}", "Age": 20 + i % 14, "Team": "AAA", "Pos": POS[i % len(POS)], "G": g,
                     "MP": mpg if i not in rookie_low_minutes else 5.0, "FG": pts / 2.4, "FGA": pts / 1.15, "FT": pts * 0.14, "FTA": pts * 0.18, "3P": pts * 0.08,
                     "TRB": mpg * 0.2 * t / 2, "AST": mpg * 0.1 * t / 2, "STL": mpg * 0.03 * t / 2, "BLK": mpg * 0.02 * t / 2, "TOV": mpg * 0.06, "PTS": pts})
    pd.DataFrame(rows).round(3).to_csv(path, index=False)
    return path


def write_draft(path, picks_per_team=4, teams=12, my_slot=3, extra=None, block=True, sport="nba"):
    """Block draft: team 1 gets the 4 best players, team 2 the next 4, ... so the true order of the teams is known."""
    log, n = [], 0
    for rnd in range(picks_per_team):
        for t in range(1, teams + 1):
            idx = (t - 1) * picks_per_team + rnd if block else n
            n += 1
            log.append({"pick_number": n, "player_id": f"n:player{idx:03d}", "team_id": t, "source": "manual"})
    log += extra or []
    Path(path).write_text(json.dumps({"version": 1, "sport": sport, "setup": {"draft_type": "snake", "teams": teams, "my_slot": my_slot}, "draft_log": log}), encoding="utf-8")
    return path


@pytest.fixture
def season(tmp_path):
    return fake_season(tmp_path / "per_game_2027.csv")


def run(mod, *args):
    return mod.main([str(a) for a in args])


# ---------------------------------------------------------------------------------------- ruler
def test_the_ruler_scores_better_players_higher(season):
    cfg = LeagueConfig.for_sport("nba", "espn_default"); cfg.roster.teams = 12
    sc, ids, val = build_ruler(om.load_season(season), cfg)
    best = [sc.pos[ids[om.norm_name(f"Player{i:03d}")]] for i in range(8)]
    worse = [sc.pos[ids[om.norm_name(f"Player{i:03d}")]] for i in range(100, 108)]
    assert sc.score(best) > sc.score(worse) > 0


def test_building_a_ruler_does_not_leave_the_data_folder_pointing_at_its_temp_folder(season):
    before = ingestion.DATA_DIR
    build_ruler(om.load_season(season), LeagueConfig.for_sport("nba", "espn_default"))
    assert ingestion.DATA_DIR == before


def test_actual_frame_turns_per_game_into_season_totals_and_drops_tiny_samples(tmp_path):
    f = fake_season(tmp_path / "s.csv", n=30, games=7, rookie_low_minutes=(5,))
    raw = om.load_season(f); fr = actual_frame(raw)
    assert "Player005" not in set(fr["Name"]) and len(fr) == 29                     # 1 game x 5 minutes is under 100 total minutes
    row = fr[fr["Name"] == "Player000"].iloc[0]
    src = raw[raw["Player"] == "Player000"].iloc[0]
    assert row["GP"] == 7 and row["PTS"] == pytest.approx(src["PTS"] * 7)
    assert row["playerid"] == "n:player000"


# ---------------------------------------------------------------------------------------- ranking
def test_the_team_that_drafted_the_best_players_ranks_first_and_the_worst_last(tmp_path, season, capsys):
    mod = _load(); d = write_draft(tmp_path / "d.json")
    assert run(mod, d, season) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip().startswith(tuple(str(i) for i in range(1, 13))) and "team" in ln]
    order = [int(ln.split("team")[1].split()[0]) for ln in lines]
    assert len(order) == 12 and order[0] == 1 and order[-1] == 12 and set(order[:3]) <= {1, 2, 3, 4}
    assert "Your team (slot 3) ranks" in out and "<- you" in out
    assert "each team's first 4 picks" in out and "Partial season: about 7 games" in out


def test_rank_teams_is_deterministic_and_rank_one_has_the_top_score(tmp_path, season):
    mod = _load(); teams, _, picks, _ = mod.load_draft(write_draft(tmp_path / "d.json"))
    cfg = LeagueConfig.for_sport("nba", "espn_default"); cfg.roster.teams = teams
    sc, ids, _ = build_ruler(om.load_season(season), cfg)
    a, b = mod.rank_teams(picks, sc, ids, 4), mod.rank_teams(picks, sc, ids, 4)
    assert a == b and a[0]["rank"] == 1 and a[0]["score"] == max(r["score"] for r in a) and [r["rank"] for r in a] == list(range(1, 13))


def test_unequal_pick_counts_compare_the_first_n_picks_unless_asked_otherwise(tmp_path, season, capsys):
    mod = _load()
    extra = [{"pick_number": 49 + i, "player_id": f"n:player{200 + i:03d}", "team_id": i + 1, "source": "manual"} for i in range(3)]   # teams 1-3 have a 5th pick
    d = write_draft(tmp_path / "d.json", extra=extra)
    assert run(mod, d, season) == 0 and "first 4 picks" in capsys.readouterr().out
    assert run(mod, d, season, "--all") == 0 and "different numbers of picks" in capsys.readouterr().out
    assert run(mod, d, season, "--picks", 2) == 0 and "first 2 picks" in capsys.readouterr().out
    assert run(mod, d, season, "--picks", 9) == 0 and "you asked for 9; the shortest team has 4" in capsys.readouterr().out
    assert run(mod, d, season, "--picks", 0) == 1 and "1 or more" in capsys.readouterr().out


def test_a_pick_who_has_not_played_counts_as_nothing_and_is_named(tmp_path, capsys):
    mod = _load(); season = fake_season(tmp_path / "per_game_2027.csv", rookie_low_minutes=(0,))     # the best player is hurt: 1 game, 5 minutes
    d = write_draft(tmp_path / "d.json")
    assert run(mod, d, season) == 0
    out = capsys.readouterr().out
    team1 = next(ln for ln in out.splitlines() if "team 1 " in ln or ln.split()[1:3] == ["team", "1"])
    assert "player000" in team1 and "<-" not in team1
    teams, _, picks, _ = mod.load_draft(d)
    cfg = LeagueConfig.for_sport("nba", "espn_default"); cfg.roster.teams = teams
    sc, ids, _ = build_ruler(om.load_season(season), cfg)
    t1 = next(r for r in mod.rank_teams(picks, sc, ids, 4) if r["team"] == 1)
    assert t1["missing"] == ["player000"] and t1["counted"] == 4


def test_picks_the_app_could_not_match_are_skipped_and_reported(tmp_path, season, capsys):
    mod = _load()
    extra = [{"pick_number": 50, "player_id": None, "team_id": 5, "source": "yahoo"}, {"pick_number": 51, "player_id": "2238", "team_id": 6, "source": "yahoo"}]
    assert run(mod, write_draft(tmp_path / "d.json", extra=extra), season) == 0
    assert "2 pick(s) in the log had no player the script could read" in capsys.readouterr().out


def test_accents_and_punctuation_in_names_still_match(tmp_path):
    mod = _load(); f = tmp_path / "per_game_2027.csv"; fake_season(f)
    t = pd.read_csv(f); t.loc[0, "Player"] = "Nikola Jokić"; t.loc[1, "Player"] = "De'Aaron Fox"; t.to_csv(f, index=False, encoding="utf-8")
    d = write_draft(tmp_path / "d.json")                                   # team 1 owns Player000..Player003
    doc = json.loads(d.read_text(encoding="utf-8"))
    for pick in doc["draft_log"]:                                          # the draft knows them the way the app saved them: lowercase, accents kept
        pick["player_id"] = {"n:player000": "n:nikola joki\u0107", "n:player001": "n:de'aaron fox"}.get(pick["player_id"], pick["player_id"])
    d.write_text(json.dumps(doc), encoding="utf-8")
    teams, _, picks, _ = mod.load_draft(d)
    assert picks[1][:2] == ["nikola joki\u0107", "de'aaron fox"] or set(picks[1]) >= {"nikola joki\u0107", "de'aaron fox"}
    sc, ids, _ = build_ruler(om.load_season(f), LeagueConfig.for_sport("nba", "espn_default"))
    res = {r["team"]: r for r in mod.rank_teams(picks, sc, ids, 4)}
    assert res[1]["missing"] == [] and res[1]["counted"] == 4


# ---------------------------------------------------------------------------------------- input errors
@pytest.mark.parametrize("content,needle", [
    (None, "was not found"),
    ("{not json", "not a valid draft file"),
    ('{"hello": 1}', "no draft_log"),
    ('{"sport": "mlb", "draft_log": [{"team_id": 1}]}', "basketball only"),
    ('{"draft_log": []}', "no picks"),
    ('{"setup": {"teams": 12}, "draft_log": [{"pick_number": 1, "player_id": "n:x", "team_id": 99}]}', "team_id must be a whole number from 1 to 12"),
    ('{"setup": {"teams": 12}, "draft_log": [{"pick_number": 1, "player_id": "n:x", "team_id": "one"}]}', "team_id must be a whole number"),
])
def test_a_bad_draft_file_gives_a_message_not_a_traceback(tmp_path, season, capsys, content, needle):
    mod = _load(); p = tmp_path / "d.json"
    if content is not None:
        p.write_text(content, encoding="utf-8")
    assert run(mod, p, season) == 1 and needle in capsys.readouterr().out


def test_a_team_with_no_picks_cannot_be_compared(tmp_path, season, capsys):
    mod = _load(); p = tmp_path / "d.json"
    p.write_text(json.dumps({"setup": {"teams": 12, "my_slot": 1}, "draft_log": [{"pick_number": 1, "player_id": "n:player000", "team_id": 1}]}), encoding="utf-8")
    assert run(mod, p, season) == 1 and "have no picks" in capsys.readouterr().out


def test_a_missing_or_wrong_season_file_is_explained(tmp_path, season, capsys):
    mod = _load(); d = write_draft(tmp_path / "d.json")
    assert run(mod, d, tmp_path / "nope.csv") == 1 and "was not found" in capsys.readouterr().out
    junk = tmp_path / "junk.csv"; junk.write_text("a,b\n1,2\n", encoding="utf-8")
    assert run(mod, d, junk) == 1 and "does not look like a Basketball-Reference per-game table" in capsys.readouterr().out


def test_without_a_season_argument_it_looks_in_the_inseason_folder_and_says_so_when_empty(tmp_path, monkeypatch, request, capsys):
    old_i, old_p = ingestion.DATA_DIR, persist.DATA_DIR
    ingestion.DATA_DIR = tmp_path / "data"; persist.DATA_DIR = tmp_path / "data"
    request.addfinalizer(lambda: (setattr(ingestion, "DATA_DIR", old_i), setattr(persist, "DATA_DIR", old_p)))
    mod = _load(); write_draft(tmp_path / "data" / "draft_state_nba.json") if (tmp_path / "data").mkdir() is None else None
    assert run(mod) == 1 and "data/nba/inseason" in capsys.readouterr().out
    fake_season((tmp_path / "data" / "nba" / "inseason").mkdir(parents=True) or tmp_path / "data" / "nba" / "inseason" / "per_game_2027.csv")
    assert run(mod) == 0 and "per_game_2027.csv" in capsys.readouterr().out


def test_limiting_to_the_first_n_picks_really_changes_the_numbers(tmp_path, season):
    mod = _load()
    extra = [{"pick_number": 49, "player_id": "n:player048", "team_id": 1, "source": "manual"}]          # a good fifth player for team 1 only
    teams, _, picks, _ = mod.load_draft(write_draft(tmp_path / "d.json", extra=extra))
    cfg = LeagueConfig.for_sport("nba", "espn_default"); cfg.roster.teams = teams
    sc, ids, _ = build_ruler(om.load_season(season), cfg)
    four = next(r for r in mod.rank_teams(picks, sc, ids, 4) if r["team"] == 1)
    everyone = next(r for r in mod.rank_teams(picks, sc, ids, 0) if r["team"] == 1)
    manual = sc.score([sc.pos[ids[om.norm_name(n)]] for n in picks[1][:4]])
    assert four["counted"] == 4 and everyone["counted"] == 5
    assert four["score"] == pytest.approx(manual) and everyone["score"] > four["score"]

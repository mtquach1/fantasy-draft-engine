"""The app must never strand a starting slot, and a pick that cannot help you finish a legal roster is a wasted pick."""
import tempfile
from pathlib import Path

import pytest

import make_sample_sport
from app import ingestion, validate
from app.config import LeagueConfig, ScoringType
from app.draft import slot_for_pick
from app.optimizer import open_starters
from app.recommend import recommend
from app.valuation import calculate_player_valuations

GUARDS = {"PG", "SG"}


@pytest.fixture(scope="module")
def nba_raw():
    old = ingestion.DATA_DIR
    with tempfile.TemporaryDirectory() as d:
        ingestion.DATA_DIR = Path(d)
        try:
            make_sample_sport.generate("nba", Path(d) / "nba", n=300)
            raw, _ = validate.clean_projections(ingestion.load_projections("nba"), "nba")
        finally:
            ingestion.DATA_DIR = old
    return raw


def _cfg(teams=8, fmt=ScoringType.ROTO, **slots):
    cfg = LeagueConfig.for_sport("nba")
    cfg.scoring_type, cfg.roster.teams = fmt, teams
    if slots:
        cfg.roster.slots = {k: v for k, v in {**cfg.roster.slots, **slots}.items() if v}
    return cfg


def _val(raw, cfg, drafted=()):
    return calculate_player_valuations(raw, cfg, frozenset(drafted))


def _pure_guards(df, n):
    g = df[df["eligible_positions"].map(lambda p: set(p) <= GUARDS)].nlargest(n, "vorp")
    return g["player_id"].tolist()


def test_open_starters_reports_the_positions_a_guard_roster_still_needs(nba_raw):
    cfg = _cfg()
    df, _ = _val(nba_raw, cfg)
    team = df[df["player_id"].isin(_pure_guards(df, 6))]
    assert sorted(open_starters(team, cfg.roster, "nba")) == ["C", "C", "F", "PF", "SF"]
    assert open_starters(team.iloc[0:0], cfg.roster, "nba") == [s for s in cfg.roster.slot_list() if s != "BN"]


def test_guard_is_a_wasted_pick_once_only_other_positions_can_finish_the_roster(nba_raw):
    cfg = _cfg()                                                 # PG SG G SF PF F C C UTIL UTIL + 3 bench = 13 rounds
    df0, _ = _val(nba_raw, cfg)
    mine = _pure_guards(df0, 8)                                  # fills PG SG G UTIL UTIL and all 3 bench spots
    df, repl = _val(nba_raw, cfg, mine)
    rows = recommend(df, repl, cfg, mine, limit=200, rounds_total=13)
    assert rows[0]["fills"] and not rows[0]["wasted"]
    guards = [r for r in rows if set(r["positions"]) <= GUARDS]
    assert guards and all(r["wasted"] and r["team_gain"] < 0 for r in guards)
    assert all(rows[i]["team_gain"] >= rows[i + 1]["team_gain"] for i in range(len(rows) - 1) if not rows[i]["wasted"] and not rows[i + 1]["wasted"])
    assert all(not r["wasted"] for r in rows[:10])               # every useful pick outranks every wasted one
    assert "Wasted pick" in guards[0]["why"]


def test_early_in_the_draft_the_best_player_still_wins(nba_raw):
    cfg = _cfg()
    df, repl = _val(nba_raw, cfg)
    top = recommend(df, repl, cfg, [], limit=3, rounds_total=13)
    assert top[0]["vorp"] == pytest.approx(df["vorp"].max(), rel=1e-6) or top[0]["vorp"] >= df["vorp"].nlargest(3).iloc[-1]
    assert not any(r["wasted"] for r in top)


@pytest.mark.parametrize("label,slots,fmt", [
    ("default", {}, ScoringType.ROTO),
    ("no UTIL", {"UTIL": 0}, ScoringType.ROTO),
    ("no bench", {"BN": 0}, ScoringType.ROTO),
    ("four UTIL", {"UTIL": 4}, ScoringType.POINTS),
])
def test_following_the_recommendations_always_fills_every_starting_slot(nba_raw, label, slots, fmt):
    teams, my_slot = 8, 3
    cfg = _cfg(teams, fmt, **slots)
    rounds = len(cfg.roster.slot_list())
    mine, drafted = [], set()
    for pick in range(1, teams * rounds + 1):
        if slot_for_pick("snake", teams, pick) == my_slot:
            df, repl = _val(nba_raw, cfg, drafted)
            rec = recommend(df, repl, cfg, mine, limit=1, rounds_total=rounds)[0]
            mine.append(rec["player_id"]); drafted.add(rec["player_id"])
        else:                                                    # everyone else takes the best player by value (a strong, need-blind drafter)
            df, _ = _val(nba_raw, cfg, drafted)
            drafted.add(df[df["available"]].nlargest(1, "vorp").iloc[0]["player_id"])
    df, _ = _val(nba_raw, cfg, drafted)
    team = df[df["player_id"].isin(mine)]
    assert len(mine) == rounds
    assert open_starters(team, cfg.roster, "nba") == [], f"{label}: starting slots left empty"


def test_roster_validation_rules():
    cfg = _cfg()
    ok = {"PG": 1, "SG": 1, "G": 1, "SF": 1, "PF": 1, "F": 1, "C": 2, "UTIL": 2, "BN": 3}
    cfg.roster.slots = ok
    assert validate.check_roster(cfg) == ([], [])
    cfg.roster.slots = {**ok, "XX": 1}
    assert "not a" in validate.check_roster(cfg)[0][0]
    cfg.roster.slots = {**ok, "PG": -1}
    assert validate.check_roster(cfg)[0]
    cfg.roster.slots = {"BN": 4}
    assert "at least one starting slot" in validate.check_roster(cfg)[0][0]
    cfg.roster.slots = {k: v for k, v in ok.items() if k not in ("C", "UTIL")}
    errs, warns = validate.check_roster(cfg)
    assert not errs and any("C players" in w for w in warns)
    cfg.roster.slots = {**ok, "BN": 0}
    assert any("no bench" in w for w in validate.check_roster(cfg)[1])

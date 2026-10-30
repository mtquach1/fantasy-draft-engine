"""Tests for the look-ahead (app/lookahead.py), its use in recommend(), and the IL-slot fix."""
import numpy as np
import pandas as pd

from app import lookahead, recommend as rec, valuation
from app.config import LeagueConfig
from app.draft import DraftSetup
from app.state import GlobalState
from tests.test_math import fake_projections

RAW = fake_projections()


def board(drafted=()):
    cfg = LeagueConfig()
    df, repl = valuation.calculate_player_valuations(RAW, cfg, set(), 2)
    pre = df.copy()
    df, repl = valuation.calculate_player_valuations(RAW, cfg, set(drafted), 2)
    return valuation.add_ranks(df, pre), repl, cfg


# ---------- turn math ----------
def test_pick_gaps_snake_waiting_and_on_the_clock():
    s = DraftSetup("snake", 12, 5, 25, None)
    g = lookahead.pick_gaps(s, 0)                       # pick 1 is up, my first pick is #5, then #20
    assert (g["my_pick"], g["next_pick"], g["k1"], g["k2"], g["on_the_clock"]) == (5, 20, 4, 14, False)
    g = lookahead.pick_gaps(s, 4)                       # picks 1-4 made: I'm on the clock
    assert (g["my_pick"], g["k1"], g["on_the_clock"]) == (5, 0, True)
    g = lookahead.pick_gaps(s, 5)                       # I just picked: next is #20, 14 picks away
    assert (g["my_pick"], g["k1"], g["k2"]) == (20, 14, 8)       # picks 21-28 belong to the others


def test_pick_gaps_snake_turn_has_back_to_back_picks():
    s = DraftSetup("snake", 12, 12, 25, None)           # slot 12 picks #12 and #13 in a row
    g = lookahead.pick_gaps(s, 11)
    assert (g["my_pick"], g["next_pick"], g["k2"]) == (12, 13, 0)


def test_pick_gaps_last_pick_and_not_applicable():
    s = DraftSetup("snake", 12, 5, 2, None)             # a 2-round draft: my picks are #5 and #20
    g = lookahead.pick_gaps(s, 6)
    assert g["my_pick"] == 20 and g["next_pick"] is None and g["k2"] == 0
    assert lookahead.pick_gaps(s, 21) is None           # none left
    assert lookahead.pick_gaps(None, 0) is None
    assert lookahead.pick_gaps(DraftSetup("auction", 12, None, 25, 260), 0) is None


# ---------- simulation ----------
def test_better_ranked_players_are_less_likely_to_last():
    pos = lookahead.simulate_order(np.arange(1, 201, dtype=float))
    survive = (pos >= 40).mean(0)                         # 40 picks by the other teams
    assert survive[0] < 0.01 and survive[-1] > 0.99
    assert survive[:10].mean() < survive[35:45].mean() < survive[150:160].mean()


def test_no_picks_in_between_means_everyone_is_there():
    df, repl, cfg = board()
    g = {"k1": 0, "k2": 0, "my_pick": 1, "next_pick": None, "current_pick": 1, "on_the_clock": True}
    assert set(lookahead.survival_by_id(df, g).values()) == {1.0}


def test_simulation_is_steady_between_calls():
    r = np.arange(1, 101, dtype=float)
    assert (lookahead.simulate_order(r) == lookahead.simulate_order(r)).all()


# ---------- recommendations ----------
def test_recommend_with_lookahead_adds_timing_fields_and_a_headline():
    df, repl, cfg = board()
    gaps = lookahead.pick_gaps(DraftSetup("snake", 12, 5, 25, None), 4)
    rows, plan = rec.recommend(df, repl, cfg, [], gaps=gaps, with_plan=True)
    assert plan["status"] == "ok" and plan["take"] == rows[0]["name"] and plan["headline"].startswith("Take ")
    top = rows[0]
    assert top["p_at_my_pick"] == 1.0 and 0.0 <= top["p_next_turn"] <= 1.0 and top["note"]
    assert len(rows) == 10


def test_plain_recommend_is_unchanged_without_gaps():
    df, repl, cfg = board()
    plain = rec.recommend(df, repl, cfg, [])
    assert "p_at_my_pick" not in plain[0]
    rows, plan = rec.recommend(df, repl, cfg, [], with_plan=True)
    assert [r["player_id"] for r in rows] == [r["player_id"] for r in plain] and plan["status"] == "off"


def test_far_from_my_turn_only_lists_players_who_can_last():
    df, repl, cfg = board()
    gaps = {"k1": 40, "k2": 20, "my_pick": 41, "next_pick": 62, "current_pick": 1, "on_the_clock": False}
    rows, plan = rec.recommend(df, repl, cfg, [], gaps=gaps, with_plan=True)
    assert plan["status"] == "ok"
    assert all(r["p_at_my_pick"] >= 0.10 for r in rows[:5])      # the near-certain-to-be-gone stars are not suggested


def test_scarce_player_is_taken_now_when_the_alternative_will_wait():
    """Two equal-looking options: the one who will last should not outrank the one who won't."""
    df, repl, cfg = board()
    gaps = lookahead.pick_gaps(DraftSetup("snake", 12, 12, 25, None), 11)   # on the clock, back-to-back picks
    rows, plan = rec.recommend(df, repl, cfg, [], gaps=gaps, with_plan=True)
    assert plan["on_the_clock"] and rows[0]["lookahead_value"] >= rows[1]["lookahead_value"] - 1e-9


def test_last_pick_of_the_draft_still_works():
    df, repl, cfg = board()
    gaps = lookahead.pick_gaps(DraftSetup("snake", 12, 5, 2, None), 6)
    rows, plan = rec.recommend(df, repl, cfg, [], gaps=gaps, with_plan=True)
    assert plan["status"] == "ok" and plan["next_pick"] is None and rows[0]["p_next_turn"] is None


def test_lookahead_error_never_breaks_recommendations(monkeypatch):
    df, repl, cfg = board()
    monkeypatch.setattr(lookahead, "rerank", lambda *a, **k: 1 / 0)
    gaps = lookahead.pick_gaps(DraftSetup("snake", 12, 5, 25, None), 4)
    rows, plan = rec.recommend(df, repl, cfg, [], gaps=gaps, with_plan=True)
    assert rows and plan["status"] == "error"


# ---------- IL slots ----------
def test_open_il_slots_shrink_with_injured_players_on_my_roster():
    raw = RAW.copy()
    raw["is_injured"] = False
    ids = list(raw["player_id"][:3])
    raw.loc[raw["player_id"].isin(ids[:2]), "is_injured"] = True
    st = GlobalState(config=LeagueConfig(), raw_df=raw)
    st.config.roster.il_slots = 2
    assert st.my_open_il_slots() == 2                                   # nobody drafted yet
    st.draft_log[:] = [{"pick_number": 1, "player_id": ids[0], "team_id": 1, "source": "manual"}]
    assert st.my_open_il_slots() == 1
    st.draft_log.append({"pick_number": 2, "player_id": ids[1], "team_id": 1, "source": "manual"})
    st.draft_log.append({"pick_number": 3, "player_id": ids[2], "team_id": 1, "source": "manual"})  # healthy: no change
    assert st.my_open_il_slots() == 0
    st.draft_log.append({"pick_number": 4, "player_id": ids[0], "team_id": 2, "source": "manual"})  # another team's: ignored
    assert st.my_open_il_slots() == 0
    st.config.roster.il_slots = 0
    assert st.my_open_il_slots() == 0


# ---------- auction nomination advice ----------
from app import auction


def _rows():
    return [{"player_id": f"x{i}", "name": f"P{i}", "bid": b, "team_gain": g}
            for i, (b, g) in enumerate([(50, 9.0), (40, 7.0), (35, 1.0), (30, 0.5), (25, 6.0), (20, 0.2), (1, 0.0), (12, 3.0)])]


def test_nominate_early_drains_with_pricey_low_fit_players_and_names_targets():
    out = auction.nominate_ideas(_rows(), {"max_bid": 100}, {"next_team": 3}, 3, filled=0.1)
    assert out["my_turn"] and out["stage"] == "early"
    assert [r["name"] for r in out["drain"]][:2] == ["P2", "P3"]            # expensive, little use to me
    assert [r["name"] for r in out["targets"]][:2] == ["P0", "P1"]          # biggest gain
    assert out["headline"].startswith("It's your turn to nominate.") and "drain" in out["headline"]


def test_nominate_late_leads_with_targets_and_respects_my_max_bid():
    out = auction.nominate_ideas(_rows(), {"max_bid": 30}, {"next_team": 5}, 3, filled=0.8)
    assert not out["my_turn"] and out["stage"] == "late"
    assert all(r["bid"] <= 30 for r in out["targets"]) and out["targets"][0]["name"] == "P4"
    assert out["headline"].startswith("Team 5 nominates next.") and "you want" in out["headline"]


def test_nominate_never_suggests_a_dollar_player_to_drain_and_handles_nothing_left():
    out = auction.nominate_ideas(_rows(), {"max_bid": 100}, {"next_team": 1}, 1, filled=0.1)
    assert all(r["bid"] > 1 for r in out["drain"])
    empty = auction.nominate_ideas([], None, {"next_team": None}, 1, filled=1.0)
    assert empty["targets"] == [] and empty["drain"] == [] and empty["headline"] == ""


# ---------- ADP (optional adp.csv) ----------
import tempfile
from pathlib import Path

from app import ingestion, views


def _adp_dir(text):
    d = Path(tempfile.mkdtemp())
    if text is not None:
        (d / "adp.csv").write_text(text, encoding="utf-8")
    return d


def _load(text, monkeypatch):
    monkeypatch.setattr(ingestion, "DATA_DIR", _adp_dir(text))
    raw = RAW.copy()
    raw["clean_name"] = raw["name"].map(ingestion.clean_name)
    return ingestion.load_adp("mlb", raw)


def test_adp_missing_file_is_silent(monkeypatch):
    assert _load(None, monkeypatch) == ({}, None)


def test_adp_by_id_and_aliases_and_bad_rows(monkeypatch):
    out, note = _load("Player_ID,Average Draft Position\nh0,1.5\nh1,2\nh2,abc\nh3,-4\nzzz,9\n", monkeypatch)
    assert out == {"h0": 1.5, "h1": 2.0}
    assert "2 players" in note and "1 row(s) didn't match" in note and "2 row(s) had no usable" in note


def test_adp_by_name_with_accents_and_suffix(monkeypatch):
    out, _ = _load("Name,ADP\nHitter 7 Jr.,12\nPITCHER 3,40\nNobody,5\n", monkeypatch)
    assert out == {"h7": 12.0, "p3": 40.0}


def test_adp_unusable_files_never_raise(monkeypatch):
    for text in ("foo,bar\n1,2\n", "adp\n5\n6\n", "name,adp\nNobody,3\n", "\x00\x01 not a csv"):
        out, note = _load(text, monkeypatch)
        assert out == {} and note


def test_rank_key_uses_adp_only_when_it_covers_enough_players():
    df, _, _ = board()
    pool = df[df["available"]].copy()
    base = lookahead.rank_key(pool)
    pool["adp"] = np.nan
    pool.loc[pool.index[:5], "adp"] = 3.0                              # too few: ignored
    assert (lookahead.rank_key(pool) == base).all()
    pool["adp"] = np.arange(1, len(pool) + 1, dtype=float)[::-1]       # full coverage, reversed
    key = lookahead.rank_key(pool)
    assert key[0] == len(pool) and key[-1] == 1.0


def test_lookahead_follows_adp_not_our_ranking():
    """If the market loves a player our ranking ignores, he's the one who won't last."""
    df, repl, cfg = board()
    gaps = {"k1": 20, "k2": 0, "my_pick": 21, "next_pick": None, "current_pick": 1, "on_the_clock": False}
    plain = lookahead.survival_by_id(df, gaps)
    low = df[df["available"]].nsmallest(1, "vorp")["player_id"].iloc[0]       # our worst player
    df2 = df.copy()
    df2["adp"] = df2["projected_rank"].astype(float)
    df2.loc[df2["player_id"] == low, "adp"] = 1.0                              # but the market takes him first
    with_adp = lookahead.survival_by_id(df2, gaps)
    assert plain[low] > 0.99 and with_adp[low] < 0.05


def test_adp_report_correlation_and_disagreements():
    df, _, _ = board()
    pre = df.sort_values(["vorp", "name"], ascending=[False, True]).reset_index(drop=True)
    adp = {pid: float(i + 1) for i, pid in enumerate(pre["player_id"])}      # market agrees exactly
    rep = views.adp_report(df, adp, 3)
    assert rep["rank_correlation"] > 0.999 and rep["verdict"] == "very close to the market"
    first, last = pre["player_id"].iloc[0], pre["player_id"].iloc[-1]
    adp[first], adp[last] = float(len(pre)), 1.0                               # market flips our best and worst
    rep = views.adp_report(df, adp, 3)
    assert rep["we_like_more_than_market"][0]["player_id"] == first
    assert rep["market_likes_more_than_we_do"][0]["player_id"] == last
    assert views.adp_report(df, {first: 1.0})["message"].startswith("Too few")


# ---------- site-style NBA files ----------
BR_FILE = """Rk,Player,Tm,Pos,G,MP,FG,FGA,3P,FT,FTA,TRB,AST,STL,BLK,TOV,PTS
1,Nikola Jokić,DEN,C,70,36.5,10.5,18.0,1.5,5.5,6.5,12.5,10.5,1.7,0.8,3.2,28.0
2,Trade Guy,2TM,SG,60,30.0,6.0,13.0,2.0,3.0,3.5,4.0,5.0,1.0,0.5,2.0,17.0
3,Trade Guy,AAA,SG,40,31.0,6.1,13.0,2.0,3.0,3.5,4.0,5.0,1.0,0.5,2.0,17.2
4,Trade Guy,BBB,SG,20,28.0,5.8,13.0,2.0,3.0,3.5,4.0,5.0,1.0,0.5,2.0,16.5
5,Jalen Williams,OKC,SG-SF,60,31.0,6.5,13.5,1.8,3.0,3.8,5.0,4.9,1.5,0.7,2.0,18.0
6,Jalen Williams,CHI,SG,50,22.0,3.0,7.0,1.0,1.0,1.4,3.0,2.0,0.8,0.3,1.0,8.0
"""


def _nba_frame(text, monkeypatch):
    base = Path(tempfile.mkdtemp())
    (base / "nba").mkdir()
    (base / "nba" / "players.csv").write_text(text, encoding="utf-8")
    monkeypatch.setattr(ingestion, "DATA_DIR", base)
    return ingestion.load_projections("nba")


def test_nba_per_game_site_file_loads_as_totals_with_generated_ids(monkeypatch):
    df = _nba_frame(BR_FILE, monkeypatch)
    assert len(df) == 4                                                    # traded player appears once, not three times
    jok = df[df["name"] == "Nikola Jokić"].iloc[0]
    assert jok["PTS"] == 28.0 * 70 and jok["FGM"] == 10.5 * 70 and jok["TO"] == 3.2 * 70 and jok["3PM"] == 1.5 * 70
    assert jok["player_id"] == "n:nikola jokic" and list(jok["eligible_positions"]) == ["C"]
    guy = df[df["name"] == "Trade Guy"].iloc[0]
    assert guy["team"] == "2TM" and guy["GP"] == 60                        # the season-total row was kept
    jw = df[df["name"] == "Jalen Williams"]
    assert sorted(jw["player_id"]) == ["n:jalen williams", "n:jalen williams_2"]    # two real players, distinct ids
    assert list(jw.iloc[0]["eligible_positions"]) == ["SG", "SF"]          # 'SG-SF' split on the hyphen


def test_nba_totals_file_is_left_alone(monkeypatch):
    totals = BR_FILE.replace("28.0", "1960").replace("17.0", "1020")       # a season-TOTAL file: PTS in the thousands
    df = _nba_frame(totals, monkeypatch)
    assert df[df["name"] == "Nikola Jokić"].iloc[0]["PTS"] == 1960


# ---------- the follow-up must be honest: real odds, and a player who can actually last ----------
def _gaps_pick1_slot5_of_12():
    return lookahead.pick_gaps(DraftSetup("snake", 12, 5, 25, None), 4)          # on the clock at #5, next pick #20


def test_follow_up_comes_with_its_odds_and_the_options_never_exceed_all_simulations():
    df, repl, cfg = board()
    rows, plan = rec.recommend(df, repl, cfg, [], gaps=_gaps_pick1_slot5_of_12(), with_plan=True)
    for r in rows[:10]:
        opts = r.get("next_options") or []
        assert sum(o["p"] for o in opts) <= 1.0 + 1e-9
        assert all(0 < o["p"] <= 1 for o in opts)
        if r.get("likely_next"):
            assert r["likely_next_p"] >= lookahead.MIN_FOLLOW_SHARE and r["likely_next"] == opts[0]["name"]
            assert r["likely_next_p"] == opts[0]["p"]
    top = rows[0]
    if top.get("likely_next"):
        assert "%" in plan["headline"] and top["likely_next"] in plan["headline"]


def test_a_named_follow_up_can_really_last_until_the_next_pick():
    """Earlier the follow-up was the least unlikely of top players who would be gone. Now it must have a real chance of lasting."""
    df, repl, cfg = board()
    gaps = _gaps_pick1_slot5_of_12()
    rows, plan = rec.recommend(df, repl, cfg, [], gaps=gaps, with_plan=True)
    pool = df[df["available"]].reset_index(drop=True)
    surv = (lookahead.simulate_order(lookahead.rank_key(pool)) >= gaps["k1"] + gaps["k2"]).mean(0)
    by_name = dict(zip(pool["name"], surv))
    named = [r["likely_next"] for r in rows[:10] if r.get("likely_next")]
    assert named, "expected at least one named follow-up on a deep board"
    for n in named:
        assert by_name[n] >= lookahead.MIN_SURVIVE - 1e-9


def test_no_clear_follow_up_is_said_plainly_instead_of_naming_a_long_shot(monkeypatch):
    monkeypatch.setattr(lookahead, "MIN_FOLLOW_SHARE", 1.01)                     # nobody can reach this share
    df, repl, cfg = board()
    rows, plan = rec.recommend(df, repl, cfg, [], gaps=_gaps_pick1_slot5_of_12(), with_plan=True)
    assert all(r["likely_next"] is None for r in rows[:10])
    assert "No clear follow-up" in plan["headline"] and "depends on who falls" in plan["headline"]


# ---------- the speed shortcut must not change a single number ----------
def test_fast_lineup_value_matches_the_full_path_in_every_format():
    import pytest
    from app.config import ScoringType
    from app.recommend import _ghosts, _lineup_value
    for st in (ScoringType.ROTO, ScoringType.POINTS, ScoringType.H2H_CATEGORIES):
        cfg = LeagueConfig(scoring_type=st)
        df, repl = valuation.calculate_player_valuations(RAW, cfg, set(), 2)
        ghosts = _ghosts(df[df["available"] & ~df["avoid"]], cfg, repl)
        score_of = dict(zip(df["player_id"], df["score"].astype(float)))
        for k in (0, 1, 4, 9):                                   # an empty roster, a few players, a full set
            team = df.sort_values("vorp", ascending=False).head(k)
            assert _lineup_value(df, team, cfg, repl, ghosts, score_of) == pytest.approx(_lineup_value(df, team, cfg, repl, ghosts), abs=1e-9)

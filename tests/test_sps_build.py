import pandas as pd
import pytest

import build_sps_projections as b
from app import nba_build

SPS = """,,,Per 36 Minutes,Per 36 Minutes,Per 36 Minutes,Per 36 Minutes,Per 36 Minutes,Per 36 Minutes,Per 36 Minutes,Per 36 Minutes,Per 36 Minutes,Per 36 Minutes,Per 36 Minutes,Per 36 Minutes,Per 36 Minutes,Per 36 Minutes,Shooting,Shooting,Shooting,-additional
Rk,Player,Type,FG,FGA,3P,3PA,FT,FTA,ORB,TRB,AST,STL,BLK,TOV,PF,PTS,FG%,3P%,FT%,-9999
1,Nikola Jokić,Projected,10.3,18.2,1.7,4.5,5.6,6.8,2.9,12.6,10.2,1.5,0.8,3.5,2.6,28.0,.567,.386,.819,jokicni01
2,Stephen Curry,Projected,9.3,20.6,4.8,12.5,4.7,5.1,0.6,4.6,5.8,1.2,0.5,3.2,1.9,28.1,.449,.386,.919,curryst01
3,Rookie Guy,Projected,5.0,11.0,2.0,5.0,2.0,2.5,1.0,5.0,3.0,1.0,0.5,2.0,3.0,14.0,.450,.400,.800,rookigu01
4,No Position,Projected,5.0,11.0,2.0,5.0,2.0,2.5,1.0,5.0,3.0,1.0,0.5,2.0,3.0,14.0,.450,.400,.800,nopos01
"""
PG = """Rk,Player,Pos,Age,Team,G,GS,MP,FG,FGA,FG%,3P,3PA,3P%,2P,2PA,2P%,eFG%,FT,FTA,FT%,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS
1,Nikola Jokić,C,30,DEN,70,70,36.0,10.0,18.0,.555,1.6,4.4,.360,8.4,13.6,.617,.600,5.5,6.6,.830,3.0,9.0,12.0,10.0,1.4,0.8,3.3,2.5,27.0
2,Stephen Curry,PG,38,GSW,40,40,32.0,8.0,18.0,.444,4.0,10.0,.400,4.0,8.0,.500,.555,4.0,4.4,.909,0.5,4.0,4.5,5.5,1.1,0.4,3.0,1.8,24.0
3,No Position,,25,XXX,60,10,20.0,3.0,7.0,.430,1.0,3.0,.333,2.0,4.0,.500,.500,1.0,1.2,.830,1.0,3.0,4.0,2.0,0.5,0.3,1.0,2.0,8.0
"""


@pytest.fixture
def files(tmp_path):
    (tmp_path / "sps.csv").write_text(SPS, encoding="utf-8")
    (tmp_path / "per_game_2026.csv").write_text(PG, encoding="utf-8")
    return tmp_path


def test_sps_conversion(files):
    rep = b.build(files / "sps.csv", [files / "per_game_2026.csv"], files / "out", 2026)
    out = pd.read_csv(files / "out" / "players.csv")
    assert set(out["Name"]) == {"Nikola Jokić", "Stephen Curry"}          # rookie (no last season) and no-position left out
    assert "Rookie Guy" in rep["no_last_season_data"] and "No Position" in rep["needs_position"]
    j = out[out["Name"] == "Nikola Jokić"].iloc[0]
    games = 0.3 * 70 + 0.7 * 68                                           # one season: 30% history, 70% baseline for a 36-mpg player
    assert j["GP"] == round(games)
    assert j["PTS"] == pytest.approx(28.0 * 36.0 / 36.0 * games, abs=0.1)  # 36 mpg -> per-36 equals per-game
    c = out[out["Name"] == "Stephen Curry"].iloc[0]
    g_c = 0.3 * 40 + 0.7 * 68
    assert c["MIN"] == pytest.approx(32.0 * g_c, abs=1)
    assert c["3PM"] == pytest.approx(4.8 * 32 / 36 * g_c, abs=0.1)
    assert (files / "out" / "history.csv").exists()


def test_positions_and_injuries_files(files):
    (files / "pos.csv").write_text("name,pos\nNo Position,SF\n")
    (files / "inj.csv").write_text("name,games_out\nStephen Curry,20\n")
    b.build(files / "sps.csv", [files / "per_game_2026.csv"], files / "out", 2026, files / "pos.csv", files / "inj.csv")
    out = pd.read_csv(files / "out" / "players.csv").set_index("Name")
    assert "No Position" in out.index and bool(out.loc["Stephen Curry", "is_injured"])


def test_needs_per_game_file(files):
    with pytest.raises(nba_build.BuildError):
        b.build(files / "sps.csv", [], files / "out", 2026)


HDR = "Rk,Player,Age,Team,Pos,G,GS,MP,FG,FGA,FG%,3P,3PA,3P%,2P,2PA,2P%,eFG%,FT,FTA,FT%,ORB,DRB,TRB,AST,STL,BLK,TOV,PF,PTS,Awards,Player-additional\n"


def _pg(path, rows):
    lines = [f"{i + 1},{n},27,BOS,PF,{g},{g},{mp},1,2,.5,1,2,.5,1,2,.5,.5,1,2,.8,1,1,2,1,1,1,1,1,10,,x" for i, (n, g, mp) in enumerate(rows)]
    path.write_text(HDR + "\n".join(lines) + "\n", encoding="utf-8")


def test_star_with_one_injured_season_is_not_crushed(tmp_path):
    """16 games after 72 and 74: a baseline-weighted estimate, not 'he will miss two thirds of the season'."""
    sps = tmp_path / "sps.csv"
    sps.write_text(SPS.split("3,Rookie")[0], encoding="utf-8")
    _pg(tmp_path / "p26.csv", [("Nikola Jokić", 16, 32.6), ("Stephen Curry", 60, 33.0)])
    _pg(tmp_path / "p25.csv", [("Nikola Jokić", 72, 36.0), ("Stephen Curry", 60, 33.0)])
    _pg(tmp_path / "p24.csv", [("Nikola Jokić", 74, 35.7), ("Stephen Curry", 60, 33.0)])
    paths = [tmp_path / "p26.csv", tmp_path / "p25.csv", tmp_path / "p24.csv"]
    b.build(sps, paths, tmp_path / "out", 2026)
    out = pd.read_csv(tmp_path / "out" / "players.csv").set_index("Name")
    star = out.loc["Nikola Jokić"]
    assert 57 <= star["GP"] <= 63                                       # ~61, not 16 and not the 42 of the old model
    assert star["MIN"] / star["GP"] == pytest.approx(35.1, abs=0.6)    # minutes lean on the healthy seasons
    assert out.loc["Stephen Curry", "GP"] == pytest.approx(0.3 * 60 + 0.7 * 68, abs=1)


def test_games_expected_override_and_history_gaps(tmp_path):
    sps = tmp_path / "sps.csv"
    sps.write_text(SPS.split("3,Rookie")[0], encoding="utf-8")
    _pg(tmp_path / "p26.csv", [("Nikola Jokić", 16, 32.6), ("Stephen Curry", 60, 33.0)])
    (tmp_path / "inj.csv").write_text("name,games_expected\nNikola Jokić,66\n", encoding="utf-8")
    b.build(sps, [tmp_path / "p26.csv"], tmp_path / "out", 2026, injuries_file=tmp_path / "inj.csv")
    out = pd.read_csv(tmp_path / "out" / "players.csv").set_index("Name")
    assert out.loc["Nikola Jokić", "GP"] == 66
    # a season missing between two he played counts as 0 games; a season before his first is ignored
    assert b.history_estimate([(60, 30.0), None, (70, 30.0)])[0] == pytest.approx((3 * 60 + 0 + 70) / 6)
    assert b.history_estimate([(60, 30.0), None, None])[0] == pytest.approx(60)

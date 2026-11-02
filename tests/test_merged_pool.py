"""merged_pool: the model's top n plus last season's top n by production, in the model's order."""
import pandas as pd

from app.own_model import merged_pool


def frames():
    ids = [f"p{i}" for i in range(10)]
    proj = pd.DataFrame({"Name": ids}, index=ids)                       # the model's order is p0 (best) .. p9
    last = pd.DataFrame({"fpg": [10, 9, 8, 7, 6, 5, 4, 3, 50, 40.0], "G": [70] * 10}, index=ids)   # last season: p8 and p9 were huge
    return proj, last


def test_players_the_model_dislikes_but_last_season_loved_are_added_and_the_model_order_is_kept():
    proj, last = frames()
    got = list(merged_pool(proj, last, 3).index)
    assert got == ["p0", "p1", "p2", "p8", "p9"]                       # model top 3 + last season's top 3 (p8, p9, p0), model order


def test_overlap_is_not_duplicated_and_n_larger_than_the_table_returns_everyone():
    proj, last = frames()
    assert len(set(merged_pool(proj, last, 3).index)) == 5
    assert list(merged_pool(proj, last, 99).index) == list(proj.index)


def test_a_short_season_is_floored_so_one_injury_year_is_not_a_verdict():
    proj, last = frames()
    last.loc["p5", "G"] = 5; last.loc["p5", "fpg"] = 80.0              # 80 a game over only 5 games: floored to 55 games = 4400, above p8's 50 x 70 = 3500 (unfloored it would be 400)
    assert "p5" in merged_pool(proj, last, 1).index                    # n=1: the model's p0 plus last season's single best (p5)
    last.loc["p5", "G"] = 0
    assert "p5" in merged_pool(proj, last, 1).index                    # floored at min_games even with no games


def test_players_missing_from_last_season_are_simply_not_added():
    proj, last = frames()
    got = merged_pool(proj, last.drop(index=["p8"]), 2)
    assert "p8" not in got.index and list(got.index)[:2] == ["p0", "p1"]

"""The own projection model: file reading, the debut rule, scoring consistency and hand-off to the app's loader."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import build_own_model
from app import ingestion, own_model as om, validate

COLS = ["Rk", "Player", "Age", "Team", "Pos", "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%", "2P", "2PA", "2P%", "eFG%",
        "FT", "FTA", "FT%", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "Awards", "Player-additional"]


def _row(pid, name, age, team, pos, g, mp, rng, scale=1.0):
    fga, fta = 8 * scale * mp / 30, 3 * scale * mp / 30
    fg, ft, tp = fga * .47, fta * .78, 1.5 * scale * mp / 30
    pts = 2 * fg + ft + tp
    r = dict.fromkeys(COLS, 0)
    r.update(Rk=1, Player=name, Age=age, Team=team, Pos=pos, G=g, GS=g, MP=mp, FG=fg, FGA=fga, FT=ft, FTA=fta, TRB=5 * scale * mp / 30,
             AST=3 * scale * mp / 30, STL=1 * mp / 30, BLK=.6 * mp / 30, TOV=2 * mp / 30, PTS=pts, **{"3P": tp})
    r["Player-additional"] = pid
    r["Awards"] = ""
    return r


def _write(dirpath, years=(2024, 2025, 2026)):
    rng = np.random.default_rng(3)
    for k, y in enumerate(years):
        rows = []
        for i in range(60):                                                    # veterans, in every season
            rows.append(_row(f"vet{i:02d}", f"Vet {i}", 24 + i % 10 + k, "AAA", "SG", 70, 20 + (i % 15), rng, 0.8 + (i % 7) / 10))
        rows.append(_row("trade01", "Traded Guy", 27 + k, "2TM", "PG", 70, 30, rng))      # combined row first, then team rows
        rows.append(_row("trade01", "Traded Guy", 27 + k, "AAA", "PG", 30, 30, rng))
        rows.append(_row("trade01", "Traded Guy", 27 + k, "BBB", "PG", 40, 30, rng))
        if y == years[-1]:
            rows.append(_row("rook01", "Rookie Rick", 20, "CCC", "SF", 70, 30, rng))     # debuts in the newest season
        rows.append(dict.fromkeys(COLS, "") | {"Player": "League Average"})
        pd.DataFrame(rows, columns=COLS).to_csv(Path(dirpath) / f"per_game_{y}.csv", index=False)


@pytest.fixture(scope="module")
def seasons():
    with tempfile.TemporaryDirectory() as d:
        _write(d)
        files = om.find_seasons(d)
        return {y: om.load_season(p) for y, p in files.items()}


def test_files_are_found_by_year_even_with_a_plus_in_the_name():
    with tempfile.TemporaryDirectory() as d:
        for n in ("per_game_2024.csv", "per_game+2025.csv", "per_game_2026.csv", "players.csv"):
            (Path(d) / n).write_text("x")
        assert list(om.find_seasons(d)) == [2024, 2025, 2026]


def test_traded_player_has_one_row_with_full_season_and_league_average_is_dropped(seasons):
    s = seasons[2026]
    assert s.index.is_unique and "League Average" not in set(s["Player"])
    assert s.loc["trade01", "G"] == 70


def test_rookie_is_not_penalised_for_seasons_before_his_debut(seasons):
    p = om.Params()
    beta, _ = om.fit_age_from_transition(seasons, [2024, 2025], p)
    proj = om.project_next(seasons, [2026, 2025, 2024], p, beta, min_total_mins=500)
    # a 70-game debut season must project like a 70-game veteran, not like a player who skipped two years
    assert proj.loc["rook01", "GP"] > 60
    assert abs(proj.loc["rook01", "GP"] - proj.loc["trade01", "GP"]) < 8


def test_a_season_missed_after_debut_counts_as_zero_games():
    h = pd.DataFrame({"Player": ["a"], "Tm": ["x"], "Age": [28.], "Pos": ["C"], "G": [70.], "MPG": [30.], "mins": [2100.], "fpg": [30.],
                      "fpm": [1.0]}, index=["a"])
    for s in om.STATS:
        h[s] = 1.0
    gone = h.iloc[0:0]                                                          # newest season: player absent (e.g. missed the year)
    core = om.project_core([gone, h], pd.Index(["a"]), om.Params(), {"fpm": 1.0, "mpg": 25.})
    assert core.loc["a", "games"] < 55


def test_projected_points_equal_the_scoring_weights_applied_to_the_stat_columns(seasons):
    p = om.Params()
    beta, n = om.fit_age_from_transition(seasons, [2024, 2025], p)
    proj = om.project_next(seasons, [2026, 2025, 2024], p, beta, min_total_mins=500)
    again = sum(om.W[s] * proj[s] for s in om.STATS)
    assert np.allclose(again, proj["fpts"]) and n >= 30
    assert (proj["GP"].between(20, 82)).all() and (proj["MPG"].between(8, 38)).all()


def test_output_loads_through_the_apps_own_loader_and_validation():
    with tempfile.TemporaryDirectory() as d:
        raw, out = Path(d) / "nba" / "raw", Path(d) / "nba"
        raw.mkdir(parents=True)
        _write(raw)
        assert build_own_model.main(raw_dir=raw, out_dir=out, top_n=50) == 0
        old = ingestion.DATA_DIR
        ingestion.DATA_DIR = Path(d)
        try:
            df, _ = validate.clean_projections(ingestion.load_projections("nba"), "nba")
        finally:
            ingestion.DATA_DIR = old
    assert len(df) >= 40 and {"FGA", "FTA", "GP"} <= set(df.columns)


def test_build_refuses_with_fewer_than_three_seasons(capsys):
    with tempfile.TemporaryDirectory() as d:
        _write(d, years=(2025, 2026))
        assert build_own_model.main(raw_dir=d, out_dir=d) == 1
    assert "at least 3" in capsys.readouterr().out


# ----------------------------------------------------------------------------------------------------------- overrides
@pytest.fixture(scope="module")
def projection(seasons):
    p = om.Params()
    beta, _ = om.fit_age_from_transition(seasons, [2024, 2025], p)
    return om.project_next(seasons, [2026, 2025, 2024], p, beta, min_total_mins=500)


def _ov(rows):
    return pd.DataFrame(rows).astype(str)


def test_games_override_rescales_the_counting_stats_and_points(projection):
    i = projection.index[projection["Name"] == "Vet 10"][0]
    before = projection.loc[i]
    new, rep = om.apply_overrides(projection, _ov([{"name": "vet 10", "games": "41", "note": "returning from injury"}]))
    after = new.loc[i]
    assert after["GP"] == 41 and after["MPG"] == before["MPG"]
    assert after["PTS"] == pytest.approx(before["PTS"] * 41 / before["GP"])
    assert after["fpts"] == pytest.approx(sum(om.W[s] * after[s] for s in om.STATS))
    assert after["note"] == "returning from injury" and rep[0].startswith("UPDATED")


def test_minutes_override_keeps_the_per_minute_rate(projection):
    i = projection.index[projection["Name"] == "Vet 20"][0]
    new, _ = om.apply_overrides(projection, _ov([{"name": "Vet 20", "mpg": "30"}]))
    a, b = projection.loc[i], new.loc[i]
    assert b["MPG"] == 30 and b["PTS"] / (b["GP"] * 30) == pytest.approx(a["PTS"] / (a["GP"] * a["MPG"]))


def test_team_and_position_labels_can_be_replaced(projection):
    i = projection.index[projection["Name"] == "Vet 5"][0]
    new, _ = om.apply_overrides(projection, _ov([{"name": "Vet 5", "team": "ZZZ", "pos": "PF,C"}]))
    assert new.loc[i, "Team"] == "ZZZ" and new.loc[i, "Pos"] == "PF,C"


def test_names_match_without_accents_dots_or_case():
    assert om.norm_name("Nikola Jokić") == om.norm_name("nikola  jokic") == "nikola jokic"
    assert om.norm_name("P.J. Washington") == om.norm_name("PJ Washington")
    assert om.norm_name("De'Aaron Fox") == om.norm_name("DeAaron Fox")


STATS_ROW = {"games": "70", "mpg": "30", "pts": "18", "reb": "7", "ast": "3", "stl": "1", "blk": "1", "tpm": "1.5", "tov": "2",
             "fgm": "6.5", "fga": "14", "ftm": "3", "fta": "4"}


def test_an_unknown_player_is_added_only_with_complete_stats(projection):
    new, rep = om.apply_overrides(projection, _ov([{"name": "Rookie Ann", "team": "NYK", "pos": "SF", **STATS_ROW},
                                                   {"name": "Rookie Half", "games": "70"}]))
    assert len(new) == len(projection) + 1
    row = new[new["Name"] == "Rookie Ann"].iloc[0]
    assert row["PTS"] == pytest.approx(18 * 70) and row["fpts"] == pytest.approx(sum(om.W[s] * row[s] for s in om.STATS))
    assert any(r.startswith("SKIPPED Rookie Half") for r in rep)


def test_bad_values_are_skipped_with_a_clear_message_and_never_crash(projection):
    new, rep = om.apply_overrides(projection, _ov([{"name": "Vet 1", "games": "abc"}, {"name": "Vet 2", "games": "200"},
                                                   {"name": "Vet 3", "mpg": "-4"}]))
    assert len(rep) == 3 and all(r.startswith("SKIPPED") for r in rep)
    assert new.loc[new["Name"] == "Vet 1", "GP"].iloc[0] == projection.loc[projection["Name"] == "Vet 1", "GP"].iloc[0]


def test_overrides_file_needs_a_name_column():
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "o.csv").write_text("player,games\nX,10\n")
        with pytest.raises(ValueError):
            om.load_overrides(Path(d) / "o.csv")


def test_build_applies_overrides_csv_found_next_to_the_raw_folder(capsys):
    with tempfile.TemporaryDirectory() as d:
        raw, out = Path(d) / "nba" / "raw", Path(d) / "nba"
        raw.mkdir(parents=True)
        _write(raw)
        pd.DataFrame([{"name": "Rookie Ann", "team": "NYK", "pos": "SF", "note": "No. 3 pick", **STATS_ROW},
                      {"name": "Vet 7", "games": "30", "note": "knee"}]).to_csv(out / "overrides.csv", index=False)
        assert build_own_model.main(raw_dir=raw, out_dir=out, top_n=400) == 0
        df = pd.read_csv(out / "players.csv")
    assert "Rookie Ann" in set(df["Name"]) and "Applied overrides" in capsys.readouterr().out
    assert df.loc[df["Name"] == "Vet 7", "GP"].iloc[0] == 30 and "knee" in df.loc[df["Name"] == "Vet 7", "note"].iloc[0]


# ------------------------------------------------------------------------------------------------------------- blending
def _ref_for(projection, scale):
    ref = om.to_app_frame(projection, 500)
    for c in ("PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"):
        ref[c] = ref[c] * scale
    ref["GP"] = ref["GP"] * 0.8
    ref["MIN"] = ref["MIN"] * 0.8 * 1.1                     # games x 0.8, minutes per game x 1.1
    return ref


def test_blend_zero_changes_nothing_and_one_matches_the_reference(projection):
    ref = _ref_for(projection, 1.5)
    same, n0 = om.blend_with_reference(projection, ref, 0.0)
    assert n0 == 0 and same["fpts"].equals(projection["fpts"])
    full, n1 = om.blend_with_reference(projection, ref, 1.0)
    i = full.index[full["Name"] == "Vet 10"][0]
    r = ref[ref["Name"] == "Vet 10"].iloc[0]
    assert n1 > 50 and full.at[i, "PTS"] == pytest.approx(r["PTS"]) and full.at[i, "GP"] == pytest.approx(r["GP"])
    assert full.at[i, "MPG"] == pytest.approx(r["MIN"] / r["GP"])


def test_blend_halfway_is_the_midpoint_and_points_are_recomputed(projection):
    ref = _ref_for(projection, 1.5)
    half, _ = om.blend_with_reference(projection, ref, 0.5)
    i = half.index[half["Name"] == "Vet 3"][0]
    r = ref[ref["Name"] == "Vet 3"].iloc[0]
    assert half.at[i, "PTS"] == pytest.approx(0.5 * projection.at[i, "PTS"] + 0.5 * r["PTS"])
    assert half.at[i, "fpts"] == pytest.approx(sum(om.W[s] * half.at[i, s] for s in om.STATS))


def test_blend_leaves_players_missing_from_the_reference_alone_and_validates_the_weight(projection):
    ref = _ref_for(projection, 1.5)
    ref = ref[ref["Name"] != "Vet 4"]
    out, _ = om.blend_with_reference(projection, ref, 0.7)
    i = projection.index[projection["Name"] == "Vet 4"][0]
    assert out.at[i, "PTS"] == projection.at[i, "PTS"]
    with pytest.raises(ValueError):
        om.blend_with_reference(projection, ref, 1.5)


def test_build_with_blend_needs_its_reference_file(capsys):
    with tempfile.TemporaryDirectory() as d:
        raw, out = Path(d) / "nba" / "raw", Path(d) / "nba"
        raw.mkdir(parents=True)
        _write(raw)
        assert build_own_model.main(raw_dir=raw, out_dir=out, blend=0.5) == 1
    assert "Not found" in capsys.readouterr().out


def test_build_writes_the_pure_own_file_and_the_blend_weight_for_the_live_slider():
    import json
    with tempfile.TemporaryDirectory() as d:
        raw, out = Path(d) / "nba" / "raw", Path(d) / "nba"
        raw.mkdir(parents=True)
        _write(raw)
        ref = om.to_app_frame(om.project_next(*_proj_args()), 100)
        for c in ("PTS", "REB", "AST"):
            ref[c] = ref[c] * 1.5
        ref.to_csv(out / "players_third_party.csv", index=False)
        assert build_own_model.main(raw_dir=raw, out_dir=out, top_n=60, blend=0.5) == 0
        final, own = pd.read_csv(out / "players.csv"), pd.read_csv(out / "players_own.csv")
        assert set(final["Name"]) == set(own["Name"]) and len(final) == 60                     # one player set, so the slider can mix them
        assert json.loads((out / "blend.json").read_text())["weight"] == 0.5
        m = final.merge(own, on="Name", suffixes=("_f", "_o"))
        assert (m["PTS_f"] > m["PTS_o"]).mean() > 0.9                                          # the final file is pulled toward the stronger reference


def _proj_args():
    with tempfile.TemporaryDirectory() as d:
        _write(d)
        s = {y: om.load_season(p) for y, p in om.find_seasons(d).items()}
    p = om.Params()
    beta, _ = om.fit_age_from_transition(s, [2024, 2025], p)
    return s, [2026, 2025, 2024], p, beta

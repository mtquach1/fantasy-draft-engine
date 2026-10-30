"""The two holes in 'last season's stats': injured stars and rookies."""
import csv
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import yahoo_check
from app import ingestion, nba_build, validate, valuation
from app.config import LeagueConfig
from tests.test_yahoo_check import FakeLeague, source

HEAD = "Rk,Player,Age,Team,Pos,G,GS,MP,FG,FGA,FG%,3P,3PA,3P%,FT,FTA,FT%,TRB,AST,STL,BLK,TOV,PF,PTS"
POS = ["PG", "SG", "SF", "PF", "C"]


def line(name, team, pos, g, q):
    """Per-game line whose production scales with q (1.0 = star)."""
    fga, fta = 18 * q + 3, 6 * q + 1
    return [name, team, pos, g, 34 * q, fga * 0.5, fga, 3 * q, 6 * q, fta * .8, fta, 8 * q, 6 * q, 1.5 * q, 1.2 * q, 2.5 * q, 2, 28 * q]


def season_csv(rows):
    out = [HEAD]
    for i, (name, team, pos, g, mp, fg, fga, tp, _, ft, fta, trb, ast, stl, blk, tov, pf, pts) in enumerate(rows, 1):
        out.append(f"{i},{name},25,{team},{pos},{g},{g},{mp:.1f},{fg:.2f},{fga:.2f},.450,{tp:.2f},5,.350,{ft:.2f},{fta:.2f},.800,"
                   f"{trb:.2f},{ast:.2f},{stl:.2f},{blk:.2f},{tov:.2f},{pf},{pts:.2f}")
    return "\n".join(out) + "\n"


def league(tmp):
    """80 ranked veterans (production falls with rank) + an injured star + traded player; returns paths and ranking text."""
    rng = np.random.default_rng(1)
    names = [f"Vet {i:02d}" for i in range(1, 81)]
    names[4] = "Star Hurt"                                  # ranked #5
    ranking = names.copy()
    ranking.insert(20, "Rookie Guy")                        # ranked #21, no NBA stats
    ranking.insert(29, "Rookie NoPos")                      # ranked #30, no position anywhere
    rank = {n: i + 1 for i, n in enumerate(ranking)}

    def vets(season, hurt_games):
        rows = []
        for n in names:
            q = max(1.15 - rank[n] * 0.0105, 0.25) * (1 + rng.normal(0, .02))
            g = hurt_games if n == "Star Hurt" else int(rng.integers(66, 78))
            rows.append(line(n, "AAA", POS[rank[n] % 5], g, q))
        rows.append(line("Traded Guy", "2TM", "SG", 60, 0.5))
        rows.append(line("Traded Guy", "AAA", "SG", 40, 0.5))
        rows.append(line("Traded Guy", "BBB", "SG", 20, 0.5))
        return rows
    raw = Path(tmp)
    files = []
    for yr, hurt in ((2026, 20), (2025, 70), (2024, 72)):
        f = raw / f"per_game_{yr}.csv"
        f.write_text(season_csv(vets(yr, hurt)), encoding="utf-8")
        files.append(f)
    pos = raw / "positions.csv"
    pos.write_text("name,pos\nRookie Guy,SG\n", encoding="utf-8")
    return files, "\n".join(f"{i}. {i}. {n}" for i, n in enumerate(ranking, 1)), pos, rank


@pytest.fixture
def built(tmp_path):
    files, ranking, pos, rank = league(tmp_path)
    out = tmp_path / "out"
    rep = nba_build.build(files, ranking, 2026, out, positions_file=pos)
    return rep, pd.read_csv(out / "players.csv"), out, rank, files, ranking, pos


def test_parse_rankings_formats():
    got = nba_build.parse_rankings("Top 200 Default Rankings - Standard\n1. 1. Nikola Jokić\n2. 2. Shai Gilgeous-Alexander\n\n3) Victor Wembanyama\nJayson Tatum")
    assert got == [(1, "Nikola Jokić"), (2, "Shai Gilgeous-Alexander"), (3, "Victor Wembanyama"), (4, "Jayson Tatum")]


def test_injured_star_is_not_buried_by_one_bad_season(built):
    rep, df, out, rank, *_ = built
    star = df[df["Name"] == "Star Hurt"].iloc[0]
    assert 50 <= star["GP"] <= 70, "projected games come from average availability (20, 70, 72), not last year's 20"
    per_game = star["PTS"] / star["GP"]
    assert per_game > 26, "rates use all three seasons, so he keeps his real per-game production"
    last_only = 20 * (1.15 - 5 * 0.0105) * 28                    # what a last-season-only file would have given
    assert star["PTS"] > 2 * last_only
    assert any(n == "Star Hurt" and last == 20 for n, rk, last, avg, proj in rep["injured_star_fixes"])


def test_rookie_gets_a_line_like_the_players_ranked_next_to_him(built):
    rep, df, out, rank, *_ = built
    rk = df[df["Name"] == "Rookie Guy"].iloc[0]
    assert "rookie" in rk["note"] and rk["GP"] == 65 and rk["Pos"] == "SG"
    nbrs = df[df["Name"].isin([f"Vet {i:02d}" for i in range(16, 27)])]
    pg = rk["PTS"] / rk["GP"]
    per_game = nbrs["PTS"] / nbrs["GP"]
    assert per_game.min() - 1 <= pg <= per_game.max() + 1
    assert any(n == "Rookie Guy" and r == 21 for n, r, c in rep["rookies_estimated"])


def test_player_without_a_position_is_reported_not_silently_dropped(built):
    rep, df, *_ = built
    assert "Rookie NoPos" in rep["needs_position"] and "Rookie NoPos" not in set(df["Name"])
    assert "NEED A POSITION" in (built[2] / "build_report.txt").read_text(encoding="utf-8")


def test_traded_player_counted_once_and_history_written(built):
    rep, df, out, *_ = built
    assert (df["Name"] == "Traded Guy").sum() == 1 and df[df["Name"] == "Traded Guy"].iloc[0]["Team"] == "2TM"
    hist = pd.read_csv(out / "history.csv")
    star = hist[hist["player_id"] == "n:star hurt"].sort_values("season")
    assert list(star["season"]) == [2024, 2025, 2026] and list(star["games_played"]) == [72, 70, 20]
    adp = pd.read_csv(out / "adp.csv")
    assert adp.iloc[0]["name"] == "Vet 01" and adp.iloc[20]["name"] == "Rookie Guy" and adp.iloc[20]["adp"] == 21


def test_thin_data_player_is_blended_part_way(tmp_path):
    files, ranking, pos, rank = league(tmp_path)
    # give "Vet 40" only 15 games in one season, nothing before
    extra = tmp_path / "t"
    extra.mkdir()
    for f in files[1:]:
        df = pd.read_csv(f)
        df = df[df["Player"] != "Vet 40"]
        df.to_csv(extra / f.name, index=False)
    df = pd.read_csv(files[0])
    df.loc[df["Player"] == "Vet 40", "G"] = 15
    df.loc[df["Player"] == "Vet 40", "PTS"] = 4.0                # a weak small-sample line
    df.to_csv(extra / files[0].name, index=False)
    rep = nba_build.build([extra / f.name for f in files], ranking, 2026, tmp_path / "o2", positions_file=pos)
    assert any(n == "Vet 40" and 0.5 < w < 1 for n, r, g, w in rep["thin_data"])
    out = pd.read_csv(tmp_path / "o2" / "players.csv")
    v = out[out["Name"] == "Vet 40"].iloc[0]
    assert v["PTS"] / v["GP"] > 6, "pulled toward its ranked neighbours, not left at the 4.0 small-sample line"


def test_injuries_file_reduces_games_and_flags_the_player(tmp_path):
    files, ranking, pos, rank = league(tmp_path)
    inj = tmp_path / "injuries.csv"
    inj.write_text("name,games_out\nVet 03,25\nVet 04,4\n", encoding="utf-8")
    nba_build.build(files, ranking, 2026, tmp_path / "o3", positions_file=pos, injuries_file=inj)
    out = pd.read_csv(tmp_path / "o3" / "players.csv").set_index("Name")
    assert out.loc["Vet 03", "is_injured"] and not out.loc["Vet 04", "is_injured"]
    assert out.loc["Vet 03", "GP"] < out.loc["Vet 02", "GP"] - 10


def test_yahoo_positions_file_beats_site_positions(tmp_path):
    files, ranking, pos, rank = league(tmp_path)
    yp = tmp_path / "positions_yahoo.csv"
    yp.write_text("player_id,eligible_positions\nn:rookie nopos,\"PF,C\"\nn:vet 10,\"PG,SG\"\n", encoding="utf-8")
    rep = nba_build.build(files, ranking, 2026, tmp_path / "o4", positions_file=pos, yahoo_positions_file=yp)
    out = pd.read_csv(tmp_path / "o4" / "players.csv").set_index("Name")
    assert rep["needs_position"] == [] and out.loc["Rookie NoPos", "Pos"] == "PF,C" and out.loc["Vet 10", "Pos"] == "PG,SG"


def test_bad_inputs_give_clear_messages(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("Name,Foo\nx,1\n", encoding="utf-8")
    with pytest.raises(nba_build.BuildError, match="no 'GP' column|no 'name' column"):
        nba_build.build([bad], "1. A", 2026, tmp_path / "o")
    with pytest.raises(nba_build.BuildError, match="at least one"):
        nba_build.build([], "1. A", 2026, tmp_path / "o")
    files, *_ = league(tmp_path)
    with pytest.raises(nba_build.BuildError, match="no player names"):
        nba_build.build(files, "\n\n", 2026, tmp_path / "o")


def test_built_files_run_through_the_whole_engine_and_rookie_lands_near_his_ranking(built, monkeypatch):
    rep, df, out, rank, *_ = built
    monkeypatch.setattr(ingestion, "DATA_DIR", out.parent / "base")
    (out.parent / "base").mkdir(exist_ok=True)
    import shutil
    shutil.copytree(out, out.parent / "base" / "nba", dirs_exist_ok=True)
    raw = ingestion.load_projections("nba")
    raw, actions = validate.clean_projections(raw, "nba")
    cfg = LeagueConfig.for_sport("nba")
    board, repl = valuation.calculate_player_valuations(raw, cfg, set(), 3)
    board = valuation.add_ranks(board, board)
    assert len(board) >= 80
    star = board[board["name"] == "Star Hurt"].iloc[0]
    rook = board[board["name"] == "Rookie Guy"].iloc[0]
    assert star["projected_rank"] <= 25, "an injured star must still rank near the top"
    assert abs(int(rook["projected_rank"]) - 21) <= 15, "the rookie lands near his Yahoo ranking"
    # the injury history from the files flags him as injury-prone-ish data, not a missing player
    assert "injury_prone" in board.columns


def test_export_positions_from_yahoo(tmp_path, capsys):
    league = FakeLeague()
    league.free_agents = lambda pos: [
        {"player_id": 1, "name": "Cooper Flagg", "eligible_positions": ["SF", "PF", "F", "Util"]},
        {"player_id": 2, "name": "Nikola Jokić", "eligible_positions": ["C", "Util"]},
        {"player_id": 3, "name": "Jalen Williams", "eligible_positions": ["SG"]},
        {"player_id": 4, "name": "Jalen Williams", "eligible_positions": ["PG"]}] if pos in ("SF", "C", "SG", "PG", "PF") else []
    res = yahoo_check.export_positions(source(league), "nba", tmp_path / "positions_yahoo.csv")
    rows = {r["player_id"]: r["eligible_positions"] for r in csv.DictReader(open(tmp_path / "positions_yahoo.csv", encoding="utf-8"))}
    assert rows["n:cooper flagg"] == "PF,SF" and rows["n:nikola jokic"] == "C"
    assert "n:jalen williams" not in rows and res["ambiguous"] == ["jalen williams"]
    assert "skipped" in capsys.readouterr().out


def test_export_positions_before_draft_only(tmp_path, capsys):
    league = FakeLeague()
    league.free_agents = lambda pos: []
    assert yahoo_check.export_positions(source(league), "nba", tmp_path / "p.csv")["saved"] == 0
    assert "BEFORE the draft" in capsys.readouterr().out

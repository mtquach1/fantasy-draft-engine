"""The quality gate and the source adapters: bad tables must be refused with a message a person can act on; odd-but-harmless ones only warned about."""
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from pipeline.quality import check_br_per_game, check_nba_api, check_table
from pipeline.sources import BRPerGameCsv, NbaApiSource, SourceError
from tests.pipeline_helpers import br_frame, csv_text, with_traded_player, write_br

FIXTURE = Path(__file__).parent / "fixtures" / "nba_api_leaguedash_sample.json"


# ------------------------------------------------------------------------------------------------ quality gate
def test_a_valid_table_passes_with_no_errors_and_no_warnings():
    rep = check_br_per_game(br_frame())
    assert rep.ok and rep.errors == [] and rep.warnings == []
    assert rep.stats["rows"] == 251                                   # 250 players + the League Average row


def test_the_league_average_row_and_a_traded_player_are_not_defects():
    rep = check_br_per_game(with_traded_player(br_frame()))
    assert rep.ok, rep.errors


def test_the_old_team_column_name_is_accepted():
    assert check_br_per_game(br_frame().rename(columns={"Team": "Tm"})).ok


@pytest.mark.parametrize("col", ["PTS", "Player-additional", "TOV", "MP"])
def test_a_missing_required_column_is_schema_drift_and_names_the_column(col):
    rep = check_br_per_game(br_frame().drop(columns=[col]))
    assert not rep.ok and len(rep.errors) == 1
    assert "Schema drift" in rep.errors[0] and col in rep.errors[0]


def test_missing_team_column_entirely_is_schema_drift():
    rep = check_br_per_game(br_frame().drop(columns=["Team"]))
    assert not rep.ok and "Team or Tm" in rep.errors[0]


def test_extra_columns_are_fine_and_counted():
    df = br_frame(); df["Rk"] = range(len(df)); df["Awards"] = ""
    rep = check_br_per_game(df)
    assert rep.ok and rep.stats["extra_columns"] == 2


def test_column_changes_since_the_current_snapshot_warn_but_do_not_block():
    old = check_br_per_game(br_frame()).to_dict()
    df = br_frame(); df["Shiny"] = 1
    rep = check_br_per_game(df, previous=old)
    assert rep.ok and any("Columns changed" in w and "Shiny" in w for w in rep.warnings)


def test_a_short_table_only_warns():
    rep = check_br_per_game(br_frame(n=40))
    assert rep.ok and any("Only 41 rows" in w for w in rep.warnings)


def test_duplicate_player_rows_are_an_error_with_an_example():
    df = br_frame(); df = pd.concat([df, df.iloc[[5]]], ignore_index=True)
    rep = check_br_per_game(df)
    assert not rep.ok and any("duplicates" in e and "pl005" in e for e in rep.errors)


@pytest.mark.parametrize("col,bad", [("G", 90), ("MP", 50), ("PTS", 75), ("Age", 9)])
def test_values_outside_the_possible_range_are_errors(col, bad):
    df = br_frame(); df.loc[3, col] = bad
    rep = check_br_per_game(df)
    assert not rep.ok and any(f"Column {col}" in e and "outside" in e for e in rep.errors)


def test_the_range_edges_themselves_are_allowed():
    df = br_frame(); df.loc[3, "G"] = 82; df.loc[4, "MP"] = 48; df.loc[5, "PTS"] = 60; df.loc[6, "G"] = 0
    assert check_br_per_game(df).ok


def test_a_few_non_numbers_warn_but_many_are_an_error():
    few = br_frame().astype({"G": object}); few.loc[2, "G"] = "n/a"
    rep = check_br_per_game(few)
    assert rep.ok and any("Column G: 1 value" in w for w in rep.warnings)
    many = br_frame().astype({"G": object}); many.loc[:100, "G"] = "n/a"
    rep = check_br_per_game(many)
    assert not rep.ok and any("Column G" in e and "aren't numbers" in e for e in rep.errors)


def test_negative_stats_are_an_error_and_named():
    df = br_frame(); df.loc[7, "AST"] = -1.0
    rep = check_br_per_game(df)
    assert not rep.ok and any("Negative values" in e and "AST" in e for e in rep.errors)


def test_a_big_change_in_row_count_warns_and_a_small_one_does_not():
    prev = check_br_per_game(br_frame(n=250)).to_dict()
    assert any("Row count moved" in w for w in check_br_per_game(br_frame(n=150), previous=prev).warnings)      # -40%
    assert not any("Row count moved" in w for w in check_br_per_game(br_frame(n=230), previous=prev).warnings)  # -8%


def test_the_nba_api_table_is_checked_too():
    df = pd.DataFrame(json.loads(FIXTURE.read_text(encoding="utf-8"))["resultSets"][0]["rowSet"],
                      columns=json.loads(FIXTURE.read_text(encoding="utf-8"))["resultSets"][0]["headers"])
    assert check_nba_api(df).ok
    assert not check_nba_api(df.drop(columns=["FG3M"])).ok
    dup = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    assert not check_nba_api(dup).ok


def test_check_table_reports_the_schema_error_alone_and_stops():
    rep = check_table(pd.DataFrame({"a": [1]}), required=["a", "b"], ranges={"a": (5, 6)}, id_col="a")
    assert len(rep.errors) == 1 and "b" in rep.errors[0]               # the out-of-range 'a' is not also reported: nothing else is reliable


# ------------------------------------------------------------------------------------------------ BRPerGameCsv
def test_the_br_source_names_its_snapshot_kind_by_season():
    assert BRPerGameCsv("x.csv", 2026).kind == "per_game_2026"


def test_a_file_with_a_bom_and_windows_line_endings_is_normalized_and_keeps_accents(tmp_path):
    df = br_frame(n=3); df.loc[0, "Player"] = "Nikola Jokić"
    p = tmp_path / "per_game_2026.csv"
    text = write_br(p, df, bom=True, crlf=True)
    got = BRPerGameCsv(p, 2026).fetch_text()
    assert got == text and "\r" not in got and not got.startswith("\ufeff") and "Jokić" in got
    assert list(BRPerGameCsv(p, 2026).fetch()["Player"])[0] == "Nikola Jokić"


def test_a_missing_file_is_a_source_error_naming_the_path(tmp_path):
    with pytest.raises(SourceError, match="does not exist"):
        BRPerGameCsv(tmp_path / "nope.csv", 2026).fetch_text()


def test_a_file_that_is_not_utf8_gets_actionable_advice(tmp_path):
    p = tmp_path / "per_game_2026.csv"
    p.write_bytes("Player\nJos\u00e9\n".encode("latin-1"))
    with pytest.raises(SourceError, match="isn't UTF-8"):
        BRPerGameCsv(p, 2026).fetch_text()


# ------------------------------------------------------------------------------------------------ NbaApiSource
def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_the_nba_api_source_turns_the_documented_shape_into_a_table():
    src = NbaApiSource("2025-26", fetcher=lambda season: _fixture())
    df = src.fetch()
    assert src.kind == "nba_api_leaguedash_2025-26"
    assert list(df["PLAYER_NAME"]) == ["Jaylen Brown", "Jayson Tatum"] and int(df["GP"].iloc[0]) == 71


def test_the_fetcher_is_told_which_season():
    seen = []
    NbaApiSource("2024-25", fetcher=lambda s: (seen.append(s), _fixture())[1]).fetch_text()
    assert seen == ["2024-25"]


def test_the_dict_style_result_set_is_accepted_too():
    block = _fixture()["resultSets"][0]
    df = NbaApiSource("2025-26", fetcher=lambda s: {"resultSets": {"headers": block["headers"], "rowSet": block["rowSet"]}}).fetch()
    assert len(df) == 2


@pytest.mark.parametrize("payload", [
    {},                                                                            # no resultSets
    {"resultSets": []},                                                            # no block at all
    {"resultSets": [{"name": "SomethingElse", "headers": [], "rowSet": []}]},      # block not found
    {"resultSets": [{"name": "LeagueDashPlayerStats", "headers": ["A", "B"], "rowSet": [[1, 2, 3]]}]},   # row wider than headers
    {"resultSets": [{"name": "LeagueDashPlayerStats", "headers": ["A"]}]},         # no rowSet
    None,
])
def test_a_malformed_response_is_a_source_error_not_a_crash(payload):
    with pytest.raises(SourceError, match="expected shape"):
        NbaApiSource("2025-26", fetcher=lambda s: payload).fetch_text()


def test_when_the_live_service_cannot_be_reached_the_error_says_what_to_do(monkeypatch):
    monkeypatch.setitem(sys.modules, "nba_api.stats.endpoints", None)               # as if the package were not installed: never touches the network
    with pytest.raises(SourceError, match="own computer"):
        NbaApiSource("2025-26").fetch_text()

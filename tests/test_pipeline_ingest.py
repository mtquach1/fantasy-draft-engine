"""Ingest: source -> quality gate -> versioned snapshot. A rejected or failed table must leave the store exactly as it was."""
import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.ingest import app_player_id, ingest_raw_dir, ingest_source, materialize, player_key
from pipeline.quality import check_br_per_game, check_nba_api
from pipeline.sources import BRPerGameCsv, NbaApiSource, SourceError
from pipeline.store import Store
from tests.pipeline_helpers import br_frame, with_traded_player, write_br

FIXTURE = Path(__file__).parent / "fixtures" / "nba_api_leaguedash_sample.json"


@pytest.fixture
def store(request):
    s = Store(":memory:")
    request.addfinalizer(s.close)
    return s


def _ingest(store, path, season=2026):
    return ingest_source(store, BRPerGameCsv(path, season), check_br_per_game, season)


def test_a_good_file_is_stored_linked_and_its_team_facts_recorded(store, tmp_path):
    write_br(tmp_path / "per_game_2026.csv", with_traded_player(br_frame()))
    r = _ingest(store, tmp_path / "per_game_2026.csv")
    assert r["status"] == "stored" and r["kind"] == "per_game_2026" and r["quality"]["ok"]
    cur = store.current_snapshot("per_game_2026")
    assert cur["id"] == r["snapshot_id"] and cur["rows"] == 254
    assert store.key_for("br", "pl007") == player_key("Player 007") == "player 007"
    assert store.key_for("br", "tr001") == "trade guy"                              # traded players are linked ...
    assert store.team_history(app_player_id("Player 000")) == [dict(team="AAA", effective_from="2025-10-01T00:00:00.000000Z", source="br_per_game", event_id=None)]
    assert store.team_history(app_player_id("Trade Guy")) == []                     # ... but get no single-team fact
    assert r["linked"] == 251 and r["team_facts"] == 250                            # 250 players + the traded one; League Average has no id


def test_the_same_file_again_stores_nothing_new(store, tmp_path):
    p = tmp_path / "per_game_2026.csv"; write_br(p, br_frame())
    first = _ingest(store, p); again = _ingest(store, p)
    assert again["status"] == "unchanged" and again["snapshot_id"] == first["snapshot_id"]
    assert len(store.snapshots()) == 1


def test_a_changed_file_becomes_a_new_current_snapshot_and_the_old_one_is_kept(store, tmp_path):
    p = tmp_path / "per_game_2026.csv"
    write_br(p, br_frame()); a = _ingest(store, p)
    write_br(p, br_frame(shift=1.0)); b = _ingest(store, p)
    assert b["status"] == "stored" and b["snapshot_id"] != a["snapshot_id"]
    assert [s["current"] for s in store.snapshots()] == [False, True]
    assert store.current_snapshot("per_game_2026")["id"] == b["snapshot_id"]


def test_a_rejected_table_leaves_the_store_exactly_as_it_was(store, tmp_path):
    p = tmp_path / "per_game_2026.csv"
    write_br(p, br_frame()); good = _ingest(store, p)
    before = (store.snapshots(), store.current_snapshot("per_game_2026"), store.key_for("br", "pl000"))
    bad = br_frame(n=260).drop(columns=["BLK"])                                    # schema drift, and it would add new players if it got in
    write_br(p, bad)
    r = _ingest(store, p)
    assert r["status"] == "rejected" and r["snapshot_id"] is None
    assert any("Schema drift" in e and "BLK" in e for e in r["quality"]["errors"])
    assert (store.snapshots(), store.current_snapshot("per_game_2026"), store.key_for("br", "pl000")) == before
    assert store.key_for("br", "pl255") is None                                     # the rejected table's new players were not linked
    assert store.current_snapshot("per_game_2026")["id"] == good["snapshot_id"]


def test_the_first_ever_table_can_also_be_rejected_and_nothing_is_created(store, tmp_path):
    p = tmp_path / "per_game_2026.csv"; df = br_frame(); df.loc[0, "G"] = 200; write_br(p, df)
    assert _ingest(store, p)["status"] == "rejected"
    assert store.snapshots() == [] and store.current_snapshot("per_game_2026") is None


@pytest.mark.parametrize("content,needle", [
    (None, "does not exist"),                                                       # no file
    (b"", "No columns"),                                                            # empty file
    (b"a,b\n1,2\n3,4,5,6,7\n", "Expected"),                                          # unparsable rows
    ("Player\nJos\u00e9\n".encode("latin-1"), "isn't UTF-8"),
])
def test_an_unreadable_source_fails_cleanly_and_stores_nothing(store, tmp_path, content, needle):
    p = tmp_path / "per_game_2026.csv"
    if content is not None:
        p.write_bytes(content)
    r = _ingest(store, p)
    assert r["status"] == "failed" and r["snapshot_id"] is None and needle in r["error"]
    assert store.snapshots() == []


def test_a_table_without_the_right_columns_that_still_parses_is_rejected_not_failed(store, tmp_path):
    p = tmp_path / "per_game_2026.csv"; p.write_bytes(b"a,b\n1,2,3,4\n")
    r = _ingest(store, p)
    assert r["status"] == "rejected" and "Schema drift" in r["quality"]["errors"][0] and store.snapshots() == []


def test_a_new_column_since_the_current_snapshot_is_warned_about_during_a_real_ingest(store, tmp_path):
    p = tmp_path / "per_game_2026.csv"
    write_br(p, br_frame()); _ingest(store, p)
    df = br_frame(shift=1); df["Shiny"] = 1; write_br(p, df)
    r = _ingest(store, p)
    assert r["status"] == "stored" and any("Columns changed" in w and "Shiny" in w for w in r["quality"]["warnings"])


def test_a_big_drop_in_rows_is_stored_with_a_warning(store, tmp_path):
    p = tmp_path / "per_game_2026.csv"
    write_br(p, br_frame(n=250)); _ingest(store, p)
    write_br(p, br_frame(n=120))
    r = _ingest(store, p)
    assert r["status"] == "stored" and any("Row count moved" in w for w in r["quality"]["warnings"])


def test_ingest_raw_dir_reads_every_season_file_including_the_plus_spelling(store, tmp_path):
    for name, shift in (("per_game_2024.csv", 0), ("per_game+2025.csv", 1), ("per_game_2026.csv", 2)):
        write_br(tmp_path / name, br_frame(shift=shift))
    results = ingest_raw_dir(store, tmp_path)
    assert [r["kind"] for r in results] == ["per_game_2024", "per_game_2025", "per_game_2026"]
    assert all(r["status"] == "stored" for r in results)


def test_one_bad_season_does_not_stop_the_others(store, tmp_path):
    write_br(tmp_path / "per_game_2024.csv", br_frame())
    write_br(tmp_path / "per_game_2025.csv", br_frame().drop(columns=["PTS"]))
    write_br(tmp_path / "per_game_2026.csv", br_frame(shift=1))
    results = {r["kind"]: r["status"] for r in ingest_raw_dir(store, tmp_path)}
    assert results == {"per_game_2024": "stored", "per_game_2025": "rejected", "per_game_2026": "stored"}


def test_an_empty_folder_is_an_error_that_names_the_folder(store, tmp_path):
    with pytest.raises(SourceError, match="No per_game"):
        ingest_raw_dir(store, tmp_path)


def test_materialize_round_trips_the_text_exactly_including_accents(store, tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "out"
    df = br_frame(); df.loc[0, "Player"] = "Nikola Jokić"
    texts = {y: write_br(raw / f"per_game_{y}.csv", df.assign(PTS=df["PTS"] + i), crlf=bool(i % 2)) for i, y in enumerate((2024, 2025, 2026))}
    ingest_raw_dir(store, raw)
    written = materialize(store, out)
    assert sorted(Path(w).name for w in written) == ["per_game_2024.csv", "per_game_2025.csv", "per_game_2026.csv"]
    for y, text in texts.items():
        assert (out / f"per_game_{y}.csv").read_bytes().decode("utf-8") == text   # byte for byte, no line-ending surprises


def test_a_rollback_takes_effect_in_the_materialized_files(store, tmp_path):
    p, out = tmp_path / "per_game_2026.csv", tmp_path / "out"
    old = write_br(p, br_frame()); a = _ingest(store, p)
    new = write_br(p, br_frame(shift=5)); b = _ingest(store, p)
    materialize(store, out)
    assert (out / "per_game_2026.csv").read_bytes().decode("utf-8") == new
    store.set_current("per_game_2026", a["snapshot_id"])
    materialize(store, out)
    assert (out / "per_game_2026.csv").read_bytes().decode("utf-8") == old


def test_materialize_writes_only_per_game_tables_and_nothing_when_empty(store, tmp_path):
    assert materialize(store, tmp_path / "empty") == []
    src = NbaApiSource("2025-26", fetcher=lambda s: json.loads(FIXTURE.read_text(encoding="utf-8")))
    assert ingest_source(store, src, check_nba_api)["status"] == "stored"
    assert materialize(store, tmp_path / "out") == []                              # the nba_api table is not a build_own_model input


def test_an_nba_api_table_is_stored_and_its_ids_linked_under_their_own_source(store):
    src = NbaApiSource("2025-26", fetcher=lambda s: json.loads(FIXTURE.read_text(encoding="utf-8")))
    r = ingest_source(store, src, check_nba_api)
    assert r["status"] == "stored" and r["kind"] == "nba_api_leaguedash_2025-26"
    assert store.key_for("nba_api", "1627759") == "jaylen brown" and store.key_for("br", "1627759") is None
    assert ingest_source(store, src, check_nba_api)["status"] == "unchanged"


def test_a_malformed_nba_api_response_fails_without_touching_the_store(store):
    r = ingest_source(store, NbaApiSource("2025-26", fetcher=lambda s: {"resultSets": []}), check_nba_api)
    assert r["status"] == "failed" and "expected shape" in r["error"] and store.snapshots() == []

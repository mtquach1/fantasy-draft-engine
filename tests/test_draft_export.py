"""Exporting a platform's picks as a draft file, and feeding that file to the calibration code."""
import json
from pathlib import Path

import pytest

import espn_check
from app import calibration as cal
from app.connectors.base import RemotePick
from app.connectors.espn import EspnSource
from app.draft_export import export_log, unresolved
from tests.test_espn import ME, Getter, pk


def picks3():
    return [RemotePick(2, "3032977", "2", "Giannis Antetokounmpo"), RemotePick(1, "3112335", "1", "Nikola Jokić"), RemotePick(3, "9", "4", "")]


def test_export_log_sorts_picks_maps_teams_and_keeps_unreadable_picks():
    doc = export_log(picks3(), lambda t: {"1": 1, "2": 2, "4": 3}[t], teams=3, my_team=3, draft_type="SNAKE")
    assert [(p["pick_number"], p["player_id"], p["team_id"]) for p in doc["draft_log"]] == [(1, "n:nikola jokić", 1), (2, "n:giannis antetokounmpo", 2), (3, None, 3)]
    assert doc["setup"] == {"draft_type": "snake", "teams": 3, "my_slot": 3, "rounds": 1, "budget": None} and unresolved(doc) == 1
    assert export_log(picks3(), lambda t: 1, teams=2, draft_type="auction")["setup"]["draft_type"] == "auction"
    assert export_log(picks3(), lambda t: 1, teams=2, draft_type="weird")["setup"]["draft_type"] == "snake"
    assert export_log([], lambda t: 1, teams=2)["setup"]["rounds"] is None


def test_the_exported_file_is_a_valid_input_for_the_calibration_reader(tmp_path):
    doc = export_log(picks3(), lambda t: {"1": 1, "2": 2, "4": 3}[t], teams=3, my_team=3)
    p = tmp_path / "x.json"; p.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    d = cal.load_picks(p)
    assert d["teams"] == 3 and d["my_slot"] == 3 and [x["key"] for x in d["picks"]] == ["nikola jokic", "giannis antetokounmpo", None]


def test_the_espn_export_command_writes_the_file_and_says_what_to_do_next(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("ESPN_SWID", ME); monkeypatch.setenv("ESPN_S2", "S" * 100)
    src = EspnSource("nba", getter=Getter([pk(1, 1, 3112335), pk(2, 2, 3032977), pk(3, 4, 4278073)], drafted=True, in_progress=False))
    out = tmp_path / "sub" / "e.json"
    assert espn_check.export_draft(src, "nba", out) == 0
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert [p["team_id"] for p in doc["draft_log"]] == [1, 2, 3] and doc["setup"]["my_slot"] == 3 and doc["setup"]["draft_type"] == "snake"
    assert "Jokić".lower() in doc["draft_log"][0]["player_id"]
    text = capsys.readouterr().out
    assert "3 picks from 3 teams" in text and "calibrate_lookahead.py" in text and "not draft slots" in text


def test_export_before_the_draft_writes_nothing_and_says_why(monkeypatch, tmp_path, capsys):
    src = EspnSource("nba", getter=Getter(drafted=False, in_progress=False))
    out = tmp_path / "e.json"
    assert espn_check.export_draft(src, "nba", out) == 1 and not out.exists()
    assert "no picks yet" in capsys.readouterr().out


def test_export_without_cookies_warns_that_my_team_is_unknown(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ESPN_SWID", raising=False)
    src = EspnSource("nba", getter=Getter([pk(1, 1, 3112335)], drafted=False, in_progress=True))
    assert espn_check.export_draft(src, "nba", tmp_path / "e.json") == 0
    assert "Your team was not found" in capsys.readouterr().out
    assert json.loads((tmp_path / "e.json").read_text(encoding="utf-8"))["setup"]["my_slot"] is None


def test_unresolved_names_are_reported(monkeypatch, tmp_path, capsys):
    g = Getter([pk(1, 1, 3112335), pk(2, 2, 999999)], drafted=True, in_progress=False)        # 999999 is not in ESPN's name list
    assert espn_check.export_draft(EspnSource("nba", getter=g), "nba", tmp_path / "e.json") == 0
    assert "1 pick(s) have no player name" in capsys.readouterr().out


def test_the_export_command_line_entry_point(monkeypatch, tmp_path, capsys):
    from app.connectors import espn as espn_mod
    monkeypatch.setenv("ESPN_LEAGUE_ID", "12345"); monkeypatch.setenv("ESPN_YEAR", "2027")
    monkeypatch.setenv("ESPN_S2", "S" * 100); monkeypatch.setenv("ESPN_SWID", ME)
    monkeypatch.setattr(espn_check.common, "OUT_FILE", tmp_path / "check_output.txt")           # never write into the project folder
    getter = Getter([pk(1, 1, 3112335), pk(2, 2, 3032977)], drafted=True, in_progress=False)
    monkeypatch.setattr(espn_mod, "EspnSource", lambda sport: EspnSource(sport, getter=getter))
    out = tmp_path / "e.json"
    assert espn_check.main(["export", "--sport", "nba", "--out", str(out)]) == 0
    assert len(json.loads(out.read_text(encoding="utf-8"))["draft_log"]) == 2
    assert "Connecting to ESPN" in capsys.readouterr().out and (tmp_path / "check_output.txt").exists()

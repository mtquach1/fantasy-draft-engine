"""Runs the REAL YahooSource code and the checker against a fake of the yahoo_fantasy_api League whose return shapes
were copied from that library's source/docs (draft_results, settings, positions, player_details, free_agents)."""
import pytest

import yahoo_check
from app.connectors.base import SourceError
from app.connectors.yahoo import YahooSource, yahoo_to_league_config
from app.config import LeagueConfig, ScoringType

KEY = "466.l.123456"
PLAYERS = {101: "Nikola Jokić", 102: "Shai Gilgeous-Alexander", 103: "Totally Unknown Rookie", 104: "Victor Wembanyama"}


class FakeOAuth:
    def token_is_valid(self):
        return True

    def refresh_access_token(self):
        raise AssertionError("token is valid: no refresh expected")


class FakeLeague:
    def __init__(self, picks=None, draft_status="draft"):
        self.picks, self.draft_status = picks if picks is not None else [], draft_status

    def settings(self):          # NO roster_positions / stat_categories: exactly like the real library
        return {"league_key": KEY, "name": "Mock", "draft_status": self.draft_status, "num_teams": 12,
                "scoring_type": "head", "is_auction_draft": "0", "draft_type": "live", "game_code": "nba"}

    def positions(self):         # shape from the library's docstring (counts can be str)
        return {"PG": {"position_type": "P", "count": 1}, "SG": {"position_type": "P", "count": 1},
                "G": {"position_type": "P", "count": 1}, "SF": {"position_type": "P", "count": 1},
                "PF": {"position_type": "P", "count": 1}, "F": {"position_type": "P", "count": 1},
                "C": {"position_type": "P", "count": 2}, "Util": {"position_type": "P", "count": 2},
                "BN": {"count": 3}, "IL": {"count": "1"}, "IL+": {"count": "2"}}

    def stat_categories(self):
        names = ["FG%", "FT%", "3PTM", "PTS", "REB", "AST", "ST", "BLK", "TO"]
        return [{"display_name": n, "position_type": "P"} for n in names]

    def teams(self):
        return {f"{KEY}.t.{i}": {} for i in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13)}      # gap: no team 4

    def team_key(self):
        return f"{KEY}.t.11"

    def draft_results(self):     # shape from the library docstring: pick/round/cost(str)/team_key/player_id(int)
        return list(self.picks)

    def player_details(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        out = []
        for i in ids:
            if i not in PLAYERS:
                raise RuntimeError(f"Player {i} not found")       # the real library raises for an unknown id
            out.append({"player_key": f"466.p.{i}", "player_id": str(i), "name": {"full": PLAYERS[i]}})
        return out

    def free_agents(self, pos):
        return [{"player_id": i, "name": n, "eligible_positions": ["C", "Util"]} for i, n in PLAYERS.items()]


def source(league):
    s = YahooSource.__new__(YahooSource)          # skip the network sign-in, keep every real method
    s.oauth, s.league, s.league_key, s._my_team_key, s._dense = FakeOAuth(), league, KEY, None, None
    return s


def pick(n, team, pid, cost=None):
    d = {"pick": n, "round": (n - 1) // 12 + 1, "team_key": f"{KEY}.t.{team}", "player_id": pid}
    if cost is not None:
        d["cost"] = cost
    return d


def test_settings_roster_comes_from_positions_not_settings():
    s = source(FakeLeague())
    settings, cats = s.fetch_settings()
    assert {r["position"]: r["count"] for r in settings["roster_positions"]}["C"] == 2
    cfg, warns = yahoo_to_league_config(settings, cats, LeagueConfig.for_sport("nba"))
    assert cfg.scoring_type == ScoringType.H2H_CATEGORIES and cfg.roster.teams == 12
    assert cfg.roster.slots["C"] == 2 and cfg.roster.slots["UTIL"] == 2 and cfg.roster.slots["BN"] == 3
    assert cfg.roster.il_slots == 3
    assert cfg.categories["X"] == ["FG%", "FT%", "3PM", "PTS", "REB", "AST", "STL", "BLK", "TO"] and warns == []


def test_picks_read_with_team_gap_and_my_team():
    s = source(FakeLeague([pick(2, 5, 102, "0"), pick(1, 13, 101), pick(3, 11, 104)]))
    picks = s._fetch_sync()
    assert [p.pick_number for p in picks] == [1, 2, 3]                       # sorted
    assert [s.team_index(p.platform_team_id) for p in picks] == [12, 4, 10]  # Yahoo teams 1,2,3,5,...,13 -> 1..12
    assert s.my_team_id() == 10                                              # Yahoo team 11 is the 10th team (4 is missing)
    assert picks[0].platform_player_id == "101"


def test_names_batch_failure_falls_back_to_one_by_one():
    s = source(FakeLeague())
    assert s.resolve_names(["101", "999", "104"]) == {"101": "Nikola Jokić", "104": "Victor Wembanyama"}


def test_check_league_runs_every_step_before_the_draft(monkeypatch, capsys):
    monkeypatch.setattr(yahoo_check, "_name_matcher", lambda sport: ({"nikola jokic": "n:1", "shai gilgeous-alexander": "n:2",
                                                                      "victor wembanyama": "n:3"}, None))
    counts = yahoo_check.check_league(source(FakeLeague(draft_status="predraft")), "nba")
    out = capsys.readouterr().out
    assert counts["FAIL"] == 0 and "0 picks so far" in out and "3 of 4 matched by name" in out
    assert "Totally Unknown Rookie" in out and "team 10" in out


def test_check_league_during_a_draft_flags_gaps_and_unmatched(monkeypatch, capsys):
    monkeypatch.setattr(yahoo_check, "_name_matcher", lambda sport: ({"nikola jokic": "n:1"}, None))
    league = FakeLeague([pick(1, 1, 101), pick(2, 2, 103), pick(4, 3, 104)])          # pick 3 missing
    counts = yahoo_check.check_league(source(league), "nba")
    out = capsys.readouterr().out
    assert counts["WARN"] >= 2 and "not 1..3 without gaps" in out and "1 of 3 matched" in out


def test_check_league_reports_failures_instead_of_crashing(monkeypatch):
    class Broken(FakeLeague):
        def draft_results(self):
            raise RuntimeError("boom")

        def team_key(self):
            return None
    monkeypatch.setattr(yahoo_check, "_name_matcher", lambda sport: (None, "no file"))
    counts = yahoo_check.check_league(source(Broken()), "nba")
    assert counts["FAIL"] >= 2


def test_watch_prints_new_picks_once_and_notices_an_undo(monkeypatch, capsys):
    league = FakeLeague([pick(1, 1, 101)])
    s = source(league)
    monkeypatch.setattr(yahoo_check, "_name_matcher", lambda sport: ({"nikola jokic": "n:1"}, None))
    script = iter([lambda: league.picks.append(pick(2, 11, 102)),                # my team takes pick 2
                   lambda: league.picks.pop(),                                    # commissioner undoes it
                   lambda: None])
    def fake_sleep(_):
        try:
            next(script)()
        except StopIteration:
            pass
    ticks = iter(range(0, 100))
    n = yahoo_check.watch(s, "nba", poll=3, max_seconds=4, sleep=fake_sleep, clock=lambda: next(ticks))
    out = capsys.readouterr().out
    assert out.count("took Nikola Jokić") == 1 and "[matches your file]" in out
    assert "took Shai Gilgeous-Alexander" in out and "<-- YOU" in out and "[NOT in your file by name]" in out
    assert "pick count dropped" in out and n >= 1


def test_setup_writes_credentials_without_echoing_them(tmp_path, capsys):
    answers = iter(["KEY123", "SECRET456", ""])
    yahoo_check.setup_credentials(ask=lambda _: next(answers), path=tmp_path / "oauth2.json")
    import json
    saved = json.loads((tmp_path / "oauth2.json").read_text())
    assert saved == {"consumer_key": "KEY123", "consumer_secret": "SECRET456", "callback_uri": "oob"}
    assert "SECRET456" not in capsys.readouterr().out


def test_setup_refuses_empty_values(tmp_path):
    answers = iter(["", "x", ""])
    yahoo_check.setup_credentials(ask=lambda _: next(answers), path=tmp_path / "oauth2.json")
    assert not (tmp_path / "oauth2.json").exists()


def test_connect_reports_missing_file_cleanly(tmp_path, capsys):
    assert yahoo_check.connect_oauth(tmp_path / "nope.json") is None
    assert "python yahoo_check.py setup" in capsys.readouterr().out


def test_leagues_prints_season_code_and_handles_no_leagues(monkeypatch, capsys):
    import sys
    import types

    class FakeGame:
        def __init__(self, oauth, sport):
            pass

        def league_ids(self):
            return []

        def game_id(self):
            return "466"
    monkeypatch.setitem(sys.modules, "yahoo_fantasy_api", types.SimpleNamespace(Game=FakeGame))
    assert yahoo_check.list_leagues(object(), "nba") == []
    out = capsys.readouterr().out
    assert "466.l.<league number>" in out and "466.l.12345" in out and "Mock drafts often aren't listed" in out
    FakeGame.league_ids = lambda self: ["466.l.111", "466.l.222"]
    assert yahoo_check.list_leagues(object(), "nba") == ["466.l.111", "466.l.222"]


def test_explain_gives_a_plain_next_step_for_common_errors():
    msg = yahoo_check.explain(RuntimeError('{"description": "This application is not authorized to perform this action."}'))
    assert "Fantasy Sports" in msg and "setup" in msg
    assert "Client ID" in yahoo_check.explain(RuntimeError("invalid_grant"))
    assert yahoo_check.explain(RuntimeError("something odd")) is None


def test_leagues_failure_prints_the_fix(monkeypatch, capsys):
    import sys
    import types

    class Denied:
        def __init__(self, oauth, sport):
            pass

        def league_ids(self):
            raise RuntimeError("This application is not authorized to perform this action.")
    monkeypatch.setitem(sys.modules, "yahoo_fantasy_api", types.SimpleNamespace(Game=Denied))
    assert yahoo_check.list_leagues(object(), "nba") == []
    out = capsys.readouterr().out
    assert "[FAIL]" in out and "What to do:" in out and "API Permissions" in out

"""ESPN connector, against fake ESPN JSON (field names as ESPN and the espn-api package use them). No network."""
import asyncio

import httpx
import pytest

from app.connectors import check_env
from app.connectors import espn as espn_mod
from app.connectors.base import SourceError
from app.connectors.espn import EspnSource

ME = "{ABCDEF01-2345-6789-ABCD-EF0123456789}"
PLAYERS = [{"id": 3032977, "fullName": "Giannis Antetokounmpo"}, {"id": 3112335, "fullName": "Nikola Jokić"},
           {"id": 4278073, "fullName": "Shai Gilgeous-Alexander"}]


def league_json(draft_picks=None, drafted=False, in_progress=True):
    return {"settings": {"name": "Mock League", "size": 3, "draftSettings": {"type": "SNAKE"}},
            "teams": [{"id": 1, "owners": ["{11111111-1111-1111-1111-111111111111}"]},
                      {"id": 2, "owners": []},
                      {"id": 4, "owners": [ME]}],                    # team 3 was removed: ids have a gap
            "draftDetail": {"drafted": drafted, "inProgress": in_progress, "picks": draft_picks or []}}


class Getter:
    def __init__(self, picks=None, **kw):
        self.picks, self.kw, self.calls, self.fail_names = picks or [], kw, [], False

    def __call__(self, league, params=None, headers=None, extend=""):
        self.calls.append((league, params, extend))
        if not league and extend == "/players":
            if self.fail_names:
                raise SourceError("names down")
            return PLAYERS
        v = params.get("view")
        return league_json(self.picks, **self.kw)


def pk(overall, team, pid=None, rnd=None, slot=None, bid=None):
    d = {"overallPickNumber": overall, "teamId": team, "roundId": rnd or (overall - 1) // 3 + 1,
         "roundPickNumber": slot or (overall - 1) % 3 + 1}
    if pid is not None:
        d["playerId"] = pid
    if bid is not None:
        d["bidAmount"] = bid
    return d


def test_live_draft_in_progress_returns_the_picks_made_so_far():
    """The espn-api package returns NOTHING here (drafted=False until the draft ends). This must not."""
    g = Getter([pk(1, 1, 3112335), pk(2, 2, 3032977), pk(3, 4)], drafted=False, in_progress=True)       # pick 3: not made yet
    src = EspnSource("nba", getter=g)
    picks = src._fetch_sync()
    assert [(p.pick_number, p.platform_player_id, p.player_name) for p in picks] == [(1, "3112335", "Nikola Jokić"), (2, "3032977", "Giannis Antetokounmpo")]


def test_teams_dense_numbers_and_my_team_from_swid(monkeypatch):
    monkeypatch.setenv("ESPN_SWID", "abcdef01-2345-6789-abcd-ef0123456789")      # pasted without braces, other case
    monkeypatch.setenv("ESPN_S2", "'AEB123'")
    src = EspnSource("nba", getter=Getter())
    assert src.cookies == {"espn_s2": "AEB123", "SWID": "{abcdef01-2345-6789-abcd-ef0123456789}"}
    assert [src.team_index(t) for t in ("1", "2", "4")] == [1, 2, 3]                  # id 4 becomes team 3
    assert src.my_team_id() == 3
    monkeypatch.delenv("ESPN_SWID")
    assert EspnSource("nba", getter=Getter()).my_team_id() is None


def test_overall_pick_is_computed_when_espn_omits_it_and_sorted():
    g = Getter([{"teamId": 2, "playerId": 3032977, "roundId": 2, "roundPickNumber": 1},
                {"teamId": 1, "playerId": 3112335, "roundId": 1, "roundPickNumber": 1}])
    assert [p.pick_number for p in EspnSource("nba", getter=g)._fetch_sync()] == [1, 4]       # round 2, slot 1, 3 teams -> 4


def test_auction_price_is_read_and_snake_zero_is_ignored():
    g = Getter([pk(1, 1, 3112335, bid=45), pk(2, 2, 3032977, bid=0)])
    picks = EspnSource("nba", getter=g)._fetch_sync()
    assert [p.cost for p in picks] == [45, None]


def test_names_are_downloaded_once_and_unknown_ids_stay_blank():
    g = Getter([pk(1, 1, 3112335), pk(2, 2, 999999)])
    src = EspnSource("nba", getter=g)
    first = src._fetch_sync()
    assert [p.player_name for p in first] == ["Nikola Jokić", ""]
    src._fetch_sync()
    src._fetch_sync()
    name_calls = [c for c in g.calls if c[2] == "/players"]
    assert len(name_calls) == 1, "the unknown id must not trigger a re-download on every poll"


def test_name_service_failure_never_breaks_pick_sync():
    g = Getter([pk(1, 1, 3112335)])
    g.fail_names = True
    src = EspnSource("nba", getter=g)
    picks = src._fetch_sync()
    assert len(picks) == 1 and picks[0].player_name == "" and picks[0].platform_player_id == "3112335"
    g.fail_names = False
    src._names_at = 0                                    # pretend the retry delay has passed
    assert src._fetch_sync()[0].player_name == "Nikola Jokić"


def test_league_info_and_async_fetch():
    src = EspnSource("nba", getter=Getter([pk(1, 1, 3112335)]))
    assert src.info() == {"name": "Mock League", "teams": 3, "size": 3, "draft_type": "SNAKE"}
    assert [p.pick_number for p in asyncio.run(src.fetch_picks())] == [1]


def test_setup_errors_are_worded_for_a_person(monkeypatch):
    monkeypatch.delenv("ESPN_LEAGUE_ID", raising=False)
    with pytest.raises(SourceError, match="ESPN_LEAGUE_ID"):
        EspnSource("nba")
    with pytest.raises(SourceError, match="isn't supported"):
        EspnSource("hockey", getter=Getter())
    with pytest.raises(SourceError, match="no teams"):
        EspnSource("nba", getter=lambda *a, **k: {"teams": [], "settings": {}})


class FakeResp:
    def __init__(self, status, payload=None, bad_json=False):
        self.status_code, self._p, self._bad = status, payload, bad_json

    def json(self):
        if self._bad:
            raise ValueError("html")
        return self._p


@pytest.mark.parametrize("status,words", [(401, "access denied"), (403, "ESPN_S2"), (404, "can't find league"), (500, "error 500")])
def test_http_statuses_become_clear_messages(monkeypatch, status, words):
    monkeypatch.setattr(espn_mod.httpx, "get", lambda *a, **k: FakeResp(status))
    with pytest.raises(SourceError, match=words):
        EspnSource("nba", league_id=123, year=2027)


def test_http_html_timeout_and_network_errors(monkeypatch):
    monkeypatch.setattr(espn_mod.httpx, "get", lambda *a, **k: FakeResp(200, bad_json=True))
    with pytest.raises(SourceError, match="login page"):
        EspnSource("nba", league_id=123, year=2027)
    def slow(*a, **k):
        raise httpx.ReadTimeout("slow")
    monkeypatch.setattr(espn_mod.httpx, "get", slow)
    with pytest.raises(SourceError, match="too long"):
        EspnSource("nba", league_id=123, year=2027)
    def down(*a, **k):
        raise httpx.ConnectError("no network")
    monkeypatch.setattr(espn_mod.httpx, "get", down)
    with pytest.raises(SourceError, match="internet"):
        EspnSource("nba", league_id=123, year=2027)


def test_http_request_uses_espns_address_cookies_and_repeated_view_params(monkeypatch):
    seen = {}

    def fake_get(url, params=None, headers=None, cookies=None, **k):
        seen.setdefault("calls", []).append((url, params, cookies))
        return FakeResp(200, [league_json()])                      # ESPN sometimes wraps the answer in a list
    monkeypatch.setattr(espn_mod.httpx, "get", fake_get)
    monkeypatch.setenv("ESPN_S2", "S2VAL")
    monkeypatch.setenv("ESPN_SWID", ME)
    src = EspnSource("nba", league_id=777, year=2027)
    url, params, cookies = seen["calls"][0]
    assert url == "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba/seasons/2027/segments/0/leagues/777"
    assert params == {"view": ["mTeam", "mSettings"]} and cookies == {"espn_s2": "S2VAL", "SWID": ME}
    src.draft_detail()
    assert seen["calls"][1][1] == {"view": "mDraftDetail"}
    src.resolve_names(["3112335"])
    assert seen["calls"][2][0].endswith("/fba/seasons/2027/players")


def test_check_env_no_longer_demands_the_espn_api_package(monkeypatch):
    monkeypatch.setenv("ESPN_LEAGUE_ID", "123456")
    monkeypatch.setenv("ESPN_YEAR", "2027")
    problems, warnings = check_env("espn")
    assert problems == [] and any("ESPN_S2" in w for w in warnings)


# ---------- the checker ----------
import espn_check
import yahoo_check


def test_env_value_checks_never_print_the_cookies(capsys):
    env = {"ESPN_LEAGUE_ID": "12x", "ESPN_YEAR": "27", "ESPN_S2": "A" * 200, "ESPN_SWID": "nope"}
    assert espn_check.check_env_values(env) == 3
    out = capsys.readouterr().out
    assert "AAAA" not in out and "nope" not in out and "END year" in out
    good = {"ESPN_LEAGUE_ID": "123", "ESPN_YEAR": "2027", "ESPN_S2": "B" * 120, "ESPN_SWID": "abcdef01-2345-6789-abcd-ef0123456789"}
    assert espn_check.check_env_values(good) == 0
    assert espn_check.check_env_values({"ESPN_LEAGUE_ID": "1", "ESPN_YEAR": "2027"}) == 0          # public league: a warning only
    assert espn_check.check_env_values({"ESPN_LEAGUE_ID": "1", "ESPN_YEAR": "2027", "ESPN_S2": "x" * 90}) == 1


def test_check_league_during_a_live_draft(monkeypatch, capsys):
    monkeypatch.setenv("ESPN_SWID", ME)
    monkeypatch.setenv("ESPN_S2", "S" * 100)
    monkeypatch.setattr(yahoo_check, "_name_matcher", lambda sport: ({"nikola jokic": "n:1", "giannis antetokounmpo": "n:2"}, None))
    src = EspnSource("nba", getter=Getter([pk(1, 1, 3112335), pk(2, 2, 3032977)], drafted=False, in_progress=True))
    counts = espn_check.check_league(src, "nba")
    out = capsys.readouterr().out
    assert counts["FAIL"] == 0 and "in progress; 2 pick(s) made" in out and "team 3" in out
    assert "2 of 2 picks have a name" in out and "ESPN lists 3 active players" in out


def test_check_league_before_the_draft_and_when_it_is_over(monkeypatch, capsys):
    monkeypatch.setattr(yahoo_check, "_name_matcher", lambda sport: (None, "no file"))
    monkeypatch.delenv("ESPN_SWID", raising=False)
    counts = espn_check.check_league(EspnSource("nba", getter=Getter(drafted=False, in_progress=False)), "nba")
    out = capsys.readouterr().out
    assert "not started; 0 pick(s)" in out and "set ESPN_SWID" in out and counts["WARN"] >= 2
    espn_check.check_league(EspnSource("nba", getter=Getter([pk(1, 1, 3112335)], drafted=True, in_progress=False)), "nba")
    assert "already finished" in capsys.readouterr().out


def test_watch_works_with_the_espn_source(monkeypatch, capsys):
    g = Getter([pk(1, 1, 3112335)])
    src = EspnSource("nba", getter=g)
    monkeypatch.setattr(yahoo_check, "_name_matcher", lambda sport: ({"nikola jokic": "n:1"}, None))
    steps = iter([lambda: g.picks.append(pk(2, 2, 3032977)), lambda: None])

    def fake_sleep(_):
        try:
            next(steps)()
        except StopIteration:
            pass
    ticks = iter(range(100))
    yahoo_check.watch(src, "nba", poll=3, max_seconds=3, sleep=fake_sleep, clock=lambda: next(ticks))
    out = capsys.readouterr().out
    assert "took Nikola Jokić" in out and "[matches your file]" in out and "took Giannis Antetokounmpo" in out and "[NOT in your file by name]" in out

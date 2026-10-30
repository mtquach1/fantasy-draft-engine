"""Tests for auction drafts: budgets, suggested bids, nomination, sale checks, saving, live prices, mock auction."""
import asyncio
import json
import tempfile
from pathlib import Path

import pandas as pd

from app import auction, persist, sync
from app.config import LeagueConfig
from app.connectors.base import RemotePick, parse_cost
from app.connectors.mock import MockSource
from app.draft import DraftSetup
from app.state import GlobalState
from app.valuation import calculate_player_valuations
from tests.test_math import fake_projections

RAW = fake_projections()
SETUP = DraftSetup("auction", 4, 2, 3, 20)          # 4 teams, 3 roster spots each, $20 budget, you are team 2


def pick(n, pid, team, price=None, source="manual"):
    p = {"pick_number": n, "player_id": pid, "team_id": team, "source": source}
    if price is not None:
        p["price"] = price
    return p


# ---------- budgets ----------
def test_budgets_spent_remaining_open_spots_and_max_bid():
    log = [pick(1, "h0", 1, 12), pick(2, "h1", 1, 3), pick(3, "h2", 2, 1)]
    t = auction.team_budgets(SETUP, log)["teams"]
    assert (t[0]["spent"], t[0]["remaining"], t[0]["open_spots"]) == (15, 5, 1)
    assert t[0]["max_bid"] == 5                        # one spot left: it can spend everything
    assert (t[1]["remaining"], t[1]["open_spots"], t[1]["max_bid"]) == (19, 2, 18)    # must keep $1 for the other spot
    assert (t[2]["remaining"], t[2]["open_spots"], t[2]["max_bid"]) == (20, 3, 18)


def test_full_roster_has_no_max_bid_and_unpriced_picks_count_as_one_dollar():
    log = [pick(1, "a", 3, 5), pick(2, "b", 3, 5), pick(3, "c", 3, None, "live")]
    res = auction.team_budgets(SETUP, log)
    t3 = res["teams"][2]
    assert t3["open_spots"] == 0 and t3["max_bid"] == 0 and t3["spent"] == 11 and res["unpriced"] == 1


def test_picks_with_unknown_team_are_ignored_not_crashed():
    log = [{"pick_number": 1, "player_id": "a", "team_id": None, "source": "live"}, pick(2, "b", 9, 4)]
    assert all(t["spent"] == 0 for t in auction.team_budgets(SETUP, log)["teams"])


def test_free_money_is_budget_above_one_dollar_per_open_spot():
    assert auction.free_money(auction.team_budgets(SETUP, [])["teams"]) == 4 * (20 - 3)
    log = [pick(1, "a", 1, 10)]
    assert auction.free_money(auction.team_budgets(SETUP, log)["teams"]) == (10 + 60) - 11


# ---------- sale checks ----------
def test_check_sale_rules():
    log = [pick(1, "a", 1, 15)]
    assert auction.check_sale(SETUP, log, 2, 18) is None
    assert "at most $18" in auction.check_sale(SETUP, log, 2, 19)
    assert auction.check_sale(SETUP, log, 1, 4) is None                      # $5 left, 2 spots: keep $1, so $4 is the limit
    assert "at most $4" in auction.check_sale(SETUP, log, 1, 5)
    assert "lower than $1" in auction.check_sale(SETUP, log, 2, 0)
    assert "whole number" in auction.check_sale(SETUP, log, 2, 2.5)
    assert "whole number" in auction.check_sale(SETUP, log, 2, True)
    full = [pick(1, "a", 3, 1), pick(2, "b", 3, 1), pick(3, "c", 3, 1)]
    assert "already full" in auction.check_sale(SETUP, full, 3, 1)


def test_setup_conflict_when_budget_or_rounds_shrink_below_what_was_spent():
    log = [pick(1, "a", 1, 15), pick(2, "b", 1, 1)]
    ok = DraftSetup("auction", 4, 2, 3, 20)
    assert auction.setup_conflict(ok, log) == {}
    low = DraftSetup("auction", 4, 2, 3, 17)                              # spent 16 + 1 open spot = 17 is fine
    assert auction.setup_conflict(low, log) == {}
    assert "budget" in auction.setup_conflict(DraftSetup("auction", 4, 2, 3, 16), log)
    assert "rounds" in auction.setup_conflict(DraftSetup("auction", 4, 2, 1, 20), log)
    assert auction.setup_conflict(DraftSetup("snake", 4, 2, 3, None), log) == {}


# ---------- nomination ----------
def test_nomination_rotates_one_to_n_and_skips_full_rosters():
    s = DraftSetup("auction", 3, 3, 2, 20)
    assert auction.nomination(s, []) == {"next_team": 1, "until_mine": 2}
    log = [pick(1, "a", 3, 5), pick(2, "b", 3, 5)]                          # team 3 won both and is now full
    n = auction.nomination(s, log)
    # after two sales the rotation is back to team 3, who is full, so team 1 is next
    assert n["next_team"] == 1
    log2 = log + [pick(3, "c", 1, 5), pick(4, "d", 1, 5)]
    assert auction.nomination(s, log2)["next_team"] == 2
    log3 = log2 + [pick(5, "e", 2, 5), pick(6, "f", 2, 5)]
    assert auction.nomination(s, log3) == {"next_team": None, "until_mine": None}


def test_nomination_without_my_slot_has_no_countdown():
    s = DraftSetup("auction", 3, None, 2, 20)
    assert auction.nomination(s, [])["until_mine"] is None


# ---------- bid values ----------
def board():
    cfg = LeagueConfig()
    cfg.roster.teams = 12
    return calculate_player_valuations(RAW, cfg, set(), 2)[0], cfg


def test_bids_are_whole_dollars_at_least_one_and_spend_the_free_money():
    df, _ = board()
    free = 12 * (260 - 25)
    b = auction.bid_values(df, free)
    assert (b >= 1).all() and (b == b.round()).all()
    paid_above_min = (b - 1).sum()
    assert abs(paid_above_min - free) / free < 0.02                       # rounding only
    assert b[df["vorp"].idxmax()] == b.max()                              # best player costs the most
    assert (b[df["vorp"] <= 0] == 1).all()


def test_drafted_players_get_no_bid_and_avoided_players_soak_up_no_money():
    df, cfg = board()
    df = df.copy()
    df.loc[df.index[0], "available"] = False
    assert pd.isna(auction.bid_values(df, 1000)[df.index[0]])
    avoided = df["vorp"].idxmax() if df.index[0] != df["vorp"].idxmax() else df["vorp"].nlargest(2).index[1]
    df["avoid"] = False
    df.loc[avoided, "avoid"] = True
    assert auction.bid_values(df, 1000)[avoided] == 1


def test_no_positive_value_or_no_money_means_everyone_costs_one_dollar():
    df, _ = board()
    assert (auction.bid_values(df, 0).dropna() == 1).all()
    flat = df.copy()
    flat["vorp"] = -1.0
    assert (auction.bid_values(flat, 500) == 1).all()


def test_inflation_is_zero_at_list_price_negative_when_overspent_positive_when_underspent():
    df, _ = board()
    free0 = 12 * (260 - 25)
    pre = auction.bid_values(df, free0)
    avail = df["available"]
    assert abs(auction.inflation(free0, pre, avail)) < 0.02
    assert auction.inflation(free0 * 0.6, pre, avail) < -0.3
    assert auction.inflation(free0 * 1.4, pre, avail) > 0.3
    assert auction.inflation(100, pre, pd.Series(False, index=df.index)) is None


def test_overspending_lowers_later_bids():
    df, cfg = board()
    free0 = 12 * (260 - 25)
    hot = auction.bid_values(df, free0 * 0.5)
    base = auction.bid_values(df, free0)
    top = df["vorp"].idxmax()
    assert hot[top] < base[top]


def test_summary_has_everything_the_page_needs():
    df, _ = board()
    pre = auction.bid_values(df, 100)
    s = auction.summary(SETUP, [pick(1, "a", 2, 7)], 2, df, pre)
    assert s["mine"]["team_id"] == 2 and s["mine"]["remaining"] == 13 and len(s["teams"]) == 4
    assert s["nomination"]["next_team"] == 2 and s["free_money"] == 4 * 20 - 7 - 11


# ---------- saving ----------
def test_prices_survive_save_and_restore_and_old_files_without_prices_still_load():
    with tempfile.TemporaryDirectory() as d:
        a = GlobalState(config=LeagueConfig(), raw_df=RAW.copy())
        a.setup = DraftSetup("auction", 12, 5, 25, 260)
        a.config.roster.teams = 12
        a.draft_log[:] = [pick(1, "h0", 1, 41), pick(2, "p1", 2, 7)]
        persist.save(a, directory=d)
        assert [p["price"] for p in json.loads(Path(d, "draft_state_mlb.json").read_text())["draft_log"]] == [41, 7]
        b = GlobalState(config=LeagueConfig(), raw_df=RAW.copy())
        res = persist.restore(b, directory=d)
        assert res["picks"] == 2 and b.draft_log == a.draft_log and res["warning"] is None

        a.setup = DraftSetup("snake", 12, 5, 25, None)             # a snake draft's picks carry no price key
        a.draft_log[:] = [pick(1, "h0", 1)]
        persist.save(a, directory=d)
        assert "price" not in json.loads(Path(d, "draft_state_mlb.json").read_text())["draft_log"][0]


def test_check_picks_rejects_bad_prices_and_missing_manual_auction_prices():
    ids = {"a", "b"}
    assert persist.check_picks([pick(1, "a", 1, 5)], ids, 4, 12, auction=True) is None
    assert "invalid price" in persist.check_picks([pick(1, "a", 1, 0)], ids, 4, 12, auction=True)
    assert "invalid price" in persist.check_picks([pick(1, "a", 1, 2.5)], ids, 4, 12)
    assert "no auction price" in persist.check_picks([pick(1, "a", 1)], ids, 4, 12, auction=True)
    assert persist.check_picks([pick(1, "a", 1, None, "live")], ids, 4, 12, auction=True) is None   # live feed may lack prices
    assert persist.check_picks([pick(1, "a", 1)], ids, 4, 12) is None                               # snake: no price needed


# ---------- live prices ----------
def test_parse_cost_accepts_what_platforms_send_and_nothing_else():
    assert [parse_cost(x) for x in (23, "23", 23.0, "$23", "1,200", " 7 ")] == [23, 23, 23, 23, 1200, 7]
    assert [parse_cost(x) for x in (None, "", "abc", 0, -4, 2.5, True, "0")] == [None] * 8


def test_live_sync_carries_prices_into_the_pick_log():
    st = GlobalState()
    st.names = {"h1": "One", "h2": "Two"}

    class Remote:
        async def fetch_picks(self):
            return [RemotePick(1, "y1", "1", cost=30), RemotePick(2, "y2", "2")]
        def team_index(self, t): return int(t)
        def my_team_id(self): return 2

    async def go():
        t = asyncio.create_task(sync.run(st, lambda: Remote(), lambda: {"y1": "h1", "y2": "h2"}, 0.01, retry_delay=0.01))
        await asyncio.sleep(0.15)
        t.cancel()
    asyncio.run(go())
    assert st.draft_log[0]["price"] == 30 and "price" not in st.draft_log[1]


# ---------- mock auction ----------
def test_mock_auction_obeys_budgets_and_roster_sizes():
    ids = [f"h{i}" for i in range(200)]
    prices = {pid: max(1, 40 - i // 4) for i, pid in enumerate(ids)}
    m = MockSource(ids, 4, 10, "auction", 2, seconds_per_pick=0.0001, budget=100, prices=prices)
    import time
    time.sleep(0.2)
    picks = asyncio.run(m.fetch_picks())
    assert len(picks) == 40 and len({p.platform_player_id for p in picks}) == 40
    spent, count = {}, {}
    for p in picks:
        spent[p.platform_team_id] = spent.get(p.platform_team_id, 0) + p.cost
        count[p.platform_team_id] = count.get(p.platform_team_id, 0) + 1
    assert all(c == 10 for c in count.values()) and all(s <= 100 for s in spent.values())
    assert all(p.cost >= 1 for p in picks)


# ---------- endpoints (need the sample data files; skipped quietly without them) ----------
def _have_data() -> bool:
    return Path(persist.DATA_DIR, "hitters.csv").exists()


def test_endpoints_run_an_auction_from_setup_to_prices_and_restart():
    if not _have_data():
        return
    from fastapi import HTTPException
    from app import main
    from app.state import state
    with tempfile.TemporaryDirectory() as d:
        orig = persist.DATA_DIR
        persist.DATA_DIR = Path(d)
        try:
            main.load_sport("mlb")
            state.sync["platform"] = "manual"
            def fails(fn, code, text):
                try:
                    fn()
                except HTTPException as e:
                    assert e.status_code == code and text in str(e.detail), e.detail
                else:
                    raise AssertionError("expected an error")
            fails(main.auction_state, 400, "isn't an auction")
            main.put_setup({"draft_type": "auction", "teams": 12, "my_slot": 4, "budget": 260})
            st = main.auction_state()
            assert st["free_money"] == 12 * (260 - 25) and st["mine"]["max_bid"] == 236 and st["nomination"]["next_team"] == 1

            rows = main.available(limit=5)
            assert rows[0]["bid"] >= rows[1]["bid"] >= 1 and rows[0]["pre_bid"] == rows[0]["bid"]
            assert main.recommend(limit=3)[0]["bid"] >= 1

            top = rows[0]["player_id"]
            fails(lambda: main.pick(main.PickRequest(player_id=top, team_id=7)), 400, "Enter the price")
            fails(lambda: main.pick(main.PickRequest(player_id=top, team_id=7, price=0)), 400, "lower than $1")
            fails(lambda: main.pick(main.PickRequest(player_id=top, team_id=7, price=999)), 400, "at most $236")
            fails(lambda: main.pick(main.PickRequest(player_id=top, price=10)), 400, "team_id is required")
            assert state.draft_log == []
            r = main.pick(main.PickRequest(player_id=top, team_id=7, price=52))
            assert r["price"] == 52
            assert main.auction_state()["teams"][6]["remaining"] == 208
            assert main.recent_picks()["picks"][0]["price"] == 52
            assert all(x["player_id"] != top for x in main.available(limit=300))

            fails(lambda: main.put_setup({"draft_type": "auction", "teams": 12, "my_slot": 4, "budget": 30}), 400, "budget")
            main.put_setup({"draft_type": "auction", "teams": 12, "my_slot": 4, "budget": 300})
            assert main.auction_state()["teams"][6]["remaining"] == 248

            saved = [dict(p) for p in state.draft_log]
            main.load_sport("mlb")
            main._restore("manual")
            assert state.draft_log == saved and state.setup.budget == 300

            main.undo()
            assert main.auction_state()["teams"][6]["remaining"] == 300

            main.put_setup({"draft_type": "snake", "teams": 12, "my_slot": 4})
            fails(lambda: main.pick(main.PickRequest(player_id=top, price=5)), 400, "only apply to auction")
            assert "bid" not in main.available(limit=1)[0]
            assert main.get_setup()["auction"] is None
        finally:
            persist.DATA_DIR = orig
            main.load_sport("mlb")


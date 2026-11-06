"""app/calibration.py and research/calibrate_lookahead.py: the calibration must recover a noise model we planted, and fail loudly on bad input."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import make_sample_sport
from app import calibration as cal, ingestion, persist

ROOT = Path(__file__).resolve().parents[1]
NAMES = [f"Player {i:03d}" for i in range(1, 151)]                 # board: Player 001 is ranked first
RANKS = cal.board_from_names(NAMES)
SMALL_BASE, SMALL_SLOPE = (1.0, 3.0, 6.0), (0.05, 0.20, 0.40)


def simulate(a, b, seed, teams=12, rounds=8, rule="noise"):
    """A snake draft by drafters whose pick is argmin(rank + (a + b*rank) * N(0,1)) ('noise'), the best player left ('chalk'), or anyone ('random')."""
    r = np.random.default_rng(seed); avail = list(range(len(NAMES))); picks = []
    for n in range(1, teams * rounds + 1):
        rnd, pos = divmod(n - 1, teams); team = pos + 1 if rnd % 2 == 0 else teams - pos
        if rule == "chalk":
            j = avail.pop(0)
        elif rule == "random":
            j = avail.pop(int(r.integers(0, len(avail))))
        else:
            j = avail.pop(int(np.argmin([(i + 1) + (a + b * (i + 1)) * r.standard_normal() for i in avail])))
        picks.append({"n": n, "team": team, "key": cal.norm_name(NAMES[j])})
    return picks


def write_log(path, picks, my_slot=None, teams=12):
    log = [{"pick_number": p["n"], "player_id": None if p["key"] is None else "n:" + p["key"], "team_id": p["team"]} for p in picks]
    Path(path).write_text(json.dumps({"sport": "nba", "setup": {"teams": teams, "my_slot": my_slot}, "draft_log": log}), encoding="utf-8")
    return path


def best_sd50(states, runs=150):
    L, combos = cal.loglik_matrix(states, SMALL_BASE, SMALL_SLOPE, runs=runs)
    a, b = combos[int(np.argmax(L.sum(0)))]
    return a + b * 50, (a, b)


# ---------------------------------------------------------------------------------------- reading
def test_board_names_are_normalized_and_the_first_appearance_wins():
    r = cal.board_from_names(["Nikola Jokić", "De'Aaron Fox", "nikola jokic"])
    assert r == {"nikola jokic": 1, "deaaron fox": 2}


def test_a_saved_draft_is_read_in_pick_order_with_accents_and_unreadable_picks_handled(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"sport": "nba", "setup": {"teams": 4, "my_slot": 2}, "draft_log": [
        {"pick_number": 2, "player_id": "n:nikola joki\u0107", "team_id": 2}, {"pick_number": 1, "player_id": "n:de'aaron fox", "team_id": 1},
        {"pick_number": 3, "player_id": None, "team_id": 3}, {"pick_number": 4, "player_id": "2238", "team_id": 4}]}), encoding="utf-8")
    d = cal.load_picks(p)
    assert d["teams"] == 4 and d["my_slot"] == 2
    assert [(x["n"], x["key"]) for x in d["picks"]] == [(1, "deaaron fox"), (2, "nikola jokic"), (3, None), (4, None)]


@pytest.mark.parametrize("content,needle", [
    (None, "was not found"), ("{nope", "not a valid draft file"), ('{"x": 1}', "no picks"), ('{"draft_log": []}', "no picks"),
    ('{"sport": "nfl", "draft_log": [{"team_id": 1}]}', "basketball"), ('{"draft_log": [{"player_id": "n:x", "team_id": "a"}]}', "no whole-number team_id"),
])
def test_a_bad_draft_file_is_explained(tmp_path, content, needle):
    p = tmp_path / "d.json"
    if content is not None:
        p.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=needle):
        cal.load_picks(p)


# ---------------------------------------------------------------------------------------- states
def test_decision_states_record_the_position_taken_among_the_players_left():
    small = cal.board_from_names(["A", "B", "C", "D", "E", "F"])
    picks = [{"n": 1, "team": 1, "key": "a"}, {"n": 2, "team": 2, "key": "c"}, {"n": 3, "team": 1, "key": "zzz"}, {"n": 4, "team": 2, "key": "c"}, {"n": 5, "team": 1, "key": "b"}]
    st = cal.decision_states(picks, small)
    assert [(s["n"], s["obs"], list(s["avail"])) for s in st] == [(1, 0, [1, 2, 3, 4, 5, 6]), (2, 1, [2, 3, 4, 5, 6]), (5, 0, [2, 4, 5, 6])]   # unknown 'zzz' and the repeat of 'c' are ignored


def test_skipped_teams_make_no_observation_but_their_players_still_leave_the_pool():
    small = cal.board_from_names(["A", "B", "C", "D"])
    picks = [{"n": 1, "team": 3, "key": "a"}, {"n": 2, "team": 1, "key": "b"}]
    st = cal.decision_states(picks, small, skip_teams=(3,))
    assert len(st) == 1 and list(st[0]["avail"]) == [2, 3, 4] and st[0]["obs"] == 0


def test_survival_states_count_the_picks_between_a_teams_turns_and_who_lasted():
    small = cal.board_from_names([f"P{i}" for i in range(1, 11)])
    picks = [{"n": 1, "team": 1, "key": "p1"}, {"n": 2, "team": 2, "key": "p2"}, {"n": 3, "team": 2, "key": "p4"}, {"n": 4, "team": 1, "key": "p3"}]
    st = cal.survival_states(picks, small, top_m=3)
    first = next(s for s in st if s["n"] == 1)                # team 1 took p1 at #1, next turn at #4: two picks (p2, p4) came between
    assert first["g"] == 2
    assert list(first["survived"]) == [False, True, False]    # best left after p1: p2 (taken), p3 (lasted), p4 (taken)
    assert next(s for s in st if s["n"] == 2)["g"] == 0       # team 2 picks back to back: nothing between
    assert all(s["n"] != 4 for s in st)                       # the last pick of a team has no "next turn"


def test_a_pick_nobody_can_read_between_two_turns_removes_that_decision_point():
    small = cal.board_from_names([f"P{i}" for i in range(1, 11)])
    picks = [{"n": 1, "team": 1, "key": "p1"}, {"n": 2, "team": 2, "key": None}, {"n": 3, "team": 1, "key": "p3"}]
    assert cal.survival_states(picks, small, top_m=3) == []


# ---------------------------------------------------------------------------------------- recovering a planted noise model
def test_the_fit_recovers_the_noise_that_generated_the_drafts():
    states = [s for seed in range(3) for s in cal.decision_states(simulate(3.0, 0.20, seed), RANKS)]
    sd50, best = best_sd50(states)
    assert 9 <= sd50 <= 17, (sd50, best)                       # truth is 3 + 0.2 * 50 = 13; the wrong neighbours give 5.5 or 23


def test_drafters_who_always_take_the_best_player_left_get_a_small_noise_fit():
    states = cal.decision_states(simulate(0, 0, 1, rule="chalk"), RANKS)
    L, combos = cal.loglik_matrix(states, (0.5, 3.0), (0.02, 0.20), runs=150)
    assert combos[int(np.argmax(L.sum(0)))] == (0.5, 0.02)


def test_drafters_who_pick_at_random_get_a_large_noise_fit():
    states = cal.decision_states(simulate(0, 0, 1, rule="random"), RANKS)
    L, combos = cal.loglik_matrix(states, (0.5, 3.0), (0.02, 0.40), runs=150)
    assert combos[int(np.argmax(L.sum(0)))] == (3.0, 0.40)


def test_fit_reports_how_much_better_than_the_current_setting_and_survives_a_grid_without_it():
    states = cal.decision_states(simulate(6.0, 0.35, 2), RANKS)
    L, combos = cal.loglik_matrix(states, cal.GRID_BASE[:4], cal.GRID_SLOPE[:6], runs=100)
    f = cal.fit(L, combos, boot=200)
    assert f["n"] == len(states) and f["gain_per_pick"] > 0 and f["gain_ci"][0] < f["gain_per_pick"] < f["gain_ci"][1]
    assert set(f["sd_at"]) == {5, 20, 50, 100} and f["current_sd_at"][50] == 8.0
    L2, c2 = cal.loglik_matrix(states[:20], (1.0, 3.0), (0.1, 0.2), runs=50)            # CURRENT (2.0, 0.12) is not on this grid
    assert "gain_per_pick" not in cal.fit(L2, c2, boot=50)


def test_the_loglik_matrix_is_deterministic():
    states = cal.decision_states(simulate(3.0, 0.2, 5), RANKS)[:15]
    a, _ = cal.loglik_matrix(states, SMALL_BASE, SMALL_SLOPE, runs=80)
    b, _ = cal.loglik_matrix(states, SMALL_BASE, SMALL_SLOPE, runs=80)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------------------- reliability
def test_the_true_noise_model_is_better_calibrated_than_a_wrong_one():
    surv = [s for seed in range(4) for s in cal.survival_states(simulate(3.0, 0.20, seed), RANKS)]
    good, bad = cal.reliability(surv, 3.0, 0.20, runs=150), cal.reliability(surv, 0.5, 0.02, runs=150)
    assert good["brier"] < bad["brier"]
    for row in good["table"]:
        if row["n"] >= 80:
            assert abs(row["predicted"] - row["observed"]) < 0.12, row
    assert sum(r["n"] for r in good["table"]) == good["n"] == sum(len(s["survived"]) for s in surv)


def test_too_little_noise_makes_the_model_overconfident_that_good_players_get_taken():
    surv = [s for seed in range(4) for s in cal.survival_states(simulate(6.0, 0.35, seed), RANKS)]
    t = cal.reliability(surv, 0.5, 0.02, runs=150)["table"]
    low = next(r for r in t if r["hi"] <= 0.10)
    assert low["observed"] > low["predicted"] + 0.05            # the players it called "almost surely gone" lasted more often than it said


# ---------------------------------------------------------------------------------------- the command-line script
def _cli():
    spec = importlib.util.spec_from_file_location("calibrate_lookahead", ROOT / "research" / "calibrate_lookahead.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _board_csv(path):
    pd.DataFrame({"Name": NAMES}).to_csv(path, index=False, encoding="utf-8")
    return path


def test_the_script_runs_end_to_end_prints_all_sections_and_leaves_my_picks_out(tmp_path, capsys):
    cli = _cli(); board = _board_csv(tmp_path / "b.csv")
    logs = [write_log(tmp_path / f"l{k}.json", simulate(3.0, 0.2, k), my_slot=3) for k in range(2)]
    assert cli.main([str(l) for l in logs] + ["--board", str(board), "--runs", "80"]) == 0
    out = capsys.readouterr().out
    assert "Board: b.csv" in out and "(slot 3, yours, left out)" in out and "88 used for the fit" in out        # 96 picks minus team 3's 8
    assert "HOW NOISY" in out and "model said" in out and "Brier" in out and "leave one log out" in out


def test_too_few_picks_is_refused_with_advice(tmp_path, capsys):
    cli = _cli(); log = write_log(tmp_path / "l.json", simulate(3.0, 0.2, 1, teams=4, rounds=5))
    assert cli.main([str(log), "--board", str(_board_csv(tmp_path / "b.csv"))]) == 1
    assert "too few to fit" in capsys.readouterr().out


@pytest.mark.parametrize("make,needle", [("nolog", "was not found"), ("nobaord", "was not found"), ("badboard", "needs a Name column")])
def test_script_input_errors_are_explained(tmp_path, capsys, make, needle):
    cli = _cli(); log = write_log(tmp_path / "l.json", simulate(3.0, 0.2, 1))
    board = _board_csv(tmp_path / "b.csv")
    if make == "nolog":
        args = [str(tmp_path / "missing.json"), "--board", str(board)]
    elif make == "nobaord":
        args = [str(log), "--board", str(tmp_path / "missing.csv")]
    else:
        (tmp_path / "bad.csv").write_text("x,y\n1,2\n", encoding="utf-8"); args = [str(log), "--board", str(tmp_path / "bad.csv")]
    assert cli.main(args) == 1 and needle in capsys.readouterr().out


def test_by_default_the_board_is_the_apps_own_ranking(tmp_path, request):
    old_i, old_p = ingestion.DATA_DIR, persist.DATA_DIR
    ingestion.DATA_DIR = tmp_path; persist.DATA_DIR = tmp_path / "persist"
    request.addfinalizer(lambda: (setattr(ingestion, "DATA_DIR", old_i), setattr(persist, "DATA_DIR", old_p)))
    cli = _cli()
    with pytest.raises(ValueError, match="No data/nba/players.csv"):
        cli.board_names(None, 12)
    make_sample_sport.generate("nba", tmp_path / "nba", n=300)
    names, note = cli.board_names(None, 12)
    assert len(names) == 300 and "projected_rank" in note and len(set(names)) == 300
    shown = pd.read_csv(tmp_path / "nba" / "players.csv")["Name"]
    assert set(names) == set(shown)


def test_with_no_noise_the_survival_chance_is_exactly_who_is_taken_in_the_next_g_picks():
    st = {"n": 1, "g": 2, "pool": np.arange(1.0, 11.0), "survived": np.array([False, False, True, True, True])}
    p = cal.predict_survival(st, 0.0, 0.0, runs=20)
    assert list(p) == [0.0, 0.0, 1.0, 1.0, 1.0]                 # the best two are taken in the two picks between; the third lasts
    st0 = dict(st, g=0)
    assert list(cal.predict_survival(st0, 0.0, 0.0, runs=20)) == [1.0] * 5          # nobody picks in between: everyone lasts

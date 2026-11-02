"""research/why_report.py and research/growth_check.py: the arithmetic behind the evidence claims, on tiny hand-made inputs.
Nothing here touches your real data or the shipped results files (all output goes to a temp folder)."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app import ingestion, persist

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "research" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


STATS = ["GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"]


def _results(tmp_path, bot_delta, n=4):
    """n paired drafts. follow_adp's lineup has fixed stats; the bot's differs by bot_delta (per stat), identically in every draft."""
    base = dict(GP=700, MIN=20000, PTS=9000, REB=3000, AST=2000, STL=500, BLK=300, **{"3PM": 800}, TO=1000, FGM=3400, FGA=7000, FTM=1500, FTA=1900)
    rows = []
    for seed in range(1, n + 1):
        for who, d in (("follow_adp", {}), ("bot", bot_delta)):
            r = dict(seed=seed, slot=3, who=who, score=50000.0 + (1000 if who == "bot" else 0) + seed, rank=1, beaten=1.0, dnp=0, st_n=10)
            r.update({f"st_{k}": float(base[k] + d.get(k, 0)) for k in STATS})
            rows.append(r)
    f = tmp_path / "results.csv"; pd.DataFrame(rows).to_csv(f, index=False)
    return f


def _run_why(tmp_path, monkeypatch, capsys, delta, argv=None):
    mod = _load("why_report")
    monkeypatch.setattr(mod, "CSV", _results(tmp_path, delta)); monkeypatch.setattr(mod, "OUT", tmp_path / "out.md")
    code = mod.main(argv or ["follow_adp", "bot"])
    return code, capsys.readouterr().out, tmp_path / "out.md"


def test_each_stat_difference_is_worth_its_weight(tmp_path, monkeypatch, capsys):
    # +50 assists (x2), +10 steals and +5 blocks (x4 each), -40 turnovers (x -2 -> +80), +100 PTS, -30 REB
    delta = {"AST": 50, "STL": 10, "BLK": 5, "TO": -40, "PTS": 100, "REB": -30}
    code, out, outfile = _run_why(tmp_path, monkeypatch, capsys, delta)
    assert code == 0
    assert "| Assists | +100 |" in out                  # 50 * 2
    assert "| Steals + blocks | +60 |" in out            # 15 * 4
    assert "| Turnovers (fewer is better) | +80 |" in out
    assert "| Rebounding | -30 |" in out
    assert "| Scoring volume (PTS + 3PM) | +100 |" in out
    assert "| Shooting efficiency (2*FGM - FGA + FTM - FTA) | +0 |" in out
    assert "| **Sum of the above (drafted starters)** | **+310** |" in out      # 100 (scoring) + 100 (assists) + 60 (stl+blk) + 80 (turnovers) - 30 (rebounds)
    assert "Season score difference: +1000 points" in out
    assert outfile.read_text(encoding="utf-8").strip() == out.strip()          # the file is exactly what was printed


def test_shooting_efficiency_counts_misses_against(tmp_path, monkeypatch, capsys):
    # same points, but 40 fewer attempts and 10 fewer free-throw attempts: efficiency is worth +40 +10
    code, out, _ = _run_why(tmp_path, monkeypatch, capsys, {"FGA": -40, "FTA": -10})
    assert code == 0 and "| Shooting efficiency (2*FGM - FGA + FTM - FTA) | +50 |" in out


def test_per_starter_rows(tmp_path, monkeypatch, capsys):
    code, out, _ = _run_why(tmp_path, monkeypatch, capsys, {"GP": -50, "MIN": 1000})
    assert code == 0
    assert "| Games played | 65.0 | 70.0 | -5.0" in out          # 650/10 vs 700/10 starters
    assert "| Minutes per game |" in out


def test_ci_is_zero_width_when_every_draft_agrees(tmp_path, monkeypatch, capsys):
    _, out, _ = _run_why(tmp_path, monkeypatch, capsys, {"AST": 50})
    assert "| Assists | +100 | +100 to +100 |" in out


def test_why_report_failures_are_readable(tmp_path, monkeypatch, capsys):
    mod = _load("why_report")
    monkeypatch.setattr(mod, "CSV", tmp_path / "nope.csv"); monkeypatch.setattr(mod, "OUT", tmp_path / "o.md")
    assert mod.main([]) == 1 and "not found" in capsys.readouterr().out
    old = pd.DataFrame([dict(seed=1, slot=1, who="bot", score=1.0), dict(seed=1, slot=1, who="follow_adp", score=1.0)])
    old.to_csv(tmp_path / "old.csv", index=False); monkeypatch.setattr(mod, "CSV", tmp_path / "old.csv")
    assert mod.main([]) == 1 and "older draft_backtest.py" in capsys.readouterr().out
    monkeypatch.setattr(mod, "CSV", _results(tmp_path, {}))
    assert mod.main(["nobody", "bot"]) == 1 and "Need both" in capsys.readouterr().out
    assert not (tmp_path / "o.md").exists()


# ---------- growth_check ----------
def _fake_seasons(raw, n=120, seed=3, planted=True):
    """Three seasons. Younger players really improve by `planted` fantasy points per game; the model sees ages, so it should expect a rise."""
    rng = np.random.default_rng(seed)
    talent = rng.uniform(0.6, 2.2, n); age0 = rng.integers(20, 34, n)
    raw.mkdir(parents=True, exist_ok=True)
    for yr in (2024, 2025, 2026):
        rows = []
        for i in range(n):
            age = int(age0[i] + yr - 2024)
            lift = (1 + 0.10 * (26 - age) * (yr - 2024) / 2) if planted else 1.0
            t = talent[i] * max(lift, 0.5)
            mpg = float(np.clip(12 + 10 * t, 8, 38)); pts = 0.5 * mpg * (0.5 + 0.12 * t)
            rows.append({"Player": f"Guy{i:03d}", "Player-additional": f"g{i:03d}", "Age": age, "Team": "AAA", "Pos": "SF", "G": 70, "MP": mpg,
                         "FG": pts / 2.5, "FGA": pts / 1.15, "FT": pts * 0.14, "FTA": pts * 0.18, "3P": pts * 0.08, "TRB": mpg * 0.2, "AST": mpg * 0.1,
                         "STL": mpg * 0.03, "BLK": mpg * 0.02, "TOV": mpg * 0.06, "PTS": pts})
        pd.DataFrame(rows).round(3).to_csv(raw / f"per_game_{yr}.csv", index=False)


@pytest.fixture
def swapped_dirs(tmp_path, request):
    old_i, old_p = ingestion.DATA_DIR, persist.DATA_DIR
    ingestion.DATA_DIR = tmp_path / "data"; persist.DATA_DIR = tmp_path / "persist"
    request.addfinalizer(lambda: (setattr(ingestion, "DATA_DIR", old_i), setattr(persist, "DATA_DIR", old_p)))
    return tmp_path


def test_growth_check_runs_and_matches_its_own_table(swapped_dirs, monkeypatch, capsys):
    _fake_seasons(swapped_dirs / "data" / "nba" / "raw")
    mod = _load("growth_check"); out_csv = swapped_dirs / "players.csv"
    monkeypatch.setattr(mod, "OUT_CSV", out_csv)
    assert mod.main() == 0
    text = capsys.readouterr().out
    t = pd.read_csv(out_csv)
    assert {"name", "proj_chg", "act_chg"} <= set(t.columns) and len(t) >= 30
    r = np.corrcoef(t["proj_chg"], t["act_chg"])[0, 1]
    assert f"{r:+.2f}" in text                                       # the printed number is the number in the table
    assert (t["proj_chg"] - (t["proj_fpg"] - t["last_fpg"])).abs().max() < 0.11     # rounding only
    assert "biggest projected risers" in text


def test_growth_check_needs_three_seasons(swapped_dirs, monkeypatch, capsys):
    raw = swapped_dirs / "data" / "nba" / "raw"; _fake_seasons(raw); (raw / "per_game_2024.csv").unlink()
    mod = _load("growth_check"); monkeypatch.setattr(mod, "OUT_CSV", swapped_dirs / "x.csv")
    assert mod.main() == 1 and "three per-game seasons" in capsys.readouterr().out
    assert not (swapped_dirs / "x.csv").exists()


def test_a_results_file_can_be_named_on_the_command_line_and_the_report_lands_beside_it(tmp_path, monkeypatch, capsys):
    mod = _load("why_report")
    src = _results(tmp_path, {"AST": 50}); merged = tmp_path / "draft_backtest_results_merged.csv"; src.rename(merged)
    monkeypatch.setattr(mod, "CSV", tmp_path / "default_not_used.csv"); monkeypatch.setattr(mod, "OUT", tmp_path / "default_out.md")
    assert mod.main(["follow_adp", "bot", str(merged)]) == 0
    assert (tmp_path / "WHY_REPORT_merged.md").exists() and not (tmp_path / "default_out.md").exists()
    assert "| Assists | +100 |" in capsys.readouterr().out

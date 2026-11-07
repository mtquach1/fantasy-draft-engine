"""The Draft Room page (app/static/index.html) and the demo notice. The page is plain HTML and JavaScript with no build step, so these checks
are what stands between a typo and a blank screen: ids it uses must exist, server text must never be inserted as HTML, and every sport needs a logo."""
import os
import re
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from app import sports
from tests.pipeline_helpers import api

ROOT = Path(__file__).resolve().parents[1]
PAGE = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------ the notice
def test_session_carries_no_notice_unless_one_is_configured(monkeypatch, tmp_path):
    monkeypatch.delenv("DEMO_NOTICE", raising=False)
    with api(tmp_path, monkeypatch) as (c, main):
        assert c.get("/session").json()["notice"] == ""
        monkeypatch.setenv("DEMO_NOTICE", "   ")
        assert c.get("/session").json()["notice"] == ""                           # whitespace is nothing


def test_the_notice_is_trimmed_and_capped(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch) as (c, main):
        monkeypatch.setenv("DEMO_NOTICE", "  Sample data only.  ")
        assert c.get("/session").json()["notice"] == "Sample data only."
        monkeypatch.setenv("DEMO_NOTICE", "x" * 1000)
        assert len(c.get("/session").json()["notice"]) == 300


# ------------------------------------------------------------------------------------------------ the entrypoint
@pytest.mark.skipif(sys.platform == "win32" or shutil.which("sh") is None, reason="needs a POSIX shell")
class TestEntrypoint:
    def run(self, tmp_path, sport="nba", with_data=False, notice=None):
        shutil.copy(ROOT / "docker-entrypoint.sh", tmp_path / "docker-entrypoint.sh")
        bindir = tmp_path / "bin"; bindir.mkdir(exist_ok=True)
        stub = bindir / "python"
        stub.write_text('#!/bin/sh\necho "STUB $* | DEMO=[${DEMO_NOTICE:-}]"\n', encoding="utf-8")
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        if with_data:
            (tmp_path / "data" / sport).mkdir(parents=True, exist_ok=True); (tmp_path / "data" / sport / "players.csv").write_text("x", encoding="utf-8")
        env = {**os.environ, "PATH": f"{bindir}:{os.environ['PATH']}", "SPORT": sport}
        env.pop("DEMO_NOTICE", None)
        if notice is not None:
            env["DEMO_NOTICE"] = notice
        return subprocess.run(["sh", "docker-entrypoint.sh"], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)

    def test_invented_data_comes_with_a_notice(self, tmp_path):
        r = self.run(tmp_path)
        assert r.returncode == 0 and "generating FAKE sample data" in r.stdout and "make_sample_sport.py nba" in r.stdout
        assert "DEMO=[Sample data: the players and stats in this copy are invented" in r.stdout

    def test_real_data_means_no_notice(self, tmp_path):
        r = self.run(tmp_path, with_data=True)
        assert "generating FAKE" not in r.stdout and "DEMO=[]" in r.stdout

    def test_a_notice_you_set_is_kept(self, tmp_path):
        assert "DEMO=[mine]" in self.run(tmp_path, notice="mine").stdout

    def test_the_baseball_sample_data_also_comes_with_a_notice(self, tmp_path):
        r = self.run(tmp_path, sport="mlb")
        assert "make_sample_data.py" in r.stdout and "DEMO=[Sample data" in r.stdout


# ------------------------------------------------------------------------------------------------ the page
def test_server_text_is_never_inserted_as_html():
    for bad in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert bad not in PAGE, f"the page uses {bad}: server text could become markup"


def test_every_element_the_script_looks_up_exists():
    ids = set(re.findall(r'\bid="([\w-]+)"', PAGE))
    used = set(re.findall(r'\$\("([\w-]+)"\)', PAGE)) | set(re.findall(r'getElementById\("([\w-]+)"\)', PAGE))
    made_in_js = set(re.findall(r'\.id\s*=\s*"([\w-]+)"', PAGE))
    missing = sorted(used - ids - made_in_js)
    assert not missing, f"the script looks up ids that are not in the page: {missing}"


def test_ids_are_unique_and_labels_point_at_real_fields():
    ids = re.findall(r'\bid="([\w-]+)"', PAGE)
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    assert not dupes, f"duplicate ids: {dupes}"
    for target in re.findall(r'<label[^>]*\bfor="([\w-]+)"', PAGE):
        assert target in ids, f"a label points at a missing field: {target}"


def test_every_sport_has_a_logo():
    block = re.search(r"const LOGOS = \{([^}]*)\}", PAGE).group(1)
    have = set(re.findall(r"(\w+):", block))
    assert set(sports.SPORTS) <= have, f"no logo for: {sorted(set(sports.SPORTS) - have)}"


def test_messages_are_announced_and_the_notice_and_toggle_start_hidden():
    assert re.search(r'id="banner"[^>]*role="status"[^>]*aria-live="polite"', PAGE)
    assert re.search(r'id="demo"[^>]*\bhidden\b', PAGE) and re.search(r'id="setupToggle"[^>]*\bhidden\b', PAGE)


def test_the_drafting_bar_stays_in_view_on_wide_screens_only():
    assert re.search(r"@media\(min-width:781px\)\{#assign\{position:sticky", PAGE)


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_script_is_valid_javascript(tmp_path):
    js = re.search(r"<script>([\s\S]*)</script>", PAGE).group(1)
    f = tmp_path / "page.js"; f.write_text(js, encoding="utf-8")
    r = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr


def test_a_background_tab_only_polls_when_picks_can_arrive_from_outside():
    tick = re.search(r"const tick = async \(\) => \{([\s\S]*?)\n\};", PAGE).group(1)
    assert "document.hidden" in tick
    assert 'livePlatform && livePlatform !== "manual"' in tick               # manual mode: nothing can change behind the page's back
    assert "hiddenTicks % 3 === 0" in tick                                    # and then only every third tick (15 s)
    assert "live()" not in tick.split("hiddenTicks = 0")[0]                   # the heavy board refresh never runs while hidden


def test_coming_back_to_the_tab_refreshes_at_once():
    assert re.search(r'addEventListener\("visibilitychange",\s*\(\) => \{ if \(!document\.hidden\) tick\(\); \}\)', PAGE)


def test_the_tab_title_shows_the_clock_the_countdown_and_the_end():
    fn = re.search(r"function pageTitle\(\) \{([\s\S]*?)\n\}", PAGE).group(1)
    assert "YOU'RE ON THE CLOCK" in fn and "picks_until_my_turn" in fn and "Draft complete" in fn and "BASE_TITLE" in fn
    assert "const BASE_TITLE = document.title" in PAGE
    assert PAGE.count("applyTitle();") >= 2                                   # refreshed after a status load and after saving the setup


# ------------------------------------------------------------------------------------------------ the default sport
def test_the_app_starts_as_basketball_when_no_sport_is_set(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch, sport=None) as (c, main):                    # SPORT is genuinely absent from the environment
        assert "SPORT" not in os.environ
        assert c.get("/sports").json()["current"] == "nba" == sports.DEFAULT_SPORT


def test_a_blank_sport_means_the_default_and_an_explicit_one_still_wins(monkeypatch, tmp_path):
    with api(tmp_path, monkeypatch, sport="") as (c, main):                      # SPORT= (blank), as an untouched .env line would leave it
        assert c.get("/sports").json()["current"] == "nba"
    with api(tmp_path, monkeypatch, sport="nfl") as (c, main):
        assert c.get("/sports").json()["current"] == "nfl"


def test_the_example_settings_and_every_shipped_default_say_basketball():
    env = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"^SPORT=nba\b", env, re.M) and not re.search(r"^SPORT=(mlb|nfl)\b", env, re.M)
    assert 'SPORT="${SPORT:-nba}"' in (ROOT / "docker-entrypoint.sh").read_text(encoding="utf-8")
    assert "SPORT: nba" in (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert sports.DEFAULT_SPORT == "nba" and "nba" in sports.SPORTS


def test_starting_with_no_data_says_which_command_to_run(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import ingestion, persist, main
    for k in ("SPORT", "LEAGUE_PRESET", "MULTI_SESSION", "DRAFT_PLATFORM"):
        monkeypatch.delenv(k, raising=False)
    old_i, old_p = ingestion.DATA_DIR, persist.DATA_DIR
    ingestion.DATA_DIR = persist.DATA_DIR = tmp_path                              # an empty folder: no players.csv anywhere
    try:
        with pytest.raises(RuntimeError) as e:
            with TestClient(main.app):
                pass
    finally:
        ingestion.DATA_DIR, persist.DATA_DIR = old_i, old_p
    msg = str(e.value)
    assert "players.csv" in msg and "python make_sample_sport.py nba" in msg and "data/README.md" in msg and "make_sample_data.py" not in msg


def test_the_baseball_hint_points_at_the_baseball_generator(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from app import ingestion, persist, main
    for k in ("LEAGUE_PRESET", "MULTI_SESSION", "DRAFT_PLATFORM"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SPORT", "mlb")
    old_i, old_p = ingestion.DATA_DIR, persist.DATA_DIR
    ingestion.DATA_DIR = persist.DATA_DIR = tmp_path
    try:
        with pytest.raises(RuntimeError) as e:
            with TestClient(main.app):
                pass
    finally:
        ingestion.DATA_DIR, persist.DATA_DIR = old_i, old_p
    assert "python make_sample_data.py" in str(e.value)

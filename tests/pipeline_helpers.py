"""Shared builders for the pipeline tests: a valid Basketball-Reference-style table, a file writer, and a throwaway running app.
Nothing here touches your real data: every test works inside a temporary folder."""
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

import make_sample_sport
from app import ingestion, persist

TEAMS = ["AAA", "BBB", "CCC"]


def br_frame(n=250, shift=0.0):
    """A valid per-game table: n players, one team each, plus the League Average row Basketball-Reference ends its files with."""
    rows = []
    for i in range(n):
        rows.append({"Player": f"Player {i:03d}", "Age": 20 + i % 15, "Team": TEAMS[i % 3], "Pos": "SF", "G": 50 + i % 30, "MP": 15.0 + i % 20,
                     "FG": 3.1, "FGA": 7.0, "FT": 1.5, "FTA": 1.9, "3P": 1.0, "TRB": 4.0, "AST": 3.0, "STL": 0.8, "BLK": 0.5, "TOV": 1.5,
                     "PTS": 10.0 + i % 15 + shift, "Player-additional": f"pl{i:03d}"})
    rows.append({"Player": "League Average", "Age": np.nan, "Team": np.nan, "Pos": np.nan, "G": np.nan, "MP": 24.0, "FG": 3.0, "FGA": 6.5, "FT": 1.4,
                 "FTA": 1.8, "3P": 1.0, "TRB": 4.0, "AST": 2.8, "STL": 0.7, "BLK": 0.4, "TOV": 1.4, "PTS": 11.0, "Player-additional": np.nan})
    return pd.DataFrame(rows)


def with_traded_player(df):
    """Add a player who changed teams mid-season: a combined '2TM' row and one row per team (the way the real files look)."""
    base = df.iloc[0].to_dict()
    extra = []
    for team in ("2TM", "AAA", "BBB"):
        r = dict(base); r.update({"Player": "Trade Guy", "Player-additional": "tr001", "Team": team}); extra.append(r)
    return pd.concat([df.iloc[:-1], pd.DataFrame(extra), df.iloc[[-1]]], ignore_index=True)


def csv_text(df) -> str:
    return df.to_csv(index=False).replace("\r\n", "\n")          # pandas uses the OS line ending; the tests want one thing everywhere


def write_br(path, df, bom=False, crlf=False) -> str:
    text = csv_text(df)
    raw = text.replace("\n", "\r\n") if crlf else text
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes((("\ufeff" if bom else "") + raw).encode("utf-8"))
    return text


@contextmanager
def api(tmp, monkeypatch, sport="nba", multi=False):
    """A running app on sample data inside `tmp`. Yields (client, app.main module). Re-entering with the same tmp = a restart."""
    for k, v in {"SPORT": sport, "LEAGUE_PRESET": "", "DRAFT_PLATFORM": "manual", "MY_TEAM_ID": "1", "MULTI_SESSION": "1" if multi else ""}.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)              # sport=None: start the app with SPORT not set at all
        else:
            monkeypatch.setenv(k, v)
    old_i, old_p = ingestion.DATA_DIR, persist.DATA_DIR
    ingestion.DATA_DIR = persist.DATA_DIR = Path(tmp)
    try:
        for sp in ("nba", "nfl"):
            if not (Path(tmp) / sp / "players.csv").exists():
                make_sample_sport.generate(sp, Path(tmp) / sp, n=300)
        from fastapi.testclient import TestClient
        from app import main
        with TestClient(main.app) as c:
            yield c, main
    finally:
        ingestion.DATA_DIR, persist.DATA_DIR = old_i, old_p


def board(c):
    rows = c.get("/players/rankings?limit=300").json()
    rows = rows if isinstance(rows, list) else next(v for v in rows.values() if isinstance(v, list))
    return rows


def explain(c, name):
    return c.get("/players/explain", params={"q": name}).json()

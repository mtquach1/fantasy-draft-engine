"""Ingest: run a source through the quality gate and, if it passes, store it as a new versioned snapshot (plus crosswalk links and team
facts). Running it again on unchanged input stores nothing."""
from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from .quality import check_br_per_game, check_nba_api
from .sources import BRPerGameCsv, NbaApiSource, Source, SourceError
from .store import Store

_MULTI = re.compile(r"^\d+TM$")


def player_key(name) -> str:
    from app.own_model import norm_name                 # one definition of "the same player" for the whole project
    return norm_name(name)


def app_player_id(name) -> str:
    return "n:" + str(name).lower()                    # how the app identifies players loaded from the model's CSV


def _link_and_record_teams(store: Store, df: pd.DataFrame, season: int) -> dict:
    """Crosswalk links for every player, and team facts for players who stayed on one team all season."""
    linked = facts = 0
    team_col = "Team" if "Team" in df.columns else "Tm"
    start = f"{season - 1}-10-01T00:00:00.000000Z"            # a season's team facts take effect at its (approximate) start
    for pid, grp in df.dropna(subset=["Player-additional"]).groupby("Player-additional"):
        name = grp["Player"].iloc[0]
        store.link_id("br", str(pid), player_key(name))
        linked += 1
        single = grp[~grp[team_col].astype(str).str.match(_MULTI)]
        if len(grp) == 1 and len(single) == 1:
            if store.add_team_fact(app_player_id(name), str(single[team_col].iloc[0]), start, "br_per_game"):
                facts += 1
    return {"linked": linked, "team_facts": facts}


def ingest_source(store: Store, source: Source, checker, season: int | None = None) -> dict:
    """Returns {"kind", "status": stored|unchanged|rejected|failed, "snapshot_id", "quality": {...}, ...}."""
    out = {"kind": source.kind, "source": source.name}
    try:
        text = source.fetch_text()
        df = pd.read_csv(io.StringIO(text))
    except (SourceError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
        return {**out, "status": "failed", "error": str(e), "snapshot_id": None}
    prev = store.current_snapshot(source.kind)
    report = checker(df, prev["quality"] if prev else None)       # the checker compares against the stored quality report (it holds the row and column counts)
    out["quality"] = report.to_dict()
    if not report.ok:
        return {**out, "status": "rejected", "snapshot_id": None}
    snap_id, created = store.add_snapshot(source.kind, source.name, text, int(len(df)), report.to_dict())
    out.update(status="stored" if created else "unchanged", snapshot_id=snap_id)
    if created and isinstance(source, BRPerGameCsv):
        out.update(_link_and_record_teams(store, df, source.season))
    elif created and isinstance(source, NbaApiSource):
        for pid, name in zip(df["PLAYER_ID"], df["PLAYER_NAME"]):
            store.link_id("nba_api", str(pid), player_key(name))
    return out


def ingest_raw_dir(store: Store, raw_dir) -> list[dict]:
    """Ingest every per_game_<year>.csv in a folder (the files build_own_model.py reads)."""
    from app.own_model import find_seasons
    files = find_seasons(raw_dir)
    if not files:
        raise SourceError(f"No per_game_<year>.csv files found in {raw_dir}.")
    return [ingest_source(store, BRPerGameCsv(p, y), check_br_per_game, y) for y, p in files.items()]


def materialize(store: Store, out_dir) -> list[str]:
    """Write the CURRENT snapshot of every per-game table back to files, so build_own_model.py can use them (this is how a rollback takes effect)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for s in store.snapshots():
        if s["current"] and s["kind"].startswith("per_game_"):
            path = out_dir / f"{s['kind']}.csv"
            path.write_text(store.snapshot_csv(s["id"]), encoding="utf-8", newline="\n")
            written.append(str(path))
    return written

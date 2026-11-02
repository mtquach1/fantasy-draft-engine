"""The quality gate between a source and the store. It refuses a table that would silently corrupt the model (a missing column, an impossible
value) and warns about anything that merely looks different from last time (new columns, a big change in row count)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

BR_REQUIRED = ["Player", "Age", "Pos", "G", "MP", "FG", "FGA", "FT", "FTA", "3P", "TRB", "AST", "STL", "BLK", "TOV", "PTS", "Player-additional"]
BR_TEAM_COLUMNS = ("Team", "Tm")
BR_RANGES = {"G": (0, 82), "MP": (0, 48), "PTS": (0, 60), "Age": (15, 55)}
NBA_API_REQUIRED = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FGA", "FTM", "FTA", "FG3M"]
NBA_API_RANGES = {"GP": (0, 82), "MIN": (0, 48), "PTS": (0, 60)}


@dataclass
class QualityReport:
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"ok": self.ok, "errors": self.errors, "warnings": self.warnings, "stats": self.stats}


def check_table(df: pd.DataFrame, *, required: list, ranges: dict, id_col: str, team_cols: tuple = (), previous: dict | None = None,
                min_rows: int = 200) -> QualityReport:
    rep = QualityReport(stats={"rows": int(len(df)), "columns": [str(c) for c in df.columns]})
    cols = set(df.columns)
    missing = [c for c in required if c not in cols]
    if team_cols and not any(t in cols for t in team_cols):
        missing.append(" or ".join(team_cols))
    if missing:
        rep.errors.append(f"Schema drift: the table is missing required column(s): {', '.join(missing)}. The source's layout has changed.")
        return rep                                              # nothing further can be checked reliably
    extra = [c for c in df.columns if c not in required and c not in team_cols]
    if previous:
        before = set(previous.get("stats", {}).get("columns", []))
        if before and before != cols:
            rep.warnings.append(f"Columns changed since the current snapshot: added {sorted(cols - before)}, removed {sorted(before - cols)}.")
    elif extra:
        rep.stats["extra_columns"] = len(extra)
    if len(df) < min_rows:
        rep.warnings.append(f"Only {len(df)} rows (expected at least {min_rows}). Is this the full table?")
    if previous:
        prev_rows = previous.get("stats", {}).get("rows")
        if prev_rows and abs(len(df) - prev_rows) / prev_rows > 0.25:
            rep.warnings.append(f"Row count moved from {prev_rows} to {len(df)} (more than 25%).")
    team = next((t for t in team_cols if t in cols), None)
    key = [id_col] + ([team] if team else [])
    dup = df.duplicated(subset=key, keep=False)
    if dup.any():
        rep.errors.append(f"{int(dup.sum())} rows share the same {' + '.join(key)} (duplicates). Example: {df.loc[dup, id_col].iloc[0]}.")
    for col, (lo, hi) in ranges.items():
        nums = pd.to_numeric(df[col], errors="coerce")
        bad_type = int(nums.isna().sum() - df[col].isna().sum())
        if bad_type:
            (rep.errors if bad_type / max(len(df), 1) > 0.2 else rep.warnings).append(f"Column {col}: {bad_type} value(s) aren't numbers.")
        out = nums[(nums < lo) | (nums > hi)]
        if len(out):
            rep.errors.append(f"Column {col}: {len(out)} value(s) outside {lo} to {hi} (for example {out.iloc[0]}).")
    stat_cols = [c for c in required if c not in ("Player", "Pos", "Player-additional", "PLAYER_NAME", "TEAM_ABBREVIATION", "Age", "PLAYER_ID")]
    negatives = [c for c in stat_cols if (pd.to_numeric(df[c], errors="coerce") < 0).any()]
    if negatives:
        rep.errors.append(f"Negative values in: {', '.join(negatives)}.")
    return rep


def check_br_per_game(df: pd.DataFrame, previous: dict | None = None) -> QualityReport:
    rep = check_table(df, required=BR_REQUIRED, ranges=BR_RANGES, id_col="Player-additional", team_cols=BR_TEAM_COLUMNS, previous=previous)
    # Basketball-Reference ends the table with a League Average row that has no player id: expected, not a defect
    return rep


def check_nba_api(df: pd.DataFrame, previous: dict | None = None) -> QualityReport:
    return check_table(df, required=NBA_API_REQUIRED, ranges=NBA_API_RANGES, id_col="PLAYER_ID", previous=previous)

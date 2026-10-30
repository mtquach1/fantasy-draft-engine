
"""Load-time cleaning + data-health report. Real projection files are messy; this makes the mess visible
(and safe) instead of letting it turn into NaN scores or a 500 error mid-draft. Works for every sport."""
from __future__ import annotations

import pandas as pd

from .config import LeagueConfig, ScoringType
from .sports import Group, get_sport
from .valuation import position_demand


def _cols_for(entry: dict, group: Group) -> list[str]:
    names = [entry["col"]] if entry["kind"] == "count" else [entry["num"], entry["den"]]
    return [c for n in names for c in group.derived.get(n, [n])]


def _has(rows: pd.DataFrame, col: str) -> bool:
    return col in rows.columns and bool(rows[col].notna().any())


def _single(g: Group) -> str:
    return g.label[:-1]                                        # "hitters" -> "hitter"


def check_config(raw: pd.DataFrame, cfg: LeagueConfig) -> list[str]:
    """Problems that would make valuation fail for this league format. Used to reject bad PUT /league/config."""
    sp = get_sport(cfg.sport)
    if cfg.scoring_type != ScoringType.POINTS and not sp.has_categories:
        return [f"{sp.label} supports points scoring only in this version: set scoring_type to 'points'"]
    problems = []
    for g in sp.groups:
        rows = raw[raw["kind"] == g.kind]
        if rows.empty:
            problems.append(f"No {g.label} in the projections")
            continue
        if cfg.scoring_type == ScoringType.POINTS:
            if not any(_has(rows, c) for c in cfg.point_weights):
                problems.append(f"Points scoring: none of your point_weights match a {_single(g)} column")
            continue
        for c in cfg.categories.get(g.kind, []):
            if c not in g.cats:
                problems.append(f"Unsupported {_single(g)} category '{c}' (supported: {', '.join(g.cats)})")
                continue
            for col in _cols_for(g.cats[c], g):
                if not _has(rows, col):
                    problems.append(f"Category '{c}' needs column '{col}' in your {g.label} projections file")
    return problems


def clean_projections(raw: pd.DataFrame, sport: str = "mlb") -> tuple[pd.DataFrame, list[str]]:
    """Fix what is safely fixable and say exactly what was changed."""
    sp = get_sport(sport)
    df, actions = raw.copy(), []
    df["player_id"] = df["player_id"].astype(str).str.strip()
    blank = df["player_id"].isin(["", "nan", "None"])
    if blank.any():
        actions.append(f"Dropped {int(blank.sum())} row(s) with no player id")
        df = df[~blank]
    dup = df.duplicated(["player_id", "kind"], keep="first")
    if dup.any():
        names = ", ".join(df.loc[dup, "name"].astype(str).head(5))
        actions.append(f"Dropped {int(dup.sum())} duplicate row(s) (kept the first): {names}")
        df = df[~dup]
    for g in sp.groups:
        m = df["kind"] == g.kind
        for c in g.components:
            if not _has(df[m], c):
                continue                                        # column not in this file: left alone (check_config reports it)
            num = pd.to_numeric(df[c], errors="coerce")
            fix = m & num.isna()
            if fix.any():
                actions.append(f"{g.label} column {c}: {int(fix.sum())} blank/non-numeric value(s) set to 0")
            df[c] = num.where(~fix, 0.0)
    return df.reset_index(drop=True), actions


def _names(df: pd.DataFrame, mask: pd.Series, n: int = 5) -> str:
    return ", ".join(df.loc[mask, "name"].astype(str).head(n))


def _sanity(sp, rows: dict, warnings: list) -> None:
    """Numbers that look wrong (wrong units, shifted columns, mixed-up files)."""
    if sp.key == "mlb":
        h, p = rows["H"], rows["P"]
        if {"H", "AB"} <= set(h.columns):
            bad = (h["H"] > h["AB"]) | ((h["AB"] >= 100) & (h["H"] / h["AB"].replace(0, float("nan")) > 0.45))
            if bad.any():
                warnings.append(f"{int(bad.sum())} hitter(s) with impossible hits/at-bats (e.g. {_names(h, bad)}). "
                                f"Check that columns are not shifted")
        if "AB" in h.columns and (h["AB"] > 800).any():
            warnings.append(f"{int((h['AB'] > 800).sum())} hitter(s) with more than 800 at-bats (e.g. {_names(h, h['AB'] > 800)})")
        if {"ER", "IP"} <= set(p.columns):
            bad = (p["IP"] >= 20) & (p["ER"] * 9 / p["IP"].replace(0, float("nan")) > 12)
            if bad.any():
                warnings.append(f"{int(bad.sum())} pitcher(s) with an ERA above 12 (e.g. {_names(p, bad)})")
        if "IP" in p.columns and (p["IP"] > 300).any():
            warnings.append(f"{int((p['IP'] > 300).sum())} pitcher(s) with more than 300 innings (e.g. {_names(p, p['IP'] > 300)})")
        if "AB" in h.columns and (h["AB"] <= 0).any():
            warnings.append(f"{int((h['AB'] <= 0).sum())} hitter(s) with 0 at-bats (kept; rate stats treat them as 0)")
        if "IP" in p.columns and (p["IP"] <= 0).any():
            warnings.append(f"{int((p['IP'] <= 0).sum())} pitcher(s) with 0 innings (kept; rate stats treat them as 0)")
    elif sp.key == "nba":
        x = rows["X"]
        for made, tried, what in (("FGM", "FGA", "field-goal"), ("FTM", "FTA", "free-throw")):
            if {made, tried} <= set(x.columns) and (x[made] > x[tried]).any():
                bad = x[made] > x[tried]
                warnings.append(f"{int(bad.sum())} player(s) with impossible {what} lines (e.g. {_names(x, bad)})")
        if "GP" in x.columns and (x["GP"] > 82).any():
            warnings.append(f"{int((x['GP'] > 82).sum())} player(s) projected for more than 82 games")
    elif sp.key == "nfl":
        x = rows["X"]
        if {"pass_cmp", "pass_att"} <= set(x.columns) and (x["pass_cmp"] > x["pass_att"]).any():
            bad = x["pass_cmp"] > x["pass_att"]
            warnings.append(f"{int(bad.sum())} player(s) with more completions than attempts (e.g. {_names(x, bad)})")
        if "G" in x.columns and (x["G"] > 17).any():
            warnings.append(f"{int((x['G'] > 17).sum())} player(s) projected for more than 17 games")


def build_report(raw: pd.DataFrame, cfg: LeagueConfig, history: dict | None = None,
                 actions: list[str] | None = None) -> dict:
    sp = get_sport(cfg.sport)
    errors, warnings, info = check_config(raw, cfg), [], {"sport": sp.label, "group_labels": [g.label for g in sp.groups]}
    rows = {g.kind: raw[raw["kind"] == g.kind] for g in sp.groups}
    for g in sp.groups:
        info[g.label] = len(rows[g.kind])
    info["two_way_players"] = int(raw["twin_id"].notna().sum() // 2) if "twin_id" in raw else 0
    if raw["player_id"].duplicated().any():
        errors.append("Duplicate player_id values remain; rankings and picks would be unreliable")

    # can the league actually fill its starting lineup from this player pool?
    cov = {}
    for pos, need in position_demand(cfg.roster, sp).items():
        n = int(raw["eligible_positions"].map(lambda ps, pos=pos: pos in ps).sum())
        cov[pos] = {"eligible_players": n, "league_starter_demand": need}
        if need and n < need:
            errors.append(f"Only {n} players are eligible at {pos} but your league starts about {need}. "
                          f"Is your positions file incomplete?")
        elif need and n < 1.5 * need:
            warnings.append(f"Thin pool at {pos}: {n} eligible for about {need} starter slots")
    info["position_coverage"] = cov
    _sanity(sp, rows, warnings)

    # optional history files (injury-prone flag and the player search)
    history = history or {}
    for g in sp.groups:
        hist, grp = history.get(g.kind), rows[g.kind]
        name = _single(g)
        if hist is None or hist.empty:
            info[f"{name}_history"] = "not loaded"
            continue
        ids = grp["base_id"] if "base_id" in grp else grp["player_id"]
        share = float(ids.isin(set(hist["player_id"])).mean()) if len(grp) else 0.0
        info[f"{name}_history"] = f"{share:.0%} of {g.label} have past-season stats"
        if share < 0.5:
            warnings.append(f"Only {share:.0%} of {g.label} match your {g.history_file}: check that its playerid "
                            f"values are the same ids as your projections")
    if "injury_seasons_used" in raw:
        g0 = sp.groups[0]
        info[f"{g0.label}_with_2plus_seasons_of_games_history"] = int((rows[g0.kind]["injury_seasons_used"] >= 2).sum())
    return {"ok": not errors, "errors": errors, "warnings": warnings, "info": info, "cleaning": actions or []}


def print_summary(report: dict) -> None:
    i = report["info"]
    counts = ", ".join(f"{i[k]} {k}" for k in i.get("group_labels", []))
    print(f"[data] {i.get('sport', '')}: {counts}, {i['two_way_players']} two-way player(s)")
    for line in report["cleaning"]:
        print(f"[data] cleaned: {line}")
    for line in report["errors"]:
        print(f"[data] ERROR: {line}")
    for line in report["warnings"]:
        print(f"[data] warning: {line}")
    print("[data] OK" if report["ok"] else "[data] problems found: see GET /health/data")





def check_roster(cfg: LeagueConfig) -> tuple[list[str], list[str]]:
    """(errors, warnings) for the roster slots the user typed. Errors reject the save; warnings are shown."""
    sp = get_sport(cfg.sport)
    slots, errs, warns = cfg.roster.slots, [], []
    valid = ", ".join(sp.slot_rules)
    for name, n in slots.items():
        if name not in sp.slot_rules:
            errs.append(f"'{name}' is not a {sp.label} roster slot. Valid slots: {valid}.")
        elif not isinstance(n, int) or isinstance(n, bool) or not 0 <= n <= 20:
            errs.append(f"{name} must be a whole number from 0 to 20.")
    if errs:
        return errs, warns
    starters = sum(n for s, n in slots.items() if s != "BN")
    if starters < 1:
        errs.append("You need at least one starting slot.")
    if sum(slots.values()) > 60:
        errs.append("A roster can have at most 60 spots in total.")
    if not 0 <= cfg.roster.il_slots <= 10:
        errs.append("Injured-list slots must be from 0 to 10.")
    pos = {p for g in sp.groups for p in g.positions if p != "DH"}
    for p in sorted(pos):
        if not any(n and s != "BN" and (not sp.rule(s).accepts or p in sp.rule(s).accepts) for s, n in slots.items()):
            warns.append(f"No starting slot accepts {p}, so {p} players can only sit on your bench.")
    if starters and not slots.get("BN"):
        warns.append("You have no bench spots: every pick must fill a starting slot.")
    return errs, warns

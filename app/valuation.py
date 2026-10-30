
"""One valuation engine, three league formats.

ROTO            player score = weighted sum of category z-scores (rate stats volume-weighted).
H2H_CATEGORIES  same player scores; team value (for /recommend) is the sum of P(win category)
                vs an average opponent, so it has diminishing returns and rewards punting.
POINTS          player score = projected points. No z-scores, no pool needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr

from .config import LeagueConfig, RosterConfig, ScoringType
from .sports import HIT_CATS, PIT_CATS, RATE_COLUMNS, Sport, get_sport    # noqa: F401  (HIT_CATS/PIT_CATS re-exported)


# ---------- helpers ----------
def _prep(df: pd.DataFrame, group) -> pd.DataFrame:
    d = df.copy()
    return group.prep(d) if group.prep else d


def _scale_playing_time(d: pd.DataFrame) -> pd.DataFrame:
    """Scale projected STATS by games_pct before any z-scoring (never scale a z-score)."""
    skip = {"player_id", "name", "team", "clean_name", "kind", "eligible_positions", "is_injured", "games_pct"} | RATE_COLUMNS
    out = d.copy()
    for c in out.columns:
        if c not in skip and pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c] * out["games_pct"]
    return out


def zscore(s: pd.Series, pool_mask: pd.Series) -> pd.Series:
    mu, sd = s[pool_mask].mean(), s[pool_mask].std(ddof=0)
    return (s - mu) / sd if sd and sd > 0 else s * 0.0


def rate_impact(df, num, den, lg_rate, invert=False) -> pd.Series:
    """Marginal contribution vs a league-average player with the same volume."""
    rate = df[num] / df[den].replace(0, np.nan)
    diff = (lg_rate - rate) if invert else (rate - lg_rate)
    return (diff * df[den]).fillna(0.0)


def _z_pass(d: pd.DataFrame, cats: list[str], spec: dict, weights: dict, mask: pd.Series) -> pd.DataFrame:
    z = pd.DataFrame(index=d.index)
    for c in cats:
        if c not in spec:
            raise ValueError(f"Unsupported category '{c}'. Supported: {sorted(spec)}")
        s = spec[c]
        needed = [s.get("col")] if s["kind"] == "count" else [s["num"], s["den"]]
        missing = [n for n in needed if n not in d]
        if missing:
            raise ValueError(f"Category '{c}' needs column(s) {missing} in your projections")
        if s["kind"] == "count":
            x = d[s["col"]].astype(float)
            x = -x if s.get("invert") else x
        else:
            lg = d.loc[mask, s["num"]].sum() / max(d.loc[mask, s["den"]].sum(), 1e-9)
            x = rate_impact(d, s["num"], s["den"], lg, invert=s.get("invert", False))
        z[f"z_{c}"] = zscore(x, mask)
    return z


def _category_scores(d, cats, spec, weights, pool_n) -> tuple[pd.Series, pd.DataFrame]:
    """Two passes: z-score against everyone, take the top pool_n, then re-z against that pool."""
    w = np.array([weights.get(c, 1.0) for c in cats])
    mask = pd.Series(True, index=d.index)
    z = _z_pass(d, cats, spec, weights, mask)
    pre = (z.to_numpy() * w).sum(axis=1)
    mask = pd.Series(False, index=d.index)
    mask[pd.Series(pre, index=d.index).nlargest(min(pool_n, len(d))).index] = True
    z = _z_pass(d, cats, spec, weights, mask)
    return pd.Series((z.to_numpy() * w).sum(axis=1), index=d.index), z


def _points_scores(d: pd.DataFrame, weights: dict) -> pd.Series:
    cols = [c for c in weights if c in d.columns]
    # hitters carry NaN in pitcher-only columns (and vice versa) after the concat, so treat NaN as 0
    return sum((d[c].fillna(0.0) * weights[c] for c in cols), pd.Series(0.0, index=d.index))


def flag_injury_prone(d: pd.DataFrame, cfg: LeagueConfig) -> pd.Series:
    """Prone = avg games missed >= threshold with enough seasons of history. The threshold lives in
    the config, so changing it takes effect immediately (no re-ingest)."""
    avg = d["avg_games_missed_pct"] if "avg_games_missed_pct" in d else pd.Series(np.nan, index=d.index)
    used = d["injury_seasons_used"] if "injury_seasons_used" in d else pd.Series(0, index=d.index)
    return ((avg >= cfg.injury_prone_threshold) & (used >= cfg.injury_prone_min_seasons)).fillna(False)


def il_adjust(value: pd.Series, is_injured: pd.Series, open_il_slots: int, cfg: LeagueConfig) -> pd.Series:
    """Penalize only the POSITIVE part of value so a negative score never improves."""
    pen = cfg.il_penalty_open_slot if open_il_slots > 0 else cfg.il_penalty_no_slot
    return value - np.where(is_injured.astype(bool), value.clip(lower=0) * (1 - pen), 0.0)


# ---------- positional demand & replacement ----------
def position_demand(r: RosterConfig, sport: Sport | None = None) -> dict[str, int]:
    """League-wide starter demand per position. Flex slots are spread over the positions they accept:
    'even' split, or 'proportional' to each position's dedicated demand (UTIL, football FLEX)."""
    sport = sport or get_sport("mlb")
    d = {p: 0.0 for g in sport.groups for p in g.positions if p != "DH"}
    flex = []
    for slot, n in r.slots.items():
        rule = sport.rule(slot)
        if not n or not rule.accepts:
            continue
        if len(rule.accepts) == 1 and not rule.pool:
            d[rule.accepts[0]] = d.get(rule.accepts[0], 0.0) + n
        else:
            flex.append((rule, n))
    for phase in ("even", "proportional"):
        for rule, n in flex:
            if rule.spread != phase:
                continue
            targets = [p for p in rule.accepts if p in d]
            tot = sum(d[p] for p in targets)
            for p in targets:
                d[p] += n * (d[p] / tot if phase == "proportional" and tot else 1 / len(targets))
    return {k: int(round(v * r.teams)) for k, v in d.items() if v}


def _nth(scores: np.ndarray, n: int, fallback: float) -> float:
    """Score of the (n+1)th best player; short pool -> worst player; empty -> fallback.
    (Never 0.0: in z-space 0 means league-average, far above replacement.)"""
    if len(scores) == 0:
        return fallback
    s = np.sort(scores)[::-1]
    return float(s[n]) if len(s) > n else float(s[-1])


def replacement_levels(df: pd.DataFrame, cfg: LeagueConfig) -> dict[str, float]:
    """Dynamic replacement: remaining demand shrinks as players are drafted, and the level is
    read from the REMAINING pool. Recomputed from scratch on every call."""
    r, sport = cfg.roster, get_sport(cfg.sport)
    demand = position_demand(r, sport)
    floor = float(df["score"].min())
    elig = lambda pos: df["eligible_positions"].map(lambda ps: pos in ps)

    # pass 1: static levels over the full pool -> each player's primary position
    lvl0 = {pos: _nth(df.loc[elig(pos), "score"].to_numpy(), n, floor) for pos, n in demand.items()}

    def primary(row):
        opts = {p: row["score"] - lvl0[p] for p in row["eligible_positions"] if p in lvl0}
        return max(opts, key=opts.get) if opts else None

    drafted = df[~df["available"]]
    drafted_counts = drafted.apply(primary, axis=1).value_counts().to_dict() if len(drafted) else {}

    # pass 2: levels from the remaining pool using remaining demand
    avail = df[df["available"] & ~df["avoid"]]     # players you would never draft do not set the floor
    repl = {}
    for pos, n in demand.items():
        remaining = max(n - drafted_counts.get(pos, 0), 0)
        sc = avail.loc[avail["eligible_positions"].map(lambda ps: pos in ps), "score"].to_numpy()
        repl[pos] = _nth(sc, remaining, floor)

    for slot, rule in sport.slot_rules.items():          # pooled slots (UTIL, FLEX, P ...): one shared level
        if not rule.pool:
            continue
        accepted = set(rule.accepts)
        mine = lambda ps: bool(accepted & set(ps))
        supply = sum(n for s, n in r.slots.items() if (q := sport.slot_rules.get(s)) and q.accepts and set(q.accepts) <= accepted)
        taken = int((~df["available"] & df["eligible_positions"].map(mine)).sum())
        pool = avail.loc[avail["eligible_positions"].map(mine), "score"].to_numpy()
        repl[slot] = _nth(pool, max(supply * r.teams - taken, 0), floor)
    return repl


def compute_vorp(df: pd.DataFrame, repl: dict[str, float], sport: Sport | None = None) -> pd.DataFrame:
    """VORP at EVERY eligible position; keep the max. A position with no dedicated slot (MLB 'DH') is measured
    against the pooled slot that accepts it (UTIL)."""
    sport = sport or get_sport("mlb")
    out = df.copy()
    pooled = {p: slot for slot, rule in sport.slot_rules.items() if rule.pool for p in rule.accepts}

    def by_pos(row):
        res = {}
        for p in row["eligible_positions"]:
            base = repl.get(p, repl.get(pooled.get(p), min(repl.values())))
            res[p] = float(row["score"] - base)
        return res

    out["vorp_by_pos"] = out.apply(by_pos, axis=1)
    out["vorp"] = out["vorp_by_pos"].map(lambda d: max(d.values()) if d else 0.0)
    out["primary_pos"] = out["vorp_by_pos"].map(lambda d: max(d, key=d.get) if d else None)
    return out


def expand_twins(raw: pd.DataFrame, ids) -> set:
    """A two-way player (hitter row + pitcher row) is ONE human: drafting either row removes both."""
    ids = set(ids)
    if not ids or "twin_id" not in raw.columns:
        return ids
    return ids | set(raw.loc[raw["player_id"].isin(ids), "twin_id"].dropna())


# ---------- main entry ----------
def calculate_player_valuations(raw: pd.DataFrame, cfg: LeagueConfig, drafted_ids=frozenset(),
                                my_open_il_slots: int = 0) -> tuple[pd.DataFrame, dict]:
    """Pure function: (raw projections, config, pick log) -> (valuated DataFrame, replacement levels)."""
    r, sport = cfg.roster, get_sport(cfg.sport)
    drafted_ids = expand_twins(raw, drafted_ids)
    parts = []
    for group in sport.groups:
        d = raw[raw["kind"] == group.kind]
        if d.empty:
            continue
        pool_n = r.teams * sport.group_starters(r.slots, group.kind)
        d = _scale_playing_time(_prep(d, group))
        if cfg.scoring_type == ScoringType.POINTS:
            score, z = _points_scores(d, cfg.point_weights), pd.DataFrame(index=d.index)
        else:
            score, z = _category_scores(d, cfg.categories.get(group.kind, []), group.cats, cfg.category_weights, pool_n)
        d = pd.concat([d, z], axis=1)
        d["injury_prone"] = flag_injury_prone(d, cfg)
        # optional discount on the POSITIVE part only (default penalty 0 = flag only)
        score = score - score.clip(lower=0) * cfg.injury_prone_penalty * d["injury_prone"]
        d["score"] = il_adjust(score, d["is_injured"], my_open_il_slots, cfg)
        parts.append(d)

    df = pd.concat(parts, ignore_index=True)
    z_cols = [c for c in df.columns if c.startswith("z_")]
    df[z_cols] = df[z_cols].fillna(0.0)
    df["available"] = ~df["player_id"].isin(set(drafted_ids))
    df["avoid"] = df["injury_prone"] & cfg.avoid_injury_prone
    repl = replacement_levels(df, cfg)
    return compute_vorp(df, repl, sport), repl


# ---------- rankings ----------
def add_ranks(df: pd.DataFrame, predraft: pd.DataFrame) -> pd.DataFrame:
    """Adds three rank columns.
    projected_rank  overall rank by VORP from the PRE-DRAFT valuation (nobody drafted), so it stays
                    fixed while you draft. It changes only if you change the league format/settings.
    pos_rank        rank within primary position from the same pre-draft board, e.g. 'SS4'.
    available_rank  rank among players still undrafted, by CURRENT VORP (None once drafted).
    Ties break on name so ranks are deterministic."""
    pre = predraft.sort_values(["vorp", "name"], ascending=[False, True], kind="stable").reset_index(drop=True)
    pre["projected_rank"] = np.arange(1, len(pre) + 1)
    pre["pos_rank"] = pre["primary_pos"].astype(str) + (pre.groupby("primary_pos").cumcount() + 1).astype(str)
    out = df.merge(pre[["player_id", "projected_rank", "pos_rank"]], on="player_id", how="left")
    avail = out[out["available"]].sort_values(["vorp", "name"], ascending=[False, True], kind="stable")
    out["available_rank"] = np.nan
    out.loc[avail.index, "available_rank"] = np.arange(1, len(avail) + 1)
    return out


# ---------- team-level value (used by /recommend) ----------
def team_value(starters: pd.DataFrame, cfg: LeagueConfig) -> float:
    """Value of a set of STARTING players for this league format."""
    if starters.empty:
        return 0.0
    if cfg.scoring_type in (ScoringType.POINTS, ScoringType.ROTO):
        # points: season points; roto: z-scores are additive, so total z ~ standings gain
        return float(starters["score"].sum())
    # H2H categories: sum of P(win category) vs an average opponent. Concave, so a category
    # you already dominate adds little and a hopeless one adds little: that's how punting emerges.
    total = 0.0
    for kind, cats in cfg.categories.items():
        g = starters[starters["kind"] == kind]
        if g.empty:
            continue
        for c in cats:
            margin = g[f"z_{c}"].sum() / np.sqrt(len(g))
            total += cfg.category_weights.get(c, 1.0) * float(ndtr(margin / cfg.h2h_noise))
    return total




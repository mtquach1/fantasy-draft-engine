

"""JSON-safe views of the valuated DataFrame (kept out of main.py so they can be unit-tested)."""
import pandas as pd

from .config import LeagueConfig
from .ingestion import clean_name

AVAILABLE_COLS = ["player_id", "name", "projected_rank", "pos_rank", "available_rank", "kind", "primary_pos",
                  "eligible_positions", "vorp",
                  "injury_prone", "avg_games_missed_pct", "injury_seasons_used",
                  "is_injured", "games_pct", "bid", "pre_bid"]       # bid/pre_bid exist only in an auction
INJURY_COLS = ["player_id", "name", "projected_rank", "pos_rank", "available_rank", "kind", "primary_pos",
               "eligible_positions", "vorp", "available",
               "injury_prone", "avg_games_missed_pct", "injury_seasons_used", "is_injured", "games_pct"]


INT_COLS = ("injury_seasons_used", "projected_rank", "available_rank", "bid", "pre_bid")
RANKING_COLS = ["projected_rank", "pos_rank", "available_rank", "player_id", "name", "kind", "primary_pos",
                "eligible_positions", "vorp", "available", "injury_prone", "is_injured"]


def _records(df: pd.DataFrame, cols: list[str]) -> list[dict]:
    out = df[[c for c in cols if c in df.columns]]
    recs = out.astype(object).where(out.notna(), None).to_dict("records")   # NaN is not valid JSON
    for r in recs:                                                          # keep whole numbers whole (3, not 3.0)
        for k in INT_COLS:
            if r.get(k) is not None:
                r[k] = int(r[k])
    return recs


def _filter(pool: pd.DataFrame, position: str | None, q: str | None) -> pd.DataFrame:
    if position:
        pos = position.strip().upper()
        pool = pool[pool["eligible_positions"].map(lambda ps: pos in ps)]
    if q and q.strip():
        want = clean_name(q) or q.strip().lower()
        names = pool["clean_name"] if "clean_name" in pool.columns else pool["name"].map(clean_name)
        pool = pool[names.str.contains(want, regex=False, na=False)]
    return pool


def available_view(df: pd.DataFrame, cfg: LeagueConfig, limit: int = 50,
                   exclude_injury_prone: bool | None = None, position: str | None = None,
                   q: str | None = None, sort: str = "value") -> list[dict]:
    """exclude_injury_prone=None follows cfg.avoid_injury_prone; True/False overrides it.
    position: any eligible position; q: part of a name.
    sort: 'value' (value over replacement, the default) or 'rank' (the overall pre-draft rank, best first)."""
    pool = df[df["available"]]
    if cfg.avoid_injury_prone if exclude_injury_prone is None else exclude_injury_prone:
        pool = pool[~pool["injury_prone"]]
    pool = _filter(pool, position, q)
    if sort == "rank":                                   # the overall board order; ties and unranked players fall back to value
        chosen = pool.assign(_r=pool["projected_rank"].fillna(1e9)).sort_values(["_r", "vorp"], ascending=[True, False], kind="stable").head(limit)
        return _records(chosen, AVAILABLE_COLS)
    return _records(pool.nlargest(limit, "vorp"), AVAILABLE_COLS)


def injury_prone_view(df: pd.DataFrame, limit: int = 50, include_drafted: bool = False) -> dict:
    """Every flagged player, best value first. Ignores the avoid setting on purpose: this
    endpoint exists to show you who is flagged. 'count' is the total before the limit."""
    pool = df[df["injury_prone"]]
    if not include_drafted:
        pool = pool[pool["available"]]
    return {"count": int(len(pool)), "players": _records(pool.nlargest(limit, "vorp"), INJURY_COLS)}


def rankings_view(df: pd.DataFrame, limit: int = 50, position: str | None = None,
                  kind: str | None = None, include_drafted: bool = True) -> list[dict]:
    """The overall board, ordered by projected_rank. position matches ANY eligible position
    (e.g. 'SS' includes 2B/SS players); kind is 'H' or 'P'."""
    pool = df
    if not include_drafted:
        pool = pool[pool["available"]]
    if kind:
        pool = pool[pool["kind"] == kind.upper()]
    if position:
        pos = position.upper()
        pool = pool[pool["eligible_positions"].map(lambda ps: pos in ps)]
    return _records(pool.nsmallest(limit, "projected_rank"), RANKING_COLS)






def adp_report(pre: pd.DataFrame, adp: dict, limit: int = 10, note: str | None = None) -> dict:
    """How closely does our pre-draft ranking agree with the ADP? Rank correlation, plus the biggest disagreements."""
    d = pre.sort_values(["vorp", "name"], ascending=[False, True], kind="stable").reset_index(drop=True)
    d["our_rank"] = d.index + 1
    d["adp"] = d["player_id"].map(adp)
    d = d[d["adp"].notna()].copy()
    if len(d) < 5:
        return {"loaded": True, "note": note, "matched": int(len(d)), "message": "Too few matched players to compare."}
    from scipy.stats import spearmanr
    rho = float(spearmanr(d["our_rank"], d["adp"]).statistic)
    d["gap"] = d["adp"].rank(method="first") - d["our_rank"].rank(method="first")   # + : we rank him higher than the market
    cols = ["player_id", "name", "our_rank", "adp"]
    row = lambda x: {"player_id": x["player_id"], "name": x["name"], "our_rank": int(x["our_rank"]), "adp": float(x["adp"])}
    return {"loaded": True, "note": note, "matched": int(len(d)), "rank_correlation": round(rho, 3),
            "verdict": ("very close to the market" if rho >= 0.9 else "reasonably close" if rho >= 0.75 else
                        "quite different from the market: check the projections, positions and category settings"),
            "we_like_more_than_market": [row(x) for _, x in d.nlargest(limit, "gap").iterrows()],
            "market_likes_more_than_we_do": [row(x) for _, x in d.nsmallest(limit, "gap").iterrows()]}


def explain_player(df: pd.DataFrame, q: str) -> dict:
    """Why is this player ranked where he is? Projected games and minutes, and what each category contributes."""
    from .ingestion import clean_name
    key = clean_name(q)
    names = df["name"].map(clean_name)
    hit = df[names == key]
    if hit.empty:
        hit = df[names.str.contains(key, regex=False)] if key else df.iloc[0:0]
    if hit.empty:
        return {"found": False, "message": f"No player matches '{q}'. Try just the last name."}
    p = hit.nsmallest(1, "projected_rank").iloc[0]
    zs = {c[2:]: round(float(p[c]), 2) for c in df.columns if c.startswith("z_") and pd.notna(p[c])}
    gp, mins = p.get("GP"), p.get("MIN")
    gp = None if gp is None or pd.isna(gp) else float(gp)
    mpg = round(float(mins) / gp, 1) if gp and mins is not None and not pd.isna(mins) else None
    notes = []
    if gp is not None and gp < 60:
        notes.append(f"Projected for only {gp:.0f} games, and counting stats scale with games. If you expect him healthy, set "
                     "games_expected for him in injuries.csv and rebuild.")
    if bool(p.get("injury_prone")):
        notes.append("Flagged injury-prone from his games missed over the last three seasons.")
    if zs:
        worst = min(zs, key=zs.get)
        if zs[worst] < -1.0:
            notes.append(f"His biggest drag is {worst} ({zs[worst]:+.1f}).")
    others = [n for n in hit["name"].tolist() if n != p["name"]][:5]
    return {"found": True, "name": p["name"], "overall_rank": _int_or_none(p.get("projected_rank")),
            "score": round(float(p["score"]), 2), "vorp": round(float(p["vorp"]), 2),
            "projected_games": gp, "minutes_per_game": mpg, "category_z": dict(sorted(zs.items(), key=lambda kv: kv[1])),
            "notes": notes, "other_matches": others}


def _int_or_none(v):
    return None if v is None or pd.isna(v) else int(v)

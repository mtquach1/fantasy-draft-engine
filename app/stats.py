
"""Player search + single-stat lookup: previous season, career and projection.

Flow (stateless, so it works in Swagger or from any client):
  /players/search?q=Eli Perez stats            -> status 'choose_stat' (lists the stats you can ask for)
  /players/search?q=Eli Perez&stat=OBP         -> status 'ok'  (previous season / career / projected)
  /players/search?q=Perez                      -> status 'choose_player' when several players match
Rate stats (AVG, OBP, ERA ...) are recomputed from COMPONENTS (hits, at-bats, earned runs ...), so a career
number is a true weighted career figure, never an average of season averages.
"""
from __future__ import annotations

import difflib
import re

import numpy as np
import pandas as pd

from .ingestion import clean_name
from .sports import get_sport


# ---------- small helpers ----------
def _py(v):
    """numpy/NaN -> plain JSON-safe python."""
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def _get(c: dict, key: str):
    v = c.get(key)
    return None if v is None or pd.isna(v) else float(v)


def _div(n, d):
    return None if n is None or d is None or d == 0 else n / d


def _need(c: dict, *keys):
    vals = [_get(c, k) for k in keys]
    return None if any(v is None for v in vals) else vals


# ---------- hitter stats ----------
def _avg(c):
    v = _need(c, "H", "AB")
    return _div(*v) if v else None


def _obp(c):
    v = _need(c, "H", "BB", "AB")
    if not v:
        return None
    h, bb, ab = v
    hbp, sf = _get(c, "HBP") or 0.0, _get(c, "SF") or 0.0          # missing HBP/SF count as 0
    return _div(h + bb + hbp, ab + bb + hbp + sf)


def _slg(c):
    v = _need(c, "H", "doubles", "triples", "HR", "AB")
    if not v:
        return None
    h, d, t, hr, ab = v
    return _div(h + d + 2 * t + 3 * hr, ab)                         # total bases = H + 2B + 2*3B + 3*HR


def _ops(c):
    o, s = _obp(c), _slg(c)
    return None if o is None or s is None else o + s


def _iso(c):
    s, a = _slg(c), _avg(c)
    return None if s is None or a is None else s - a


def _babip(c):
    v = _need(c, "H", "HR", "AB", "SO")
    if not v:
        return None
    h, hr, ab, so = v
    return _div(h - hr, ab - so - hr + (_get(c, "SF") or 0.0))


def _ratio(num, den):
    def f(c):
        v = _need(c, num, den)
        return _div(*v) if v else None
    return f


def _count(key):
    return lambda c: _get(c, key)


HITTER_STATS = {   # name -> (function of components, display format)
    "AVG": (_avg, "rate3"), "OBP": (_obp, "rate3"), "SLG": (_slg, "rate3"), "OPS": (_ops, "rate3"),
    "ISO": (_iso, "rate3"), "BABIP": (_babip, "rate3"),
    "BB%": (_ratio("BB", "PA"), "pct"), "K%": (_ratio("SO", "PA"), "pct"),
    "G": (_count("G"), "int"), "PA": (_count("PA"), "int"), "AB": (_count("AB"), "int"),
    "H": (_count("H"), "int"), "1B": (_count("singles"), "int"), "2B": (_count("doubles"), "int"),
    "3B": (_count("triples"), "int"), "HR": (_count("HR"), "int"), "R": (_count("R"), "int"),
    "RBI": (_count("RBI"), "int"), "SB": (_count("SB"), "int"), "BB": (_count("BB"), "int"),
    "SO": (_count("SO"), "int"), "HBP": (_count("HBP"), "int"),
}


# ---------- pitcher stats ----------
def _per9(num):
    def f(c):
        v = _need(c, num, "IP")
        return None if not v or v[1] == 0 else 9 * v[0] / v[1]
    return f


def _whip(c):
    v = _need(c, "BB_allowed", "H_allowed", "IP")
    return _div(v[0] + v[1], v[2]) if v else None


PITCHER_STATS = {
    "ERA": (_per9("ER"), "dec2"), "WHIP": (_whip, "dec2"), "K/9": (_per9("K"), "dec2"),
    "BB/9": (_per9("BB_allowed"), "dec2"), "HR/9": (_per9("HR_allowed"), "dec2"),
    "K/BB": (_ratio("K", "BB_allowed"), "dec2"),
    "IP": (_count("IP"), "ip"), "W": (_count("W"), "int"), "L": (_count("L"), "int"),
    "SV": (_count("SV"), "int"), "HLD": (_count("HLD"), "int"), "QS": (_count("QS"), "int"),
    "G": (_count("G"), "int"), "GS": (_count("GS"), "int"), "K": (_count("K"), "int"),
    "BB": (_count("BB_allowed"), "int"), "H": (_count("H_allowed"), "int"),
    "ER": (_count("ER"), "int"), "HR": (_count("HR_allowed"), "int"),
}

def _pg(num, den="GP"):
    return _ratio(num, den)


NBA_STATS = {
    "FG%": (_ratio("FGM", "FGA"), "rate3"), "FT%": (_ratio("FTM", "FTA"), "rate3"),
    "PPG": (_pg("PTS"), "dec2"), "RPG": (_pg("REB"), "dec2"), "APG": (_pg("AST"), "dec2"),
    "SPG": (_pg("STL"), "dec2"), "BPG": (_pg("BLK"), "dec2"), "3PG": (_pg("3PM"), "dec2"), "TOPG": (_pg("TO"), "dec2"),
    "MPG": (_pg("MIN"), "dec2"),
    **{k: (_count(k), "int") for k in ("GP", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA", "MIN")},
}
NFL_STATS = {
    "CMP%": (_ratio("pass_cmp", "pass_att"), "pct"), "Y/A": (_ratio("pass_yds", "pass_att"), "dec2"),
    "YPC": (_ratio("rush_yds", "rush_att"), "dec2"), "Y/R": (_ratio("rec_yds", "rec"), "dec2"),
    **{k: (_count(v), "int") for k, v in {
        "G": "G", "PASS ATT": "pass_att", "PASS CMP": "pass_cmp", "PASS YDS": "pass_yds", "PASS TD": "pass_tds",
        "INT": "ints", "RUSH ATT": "rush_att", "RUSH YDS": "rush_yds", "RUSH TD": "rush_tds", "REC": "rec",
        "REC YDS": "rec_yds", "REC TD": "rec_tds", "FUM": "fumbles", "FGM": "fgm", "XPM": "xpm", "SACKS": "sacks",
        "DEF INT": "def_int", "FUM REC": "fum_rec", "DEF TD": "def_td", "SAFETIES": "safeties"}.items()},
}
CATALOGS = {("mlb", "H"): HITTER_STATS, ("mlb", "P"): PITCHER_STATS, ("nba", "X"): NBA_STATS, ("nfl", "X"): NFL_STATS}


def catalog_for(sport: str, kind: str) -> dict:
    return CATALOGS[(sport, kind)]


_ALIASES = {"BA": "AVG", "AVERAGE": "AVG", "BATTINGAVERAGE": "AVG", "BATTINGAVG": "AVG",
            "ONBASE": "OBP", "SLUGGING": "SLG", "HOMERUNS": "HR", "HOMERS": "HR", "STEALS": "SB",
            "STOLENBASES": "SB", "WALKS": "BB", "RUNS": "R", "HITS": "H", "DOUBLES": "2B",
            "TRIPLES": "3B", "SINGLES": "1B", "WINS": "W", "LOSSES": "L", "SAVES": "SV", "HOLDS": "HLD",
            "QUALITYSTARTS": "QS", "INNINGS": "IP", "INNINGSPITCHED": "IP", "STARTS": "GS", "GAMES": "G",
            "EARNEDRUNS": "ER", "K9": "K/9", "BB9": "BB/9", "HR9": "HR/9", "KBB": "K/BB",
            "BBPCT": "BB%", "KPCT": "K%", "RBIS": "RBI", "PLATEAPPEARANCES": "PA", "ATBATS": "AB",
            "STRIKEOUTS": "STRIKEOUTS"}
_ALL_TOKENS = {_squash_key for c in CATALOGS.values() for _squash_key in c} | set(_ALIASES) | {"SO", "K"}
_ALL_TOKENS |= {re.sub(r"[\s_.\-]", "", k) for k in list(_ALL_TOKENS)}


def _squash(s: str) -> str:
    return re.sub(r"[\s_.\-]", "", str(s).upper())


def normalize_stat(raw: str, kind: str, sport: str = "mlb") -> str | None:
    """'obp' / 'Batting Average' / 'k9' / 'ppg' -> canonical stat for this player group, or None."""
    cat = catalog_for(sport, kind)
    s = _ALIASES.get(_squash(raw), _squash(raw))
    if sport == "mlb" and s in ("STRIKEOUTS", "K", "SO"):    # strikeouts are SO for hitters, K for pitchers
        s = "SO" if kind == "H" else "K"
    if s in cat:
        return s
    spaced = {re.sub(r"[\s_.\-]", "", k): k for k in cat}      # 'PASS YDS' can be typed as 'passyds' or 'pass_yds'
    return spaced.get(s)


def parse_query(q: str) -> tuple[str, str | None]:
    """'Eli Perez stats' -> ('Eli Perez', None);  'eli perez obp' / 'Eli Perez stats K/9' -> (name, stat token)."""
    tokens = re.sub(r"\s+", " ", q.strip()).split(" ")
    stat = None
    if len(tokens) >= 2 and _squash(tokens[-1]) in _ALL_TOKENS and _squash(tokens[-1]) not in ("STATS", "STAT"):
        stat = tokens.pop()
    if tokens and tokens[-1].lower() in ("stats", "stat", "statistics"):
        tokens.pop()
    return " ".join(tokens), stat


# ---------- formatting ----------
def _fmt(v, kind: str) -> tuple[object, str]:
    """-> (numeric value, display string)."""
    if v is None:
        return None, "n/a"
    if kind == "rate3":
        s = f"{v:.3f}"
        return round(v, 3), s.replace("0.", ".", 1) if abs(v) < 1 else s
    if kind == "pct":
        return round(v * 100, 1), f"{v * 100:.1f}%"
    if kind == "dec2":
        return round(v, 2), f"{v:.2f}"
    if kind == "ip":
        whole = int(v)
        thirds = round((v - whole) * 3)
        if thirds == 3:
            whole, thirds = whole + 1, 0
        return round(v, 1), f"{whole}.{thirds}"
    return int(round(v)), f"{v:,.0f}"


def _pack(v, kind):
    value, display = _fmt(v, kind)
    return {"value": value, "display": display}


# ---------- components ----------
def _keys(kind: str, sport: str) -> tuple:
    return get_sport(sport).group(kind).components


def _finish(c: dict, kind: str) -> dict:
    if "singles" in c and c["singles"] is None and None not in (c["H"], c["doubles"], c["triples"], c["HR"]):
        c["singles"] = c["H"] - c["doubles"] - c["triples"] - c["HR"]
    return c


def _row_components(row: pd.Series, kind: str, sport: str = "mlb") -> dict:
    return _finish({k: (None if k not in row.index or pd.isna(row[k]) else float(row[k])) for k in _keys(kind, sport)}, kind)


def _sum_components(rows: pd.DataFrame, kind: str, sport: str = "mlb") -> dict:
    out = {}
    for k in _keys(kind, sport):
        s = rows[k].sum(min_count=1) if k in rows.columns else np.nan
        out[k] = None if pd.isna(s) else float(s)
    return _finish(out, kind)


# ---------- search ----------
def _info(row: pd.Series) -> dict:
    return {"player_id": str(row["player_id"]), "name": _py(row["name"]), "team": _py(row.get("team")),
            "kind": row["kind"], "positions": _py(row.get("eligible_positions")),
            "projected_rank": None if pd.isna(row.get("projected_rank", np.nan)) else int(row["projected_rank"]),
            "pos_rank": _py(row.get("pos_rank"))}


MATCH_LABEL = {0: "exact", 1: "name starts with", 2: "partial", 3: "closest spelling", 4: "similar spelling"}


def search_players(board: pd.DataFrame, name_query: str, limit: int = 10) -> list[dict]:
    """Accent/case-insensitive. Tiers: exact > every word is the start of a name word > substring >
    fuzzy spelling (only used when nothing else matches). Ties are ordered by projected_rank."""
    q = clean_name(name_query)
    if not q:
        return []
    names = board["name"].astype(str).map(clean_name)
    words = q.split()
    exact = names == q
    prefix = names.map(lambda n: all(any(w.startswith(t) for w in n.split()) for t in words))
    contains = names.str.contains(re.escape(q), regex=True)
    tier = pd.Series(9, index=board.index)
    tier[contains] = 2
    tier[prefix] = 1
    tier[exact] = 0
    if (tier == 9).all():
        # typo tolerance: rank by spelling similarity; only the clear winner(s) are tier 3, the rest tier 4
        sim = {n: difflib.SequenceMatcher(None, q, n).ratio() for n in names.unique()}
        sim = {n: r for n, r in sim.items() if r >= 0.75}
        if sim:
            top = max(sim.values())
            tier[names.map(lambda n: n in sim and sim[n] >= top - 0.03)] = 3
            tier[names.map(lambda n: n in sim and sim[n] < top - 0.03)] = 4
    hit = board[tier < 9].copy()
    hit["_tier"] = tier[tier < 9]
    by = "projected_rank" if "projected_rank" in hit.columns else "vorp"
    hit = hit.sort_values(["_tier", by], ascending=[True, by == "projected_rank"], kind="stable")
    out = []
    for _, r in hit.head(limit).iterrows():
        d = _info(r)
        d["match"], d["_tier"] = MATCH_LABEL[int(r["_tier"])], int(r["_tier"])
        out.append(d)
    return out


# ---------- the report ----------
def _last_season(history: dict) -> int | None:
    seasons = [int(df["season"].max()) for df in history.values() if df is not None and len(df)]
    return max(seasons) if seasons else None


def _proj_row(raw: pd.DataFrame, pid: str, kind: str):
    m = raw[(raw["player_id"] == pid) & (raw["kind"] == kind)]
    return None if m.empty else m.iloc[0]


def _stat_entry(key: str, kind: str, prev_c, career_c, proj_c, sport: str = "mlb") -> dict:
    fn, f = catalog_for(sport, kind)[key]
    return {"previous_season": _pack(fn(prev_c) if prev_c else None, f),
            "career": _pack(fn(career_c) if career_c else None, f),
            "projected": _pack(fn(proj_c) if proj_c else None, f)}


def player_report(player: dict, raw: pd.DataFrame, history: dict, stat: str | None,
                  is_injured: bool = False, games_pct: float = 1.0, sport: str = "mlb") -> dict:
    kind, name = player["kind"], player["name"]
    catalog, group = catalog_for(sport, kind), get_sport(sport).group(kind)
    label = group.noun
    info = {k: player[k] for k in ("player_id", "name", "team", "kind", "positions", "projected_rank", "pos_rank")}
    if stat is None:
        return {"status": "choose_stat", "player": info,
                "prompt": f"Which stat would you like for {name}? Re-run with stat=<one of these> (or stat=all).",
                "stats": list(catalog)}
    show_all = str(stat).strip().lower() == "all"
    key = None if show_all else normalize_stat(stat, kind, sport)
    if not show_all and key is None:
        return {"status": "unknown_stat", "player": info,
                "message": f"'{stat}' isn't a {label} stat I track for {name}.", "stats": list(catalog)}

    pid, notes = player["player_id"], []
    last = _last_season(history)
    prow = _proj_row(raw, pid, kind)
    base = str(prow["base_id"]) if prow is not None and "base_id" in prow.index and pd.notna(prow["base_id"]) else pid
    hist = history.get(kind)
    hist = hist[hist["player_id"] == base] if hist is not None and len(hist) else None
    prev_c = career_c = None
    career_seasons = None
    if hist is None or hist.empty:
        notes.append(f"No past-season {label} stats on file for {name}. Add data/"
                     f"{group.history_file} to see previous season and career.")
    else:
        career_c = _sum_components(hist, kind, sport)
        lo, hi = int(hist["season"].min()), int(hist["season"].max())
        career_seasons = {"seasons": str(lo) if lo == hi else f"{lo}-{hi}", "n_seasons": int(hist["season"].nunique())}
        last_rows = hist[hist["season"] == last]
        if last_rows.empty:
            notes.append(f"No {last} stats on file for {name} (did not play, or missing from the history file).")
        else:
            prev_c = _sum_components(last_rows, kind, sport)
    proj_c = _row_components(prow, kind, sport) if prow is not None else None
    if proj_c is None:
        notes.append("No projection on file for this player.")
    elif is_injured:
        notes.append(f"Projection is the full-season figure; it is not reduced for the current injury (games_pct {games_pct}).")

    base = {"status": "ok", "player": info, "previous_season_year": last, "career": career_seasons, "notes": notes}
    if show_all:
        base["stats"] = [{"stat": k, **_stat_entry(k, kind, prev_c, career_c, proj_c, sport)} for k in catalog]
        return base
    e = _stat_entry(key, kind, prev_c, career_c, proj_c, sport)
    year = last if last is not None else "Previous"
    lines = [f"{year} (previous season) {key}: {e['previous_season']['display']}",
             f"Career {key}: {e['career']['display']}",
             f"Projected {key}: {e['projected']['display']}"]
    base.update({"stat": key, **e, "lines": lines, "display": "\n".join(lines)})
    return base


def _board_row(board: pd.DataFrame, player_id: str, kind: str | None):
    m = board[board["player_id"] == str(player_id)]
    if kind:
        m = m[m["kind"] == kind.upper()]
    return None if m.empty else m.iloc[0]


def stats_for_player(board, raw, history, player_id: str, stat: str | None = None, kind: str | None = None,
                     sport: str = "mlb") -> dict:
    row = _board_row(board, player_id, kind)
    if row is None:
        return {"status": "no_match", "message": f"No player with id '{player_id}'."}
    return player_report(_info(row), raw, history, stat, bool(row.get("is_injured", False)),
                         float(row.get("games_pct", 1.0)), sport)


def search_flow(board, raw, history, q: str, stat: str | None = None, limit: int = 10, sport: str = "mlb") -> dict:
    name_q, token = parse_query(q)
    stat = stat or token
    matches = search_players(board, name_q, limit)
    if not matches:
        return {"status": "no_match", "query": name_q, "message": f"No player found matching '{name_q}'."}
    best = matches[0]["_tier"]
    top = [m for m in matches if m["_tier"] == best]
    if len(top) > 1 and stat:
        # a hitter and a pitcher share the name: if only one of them HAS this stat (OBP vs ERA), that's the one
        valid = [m for m in top if normalize_stat(stat, m["kind"], sport)]
        if len(valid) == 1:
            top = valid
    clean = [{k: v for k, v in m.items() if k != "_tier"} for m in matches]
    if len(top) > 1:
        return {"status": "choose_player", "query": name_q,
                "message": "Several players match. Pick one and call GET /players/{player_id}/stats?stat=...",
                "matches": clean}
    row = _board_row(board, top[0]["player_id"], top[0]["kind"])
    out = player_report(_info(row), raw, history, stat, bool(row.get("is_injured", False)),
                        float(row.get("games_pct", 1.0)), sport)
    if best in (3, 4):
        out.setdefault("notes", []).insert(0, f"Showing the closest match for '{name_q}'.")
    return out




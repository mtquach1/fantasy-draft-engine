
"""Load projections -> clean DataFrame, one row per player, with a 'kind' column (the sport's player group).
File names, columns and positions come from the sport profile in sports.py."""
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from .sports import get_sport

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv)\b\.?", re.I)


def data_dir(sport: str = "mlb") -> Path:
    """MLB files live in data/ (as before); other sports in data/<sport>/ (data/nba, data/nfl)."""
    return DATA_DIR / get_sport(sport).data_subdir


def read_csv_robust(path, **kw) -> pd.DataFrame:
    """read_csv that survives how files really get saved: UTF-8 with or without a BOM (Excel and FanGraphs add one,
    which would otherwise turn the 'Name' column into '\\ufeffName'), then Windows cp1252 (accents saved by older
    Excel), then latin-1. Header names are trimmed."""
    df = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, encoding=enc, **kw)
            break
        except UnicodeDecodeError:
            continue
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    return df


def clean_name(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = SUFFIXES.sub("", s.replace(".", ""))
    return re.sub(r"\s+", " ", s).strip().lower()


def parse_positions(raw, sport: str = "mlb") -> list[str]:
    """'1B, C, LF' / 'SS/2B' -> ['1B','C','OF']. Aliases (Yahoo LF/CF/RF -> OF, NFL DEF -> DST) come from the sport."""
    sp = get_sport(sport)
    parts = [p.strip().upper() for p in re.split(r"[,/-]", str(raw)) if p.strip()]
    parts = [sp.aliases.get(p, p) for p in parts]
    parts = list(dict.fromkeys(p for p in parts if p in sp.valid_positions))   # dedupe, keep order
    real = [p for p in parts if p != "DH"]
    return real or parts


def _load_positions_override(sport: str = "mlb", platform: str | None = None) -> dict[str, list[str]]:
    """Optional positions file (player_id, eligible_positions) in the sport's data folder. Yahoo and ESPN use
    different eligibility rules (games needed at a position), so positions_<platform>.csv, e.g. positions_espn.csv,
    wins over the generic positions.csv when DRAFT_PLATFORM matches."""
    base = data_dir(sport)
    path = next((p for p in ([base / f"positions_{platform}.csv"] if platform else []) + [base / "positions.csv"]
                 if p.exists()), None)
    if path is None:
        return {}
    df = read_csv_robust(path, dtype={"player_id": str})
    return {r.player_id: parse_positions(r.eligible_positions, sport) for r in df.itertuples()}


def _common(df: pd.DataFrame) -> pd.DataFrame:
    df["player_id"] = df["player_id"].astype(str)
    df["clean_name"] = df["name"].map(clean_name)
    if "is_injured" not in df:
        df["is_injured"] = False
    if "games_pct" not in df:
        df["games_pct"] = 1.0
    df["games_pct"] = df["games_pct"].fillna(1.0).clip(0, 1)
    return df


TEAM_GAMES = {2020: 60}         # MLB's shortened season; every other year uses the group's default games
DEFAULT_TEAM_GAMES = 162
HISTORY_SEASONS = 3


def load_injury_history(path="history.csv", sport: str = "mlb") -> pd.DataFrame:
    """history.csv in the sport's data folder: player_id, season, games_played [, games_possible].
    Groups with a default season length (MLB hitters 162, NBA 82, NFL 17) need nothing else.
    Groups without one (MLB pitchers): supply games_possible yourself (games played is a poor yardstick for
    starters), otherwise they are treated as 'no history' and never flagged."""
    p = data_dir(sport) / path
    if not p.exists():
        return pd.DataFrame(columns=["player_id", "season", "games_played", "games_possible"])
    return read_csv_robust(p, dtype={"player_id": str})


def compute_games_missed(history: pd.DataFrame, kinds: dict, n_seasons: int = HISTORY_SEASONS,
                         games_by_kind: dict | None = None, season_overrides: dict | None = None) -> pd.DataFrame:
    """Average fraction of games missed over the last n seasons in the file.
    A season with NO row after a player's debut counts as a full season missed (data sources
    simply omit players who never appeared). Seasons before the debut are ignored.
    Returns player_id, avg_games_missed_pct, injury_seasons_used."""
    games_by_kind = {"H": DEFAULT_TEAM_GAMES} if games_by_kind is None else games_by_kind
    season_overrides = TEAM_GAMES if season_overrides is None else season_overrides
    cols = ["player_id", "avg_games_missed_pct", "injury_seasons_used"]
    if history.empty:
        return pd.DataFrame(columns=cols)
    h = history.copy()
    h["player_id"] = h["player_id"].astype(str)
    h["season"] = h["season"].astype(int)
    if "games_possible" not in h:
        h["games_possible"] = np.nan
    last = h["season"].max()
    window = list(range(last - n_seasons + 1, last + 1))
    rows = []
    for pid, g in h.groupby("player_id"):
        base = games_by_kind.get(kinds.get(pid))
        g = g.groupby("season").agg(games_played=("games_played", "sum"),
                                    games_possible=("games_possible", "max"))   # traded players: sum splits
        debut = g.index.min()
        fallback = g["games_possible"].mean()                  # NaN if never supplied
        missed = []
        for s in (s for s in window if s >= debut):
            default = season_overrides.get(s, base) if base is not None else fallback
            if s in g.index:
                played = g.loc[s, "games_played"]
                denom = g.loc[s, "games_possible"] if pd.notna(g.loc[s, "games_possible"]) else default
            else:
                played, denom = 0.0, default
            if pd.isna(denom) or denom <= 0:
                continue                                       # can't judge this season
            missed.append(min(max(1 - played / denom, 0.0), 1.0))
        rows.append({"player_id": pid,
                     "avg_games_missed_pct": float(np.mean(missed)) if missed else np.nan,
                     "injury_seasons_used": len(missed)})
    return pd.DataFrame(rows, columns=cols)


def link_two_way(df: pd.DataFrame) -> pd.DataFrame:
    """A player in BOTH the hitters and pitchers file (Ohtani) would share one player_id, which breaks
    rankings and picks. The pitcher row gets id '<id>_P'; both rows keep base_id (the original id, used for
    history lookups) and twin_id (each other's id, so drafting one removes both)."""
    df = df.copy()
    df["base_id"] = df["player_id"].astype(str)
    df["twin_id"] = None
    both = set(df.loc[df["kind"] == "H", "player_id"]) & set(df.loc[df["kind"] == "P", "player_id"])
    for pid in both:
        new = f"{pid}_P"
        df.loc[(df["player_id"] == pid) & (df["kind"] == "P"), "player_id"] = new
        df.loc[(df["player_id"] == pid) & (df["kind"] == "H"), "twin_id"] = new
        df.loc[df["player_id"] == new, "twin_id"] = pid
    return df


def dedupe_traded(df: pd.DataFrame) -> pd.DataFrame:
    """Traded players appear several times: a season-total row (team TOT / 2TM / 3TM) plus one row per team. Keep the total."""
    if "team" not in df.columns or "name" not in df.columns:
        return df
    total = df["team"].astype(str).str.upper().str.fullmatch(r"TOT|\dTM")
    traded = set(df.loc[total, "name"])
    return df[total | ~df["name"].isin(traded)].reset_index(drop=True) if traded else df


def _friendly(df: pd.DataFrame, sp) -> pd.DataFrame:
    """Accept files straight from common sites: (1) no id column -> make a stable id from the name (names that
    repeat get _2, _3), and (2) NBA per-game averages -> season totals (per-game PTS never exceeds ~45)."""
    if "player_id" not in df.columns:
        df = dedupe_traded(df)
    if "player_id" not in df.columns and "name" in df.columns:
        base = "n:" + df["name"].map(clean_name)
        df["player_id"] = base + df.groupby(base).cumcount().map(lambda i: "" if i == 0 else f"_{i + 1}")
    if sp.key == "nba" and {"GP", "PTS"} <= set(df.columns):
        pts, gp = pd.to_numeric(df["PTS"], errors="coerce"), pd.to_numeric(df["GP"], errors="coerce")
        if pts.notna().any() and pts.max() < 60 and gp.max() > 1:
            for c in ("MIN", "PTS", "REB", "AST", "STL", "BLK", "3PM", "TO", "FGM", "FGA", "FTM", "FTA"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce") * gp
            print("[data] Your NBA file has per-game numbers: converted to season totals (per game x games played).")
    return df


def load_projections(sport: str = "mlb", platform: str | None = None, file_overrides: dict | None = None) -> pd.DataFrame:
    """Reads the sport's projection CSVs from its data folder. Positions come from (in order):
    positions.csv (e.g. Yahoo-derived) -> a 'Pos' column -> the group's default (MLB pitchers: SP if GS/G > 0.5)."""
    sp = get_sport(sport)
    base = data_dir(sport)
    override = _load_positions_override(sport, platform)
    frames = []
    for g in sp.groups:
        path = base / (file_overrides or {}).get(g.file, g.file)     # e.g. {"players.csv": "players_own.csv"}
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. {sp.label} needs {', '.join(x.file for x in sp.groups)} in {base}")
        df = _common(_friendly(read_csv_robust(path).rename(columns=g.rename), sp))
        df["kind"] = g.kind
        pos_col = df["Pos"] if "Pos" in df else pd.Series("", index=df.index)
        default = g.default_positions(df) if g.default_positions else [None] * len(df)
        df["eligible_positions"] = [override.get(pid) or parse_positions(raw, sport) or ([d] if d else [])
                                    for pid, raw, d in zip(df["player_id"], pos_col, default)]
        frames.append(df[df["eligible_positions"].map(len) > 0])
    df = link_two_way(pd.concat(frames, ignore_index=True))
    kinds = dict(zip(df["base_id"], df["kind"]))
    first = sp.groups[0].kind
    kinds.update({b: first for b in df.loc[df["kind"] == first, "base_id"]})      # a two-way player counts as the first group
    inj = compute_games_missed(load_injury_history(sport=sport), kinds,
                               games_by_kind={g.kind: g.default_games for g in sp.groups},
                               season_overrides=TEAM_GAMES if sp.key == "mlb" else {})
    inj = inj.rename(columns={"player_id": "base_id"})
    df = df.merge(inj, on="base_id", how="left")
    df["injury_seasons_used"] = df["injury_seasons_used"].fillna(0).astype(int)
    return df


def _ip_from_baseball(s: pd.Series) -> pd.Series:
    """Baseball notation (180.1 = 180 1/3, 180.2 = 180 2/3) -> true decimal innings.
    Applied to the HISTORY files only; projections are used as they come."""
    whole = np.floor(s)
    frac = ((s - whole) * 10).round()
    return whole + np.where(frac == 1, 1 / 3, np.where(frac == 2, 2 / 3, s - whole))


def load_stats_history(sport: str = "mlb") -> dict:
    """Past-season stats for the search feature: one row per player per season, in the sport's history file(s)
    (MLB: hitter_history.csv + pitcher_history.csv with FanGraphs columns, IP in 180.1 notation;
    NBA/NFL: player_history.csv with Season, playerid and the same stat columns as the projections).
    Rows for the same player+season (e.g. a traded player) are summed. Missing files -> empty frame."""
    sp = get_sport(sport)
    out = {}
    for g in sp.groups:
        path = data_dir(sport) / g.history_file
        if not path.exists():
            out[g.kind] = pd.DataFrame()
            continue
        df = read_csv_robust(path, dtype={"playerid": str, "player_id": str})
        df = df.rename(columns={**g.rename, **g.history_extra_rename, "Season": "season"})
        df["player_id"] = df["player_id"].astype(str)
        df["season"] = df["season"].astype(int)
        if sp.key == "mlb" and g.kind == "P" and "IP" in df:
            df["IP"] = _ip_from_baseball(df["IP"].astype(float))
        num = [c for c in df.select_dtypes("number").columns if c != "season"]
        out[g.kind] = df.groupby(["player_id", "season"], as_index=False)[num].sum(min_count=1)
    return out


def load_id_crosswalk(path="id_crosswalk.csv", sport: str = "mlb") -> pd.DataFrame:
    """CSV with at least: player_id (your projections' id), yahoo_id. For MLB, Smart Fantasy Baseball's
    player ID map has both (IDFANGRAPHS and YAHOOID): rename the columns. Each sport has its own file."""
    return read_csv_robust(data_dir(sport) / path, dtype=str)


def build_platform_map(xwalk: pd.DataFrame, platform_col: str) -> dict[str, str]:
    x = xwalk.dropna(subset=[platform_col, "player_id"])
    return dict(zip(x[platform_col].astype(str).str.strip(), x["player_id"].astype(str).str.strip()))


def load_platform_map(sport: str, platform: str) -> tuple[dict[str, str], str | None]:
    """(platform player id -> your player_id, note). Never raises: a missing crosswalk file or column just means the
    live sync falls back to matching by name, and the note says so. Column is <platform>_id (yahoo_id, espn_id)."""
    col = f"{platform}_id"
    try:
        xw = load_id_crosswalk(sport=sport)
    except FileNotFoundError:
        return {}, f"No id_crosswalk.csv for {sport.upper()}: matching picks by player name instead."
    except Exception as e:
        return {}, f"id_crosswalk.csv couldn't be read ({type(e).__name__}): matching picks by player name instead."
    if "player_id" not in xw.columns or col not in xw.columns:
        return {}, f"id_crosswalk.csv has no '{col}' column: matching picks by player name instead."
    return build_platform_map(xw, col), None


def build_name_map(raw: pd.DataFrame) -> dict[str, str]:
    """clean_name -> player_id for names that point to exactly ONE human. Two rows are one human when they are
    the two halves of a two-way player (same base_id); the first row stands for them. Names shared by different
    players (two Sam Walkers) are left out, so they are reported as unmatched instead of guessed."""
    if raw.empty or "clean_name" not in raw.columns:
        return {}
    base = raw["base_id"] if "base_id" in raw.columns else raw["player_id"]
    out = {}
    for name, grp in raw.assign(_b=base.fillna(raw["player_id"])).groupby("clean_name"):
        if name and grp["_b"].nunique() == 1:
            out[name] = str(grp["player_id"].iloc[0])
    return out





ADP_ALIASES = {"adp": "adp", "avg_pick": "adp", "average_draft_position": "adp", "avg": "adp", "average": "adp",
               "name": "name", "player": "name", "playername": "name", "player_name": "name",
               "player_id": "player_id", "playerid": "player_id", "id": "player_id"}


def load_adp(sport: str, raw: pd.DataFrame, name_map: dict | None = None) -> tuple[dict[str, float], str | None]:
    """Optional `adp.csv` in the sport's data folder: average draft position from your league site or FanGraphs.
    Needs an `adp` column plus `player_id` (same ids as your projections) and/or `name`. Returns
    ({player_id: adp}, a note for people). Never raises: a bad file just means the look-ahead uses the default."""
    path = data_dir(sport) / "adp.csv"
    if not path.exists():
        return {}, None
    try:
        df = read_csv_robust(path)
        df = df.rename(columns={c: ADP_ALIASES[re.sub(r"[^a-z_]", "", str(c).strip().lower().replace(" ", "_"))]
                                for c in df.columns
                                if re.sub(r"[^a-z_]", "", str(c).strip().lower().replace(" ", "_")) in ADP_ALIASES})
        df = df.loc[:, ~df.columns.duplicated()]
        if "adp" not in df.columns or not ({"player_id", "name"} & set(df.columns)):
            return {}, "adp.csv needs an 'adp' column and a 'player_id' or 'name' column. Ignored."
        df["adp"] = pd.to_numeric(df["adp"], errors="coerce")
        bad = int(df["adp"].isna().sum() + (df["adp"] <= 0).sum())
        df = df[df["adp"] > 0]
        ids, names = set(raw["player_id"]), (name_map if name_map is not None else build_name_map(raw))
        out, unmatched = {}, 0
        for _, r in df.iterrows():
            pid = None
            if "player_id" in df.columns and pd.notna(r["player_id"]) and str(r["player_id"]).strip() in ids:
                pid = str(r["player_id"]).strip()
            elif "name" in df.columns and pd.notna(r["name"]):
                pid = names.get(clean_name(r["name"]))
            if pid is None:
                unmatched += 1
            else:
                out.setdefault(pid, float(r["adp"]))
        if not out:
            return {}, "adp.csv was read but no player matched your projections (check the ids or names). Ignored."
        note = f"ADP loaded for {len(out)} players"
        if unmatched:
            note += f"; {unmatched} row(s) didn't match a player"
        if bad:
            note += f"; {bad} row(s) had no usable ADP"
        return out, note + "."
    except Exception as e:
        return {}, f"adp.csv couldn't be read ({type(e).__name__}: {e}). Ignored."

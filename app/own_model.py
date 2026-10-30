"""Own projection model: next season's per-stat totals from past Basketball-Reference per-game seasons.

Parts (each has one fitted knob):
  production per minute   recency-weighted over seasons, shrunk toward the league rate (k_rate minutes)
  minutes per game        games-weighted average (k_min pseudo-games toward the league mean)
  games played            observed availability shrunk toward a role baseline (k_g pseudo-seasons);
                          seasons before a player's debut are ignored, seasons missed after it count as 0 games
  age / trajectory        additive correction to fantasy points per game, fitted on one season transition

Season totals are rate x minutes x games. Points are scored with ESPN default weights only to fit and check the model;
the app re-scores the stat columns with whatever scoring the league uses.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

W = dict(PTS=1, FGM=2, FGA=-1, FTM=1, FTA=-1, TPM=1, TRB=1, AST=2, STL=4, BLK=4, TOV=-2)   # ESPN default points
STATS = list(W)


def load_season(path) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig")
    d = d[d["Player"] != "League Average"].copy()
    d["pid"] = d["Player-additional"]
    tm = d["Team"] if "Team" in d else d["Tm"]
    d["_multi"] = tm.astype(str).str.match(r"^\d+TM$")
    d = d.sort_values("_multi", ascending=False).drop_duplicates("pid", keep="first")      # combined row for traded players
    d = d.rename(columns={"Team": "Tm", "3P": "TPM", "FG": "FGM", "FT": "FTM", "MP": "MPG"})
    for c in ["Age", "G", "MPG", "FGM", "FGA", "FTM", "FTA", "TPM", "TRB", "AST", "STL", "BLK", "TOV", "PTS"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["G", "MPG", "PTS"])
    d["fpg"] = sum(w * d[k] for k, w in W.items())
    d["mins"] = d["G"] * d["MPG"]
    d["fpm"] = d["fpg"] / d["MPG"]
    return d.set_index("pid")[["Player", "Tm", "Age", "Pos", "G", "MPG", "mins", "fpg", "fpm"] + STATS]


def find_seasons(raw_dir) -> dict:
    """{season_end_year: path} for files named per_game_2026.csv (also per_game+2026 / per_game-2026)."""
    out = {}
    for p in Path(raw_dir).glob("per_game*.csv"):
        m = re.search(r"per_game[_+\- ]?(\d{4})\.csv$", p.name)
        if m:
            out[int(m.group(1))] = p
    return dict(sorted(out.items()))


class Params:
    def __init__(self, lam=0.5, k_rate=109., k_min=1., k_g=2.26, base_g=68., mpg_floor=8.):
        self.lam, self.k_rate, self.k_min, self.k_g, self.base_g, self.mpg_floor = lam, k_rate, k_min, k_g, base_g, mpg_floor


def league_means(h):
    r = h[h["mins"] >= 500]
    return dict(fpm=(r["fpm"] * r["mins"]).sum() / r["mins"].sum(), mpg=(r["MPG"] * r["G"]).sum() / r["G"].sum())


def project_core(history, ids, p, league):
    """history: season frames, newest first. Returns projected mpg, games and fantasy points per minute."""
    z = lambda: pd.Series(0., index=ids)
    num_r, den_r, num_m, den_m, num_g, den_g = z(), z(), z(), z(), z(), z()
    oldest = pd.Series(0, index=ids)
    for i, h in enumerate(history):
        oldest = oldest.where(~h.reindex(ids)["G"].notna(), i)          # oldest season index the player appears in
    for i, h in enumerate(history):
        w = p.lam ** i
        h = h.reindex(ids)
        mins, G = h["mins"].fillna(0), h["G"].fillna(0)
        num_r += w * (h["fpm"].fillna(0) * mins); den_r += w * mins
        num_m += w * (h["MPG"].fillna(0) * G);    den_m += w * G
        num_g += w * G;                           den_g += w * 82 * (oldest >= i).astype(float)
    fpm = (num_r + p.k_rate * league["fpm"]) / (den_r + p.k_rate)
    mpg = (num_m + p.k_min * league["mpg"]) / (den_m + p.k_min)
    obs = (num_g / den_g.replace(0, np.nan) * 82).fillna(p.base_g)
    base = np.where(mpg >= 28, p.base_g, np.where(mpg >= 20, p.base_g - 4, p.base_g - 13))
    games = (obs * den_g / 82 + p.k_g * base) / (den_g / 82 + p.k_g)
    return pd.DataFrame({"mpg": mpg.clip(lower=p.mpg_floor, upper=38), "games": games.clip(20, 82), "fpm": fpm}, index=ids)


def fit_age_curve(frame) -> np.ndarray:
    """frame: columns base, age, y (fantasy pts per game). Fits  y - base ~ const + a + a^2 + under24, a = (age-27)/5."""
    a = (frame["age"] - 27) / 5
    X = np.column_stack([np.ones(len(a)), a, a ** 2, (frame["age"] < 24).astype(float)])
    return np.linalg.lstsq(X, frame["y"] - frame["base"], rcond=None)[0]


def apply_age_curve(base, age, beta, cap=(0.85, 1.20)):
    a = (age - 27) / 5
    adj = beta[0] + beta[1] * a + beta[2] * a ** 2 + beta[3] * (age < 24)
    return (base + adj).clip(base * cap[0], base * cap[1])


def _ages(hist, ids):
    age = (hist[0]["Age"].reindex(ids) + 1)
    for i, x in enumerate(hist[1:], 1):
        age = age.fillna(x["Age"].reindex(ids) + 1 + i)
    return age.fillna(27.)


def fit_age_from_transition(seasons: dict, years, p: Params):
    """Fit the age curve on the oldest transition we have: history = years[0] only, target = years[1]."""
    h0, tgt = seasons[years[0]], seasons[years[1]]
    ids = h0.index[h0["mins"] >= 500]
    core = project_core([h0], ids, p, league_means(h0))
    y = tgt.reindex(ids)
    df = pd.DataFrame({"base": core["fpm"] * core["mpg"], "age": _ages([h0], ids), "y": y["fpg"], "G": y["G"]}, index=ids)
    df = df[df["y"].notna() & (df["G"] >= 10)]
    return fit_age_curve(df), len(df)


def project_next(seasons: dict, years, p: Params, beta, min_total_mins=800) -> pd.DataFrame:
    """Per-stat season totals for the season after max(years), from those history years."""
    hist = [seasons[y] for y in sorted(years, reverse=True)]
    allids = sorted(set().union(*[set(h.index) for h in hist]))
    tot = sum(h.reindex(allids)["mins"].fillna(0) for h in hist)
    recent = hist[0].reindex(allids)["mins"].notna() | (hist[1].reindex(allids)["mins"].notna() if len(hist) > 1 else False)
    ids = pd.Index([i for i in allids if tot[i] >= min_total_mins and bool(recent[i])])
    core = project_core(hist, ids, p, league_means(hist[0]))
    age = _ages(hist, ids)
    lg = hist[0][hist[0]["mins"] >= 500]
    out = pd.DataFrame(index=ids)
    pm = {}
    for s in STATS:
        num = pd.Series(0., index=ids); den = pd.Series(0., index=ids)
        for i, x in enumerate(hist):
            x = x.reindex(ids); mins = x["mins"].fillna(0)
            num += p.lam ** i * (x[s] / x["MPG"].replace(0, np.nan)).fillna(0) * mins; den += p.lam ** i * mins
        lg_rate = (lg[s] * lg["G"]).sum() / lg["mins"].sum()
        pm[s] = (num + p.k_rate * lg_rate) / (den + p.k_rate)
    base_fpg = sum(W[s] * pm[s] for s in STATS) * core["mpg"]
    f = (apply_age_curve(base_fpg, age, beta) / base_fpg).fillna(1.0)
    for s in STATS:
        out[s] = pm[s] * f * core["mpg"] * core["games"]
    out["GP"], out["MPG"], out["age"] = core["games"], core["mpg"], age
    out["fpts"] = sum(W[s] * out[s] for s in STATS)
    meta = hist[0].reindex(ids)
    for x in hist[1:]:
        meta = meta.combine_first(x.reindex(ids))
    out["Name"], out["Team"], out["Pos"] = meta["Player"], meta["Tm"], meta["Pos"]
    return out.sort_values("fpts", ascending=False)


def to_app_frame(proj: pd.DataFrame, top_n=300) -> pd.DataFrame:
    """The columns data/nba/players.csv needs. Totals, not per game."""
    p = proj.head(top_n)
    return pd.DataFrame({
        "Name": p["Name"].values, "Team": p["Team"].values, "Pos": p["Pos"].astype(str).str.replace("-", ",").values,
        "playerid": ["n:" + str(n).lower() for n in p["Name"]], "GP": p["GP"].round(1).values,
        "MIN": (p["GP"] * p["MPG"]).round(1).values, "PTS": p["PTS"].round(1).values, "REB": p["TRB"].round(1).values,
        "AST": p["AST"].round(1).values, "STL": p["STL"].round(1).values, "BLK": p["BLK"].round(1).values,
        "3PM": p["TPM"].round(1).values, "TO": p["TOV"].round(1).values, "FGM": p["FGM"].round(1).values,
        "FGA": p["FGA"].round(1).values, "FTM": p["FTM"].round(1).values, "FTA": p["FTA"].round(1).values,
        "is_injured": False,
        "note": [("own model" + (f"; {n}" if isinstance(n, str) and n else "")) for n in p.get("note", pd.Series([""] * len(p), index=p.index))]})


# ---------------------------------------------------------------------------------------------------------------------
# Overrides: context the data cannot see (trades, role changes, rookies, returns from injury). One row per player.
# ---------------------------------------------------------------------------------------------------------------------
import unicodedata

PER_GAME_COLS = {"pts": "PTS", "reb": "TRB", "ast": "AST", "stl": "STL", "blk": "BLK", "tpm": "TPM", "tov": "TOV",
                 "fgm": "FGM", "fga": "FGA", "ftm": "FTM", "fta": "FTA"}


def norm_name(n) -> str:
    s = unicodedata.normalize("NFKD", str(n)).encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", s.replace(".", "").replace("'", "").replace("-", " ")).strip()


def load_overrides(path) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    d.columns = [c.strip().lower() for c in d.columns]
    if "name" not in d.columns:
        raise ValueError("overrides.csv needs a 'name' column")
    return d[d["name"].str.strip() != ""].reset_index(drop=True)


def _num(v, lo, hi, what, name):
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"{name}: {what} '{v}' is not a number.")
    if not (lo <= x <= hi):
        raise ValueError(f"{name}: {what} {x:g} is outside {lo:g}-{hi:g}.")
    return x


def apply_overrides(proj: pd.DataFrame, ov: pd.DataFrame):
    """Returns (new projection, report lines). Columns used (all optional except name):
         games, mpg          set projected games / minutes per game; counting stats rescale (per-minute rates stay the same)
         team, pos           replace the labels (pos like 'PF,C')
         note                free text kept on the player
       A name the model has never seen is ADDED as a new player when the row also has games, mpg and the per-game stats
       pts, reb, ast, stl, blk, tpm, tov, fgm, fga, ftm, fta."""
    out, report = proj.copy(), []
    out["note"] = ""
    key = {norm_name(n): i for i, n in out["Name"].items()}
    added = []
    for _, r in ov.iterrows():
        name = r["name"].strip()
        i = key.get(norm_name(name))
        get = lambda c: r.get(c, "").strip() if isinstance(r.get(c, ""), str) else ""
        try:
            if i is None:
                need = ["games", "mpg"] + list(PER_GAME_COLS)
                if any(get(c) == "" for c in need):
                    report.append(f"SKIPPED {name}: not in the model, and a new player needs games, mpg and all per-game stats.")
                    continue
                g, m = _num(get("games"), 1, 82, "games", name), _num(get("mpg"), 1, 48, "mpg", name)
                row = {col: _num(get(c), 0, 100, c, name) * g for c, col in PER_GAME_COLS.items()}
                row.update(GP=g, MPG=m, age=np.nan, Name=name, Team=get("team") or "FA", Pos=get("pos") or "SF", note=get("note") or "added by override")
                row["fpts"] = sum(W[s] * row[s] for s in STATS)
                added.append(row)
                report.append(f"ADDED   {name}: {g:g} games, {m:g} mpg")
                continue
            before = (out.at[i, "GP"], out.at[i, "MPG"])
            g = _num(get("games"), 1, 82, "games", name) if get("games") else before[0]
            m = _num(get("mpg"), 1, 48, "mpg", name) if get("mpg") else before[1]
            f = (g / before[0]) * (m / before[1])
            for s in STATS:
                out.at[i, s] = out.at[i, s] * f
            out.at[i, "GP"], out.at[i, "MPG"] = g, m
            if get("team"): out.at[i, "Team"] = get("team")
            if get("pos"): out.at[i, "Pos"] = get("pos")
            if get("note"): out.at[i, "note"] = get("note")
            out.at[i, "fpts"] = sum(W[s] * out.at[i, s] for s in STATS)
            changed = [x for x in ("games", "mpg", "team", "pos") if get(x)]
            report.append(f"UPDATED {name}: {', '.join(changed) or 'note only'}")
        except ValueError as e:
            report.append(f"SKIPPED {e}")
    if added:
        out = pd.concat([out, pd.DataFrame(added, index=[f"added_{k}" for k in range(len(added))])])
    return out.sort_values("fpts", ascending=False), report


# ---------------------------------------------------------------------------------------------------------------------
# Optional: pull the model toward a reference projection (another source's players.csv). Off by default.
# ---------------------------------------------------------------------------------------------------------------------
REF_COLS = {"PTS": "PTS", "TRB": "REB", "AST": "AST", "STL": "STL", "BLK": "BLK", "TPM": "3PM", "TOV": "TO",
            "FGM": "FGM", "FGA": "FGA", "FTM": "FTM", "FTA": "FTA"}


def blend_with_reference(proj: pd.DataFrame, ref: pd.DataFrame, w: float):
    """proj: project_next() output. ref: an app-format players.csv (season TOTALS: Name, GP, MIN, PTS, REB, ...).
    For every player in both: each total = (1-w) * own + w * reference, games and minutes blended the same way.
    Players only in the reference are ignored (add rookies through overrides). Returns (frame, number blended)."""
    if not (0.0 <= w <= 1.0):
        raise ValueError("blend must be between 0 and 1")
    out = proj.copy()
    if w == 0:
        return out, 0
    r = {norm_name(n): row for n, row in zip(ref["Name"], ref.to_dict("records"))}
    n = 0
    for i, name in out["Name"].items():
        row = r.get(norm_name(name))
        if row is None:
            continue
        gp_o, gp_r = float(out.at[i, "GP"]), float(row["GP"])
        mpg_o, mpg_r = float(out.at[i, "MPG"]), float(row["MIN"]) / gp_r
        for own_c, ref_c in REF_COLS.items():
            out.at[i, own_c] = (1 - w) * float(out.at[i, own_c]) + w * float(row[ref_c])
        out.at[i, "GP"] = (1 - w) * gp_o + w * gp_r
        out.at[i, "MPG"] = (1 - w) * mpg_o + w * mpg_r
        out.at[i, "fpts"] = sum(W[s] * out.at[i, s] for s in STATS)
        n += 1
    return out.sort_values("fpts", ascending=False), n

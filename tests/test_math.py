
import numpy as np
import pandas as pd
import pytest

from app.config import LeagueConfig, ScoringType
from app.optimizer import is_eligible, optimize_roster
from app.recommend import recommend
from app.valuation import calculate_player_valuations, zscore


def fake_projections(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    pos_pool = ["C", "1B", "2B", "3B", "SS", "OF", "OF", "OF"]
    for i in range(260):
        ab = int(rng.integers(150, 650)); h = int(ab * rng.normal(0.255, 0.025))
        bb = int(rng.integers(10, 80)); hr = int(rng.integers(2, 40))
        d, t = int(rng.integers(5, 35)), int(rng.integers(0, 6))
        rows.append(dict(player_id=f"h{i}", name=f"Hitter {i}", kind="H",
                         eligible_positions=[pos_pool[i % 8]] + (["OF"] if i % 11 == 0 else []),
                         is_injured=(i % 40 == 0), games_pct=1.0 if i % 40 else 0.6,
                         AB=ab, PA=ab + bb, H=h, BB=bb, HBP=3, R=int(rng.integers(20, 110)), HR=hr,
                         RBI=int(rng.integers(20, 110)), SB=int(rng.integers(0, 35)),
                         SO=int(rng.integers(40, 180)), doubles=d, triples=t, singles=h - hr - d - t))
    for i in range(220):
        ip = float(rng.integers(40, 200)) if i % 3 else float(rng.integers(40, 75))
        sp = i % 3 != 0
        rows.append(dict(player_id=f"p{i}", name=f"Pitcher {i}", kind="P",
                         eligible_positions=["SP"] if sp else ["RP"], is_injured=False, games_pct=1.0,
                         IP=ip, W=int(ip / 15) if sp else 3, L=int(ip / 18), SV=0 if sp else int(rng.integers(0, 35)),
                         K=int(ip * rng.normal(9, 1.5) / 9), QS=int(ip / 11) if sp else 0,
                         ER=int(ip * rng.normal(3.8, 0.7) / 9), H_allowed=int(ip * 0.95),
                         BB_allowed=int(ip * 0.33)))
    return pd.DataFrame(rows)


def test_zscore_zero_variance():
    s = pd.Series([5.0, 5.0, 5.0])
    assert (zscore(s, pd.Series([True] * 3)) == 0).all()


def test_util_excludes_pitchers():
    assert not is_eligible(["SP"], "UTIL")
    assert is_eligible(["SS"], "MI") and is_eligible(["1B"], "CI") and is_eligible(["2B"], "INF")


@pytest.mark.parametrize("fmt", list(ScoringType))
def test_every_format_runs_end_to_end(fmt):
    raw, cfg = fake_projections(), LeagueConfig(scoring_type=fmt)
    df, repl = calculate_player_valuations(raw, cfg)
    assert df["vorp"].notna().all() and len(df) == len(raw)
    # picks make replacement levels move and drafted players unavailable
    top = df.nlargest(30, "vorp")["player_id"].tolist()
    df2, repl2 = calculate_player_valuations(raw, cfg, set(top))
    assert not df2.loc[df2["player_id"].isin(top), "available"].any()
    team = df2[df2["player_id"].isin(top[:10])]
    res = optimize_roster(team, cfg.roster, repl2)
    assert all(a["slot"] != "UTIL" or "SP" not in team.set_index("player_id").loc[a["player_id"], "eligible_positions"]
               for a in res["assignments"])
    recs = recommend(df2, repl2, cfg, top[:5], top_k=15, limit=5)
    assert len(recs) == 5


def test_era_not_double_inverted():
    raw = fake_projections()
    df, _ = calculate_player_valuations(raw, LeagueConfig())
    p = df[df["kind"] == "P"]
    sp = p[p["IP"] > 100]
    corr = np.corrcoef(sp["z_ERA"], -(sp["ER"] / sp["IP"]))[0, 1]
    assert corr > 0.9   # lower ERA => higher z


def test_injury_history_math():
    from app.ingestion import compute_games_missed
    rows = []
    for s, g in [(2024, 162), (2025, 162), (2026, 162)]:
        rows.append(dict(player_id="durable", season=s, games_played=g))
    for s, g in [(2024, 100), (2025, 110), (2026, 90)]:
        rows.append(dict(player_id="fragile", season=s, games_played=g))
    for s, g in [(2024, 150), (2026, 150)]:                       # whole 2025 absent after debut
        rows.append(dict(player_id="lost_year", season=s, games_played=g))
    rows.append(dict(player_id="rookie", season=2026, games_played=140))
    rows.append(dict(player_id="arm", season=2026, games_played=20))   # pitcher, no games_possible
    kinds = {"durable": "H", "fragile": "H", "lost_year": "H", "rookie": "H", "arm": "P"}
    out = compute_games_missed(pd.DataFrame(rows), kinds).set_index("player_id")
    assert out.loc["durable", "avg_games_missed_pct"] == 0
    assert abs(out.loc["fragile", "avg_games_missed_pct"] - (1 - 100 / 162 + 1 - 110 / 162 + 1 - 90 / 162) / 3) < 1e-9
    assert out.loc["lost_year", "injury_seasons_used"] == 3 and out.loc["lost_year", "avg_games_missed_pct"] > 0.3
    assert out.loc["rookie", "injury_seasons_used"] == 1
    assert out.loc["arm", "injury_seasons_used"] == 0


def test_injury_prone_flag_and_penalty():
    raw = fake_projections()
    raw["avg_games_missed_pct"] = 0.0
    raw["injury_seasons_used"] = 3
    base_cfg = LeagueConfig()
    df0, _ = calculate_player_valuations(raw, base_cfg)
    star = df0[df0["kind"] == "H"].nlargest(1, "score").iloc[0]["player_id"]
    raw.loc[raw["player_id"] == star, "avg_games_missed_pct"] = 0.25     # exactly at threshold -> prone
    raw.loc[raw["player_id"] == "h1", ["avg_games_missed_pct", "injury_seasons_used"]] = [0.5, 1]  # too little history
    df1, _ = calculate_player_valuations(raw, base_cfg)
    assert df1.set_index("player_id").loc[star, "injury_prone"]
    assert not df1.set_index("player_id").loc["h1", "injury_prone"]
    assert df1["injury_prone"].sum() == 1
    assert abs(df1.set_index("player_id").loc[star, "score"] - df0.set_index("player_id").loc[star, "score"]) < 1e-9
    cfg2 = LeagueConfig(injury_prone_penalty=0.2)
    df2, _ = calculate_player_valuations(raw, cfg2)
    s0 = df0.set_index("player_id").loc[star, "score"]
    assert s0 > 0 and abs(df2.set_index("player_id").loc[star, "score"] - 0.8 * s0) < 1e-9


def test_avoid_injury_prone_setting():
    raw = fake_projections()
    raw["avg_games_missed_pct"] = 0.0
    raw["injury_seasons_used"] = 3
    top = calculate_player_valuations(raw, LeagueConfig())[0].nlargest(6, "vorp")["player_id"].tolist()
    raw.loc[raw["player_id"].isin(top[:3]), "avg_games_missed_pct"] = 0.4

    keep = LeagueConfig()
    avoid = LeagueConfig(avoid_injury_prone=True)
    df_keep, repl_keep = calculate_player_valuations(raw, keep)
    df_avoid, repl_avoid = calculate_player_valuations(raw, avoid)

    # default: flagged players stay in the recommendations
    assert any(r["injury_prone"] for r in recommend(df_keep, repl_keep, keep, [], top_k=40, limit=40))
    # standing setting: none of them show up, and the next best players fill in
    recs = recommend(df_avoid, repl_avoid, avoid, [], top_k=40, limit=10)
    assert len(recs) == 10 and not any(r["injury_prone"] for r in recs)
    # per-call override beats the config in both directions
    assert not any(r["injury_prone"] for r in recommend(df_keep, repl_keep, keep, [], limit=40, exclude_injury_prone=True))
    assert any(r["injury_prone"] for r in recommend(df_avoid, repl_avoid, avoid, [], limit=40, exclude_injury_prone=False))
    # replacement levels are drawn from the pool you would actually pick from
    assert all(repl_avoid[k] <= repl_keep[k] + 1e-9 for k in repl_keep)


def test_views_show_both_injury_kinds_and_list_flagged():
    import json
    from app.views import available_view, injury_prone_view
    raw = fake_projections()
    raw["avg_games_missed_pct"] = 0.0
    raw["injury_seasons_used"] = 3
    flagged = ["h5", "h6", "h7"]
    raw.loc[raw["player_id"].isin(flagged), "avg_games_missed_pct"] = 0.4
    cfg = LeagueConfig()
    df, repl = calculate_player_valuations(raw, cfg, {"h5"})        # h5 already drafted

    rows = available_view(df, cfg, limit=500)
    json.dumps(rows)                                                # must be JSON-safe
    assert {"injury_prone", "is_injured", "games_pct", "avg_games_missed_pct"} <= set(rows[0])
    injured_row = next(r for r in rows if r["player_id"] == "h0")   # fake data marks h0 is_injured
    assert injured_row["is_injured"] is True and injured_row["games_pct"] == 0.6
    assert next(r for r in rows if r["player_id"] == "h6")["injury_prone"] is True
    assert isinstance(rows[0]["injury_seasons_used"], int)

    res = injury_prone_view(df)
    json.dumps(res)
    assert res["count"] == 2 and {p["player_id"] for p in res["players"]} == {"h6", "h7"}
    vorps = [p["vorp"] for p in res["players"]]
    assert vorps == sorted(vorps, reverse=True)
    res_all = injury_prone_view(df, include_drafted=True)
    assert res_all["count"] == 3 and any(not p["available"] for p in res_all["players"])
    assert injury_prone_view(df, limit=1)["count"] == 2 and len(injury_prone_view(df, limit=1)["players"]) == 1
    # avoid setting hides them from available but NOT from the injury-prone list
    avoid = LeagueConfig(avoid_injury_prone=True)
    df2, _ = calculate_player_valuations(raw, avoid)
    assert not any(r["injury_prone"] for r in available_view(df2, avoid, limit=500))
    assert injury_prone_view(df2)["count"] == 3

    recs = recommend(df, repl, cfg, [], top_k=10, limit=3)
    assert {"is_injured", "games_pct", "injury_prone"} <= set(recs[0])


def test_rankings():
    import json
    from app.valuation import add_ranks
    from app.views import available_view, rankings_view
    raw = fake_projections()
    cfg = LeagueConfig()
    pre, _ = calculate_player_valuations(raw, cfg)
    top5 = pre.nlargest(5, "vorp")["player_id"].tolist()
    drafted = set(top5[:2])
    now, _ = calculate_player_valuations(raw, cfg, drafted)
    ranked = add_ranks(now, pre).set_index("player_id")
    base = add_ranks(pre, pre).set_index("player_id")

    # projected_rank is unique, 1..N, and does NOT move when players are drafted
    assert sorted(ranked["projected_rank"]) == list(range(1, len(raw) + 1))
    assert (ranked["projected_rank"] == base["projected_rank"]).all()
    assert ranked.loc[top5[0], "projected_rank"] == 1
    # available_rank skips drafted players, starting at 1 with the best one left
    assert ranked.loc[list(drafted), "available_rank"].isna().all()
    assert ranked.loc[top5[2], "available_rank"] == 1
    assert sorted(ranked["available_rank"].dropna()) == list(range(1, len(raw) - 2 + 1))
    # pos_rank is per-position and starts at 1
    assert ranked["pos_rank"].str.match(r"^[A-Z0-9]+\d+$").all()
    c_best = ranked[ranked["primary_pos"] == "C"].sort_values("projected_rank").iloc[0]
    assert c_best["pos_rank"] == "C1"

    full = add_ranks(now, pre)
    rows = rankings_view(full, limit=10)
    json.dumps(rows)
    assert [r["projected_rank"] for r in rows] == list(range(1, 11)) and isinstance(rows[0]["projected_rank"], int)
    assert rows[0]["available"] is False                     # drafted players stay on the board
    assert all(r["available"] for r in rankings_view(full, limit=50, include_drafted=False))
    assert all(r["kind"] == "P" for r in rankings_view(full, limit=20, kind="p"))
    assert all("SS" in r["eligible_positions"] for r in rankings_view(full, limit=20, position="ss"))
    avail = available_view(full, cfg, limit=5)
    assert avail[0]["available_rank"] == 1 and "projected_rank" in avail[0] and "pos_rank" in avail[0]
    recs = recommend(full, _, cfg, [], top_k=10, limit=3)
    assert all(r["projected_rank"] is not None for r in recs)


# ---------------- player search / stats ----------------
def _search_fixture():
    raw = pd.DataFrame([
        dict(player_id="1", name="Eli Perez", kind="H", team="TOR", AB=600, PA=680, H=180, singles=100, doubles=30,
             triples=2, HR=48, R=120, RBI=110, SB=10, BB=60, SO=100, HBP=5),
        dict(player_id="2", name="Ryan Perez", kind="H", team="SDP", AB=500, PA=560, H=125, singles=80, doubles=25,
             triples=2, HR=18, R=70, RBI=65, SB=5, BB=45, SO=110, HBP=3),
        dict(player_id="3", name="José Ramírez", kind="H", team="CLE", AB=550, PA=620, H=160, singles=95, doubles=35,
             triples=3, HR=27, R=95, RBI=90, SB=25, BB=60, SO=70, HBP=5),
        dict(player_id="4", name="Zane Reyes", kind="P", team="BOS", IP=180.0, W=14, L=8, SV=0, QS=17, K=190,
             H_allowed=160, BB_allowed=50, ER=70),
    ])
    raw["is_injured"], raw["games_pct"] = False, 1.0
    raw["eligible_positions"] = [["1B"], ["SS"], ["3B"], ["SP"]]
    raw["projected_rank"] = [1, 40, 3, 7]
    raw["pos_rank"] = ["1B1", "SS9", "3B1", "SP1"]
    hist = {"H": pd.DataFrame([
        dict(player_id="1", season=2025, G=150, PA=600, AB=500, H=150, singles=90, doubles=30, triples=0, HR=30,
             R=90, RBI=95, BB=80, SO=120, HBP=10, SF=10, SB=5),
        dict(player_id="1", season=2026, G=40, PA=110, AB=100, H=40, singles=20, doubles=10, triples=0, HR=10,
             R=25, RBI=30, BB=5, SO=20, HBP=3, SF=2, SB=1)]),
        "P": pd.DataFrame([
        dict(player_id="4", season=2025, G=30, GS=30, IP=100.0, W=8, L=5, SV=0, QS=10, K=100, H_allowed=90,
             BB_allowed=30, ER=40, HR_allowed=10),
        dict(player_id="4", season=2026, G=30, GS=30, IP=50.0, W=4, L=3, SV=0, QS=5, K=60, H_allowed=40,
             BB_allowed=10, ER=10, HR_allowed=4)])}
    return raw, hist


def test_parse_query_and_stat_names():
    from app.stats import normalize_stat, parse_query
    assert parse_query("Eli Perez stats") == ("Eli Perez", None)
    assert parse_query("eli perez obp") == ("eli perez", "obp")
    assert parse_query("Eli Perez stats K/9") == ("Eli Perez", "K/9")
    assert parse_query("Perez") == ("Perez", None)
    assert normalize_stat("obp", "H") == "OBP" and normalize_stat("Batting Average", "H") == "AVG"
    assert normalize_stat("k9", "P") == "K/9" and normalize_stat("strikeouts", "H") == "SO"
    assert normalize_stat("strikeouts", "P") == "K" and normalize_stat("ERA", "H") is None


def test_shared_name_resolved_by_the_stat_asked_for():
    from app.stats import search_flow
    raw, hist = _search_fixture()
    twin = raw.iloc[[3]].copy()                                   # a PITCHER also named Eli Perez
    twin["player_id"], twin["name"], twin["projected_rank"] = "8", "Eli Perez", 30
    raw2 = pd.concat([raw, twin], ignore_index=True)
    assert search_flow(raw2, raw2, hist, "Eli Perez stats")["status"] == "choose_player"
    assert search_flow(raw2, raw2, hist, "Eli Perez", "HR")["status"] == "choose_player"      # HR exists for both
    h = search_flow(raw2, raw2, hist, "Eli Perez obp")
    assert h["status"] == "ok" and h["player"]["kind"] == "H" and h["player"]["player_id"] == "1"
    p = search_flow(raw2, raw2, hist, "Eli Perez", "ERA")
    assert p["status"] == "ok" and p["player"]["kind"] == "P" and p["player"]["player_id"] == "8"


def test_typo_picks_the_closest_name_only():
    from app.stats import search_flow, search_players
    raw, hist = _search_fixture()
    extra = raw.iloc[[1]].copy()
    extra["player_id"], extra["name"], extra["projected_rank"] = "9", "Eli Lopez", 12
    raw2 = pd.concat([raw, extra], ignore_index=True)
    m = search_players(raw2, "Eli Perze")
    assert m[0]["name"] == "Eli Perez" and m[0]["match"] == "closest spelling"
    assert [x["match"] for x in m[1:]] == ["similar spelling"]
    r = search_flow(raw2, raw2, hist, "Eli Perze", "HR")
    assert r["status"] == "ok" and r["player"]["name"] == "Eli Perez"
    assert type(r["player"]["positions"][0]) is str


def test_search_matching():
    from app.stats import search_flow, search_players
    raw, hist = _search_fixture()
    assert [m["player_id"] for m in search_players(raw, "perez")] == ["1", "2"]          # rank order
    assert search_players(raw, "jose ramirez")[0]["player_id"] == "3"                    # accents ignored
    assert search_players(raw, "Eli Perze")[0]["match"] == "closest spelling"             # typo
    assert search_players(raw, "xyzzy") == []
    assert search_flow(raw, raw, hist, "perez")["status"] == "choose_player"
    assert search_flow(raw, raw, hist, "nobody here")["status"] == "no_match"
    typo = search_flow(raw, raw, hist, "Eli Perze", "AVG")
    assert typo["status"] == "ok" and "closest match" in typo["notes"][0]


def test_hitter_stat_report_uses_true_career_rates():
    from app.stats import search_flow
    raw, hist = _search_fixture()
    ask = search_flow(raw, raw, hist, "Eli Perez stats")
    assert ask["status"] == "choose_stat" and "OBP" in ask["stats"] and "ERA" not in ask["stats"]
    r = search_flow(raw, raw, hist, "Eli Perez", "avg")
    assert r["previous_season"]["display"] == ".400"                  # 2026: 40/100
    assert r["career"]["display"] == ".317"                           # 190/600, NOT the mean of .300 and .400
    assert r["projected"]["display"] == ".300"                        # 180/600
    assert r["previous_season_year"] == 2026
    assert r["lines"][0] == "2026 (previous season) AVG: .400"
    assert r["lines"][1] == "Career AVG: .317" and r["lines"][2] == "Projected AVG: .300"
    # OBP includes HBP and SF: career (190+85+13) / (600+85+13+12)
    obp = search_flow(raw, raw, hist, "Eli Perez obp")
    assert obp["career"]["display"] == f"{(190 + 85 + 13) / (600 + 85 + 13 + 12):.3f}".replace("0.", ".", 1)
    ops = search_flow(raw, raw, hist, "Eli Perez", "OPS")["career"]["value"]
    slg = (190 + 40 + 2 * 0 + 3 * 40) / 600
    assert abs(ops - round((288 / 710) + slg, 3)) < 0.0011
    hr = search_flow(raw, raw, hist, "Eli Perez", "HR")
    assert (hr["previous_season"]["value"], hr["career"]["value"], hr["projected"]["value"]) == (10, 40, 48)
    assert search_flow(raw, raw, hist, "Eli Perez", "ERA")["status"] == "unknown_stat"
    allr = search_flow(raw, raw, hist, "Eli Perez", "all")
    assert allr["status"] == "ok" and {s["stat"] for s in allr["stats"]} >= {"AVG", "OBP", "OPS", "HR"}


def test_pitcher_report_and_missing_history():
    from app.stats import search_flow
    raw, hist = _search_fixture()
    r = search_flow(raw, raw, hist, "Zane Reyes", "ERA")
    assert r["previous_season"]["display"] == "1.80"                  # 9*10/50
    assert r["career"]["display"] == "3.00"                           # 9*50/150, not mean(3.60, 1.80)
    assert r["projected"]["display"] == "3.50"                        # 9*70/180
    whip = search_flow(raw, raw, hist, "Zane Reyes", "whip")
    assert whip["career"]["display"] == "1.13"                         # (BB 40 + H 130) / 150 IP
    ksum = search_flow(raw, raw, hist, "Zane Reyes", "K")               # 'K' means strikeouts for a pitcher
    assert (ksum["previous_season"]["value"], ksum["career"]["value"], ksum["projected"]["value"]) == (60, 160, 190)
    # a player with no history rows: stats are n/a with an explanation, projection still works
    nohist = search_flow(raw, raw, {"H": hist["H"], "P": pd.DataFrame()}, "Zane Reyes", "ERA")
    assert nohist["previous_season"]["display"] == "n/a" and nohist["projected"]["display"] == "3.50"
    assert "hitter_history" not in nohist["notes"][0] and "pitcher_history.csv" in nohist["notes"][0]
    # played before but not in the latest season
    ramirez = search_flow(raw, raw, hist, "Jose Ramirez", "HR")
    assert ramirez["career"]["display"] == "n/a" and ramirez["projected"]["value"] == 27
    rookie_gap = {"H": pd.concat([hist["H"], pd.DataFrame([dict(player_id="3", season=2025, G=100, AB=300, H=90, HR=20)])]),
                  "P": hist["P"]}
    gap = search_flow(raw, raw, rookie_gap, "Jose Ramirez", "HR")
    assert gap["previous_season"]["display"] == "n/a" and gap["career"]["value"] == 20 and "No 2026" in gap["notes"][0]


def test_history_loader_baseball_innings_and_sums():
    import tempfile
    from pathlib import Path
    from app import ingestion
    old = ingestion.DATA_DIR
    with tempfile.TemporaryDirectory() as d:
        ingestion.DATA_DIR = Path(d)
        try:
            pd.DataFrame([
                dict(Season=2026, playerid="4", G=20, GS=20, IP=180.1, W=10, L=5, SV=0, QS=12, SO=150, H=150, ER=60, BB=40, HR=15),
                dict(Season=2025, playerid="4", G=10, GS=10, IP=50.2, W=3, L=2, SV=0, QS=4, SO=40, H=50, ER=20, BB=15, HR=5),
                dict(Season=2025, playerid="4", G=10, GS=10, IP=50.0, W=3, L=2, SV=0, QS=4, SO=40, H=50, ER=20, BB=15, HR=5),
            ]).to_csv(Path(d) / "pitcher_history.csv", index=False)
            h = ingestion.load_stats_history()
        finally:
            ingestion.DATA_DIR = old
    assert h["H"].empty                                              # missing file -> empty frame, no crash
    p = h["P"].set_index("season")
    assert abs(p.loc[2026, "IP"] - (180 + 1 / 3)) < 1e-9             # 180.1 = 180 1/3
    assert abs(p.loc[2025, "IP"] - (50 + 2 / 3 + 50)) < 1e-9         # two rows for one season are summed
    assert p.loc[2025, "K"] == 80 and p.loc[2025, "H_allowed"] == 100 and p.loc[2026, "HR_allowed"] == 15


# ---------------- data validation / two-way players ----------------
def _fp():
    raw = fake_projections()
    raw["player_id"] = raw["player_id"].astype(str)
    return raw


def test_check_config_names_the_missing_column():
    from app.validate import check_config
    raw = _fp()
    assert check_config(raw, LeagueConfig()) == []
    no_sb = raw.drop(columns=["SB"])
    msgs = check_config(no_sb, LeagueConfig())
    assert any("'SB'" in m and "SB" in m for m in msgs)
    assert any("Unsupported" in m for m in check_config(raw, LeagueConfig(categories={"H": ["R", "OPS"], "P": ["W"]})))
    assert any("'QS'" in m for m in check_config(raw.assign(QS=float("nan")), LeagueConfig(categories={"H": ["R"], "P": ["QS"]})))
    assert check_config(raw, LeagueConfig(scoring_type=ScoringType.POINTS)) == []
    assert check_config(raw[raw["kind"] == "H"], LeagueConfig()) != []        # no pitchers at all


def test_clean_projections_fixes_and_reports():
    from app.validate import clean_projections
    raw = _fp()
    raw["HR"] = raw["HR"].astype(object)
    raw.loc[raw["player_id"] == "h3", "HR"] = "n/a"                          # text in a number column
    raw.loc[raw["player_id"] == "h4", "RBI"] = float("nan")                  # blank cell
    dup = raw[raw["player_id"] == "h5"]
    raw = pd.concat([raw, dup], ignore_index=True)                           # duplicate row
    clean, actions = clean_projections(raw)
    assert len(clean) == len(raw) - 1
    assert clean.loc[clean["player_id"] == "h3", "HR"].iloc[0] == 0
    assert clean.loc[clean["player_id"] == "h4", "RBI"].iloc[0] == 0
    assert any("duplicate" in a for a in actions) and any("HR" in a for a in actions) and any("RBI" in a for a in actions)
    assert not clean[clean["kind"] == "H"][["HR", "RBI", "R"]].isna().any().any()   # pitchers legitimately have none
    df, _ = calculate_player_valuations(clean, LeagueConfig())               # and it now values cleanly
    assert df["vorp"].notna().all()
    assert not clean[clean["kind"] == "P"]["IP"].isna().any()


def test_two_way_player_is_one_human():
    from app.ingestion import link_two_way
    from app.valuation import add_ranks, expand_twins
    raw = _fp()
    pit = raw[raw["kind"] == "P"].iloc[[0]].copy()
    pit["player_id"] = "h7"                                                  # same id as hitter h7 -> a two-way player
    both = link_two_way(pd.concat([raw, pit], ignore_index=True))
    assert not both["player_id"].duplicated().any()
    h7 = both[both["base_id"] == "h7"].set_index("kind")
    assert h7.loc["P", "player_id"] == "h7_P" and h7.loc["H", "twin_id"] == "h7_P" and h7.loc["P", "twin_id"] == "h7"
    assert expand_twins(both, {"h7"}) == {"h7", "h7_P"} and expand_twins(both, {"h7_P"}) == {"h7", "h7_P"}
    assert expand_twins(both, {"h1"}) == {"h1"}
    cfg = LeagueConfig()
    pre, _ = calculate_player_valuations(both, cfg)
    now, _ = calculate_player_valuations(both, cfg, {"h7"})
    assert not now.set_index("player_id").loc[["h7", "h7_P"], "available"].any()      # drafting one removes both
    ranked = add_ranks(now, pre)
    assert len(ranked) == len(both) and sorted(ranked["projected_rank"]) == list(range(1, len(both) + 1))


def test_health_report_flags_real_problems():
    from app.validate import build_report
    raw = _fp()
    ok = build_report(raw, LeagueConfig())
    assert ok["ok"] and ok["errors"] == [] and ok["info"]["hitters"] == 260
    no_c = raw[~raw["eligible_positions"].map(lambda ps: "C" in ps)]
    rep = build_report(no_c, LeagueConfig())
    assert not rep["ok"] and any("eligible at C" in e for e in rep["errors"])
    bad = raw.copy()
    bad.loc[bad["player_id"] == "h2", "H"] = bad.loc[bad["player_id"] == "h2", "AB"] + 50
    bad.loc[bad["player_id"] == "p4", "IP"] = 400.0
    w = build_report(bad, LeagueConfig())["warnings"]
    assert any("impossible hits" in x for x in w) and any("more than 300 innings" in x for x in w)
    dup = pd.concat([raw, raw.iloc[[0]]], ignore_index=True)
    assert any("Duplicate player_id" in e for e in build_report(dup, LeagueConfig())["errors"])
    hist = {"H": pd.DataFrame([dict(player_id="zzz", season=2026, G=100)]), "P": pd.DataFrame()}
    r = build_report(raw, LeagueConfig(), hist)
    assert any("match your hitter_history.csv" in x for x in r["warnings"]) and r["info"]["pitcher_history"] == "not loaded"


def test_stats_history_follows_base_id_for_two_way_pitcher_row():
    from app.stats import search_flow
    raw, hist = _search_fixture()
    raw["base_id"] = raw["player_id"]
    raw.loc[raw["player_id"] == "4", ["player_id", "base_id"]] = ["4_P", "4"]   # pitcher half of a two-way player
    r = search_flow(raw, raw, hist, "Zane Reyes", "ERA")
    assert r["player"]["player_id"] == "4_P"
    assert r["previous_season"]["display"] == "1.80" and r["career"]["display"] == "3.00"


# ---------------- draft setup ----------------
def test_draft_setup_validation():
    from app.draft import validate_setup
    ok = lambda **kw: validate_setup({"draft_type": "snake", "teams": 12, "my_slot": 5, **kw}, 25)
    s, e = ok(); assert s and s.teams == 12 and s.my_slot == 5 and s.rounds == 25 and e == {}
    assert ok(teams="12", my_slot=" 5 ")[0].my_slot == 5                      # strings and spaces are fine
    assert "between" not in ok(my_slot=12)[1].get("my_slot", "") and ok(my_slot=12)[0]   # 12 of 12 is valid
    assert "higher than 12" in ok(my_slot=13)[1]["my_slot"]
    assert "lower than 1" in ok(my_slot=0)[1]["my_slot"]
    for bad in ("abc", "-3", "4.5", "1e3", True, [1]):
        assert "whole number" in ok(my_slot=bad)[1]["my_slot"], bad
    assert "my_slot" in ok(my_slot="")[1] and "teams" in ok(teams="")[1]
    assert "between 2 and 30" in ok(teams=1)[1]["teams"] and "between 2 and 30" in ok(teams=31)[1]["teams"]
    assert "whole number" in ok(teams="twelve")[1]["teams"]
    assert "draft_type" in ok(draft_type="draft")[1]
    assert set(ok(teams="", my_slot=99)[1]) == {"teams"}                       # no confusing second error
    assert "rounds" in ok(rounds=0)[1] and "rounds" in ok(rounds=61)[1] and ok(rounds=10)[0].rounds == 10
    a = lambda **kw: validate_setup({"draft_type": "auction", "teams": 12, **kw}, 25)
    assert a(budget=260)[0].budget == 260 and a(budget=260)[0].my_slot is None   # nomination slot optional
    assert "budget" in a()[1] and "budget" in a(budget=0)[1] and "$1" in a(budget=20)[1]["budget"]
    assert a(budget=260, my_slot=13)[1]["my_slot"]


def test_pick_order_and_status():
    from app.draft import DraftSetup, slot_for_pick, status
    assert [slot_for_pick("snake", 12, n) for n in (1, 12, 13, 24, 25)] == [1, 12, 12, 1, 1]
    assert [slot_for_pick("linear", 12, n) for n in (1, 12, 13, 24, 25)] == [1, 12, 1, 12, 1]
    s = DraftSetup("snake", 12, 3, 25, None)
    st = status(s, 0)
    assert st["my_pick_numbers"][:3] == [3, 22, 27] and st["picks_until_my_turn"] == 2 and not st["you_are_on_the_clock"]
    assert status(s, 2)["you_are_on_the_clock"] and status(s, 2)["picks_until_my_turn"] == 0
    st = status(s, 3)
    assert st["my_next_pick"] == 22 and st["picks_until_my_turn"] == 18 and st["my_next_round"] == 2
    assert st["round"] == 1 and st["pick_in_round"] == 4 and st["on_clock_slot"] == 4 and st["my_picks_made"] == 1
    assert status(s, 12)["round_order"][:2] == [12, 11] and status(s, 12)["on_clock_slot"] == 12
    last = DraftSetup("snake", 12, 12, 25, None)
    assert last_picks(last)[:2] == [12, 13]                                    # the corner gets back-to-back picks
    done = status(s, 300)
    assert done["complete"] and "current_pick" not in done
    assert status(s, 290)["my_next_pick"] == 291 and status(s, 290)["picks_until_my_turn"] == 0
    late = status(s, 299)                                                      # all of your picks are behind you
    assert not late["complete"] and late["my_next_pick"] is None and late["picks_until_my_turn"] is None
    assert status(DraftSetup("auction", 12, None, 25, 260), 4)["budget"] == 260


def last_picks(s):
    from app.draft import status
    return status(s, 0)["my_pick_numbers"]


# ---------------- team score, pace benchmark, live sync ----------------
def _scorer(fmt=ScoringType.ROTO, teams=12):
    from app.scoring import TeamScorer
    cfg = LeagueConfig(scoring_type=fmt)
    cfg.roster.teams = teams
    pre, repl = calculate_player_valuations(_fp(), cfg)
    return TeamScorer(pre, repl, cfg), pre, repl, cfg


def test_team_scorer_matches_optimizer_and_summary_lists_empty_slots():
    from app.scoring import summary
    sc, pre, repl, cfg = _scorer()
    ids = pre.nlargest(8, "vorp")["player_id"].tolist()
    total, _ = sc.assign([sc.pos[i] for i in ids])
    assert abs(total - optimize_roster(pre[pre["player_id"].isin(ids)], cfg.roster, repl)["total_value"]) < 1e-6
    s = summary(sc, ids + [None])                                       # an unmatched pick is ignored
    assert len(s["roster"]) == len(cfg.roster.slot_list()) == 25
    assert sum(r["name"] is not None for r in s["roster"]) == 8 and any(r["name"] is None for r in s["roster"])
    assert summary(sc, [])["score"] == 0.0
    for fmt in ScoringType:                                             # every format produces a number
        sc2, pre2, _, _ = _scorer(fmt)
        assert isinstance(summary(sc2, pre2.nlargest(5, "vorp")["player_id"].tolist())["score"], float)
    more = summary(sc, pre.nlargest(12, "vorp")["player_id"].tolist())["score"]
    assert more > s["score"]                                            # 12 of the best players beat 8


def test_benchmark_simulation_and_verdicts():
    import numpy as np
    from app.scoring import compare, simulate
    sc, *_ = _scorer(teams=4)
    a = simulate(sc, "snake", 4, 10, runs=6)
    b = simulate(sc, "snake", 4, 10, runs=6)
    assert a["mean"].shape == (4, 10) and (a["mean"] == b["mean"]).all()          # deterministic
    assert a["mean"][0, 0] >= a["mean"][3, 0]                                     # slot 1 gets the better first player
    assert a["mean"][0, -1] > a["mean"][0, 0]                                     # a full roster beats one player
    bench = {"mean": np.array([[10.0, 20.0], [8.0, 16.0], [6.0, 12.0]]), "sd": np.array([[1.0, 2.0]] * 3), "runs": 5, "teams": 3, "rounds": 2}
    assert compare(bench, 2, 0, 0)["status"] == "no_picks"
    assert compare(bench, 1, 10.0, 1)["verdict"] == "on track"
    assert compare(bench, 1, 11.0, 1)["verdict"] == "ahead" and compare(bench, 1, 11.0, 1)["gap"] == 1.0
    assert compare(bench, 1, 8.5, 1)["verdict"] == "behind" and compare(bench, 1, 6.0, 1)["verdict"] == "well behind"
    assert compare(bench, 3, 12.0, 2)["expected_final"] == 12.0 and len(compare(bench, 3, 12.0, 2)["expected_final_by_slot"]) == 3
    assert compare(bench, 1, 20.0, 5)["expected_now"] == 20.0                     # more picks than rounds is capped


class _FakeRemote:
    def __init__(self, picks): self.picks, self.fail = picks, False
    async def fetch_picks(self):
        if self.fail: raise ConnectionError("boom")
        from app.connectors.base import RemotePick
        return [RemotePick(n, pid, str(team)) for n, (pid, team) in enumerate(self.picks, 1)]
    def team_index(self, t): return int(t)
    def my_team_id(self): return 2


def test_sync_builds_log_events_and_survives_failures():
    import asyncio
    from app import sync
    from app.state import GlobalState
    from app.connectors.base import RemotePick
    log, unmapped = sync.build_log([RemotePick(1, "y1", "1"), RemotePick(2, "yX", "2")], {"y1": "h1"}, int)
    assert [p["player_id"] for p in log] == ["h1", None] and unmapped == ["yX"]       # unmatched keeps its pick slot
    ev = sync.diff_events(log[:1], log, 2, {"h1": "A"})
    assert len(ev) == 1 and ev[0]["mine"] and ev[0]["pick_number"] == 2
    assert sync.diff_events(log, log[:1], 2, {})[0]["type"] == "undo"

    st = GlobalState(); st.names = {"h1": "One", "h2": "Two", "h3": "Three"}
    remote = _FakeRemote([("y1", 1), ("y2", 2)])
    async def go():
        t = asyncio.create_task(sync.run(st, lambda: remote, lambda: {"y1": "h1", "y2": "h2", "y3": "h3"}, 0.01, retry_delay=0.01))
        await asyncio.sleep(0.15)
        assert [p["player_id"] for p in st.draft_log] == ["h1", "h2"] and st.my_team_id == 2 and st.sync["connected"]
        remote.picks.append(("y3", 2)); await asyncio.sleep(0.1)
        assert st.events[-1]["player"] == "Three" and st.events[-1]["mine"] and st.team_player_ids(2) == ["h2", "h3"]
        remote.fail = True; await asyncio.sleep(0.1)
        assert not st.sync["connected"] and "boom" in st.sync["last_error"] and len(st.draft_log) == 3   # picks kept
        remote.fail = False; remote.picks.pop(); await asyncio.sleep(0.1)
        assert st.sync["connected"] and len(st.draft_log) == 2 and st.events[-1]["type"] == "undo"
        t.cancel()
    asyncio.run(go())


def test_mock_source_follows_snake_order():
    import asyncio
    from app.connectors.mock import MockSource
    ids = [f"p{i}" for i in range(100)]
    m = MockSource(ids, teams=4, rounds=5, draft_type="snake", my_slot=2, seconds_per_pick=1e-9)
    picks = asyncio.run(m.fetch_picks())
    assert len(picks) == 20 and len({p.platform_player_id for p in picks}) == 20       # no player twice
    assert [p.platform_team_id for p in picks[:10]] == ["1", "2", "3", "4", "4", "3", "2", "1", "1", "2"]
    assert m.my_team_id() == 2


def test_vectorised_h2h_score_equals_team_value():
    from app.valuation import team_value
    sc, pre, _, cfg = _scorer(ScoringType.H2H_CATEGORIES)
    idx = [sc.pos[i] for i in pre.nlargest(14, "vorp")["player_id"].tolist()]
    _, keep = sc.assign(idx)
    starters = [idx[a] for a, b in keep if not sc.bn[b]]
    filled = {b for a, b in keep if not sc.bn[b]}      # starting slots left open are filled at waiver level
    starters += [g for j, g in enumerate(sc.ghost) if g is not None and not sc.bn[j] and j not in filled]
    assert abs(sc.score(idx) - team_value(sc.pre.iloc[starters], cfg)) < 1e-9


# ---------------- multi-sport ----------------
def _sample_sport(sport):
    """Generate fake NBA/NFL files in a temp folder and load them through the real loaders."""
    import tempfile
    from pathlib import Path
    import make_sample_sport
    from app import ingestion
    old = ingestion.DATA_DIR
    with tempfile.TemporaryDirectory() as d:
        ingestion.DATA_DIR = Path(d)
        try:
            make_sample_sport.generate(sport, ingestion.data_dir(sport))
            raw = ingestion.load_projections(sport)
            hist = ingestion.load_stats_history(sport)
        finally:
            ingestion.DATA_DIR = old
    from app.validate import clean_projections
    raw, actions = clean_projections(raw, sport)
    return raw, hist, actions


def test_sport_profiles_are_consistent():
    from app.sports import SPORTS, get_sport
    from app.valuation import position_demand
    assert set(SPORTS) == {"mlb", "nba", "nfl"}
    for key, sp in SPORTS.items():
        cfg = LeagueConfig.for_sport(key)
        for slot in cfg.roster.slots:
            assert slot == "BN" or slot in sp.slot_rules, (key, slot)
        for g in sp.groups:
            assert set(cfg.categories.get(g.kind, [])) <= set(g.cats), (key, g.kind)
            assert sp.group_starters(cfg.roster.slots, g.kind) > 0
        demand = position_demand(cfg.roster, sp)
        assert demand and all(v > 0 for v in demand.values()) and set(demand) <= sp.valid_positions
        assert len(cfg.roster.slot_list()) >= 13
    try:
        get_sport("curling"); assert False
    except ValueError as e:
        assert "Available" in str(e)
    try:
        LeagueConfig.for_sport("nfl", "nonsense"); assert False
    except ValueError as e:
        assert "ppr" in str(e)
    assert LeagueConfig.for_sport("nfl", "ppr").point_weights["rec"] == 1.0 and LeagueConfig.for_sport("nfl", "standard").point_weights["rec"] == 0.0
    assert LeagueConfig.for_sport("nfl").scoring_type == ScoringType.POINTS and LeagueConfig().sport == "mlb"


def test_nba_end_to_end():
    from app.ingestion import parse_positions
    from app.scoring import TeamScorer, simulate, summary
    from app.stats import search_flow
    from app.validate import build_report, check_config
    from app.valuation import add_ranks
    raw, hist, actions = _sample_sport("nba")
    cfg = LeagueConfig.for_sport("nba")
    rep = build_report(raw, cfg, hist, actions)
    assert rep["ok"] and rep["info"]["players"] == 200 and rep["info"]["player_history"].startswith("100%")
    assert parse_positions("PG/SG", "nba") == ["PG", "SG"] and parse_positions("PG,XX", "nba") == ["PG"]
    for fmt in ScoringType:
        c = LeagueConfig.for_sport("nba"); c.scoring_type = fmt
        df, repl = calculate_player_valuations(raw, c)
        assert df["vorp"].notna().all() and {"PG", "C", "UTIL"} <= set(repl)
        assert set(df["primary_pos"]) <= {"PG", "SG", "SF", "PF", "C"}
    df, repl = calculate_player_valuations(raw, cfg)
    best_ft = df.loc[df["z_FT%"].idxmax()]
    assert best_ft["FTA"] > df["FTA"].median()                    # free-throw % is volume-weighted
    tov = df.sort_values("TO", ascending=False).head(10)["z_TO"].mean()
    assert tov < df["z_TO"].mean()                                # turnovers count against you
    sc = TeamScorer(df, repl, cfg)
    five = df.nlargest(5, "vorp")["player_id"].tolist()
    s = summary(sc, five)
    assert len(s["roster"]) == 13 and sum(r["name"] is not None for r in s["roster"]) == 5
    b = simulate(sc, "snake", 12, 13, runs=3)
    assert b["mean"].shape == (12, 13) and b["mean"][0, 0] > b["mean"][11, 0]
    board = add_ranks(df, df)
    name = board.nlargest(1, "vorp").iloc[0]["name"]
    r = search_flow(board, raw, hist, f"{name} ppg", sport="nba")
    assert r["status"] == "ok" and r["stat"] == "PPG" and r["projected"]["value"] > 10
    assert search_flow(board, raw, hist, name, "ERA", sport="nba")["status"] == "unknown_stat"
    assert any("'STL'" in m for m in check_config(raw.drop(columns=["STL"]), cfg))


def test_nfl_end_to_end():
    from app.ingestion import parse_positions
    from app.optimizer import is_eligible
    from app.scoring import TeamScorer, summary
    from app.sports import NFL
    from app.stats import search_flow
    from app.validate import build_report, check_config
    from app.valuation import add_ranks, position_demand
    raw, hist, actions = _sample_sport("nfl")
    cfg = LeagueConfig.for_sport("nfl", "ppr")
    rep = build_report(raw, cfg, hist, actions)
    assert rep["ok"] and rep["info"]["players"] == 323
    assert parse_positions("DEF", "nfl") == ["DST"] and parse_positions("PK", "nfl") == ["K"]
    assert check_config(raw, LeagueConfig(sport="nfl", scoring_type=ScoringType.ROTO, categories={}))[0].endswith("'points'")
    df, repl = calculate_player_valuations(raw, cfg)
    assert set(df["primary_pos"]) <= set(NFL.valid_positions)
    best = df.groupby("primary_pos")["vorp"].max()
    assert best["K"] < best["RB"] and best["DST"] < best["WR"]               # kickers/defenses are low value
    # PPR vs standard: every reception is worth a point, nothing else changes
    std, _ = calculate_player_valuations(raw, LeagueConfig.for_sport("nfl", "standard"))
    gap = (df.set_index("player_id")["score"] - std.set_index("player_id")["score"])
    rec = df.set_index("player_id")["rec"]
    assert ((gap - rec).abs() < 1e-6).all()
    # flex slots: a third RB fills FLEX; superflex lets a second QB start
    sc = TeamScorer(df, repl, cfg)
    rbs = df[df["primary_pos"] == "RB"].nlargest(3, "vorp")["player_id"].tolist()
    slots = {r["slot"] for r in summary(sc, rbs)["roster"] if r["name"]}
    assert slots == {"RB", "FLEX"}
    assert is_eligible(["TE"], "FLEX", "nfl") and not is_eligible(["QB"], "FLEX", "nfl") and not is_eligible(["K"], "FLEX", "nfl")
    assert is_eligible(["QB"], "SUPERFLEX", "nfl") and not is_eligible(["K"], "SUPERFLEX", "nfl")
    sf = LeagueConfig.for_sport("nfl", "ppr"); sf.roster.slots["SUPERFLEX"] = 1
    d2, r2 = calculate_player_valuations(raw, sf)
    qbs = d2[d2["primary_pos"] == "QB"].nlargest(2, "vorp")["player_id"].tolist()
    assert {r["slot"] for r in summary(TeamScorer(d2, r2, sf), qbs)["roster"] if r["name"]} == {"QB", "SUPERFLEX"}
    dem = position_demand(sf.roster, NFL)
    assert dem["RB"] > dem["TE"] and dem["WR"] > dem["TE"]                    # FLEX demand follows the dedicated slots
    board = add_ranks(df, df)
    rb = df[df["primary_pos"] == "RB"].nlargest(1, "vorp").iloc[0]["name"]
    r = search_flow(board, raw, hist, f"{rb} ypc", sport="nfl")
    assert r["status"] == "ok" and r["stat"] == "YPC" and 3 < r["projected"]["value"] < 7
    assert search_flow(board, raw, hist, rb, "rec yds", sport="nfl")["stat"] == "REC YDS"


def test_yahoo_import_per_sport():
    from app.connectors.yahoo import yahoo_to_league_config
    nba = {"scoring_type": "head", "num_teams": 10, "roster_positions": [
        {"position": "PG", "count": 1}, {"position": "G", "count": 1}, {"position": "C", "count": 2}, {"position": "Util", "count": 2},
        {"position": "BN", "count": 3}, {"position": "IL", "count": 2}, {"position": "IL+", "count": 1}]}
    cats = [{"display_name": n, "position_type": "P"} for n in ("FG%", "FT%", "3PTM", "PTS", "REB", "AST", "ST", "BLK", "TO")]
    cfg, warn = yahoo_to_league_config(nba, cats, LeagueConfig.for_sport("nba"))
    assert cfg.scoring_type == ScoringType.H2H_CATEGORIES and cfg.roster.teams == 10 and cfg.roster.il_slots == 3
    assert cfg.roster.slots["C"] == 2 and cfg.roster.slots["UTIL"] == 2 and cfg.roster.slots["SF"] == 0
    assert cfg.categories["X"] == ["FG%", "FT%", "3PM", "PTS", "REB", "AST", "STL", "BLK", "TO"] and warn == []
    nfl = {"scoring_type": "head", "num_teams": 12, "roster_positions": [
        {"position": "QB", "count": 1}, {"position": "RB", "count": 2}, {"position": "WR", "count": 3}, {"position": "TE", "count": 1},
        {"position": "W/R/T", "count": 1}, {"position": "K", "count": 1}, {"position": "DEF", "count": 1},
        {"position": "BN", "count": 6}, {"position": "IR", "count": 1}, {"position": "D", "count": 1}]}
    cfg, warn = yahoo_to_league_config(nfl, [], LeagueConfig.for_sport("nfl"))
    assert cfg.scoring_type == ScoringType.POINTS                                # categories aren't modelled for football
    assert cfg.roster.slots["FLEX"] == 1 and cfg.roster.slots["DST"] == 1 and cfg.roster.slots["WR"] == 3 and cfg.roster.il_slots == 1
    assert any("points scoring only" in w for w in warn) and any("Unmodeled roster slot 'D'" in w for w in warn)
    mlb, _ = yahoo_to_league_config({"scoring_type": "roto", "num_teams": 14, "roster_positions": [
        {"position": "C", "count": 1}, {"position": "IF", "count": 1}, {"position": "BN", "count": 5}, {"position": "DL", "count": 2}]},
        [{"display_name": "HR", "position_type": "B"}, {"display_name": "K", "position_type": "B"}, {"display_name": "ERA", "position_type": "P"}], LeagueConfig())
    assert mlb.roster.slots["INF"] == 1 and mlb.roster.il_slots == 2 and mlb.categories["H"] == ["HR", "SO"] and mlb.categories["P"] == ["ERA"]




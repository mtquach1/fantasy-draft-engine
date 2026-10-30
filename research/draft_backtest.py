"""Out-of-sample DRAFT backtest: could the app have drafted the 2025-26 season better than a drafter following last season's ranking?

Uses only what existed before that season: your per-game files up to 2024-25 (the model never sees 2025-26).
 - projections: the own model, history through 2024-25, top 200 players
 - field (11 other teams): draft by last season's fantasy points PER GAME (floored at 55 games, so one injury year isn't a verdict) plus noise
 - three ways to draft the remaining slot, same field every time:
     follow_adp    best player left by the field's own ranking (what an unprepared drafter does)
     follow_model  best player left by the own model's board, no roster logic (isolates what the PROJECTIONS add)
     bot           the app's top recommendation, GET /recommend (projections + roster-aware logic)
 - scoring: the REAL 2025-26 stats (hindsight), lineup chosen optimally under the ESPN default roster; a drafted player who never
   played counts as nothing (the slot is filled from the waiver pool), so injuries hurt both sides, unforeseeably.
Run from the project root:  python research/draft_backtest.py [seeds=3] [slots=1-12]"""
import os, sys, tempfile, random, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["SPORT"] = "nba"; os.environ["LEAGUE_PRESET"] = "espn_default"; os.environ["DRAFT_PLATFORM"] = "manual"
from fastapi.testclient import TestClient              # noqa: E402
from app import ingestion, persist, validate, own_model as om   # noqa: E402
from app.config import LeagueConfig                     # noqa: E402
from app.draft import slot_for_pick                     # noqa: E402
from app.scoring import TeamScorer                      # noqa: E402
from app.valuation import calculate_player_valuations   # noqa: E402

TEAMS, ROUNDS, POOL = 12, 13, 200
seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
slots = range(1, TEAMS + 1)
if len(sys.argv) > 2:
    a, _, b = sys.argv[2].partition("-"); slots = range(int(a), int(b or a) + 1)

real_nba = ingestion.DATA_DIR / "nba"
files = om.find_seasons(real_nba / "raw"); yrs = sorted(files)
assert len(yrs) >= 3, "need three per-game seasons in data/nba/raw"
a_y, b_y, c_y = yrs[-3:]
S = {y: om.load_season(files[y]) for y in (a_y, b_y, c_y)}
p = om.Params(); beta, _ = om.fit_age_from_transition(S, [a_y, b_y], p)
proj = om.project_next(S, [b_y, a_y], p, beta)                       # what a drafter could have known before season c
pool = proj.head(POOL)
print(f"seasons: model history {a_y}-{b_y}, drafted for season ending {c_y}; pool {len(pool)} players", flush=True)

tmp = Path(tempfile.mkdtemp()); (tmp / "nba").mkdir()
ingestion.DATA_DIR = tmp; persist.DATA_DIR = Path(tempfile.mkdtemp())
om.to_app_frame(pool, POOL).to_csv(tmp / "nba" / "players.csv", index=False)

cfg = LeagueConfig.for_sport("nba", "espn_default"); cfg.roster.teams = TEAMS
# ruler: the real stats of season c
act = S[c_y]; act = act[act["mins"] >= 100]
actual = pd.DataFrame({"Name": act["Player"].values, "Team": act["Tm"].values, "Pos": act["Pos"].astype(str).str.replace("-", ",").values,
    "playerid": ["n:" + str(n).lower() for n in act["Player"]], "GP": act["G"].values, "MIN": act["mins"].values,
    **{k: (act[v] * act["G"]).values for k, v in {"PTS": "PTS", "REB": "TRB", "AST": "AST", "STL": "STL", "BLK": "BLK", "3PM": "TPM",
                                                  "TO": "TOV", "FGM": "FGM", "FGA": "FGA", "FTM": "FTM", "FTA": "FTA"}.items()},
    "is_injured": False, "note": "actual"})
rd = Path(tempfile.mkdtemp()); (rd / "nba").mkdir(); actual.to_csv(rd / "nba" / "players.csv", index=False)
ingestion.DATA_DIR = rd
rawR, _ = validate.clean_projections(ingestion.load_projections("nba"), "nba")
valR, replR = calculate_player_valuations(rawR, cfg, frozenset()); scR = TeamScorer(valR, replR, cfg)
ingestion.DATA_DIR = tmp
idR = {om.norm_name(n): q for n, q in zip(valR["name"], valR["player_id"])}

last = S[b_y]; adp_pts = (last["fpg"] * last["G"]).rename(index=lambda i: om.norm_name(last.loc[i, "Player"]))
pool_keys = [om.norm_name(n) for n in pool["Name"]]
fpg_last = last["fpg"].rename(index=lambda i: om.norm_name(last.loc[i, "Player"])); g_last = last["G"].rename(index=lambda i: om.norm_name(last.loc[i, "Player"]))
adp_score = {k: float(fpg_last.get(k, 0)) * max(float(g_last.get(k, 0)), 55) for k in pool_keys}      # per-game rate x at least 55 games
ADP = {k: r + 1 for r, k in enumerate(sorted(pool_keys, key=lambda k: -adp_score[k]))}
MODEL = {k: r + 1 for r, k in enumerate(om.norm_name(n) for n in pool["Name"])}                   # the own model's board order
idP = {om.norm_name(n): "n:" + str(n).lower() for n in pool["Name"]}
names_of = {v: k for k, v in idP.items()}

def noisy_pick(avail, pick_no, seed):
    rng = random.Random(seed * 100003 + pick_no)
    return min(avail, key=lambda k: ADP[k] + rng.gauss(0, 1.5 + 0.08 * ADP[k]))

def actual_score(rosters):
    return np.array([scR.score([scR.pos[idR[k]] for k in r if k in idR]) for r in rosters])

def baseline(seed, my_slot, mode):
    avail, rosters = list(pool_keys), [[] for _ in range(TEAMS)]
    for n in range(1, TEAMS * ROUNDS + 1):
        slot = slot_for_pick("snake", TEAMS, n)
        k = (min(avail, key=lambda q: MODEL[q]) if (mode == "follow_model" and slot == my_slot) else noisy_pick(avail, n, seed))
        avail.remove(k); rosters[slot - 1].append(k)
    return rosters

def bot(c, seed, my_slot):
    assert c.post("/draft/reset").status_code == 200
    assert c.put("/draft/setup", json={"draft_type": "snake", "teams": TEAMS, "my_slot": my_slot, "rounds": ROUNDS}).status_code == 200
    avail, rosters = list(pool_keys), [[] for _ in range(TEAMS)]
    for n in range(1, TEAMS * ROUNDS + 1):
        slot = slot_for_pick("snake", TEAMS, n)
        if slot == my_slot:
            pid = c.get("/recommend?limit=1").json()[0]["player_id"]; k = names_of[pid]
        else:
            k = noisy_pick(avail, n, seed); pid = idP[k]
        avail.remove(k); rosters[slot - 1].append(k)
        r = c.post("/draft/pick", json={"player_id": pid, "team_id": slot}); assert r.status_code == 200, r.text
    return rosters

rows, t0 = [], time.time()
with TestClient(__import__("app.main", fromlist=["app"]).app) as c:
    for seed in range(1, seeds + 1):
        for s in slots:
            for label, r in (("follow_adp", baseline(seed, s, "follow_adp")), ("follow_model", baseline(seed, s, "follow_model")), ("bot", bot(c, seed, s))):
                sc = actual_score(r)
                dnp = sum(1 for k in r[s - 1] if k not in idR)
                rows.append(dict(seed=seed, slot=s, who=label, score=sc[s - 1], rank=int((sc > sc[s - 1]).sum()) + 1,
                                 beaten=float((sc < sc[s - 1]).sum()) / (TEAMS - 1), dnp=dnp))
            print(f"seed {seed} slot {s:2d} done ({time.time() - t0:.0f}s)", flush=True)
df = pd.DataFrame(rows); df.to_csv(ROOT / "research" / "draft_backtest_results.csv", index=False)
pv = df.pivot_table(index=["seed", "slot"], columns="who", values=["score", "rank", "beaten", "dnp"])
rng = np.random.default_rng(0)
def cmp(a, b, what):
    d = (pv[("score", a)] - pv[("score", b)]).values
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    print(f"{what:44s} {d.mean():+7.0f} season points  (95% CI {np.percentile(bs, 2.5):+.0f} to {np.percentile(bs, 97.5):+.0f}); ahead in {(d > 0).mean():.0%} of paired drafts")
print(f"\n{len(pv)} paired drafts ({seeds} seeds x {len(list(slots))} slots), scored by REAL {c_y} stats, starting lineup, ESPN default points\n")
for w in ("follow_adp", "follow_model", "bot"):
    print(f"{w:13s} mean score {pv[('score', w)].mean():7.0f} | mean finish {pv[('rank', w)].mean():4.1f} of 12 | teams beaten {pv[('beaten', w)].mean():4.0%} | picks who never played {pv[('dnp', w)].mean():.2f}")
print()
cmp("follow_model", "follow_adp", "projections alone (model board - ADP):")
cmp("bot", "follow_model", "roster-aware logic on top (bot - model board):")
cmp("bot", "follow_adp", "everything (bot - ADP):")

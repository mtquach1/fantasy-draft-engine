"""Measured draft result: the app's recommendations vs a drafter who just follows a consensus ranking.

For each draft slot and seed: a 12-team, 13-round snake draft (ESPN default points league). Opponents draft from a THIRD-PARTY
ranking (data/nba/players_third_party.csv, list order) plus noise, so they are independent of the bot's valuation.
 - bot:      takes the app's top recommendation (GET /recommend) at its slot; same opponents.
 - baseline: the same slot, same opponents, but takes the best-ranked player left (what an unprepared drafter does).
Both final rosters are scored two ways: by the app's own valuation, and by the third-party projections (an independent ruler).

Run from the project root:  python research/draft_simulation.py [seeds=2] [slots=1-12]
Needs data/nba/players.csv (your own model output) and data/nba/players_third_party.csv."""
import os, sys, shutil, tempfile, random, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SPORT", "nba"); os.environ.setdefault("LEAGUE_PRESET", "espn_default"); os.environ.setdefault("DRAFT_PLATFORM", "manual")
from fastapi.testclient import TestClient            # noqa: E402
from app import ingestion, persist, validate         # noqa: E402
persist.DATA_DIR = Path(tempfile.mkdtemp())          # saved-draft files go to a temp folder: this script must never touch a real draft
from app.config import LeagueConfig                  # noqa: E402
from app.draft import slot_for_pick                  # noqa: E402
from app.own_model import norm_name                  # noqa: E402
from app.scoring import TeamScorer                   # noqa: E402
from app.valuation import calculate_player_valuations  # noqa: E402

TEAMS, ROUNDS = 12, 13
seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
slots = range(1, TEAMS + 1)
if len(sys.argv) > 2:
    a, _, b = sys.argv[2].partition("-"); slots = range(int(a), int(b or a) + 1)

nba = ingestion.DATA_DIR / "nba"
tp_file = nba / "players_third_party.csv"
assert tp_file.exists() and (nba / "players.csv").exists(), "need data/nba/players.csv and data/nba/players_third_party.csv"
tp = pd.read_csv(tp_file)
tp["key"] = tp["Name"].map(norm_name)
# list order = fantasy points under ESPN default weights (that is how the list is ranked)
w = dict(PTS=1, FGM=2, FGA=-1, FTM=1, FTA=-1, **{"3PM": 1}, REB=1, AST=2, STL=4, BLK=4, TO=-2)
tp["fp"] = sum(v * tp[k] for k, v in w.items())
order = tp.sort_values("fp", ascending=False)["key"].tolist()
ADP = {k: i + 1 for i, k in enumerate(order)}

cfg = LeagueConfig.for_sport("nba", "espn_default"); cfg.roster.teams = TEAMS

def scorer_for(dirpath):
    old = ingestion.DATA_DIR; ingestion.DATA_DIR = Path(dirpath)
    try:
        raw, _ = validate.clean_projections(ingestion.load_projections("nba"), "nba")
    finally:
        ingestion.DATA_DIR = old
    val, repl = calculate_player_valuations(raw, cfg, frozenset())
    return TeamScorer(val, repl, cfg), val

with tempfile.TemporaryDirectory() as d:
    (Path(d) / "nba").mkdir(); shutil.copy(tp_file, Path(d) / "nba" / "players.csv")
    scB, valB = scorer_for(d)                          # ruler B: third-party projections
scA, valA = scorer_for(ingestion.DATA_DIR)             # ruler A: the app's own valuation
idA = {norm_name(n): p for n, p in zip(valA["name"], valA["player_id"])}
idB = {norm_name(n): p for n, p in zip(valB["name"], valB["player_id"])}
common = [k for k in order if k in idA and k in idB]
print(f"players in both pools: {len(common)}; own pool {len(valA)}, third-party pool {len(valB)}")

def noisy_pick(avail, pick_no, seed):
    rng = random.Random(seed * 100003 + pick_no)
    return min(avail, key=lambda k: ADP.get(k, 400) + rng.gauss(0, 1.5 + 0.08 * ADP.get(k, 400)))

def team_scores(rosters):
    a = [scA.score([scA.pos[idA[k]] for k in r if k in idA]) for r in rosters]
    b = [scB.score([scB.pos[idB[k]] for k in r if k in idB]) for r in rosters]
    return np.array(a), np.array(b)

def baseline(seed, my_slot):
    avail, rosters = list(common), [[] for _ in range(TEAMS)]
    for pick_no in range(1, TEAMS * ROUNDS + 1):
        t = slot_for_pick("snake", TEAMS, pick_no) - 1
        k = noisy_pick(avail, pick_no, seed)
        avail.remove(k); rosters[t].append(k)
    return rosters

def bot(c, seed, my_slot):
    assert c.post("/draft/reset").status_code == 200
    assert c.put("/draft/setup", json={"draft_type": "snake", "teams": TEAMS, "my_slot": my_slot, "rounds": ROUNDS}).status_code == 200
    avail, rosters, outside = list(common), [[] for _ in range(TEAMS)], 0
    names = {p: k for k, p in idA.items()}
    for pick_no in range(1, TEAMS * ROUNDS + 1):
        slot = slot_for_pick("snake", TEAMS, pick_no)
        if slot == my_slot:
            pid = c.get("/recommend?limit=1").json()[0]["player_id"]
            k = names[pid]
            if k not in avail:
                outside += 1
            else:
                avail.remove(k)
        else:
            k = noisy_pick(avail, pick_no, seed); avail.remove(k); pid = idA[k]
        r = c.post("/draft/pick", json={"player_id": pid, "team_id": slot}); assert r.status_code == 200, r.text
        rosters[slot - 1].append(k)
    return rosters, outside

rows, t0 = [], time.time()
with TestClient(__import__("app.main", fromlist=["app"]).app) as c:
    for seed in range(1, seeds + 1):
        for s in slots:
            base_r = baseline(seed, s); bot_r, outside = bot(c, seed, s)
            for label, r in (("baseline", base_r), ("bot", bot_r)):
                a, b = team_scores(r)
                rows.append(dict(seed=seed, slot=s, who=label, A=a[s - 1], B=b[s - 1], rankA=int((a > a[s - 1]).sum()) + 1,
                                 rankB=int((b > b[s - 1]).sum()) + 1, beatenA=float((a < a[s - 1]).sum()) / (TEAMS - 1),
                                 beatenB=float((b < b[s - 1]).sum()) / (TEAMS - 1), outside=outside if label == "bot" else 0))
            print(f"seed {seed} slot {s:2d} done ({time.time() - t0:.0f}s)", flush=True)
df = pd.DataFrame(rows); df.to_csv(ROOT / "research" / "draft_simulation_results.csv", index=False)
p = df.pivot_table(index=["seed", "slot"], columns="who", values=["A", "B", "beatenA", "beatenB", "rankA", "rankB"])
rng = np.random.default_rng(0)
def ci(x):
    x = np.asarray(x); bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
print(f"\n{len(p)} paired drafts ({seeds} seeds x {len(list(slots))} slots), {TEAMS} teams, {ROUNDS} rounds, snake, ESPN default points\n")
for r, nm in (("A", "app's own valuation"), ("B", "third-party projections")):
    d = (p[(r, "bot")] - p[(r, "baseline")]); m, lo, hi = ci(d)
    print(f"[{nm}] bot minus follow-the-ranking, season points of the starting lineup: {m:+.0f}  (95% CI {lo:+.0f} to {hi:+.0f}); bot better in {(d > 0).mean():.0%} of drafts")
    print(f"   share of the 11 other teams beaten: bot {p[('beaten'+r, 'bot')].mean():.0%} vs baseline {p[('beaten'+r, 'baseline')].mean():.0%};  mean finish: bot {p[('rank'+r, 'bot')].mean():.1f} vs baseline {p[('rank'+r, 'baseline')].mean():.1f} (of 12)")
print(f"bot picks outside the shared player pool: {int(df[df.who=='bot'].outside.sum())}")

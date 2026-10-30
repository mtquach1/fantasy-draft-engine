# Fantasy Draft Engine
![tests](https://github.com/mtquach1/fantasy-draft-engine/actions/workflows/tests.yml/badge.svg)
**Live demo:** https://fantasy-draft-engine.onrender.com (free hosting: sleeps after 15 minutes idle; runs on fake sample data; drafts are temporary)

A draft assistant for fantasy basketball (baseball and football profiles are included too). It projects every player with its own
model, values them for **your** league's scoring and roster, and during a draft recommends the pick that adds the most to *your*
lineup. It never drafts for you: you make the pick on ESPN/Yahoo yourself.

Python, FastAPI, pandas, NumPy, SciPy. One static page (vanilla JS), 232 automated tests.

## What it does
- **Own projection model** (`app/own_model.py`): per-minute production, minutes and games from past seasons, with shrinkage toward
  league averages and an age curve fitted on one season transition. Backtested out of sample (below).
- **Valuation** for roto, head-to-head categories or points (`app/valuation.py`): z-scores, value over replacement at every eligible
  position, replacement levels recomputed after every pick.
- **Lineup optimizer** (`app/optimizer.py`): Hungarian assignment of players to roster slots (G, F, UTIL and bench included).
- **Recommendations** (`app/recommend.py`): marginal gain to your lineup, a rule against picks that cannot fill an open slot, and
  a look-ahead for who will probably still be there at your next turn. The follow-up for your next pick is drawn only from players
  with a real chance of lasting that long, and is shown with its odds ("best option in about 38% of simulated drafts").
- **Team score and pace**: your lineup's projected score, and whether it is ahead of or behind what 16 simulated drafts at your slot give.
- **Live what-if edits**: change a player's expected games or minutes and every value, rank, recommendation and team score is recomputed.
- **Player list sorting**: by value over replacement (the default, shifts as the draft moves) or by fixed overall rank.
- **Live projection blend**: a slider on the page mixes the own model with a reference list (0% to 100%) and the board, recommendations
  and team score update as you move it. Needs `players_own.csv`, `players_third_party.csv` and `blend.json` in `data/nba/`.
- **Safety**: strict input validation, saved drafts survive a restart, undo, and a manual mode that needs no platform connection.
- **Race-safe picks**: every change to a draft happens under one lock per draft, so 20 simultaneous picks of the same player let exactly one
  through, pick numbers never repeat or skip, and a finished draft can't overflow. Picks accept an `Idempotency-Key` header (the page sends one
  per click), so a retry or double click can never draft twice. The concurrency tests fire real threads, and were checked to fail when the lock is removed.
- **Hosting mode** (`MULTI_SESSION=1`): every browser gets its own private draft in memory (cookie-based sessions, least-recently-used
  eviction, idle expiry, nothing written to disk). Without it you get the single saved draft you use locally.

## Quick start (Windows, from the project folder)
```
pip install -r requirements.txt
pytest                                   # 232 tests
python make_sample_sport.py nba          # FAKE sample data, to try the app right away
uvicorn app.main:app --reload            # open http://127.0.0.1:8000/
```
With Docker: `docker compose up` (generates fake sample data when `data/` is empty). Not yet built or run by the author.

## Using real data
1. Download Basketball-Reference per-game tables for three seasons into `data/nba/raw/` and name them by the season's end year:
   `per_game_2024.csv`, `per_game_2025.csv`, `per_game_2026.csv`. (Check the site's terms; these files are not part of this repo.)
2. `python build_own_model.py` writes `data/nba/players.csv` from the own model.
3. Optional context the data cannot see (trades, rookies, returns from injury): `data/nba/overrides.csv`, one row per player:
   `name, team, pos, games, mpg, note`, and for a new player also `pts, reb, ast, stl, blk, tpm, tov, fgm, fga, ftm, fta` (per game).
4. Optional: `python build_own_model.py --blend 0.5` pulls the model halfway toward a reference projection file
   (`data/nba/players_third_party.csv`, same layout as `players.csv`). It also writes `players_own.csv` (the pure own model) and
   `blend.json`, which switch on the blend slider in the page's setup card; the slider starts at the weight you built with.
5. In `.env` set `SPORT=nba` and `LEAGUE_PRESET=espn_default` for ESPN's default points league (10 starters, 3 bench, 3 IR).
6. Restart the server and open `/health/data`: it must say `"ok": true`.

During a draft, the slider changes the blend, and `PUT /players/override` (see `/docs`) changes one player's games or minutes; `DELETE /players/overrides` undoes all.

## How well does it work? (measured, with the caveats)
**Projection backtest** (`research/evaluate_backtest.py`). Settings fitted on 2024-25 only; tested on 2025-26 for the top 150 players
by 2024-25 fantasy points (143 played). Average error in season fantasy points:

| method | error |
|---|---|
| own model | 612 |
| last per-game rate x a flat 68 games | 650 |
| repeat last season's total | 700 |

The own model beats "repeat last season" by 89 points (95% CI 42 to 136) and the flat-68 baseline by 38 (95% CI 5 to 71). Most of the
remaining error is injuries, which the model does not predict.

**Draft backtest** (`research/draft_backtest.py`): 36 simulated 12-team, 13-round snake drafts (all 12 slots x 3 seeds) for the 2025-26
season using only earlier data, scored by that season's real stats. The other 11 teams draft by last season's per-game production.

| drafter for my slot | mean finish (of 12) |
|---|---|
| follows last season's ranking | 6.5 |
| follows the own model's board, no roster logic | 1.3 |
| uses the app's recommendations | 1.5 |

The projections alone beat the ranking-follower by about 4,600 season points (95% CI 3,800 to 5,400; ahead in 97% of drafts). The
roster-aware recommendation logic added only +204 points on top, which is not statistically distinguishable from zero (CI -218 to +645).
Honest reading: the value so far comes from the projections, not from the recommendation logic.

Caveats: one season, and an injury-heavy one for stars. The field drafts by a simple rule; real leagues use consensus rankings that are
harder to beat. The player pool is the model's own top 200, which favours the model. Treat these as evidence, not a guarantee.

**Agreement with a third-party board** (ESPN default points, 186 shared players): rank correlation 0.87 for the pure own model, 0.97
at `--blend 0.5`. Closeness to another forecast is not accuracy; only the season itself can say which is better.

## Limits
- The draft lock is per process. Running several server workers would need a shared store (for example Redis) so they agree on the pick log;
  the app is designed and tested as a single process.
- Hosted sessions live in memory: a restart (or a free host going to sleep) loses them. Locally, your one draft is saved to disk.
  Hosted mode turns live Yahoo/ESPN sync off, and the session cap (`MAX_SESSIONS`) drops the least recently used draft when full.
- No rookies in the model itself (add them in `overrides.csv`), no trade or role-change awareness beyond what you put in overrides.
- Yahoo and ESPN live sync have never run against a real league and are experimental. Manual mode is the tested path.
- Basketball only for live overrides; no schedule effects and no automatic punting in category leagues.
- Docker, Render and the CI workflow are written but were not built or run by the author.

## Layout
```
app/own_model.py     projection model, overrides, blending
app/valuation.py     z-scores, value over replacement, ranks
app/optimizer.py     lineup assignment      app/recommend.py   pick recommendations
app/scoring.py       team score and pace    app/draft.py       snake/linear/auction order and validation
app/main.py          API                    app/static/index.html   the Draft Room page
build_own_model.py   data builder           research/          backtests (reproduce the numbers above)
tests/               232 tests              Dockerfile, docker-compose.yml, render.yaml, .github/workflows/tests.yml
```

# Fantasy Draft Engine

[![tests](https://github.com/mtquach1/fantasy-draft-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/mtquach1/fantasy-draft-engine/actions/workflows/tests.yml)

**Live demo:** https://fantasy-draft-engine.onrender.com (fake sample data, free hosting that sleeps after 15 minutes idle, so the first load can take a minute; every browser gets its own private draft that is forgotten on restart).

**Tech stack:** Python 3.12/3.13 · FastAPI · pandas · NumPy · SciPy · SQLite · vanilla JavaScript (no build step) · pytest · GitHub Actions · Docker · Render

A draft assistant for fantasy basketball (baseball and football profiles are included too). It projects every player with its own
model, values them for **your** league's scoring and roster, and during a draft recommends the pick that adds the most to *your*
lineup. It never drafts for you: you make the pick on ESPN/Yahoo yourself.

509 automated tests run in CI on Python 3.12 and 3.13. The whole front end is one static page.

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
pytest                                   # 509 tests
python make_sample_sport.py nba          # FAKE sample data, to try the app right away
uvicorn app.main:app --reload            # open http://127.0.0.1:8000/
```
With Docker: `docker compose up` (generates fake sample data when `data/` is empty). The image builds and runs on Render (see the live demo); `docker compose` itself has not been run locally by the author.

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
season using only earlier data, scored by that season's real stats. The other 11 teams draft by last season's per-game production. The
"autodraft-style" baseline below takes, at my slot, the best player left by that same ranking. It is not ESPN's actual autodraft.

| drafter for my slot | mean score | mean finish (of 12) |
|---|---:|---:|
| autodraft-style baseline (last season's ranking) | 22,053 | 6.5 |
| follows the own model's board, no roster logic | 26,657 | 1.3 |
| uses the app's recommendations | 26,737 | 1.5 |

The app's lineup beat the baseline by 4,684 season points (95% CI 3,831 to 5,525; ahead in 97% of the 36 paired drafts). The projections
alone account for nearly all of it (+4,604, CI 3,776 to 5,370). The roster-aware recommendation logic added +80 points on top, which is not
distinguishable from zero (CI -347 to +534; ahead in 47% of drafts). Honest reading: the value so far comes from the projections, not from the
recommendation logic.

**Where the gap comes from** (`research/why_report.py`, same 36 drafts, starting lineups, real 2025-26 stats). The stat breakdown explains
4,112 of the 4,684 points; the rest is probably how the score treats empty slots. Scoring volume (points plus threes) is worth +3,044,
steals plus blocks +742, rebounding +508, assists +435. Shooting efficiency (-260) and turnovers (-357) were worse than the baseline's. The
app's starters played 3.2 more minutes per game (33.1 vs 30.0) and scored 7.6 more fantasy points per game; games played were equal
(65.6 vs 65.4) and so were picks who never played (0.53 each), so the gap is not injury luck. In short: bigger roles and more volume, bought at
some cost in efficiency.

**Do projected risers actually rise?** (`research/growth_check.py`, 228 players with at least 800 minutes in both seasons.) Correlation between
the model's projected change in fantasy points per game and the real change: +0.39 (95% CI +0.28 to +0.48). Direction right for 60% of players.
Players projected to rise gained +2.3 points per game on average, players projected to fall lost -1.2; 15 of the 20 biggest projected risers rose.
Players aged 23 or under were projected +1.5 and really changed +3.0: right direction, but the model understates young players' growth.
Caveats: the minutes filter favours players whose role held or grew, which biases real changes upward, and there is no age-only baseline to
compare the 0.39 against. Rookies and role changes from trades are not forecast by the model; they enter only through `overrides.csv`, which is
hand-entered judgment.

**Current-season check.** `python research/draft_backtest.py 3 1-12 model --inseason` drafts for the season in progress, using a partial-season
table kept in `data/nba/inseason/` (never in `data/nba/raw`), and `python research/mock_rank.py` ranks the teams of a recorded mock draft by real
stats. Early in a season both are very noisy (a handful of games per player), so no results from them are reported here.

**Calibrating the look-ahead.** The look-ahead assumes the other drafters follow a ranking with noise that grows with rank (2.0 + 0.12 x rank, a
guess). `python research/calibrate_lookahead.py [draft files]` fits that noise to real draft logs and checks the "X% likely to last" odds against
what actually happened (a reliability table and Brier score, with a bootstrap interval and, given two or more drafts, a check on a draft the fit
did not see). `python espn_check.py export` saves a real ESPN league's picks, finished or in progress, in the same format. On drafts simulated from
a planted noise model the fit recovers the planted noise (`tests/test_calibration.py`). Results from real drafts are not reported here yet.

Caveats for all of the above: one season, and an injury-heavy one for stars; simulated opponents following one simple rule, while real leagues
use consensus rankings that are harder to beat; the player pool is the model's own top 200, which favours the model (`draft_backtest.py 3 1-12 merged`
reruns with last season's top 200 added to the pool; that result is not reported here yet); the test season was looked at repeatedly while the
model was built. Treat these as evidence, not a guarantee.

**Agreement with a third-party board** (ESPN default points, 186 shared players): rank correlation 0.87 for the pure own model, 0.97
at `--blend 0.5`. Closeness to another forecast is not accuracy; only the season itself can say which is better.

## Operating it
Every response carries an `X-Request-ID` header (a valid one sent by the caller is kept). With `REQUEST_LOG=1` the server prints one line per
request (JSON when `LOG_FORMAT=json`) with the method, route, status, duration and request id; query strings, cookies and bodies are never logged.
`GET /metrics` serves Prometheus-format counters and duration histograms by route, and `GET /metrics/summary` serves p50/p95/p99 per route from
the most recent requests; set `METRICS_TOKEN` to require a bearer token. Metrics are in memory and per process. Unknown URLs are all counted as
one "unmatched" route so the metrics cannot grow without bound.

A hosted copy can show a banner at the top of the page by setting `DEMO_NOTICE` (the Docker image sets a default one whenever it had to invent sample data, so
a visitor is never left wondering why the players are made up).

`python research/load_test.py --url https://your-app.onrender.com --users 5 --duration 20` drives a hosted copy with simulated browsers (each with
its own private draft): it lists players, asks for recommendations, drafts with idempotency keys, resends picks to check retries are harmless, and
reports p50/p95/p99 per endpoint, the error rate, and correctness checks. It refuses to run against a single-user server (it would overwrite the
draft) unless told to. Numbers describe that server, your network and that moment; the first request is reported separately because a sleeping
free-tier host takes a while to wake.

## Data pipeline
Ingest, quality gate, versioned snapshots with rollback, and an idempotent event feed for injury news and trades: see [docs/PIPELINE.md](docs/PIPELINE.md).

## Limits
- The draft lock is per process. Running several server workers would need a shared store (for example Redis) so they agree on the pick log;
  the app is designed and tested as a single process.
- Hosted sessions live in memory: a restart (or a free host going to sleep) loses them. Locally, your one draft is saved to disk.
  Hosted mode turns live Yahoo/ESPN sync off, and the session cap (`MAX_SESSIONS`) drops the least recently used draft when full.
- No rookies in the model itself (add them in `overrides.csv`), no trade or role-change awareness beyond what you put in overrides.
- Yahoo and ESPN live sync have never run against a real league and are experimental. Manual mode is the tested path.
- Basketball only for live overrides; no schedule effects and no automatic punting in category leagues.
- Hosting is a free tier with fake sample data: it sleeps when idle and has no monitoring or load testing yet. CI runs the tests only.
- The data pipeline has never run against a live feed (see `docs/PIPELINE.md`).

## Layout
```
app/own_model.py     projection model, overrides, blending
app/valuation.py     z-scores, value over replacement, ranks
app/optimizer.py     lineup assignment      app/recommend.py   pick recommendations
app/scoring.py       team score and pace    app/draft.py       snake/linear/auction order and validation
app/main.py          API                    app/static/index.html   the Draft Room page
pipeline/            data pipeline: sources, quality gate, snapshots, events, replay (docs/PIPELINE.md)
build_own_model.py   data builder           research/          backtests (reproduce the numbers above)
tests/               509 tests              Dockerfile, docker-compose.yml, render.yaml, .github/workflows/tests.yml
```

## License
MIT: see [LICENSE](LICENSE). The projection data you supply (Basketball-Reference tables, ESPN lists) stays on your machine, is never committed, and is not covered by this license.

# Draft-day cheat sheet (print this)

## Before the draft (do this the day before)
1. Double-click `start_draft.bat` (Windows) or run `./start_draft.sh`. The page opens at http://127.0.0.1:8000/
2. Open http://127.0.0.1:8000/health/data : it must say `"ok": true`. Fix anything it lists.
3. Draft setup card: pick your sport, format, number of teams and YOUR pick. Press Save.
4. Do one practice run with `DRAFT_PLATFORM=mock` in `.env` (restart after editing), or a Yahoo/ESPN mock draft.
5. Optional but worthwhile: put `adp.csv` (columns `name` or `player_id`, and `adp`) in your sport's data folder, restart, then open
   http://127.0.0.1:8000/health/adp . It shows how far the ranking is from your league's drafters.

## Basketball data (injured stars and rookies handled)
Put per-game CSVs and `rankings.txt` in `data/nba/raw/`, run `python yahoo_check.py positions` once, then `python build_nba_projections.py`.
Read the report it prints: every rookie/estimated player is listed there.

## Yahoo live sync (test it on a mock draft first)
`python yahoo_check.py check`, then `python yahoo_check.py watch` during the mock. Details in README ("Yahoo setup").

## ESPN live sync (no approval needed, unofficial)
Cookies + league id in `.env`, then `python espn_check.py check` and `watch`. Details in README ("ESPN setup").
Yahoo's fantasy data now requires Yahoo's approval (apply at sports.yahoo.com/developer/access); until then use manual mode there.

## During the draft
- **Top of "Recommended for you"** is the line to read. In snake/linear drafts it says who to take and who will probably be there at your next pick.
- **"x% likely to be there"** is an estimate. It assumes other teams follow the ADP (or the pre-draft ranking if no ADP file). It does not know what each team needs.
- **Manual mode:** press **Draft** next to every pick as it happens, in order. Missed one? Keep going; "Undo last pick" fixes a wrong one.
- **Auction:** type the price, choose the winning team, press Draft. The card names who to nominate (to drain rivals' money early, your targets late).
- **Yahoo/ESPN mode:** picks arrive by themselves. If the Connection card turns red, switch to manual: nothing else changes.

## If something breaks
| What you see | Do this |
|---|---|
| Page won't load | Is the black window still open? Run `start_draft.bat` again. |
| Computer crashed / restarted | Start it again. Your picks and setup are restored from `data/draft_state_<sport>.json`. |
| Numbers look frozen | Reload the page (F5). |
| "Couldn't match" players on the Connection card | Add them to `id_crosswalk.csv`, or just keep drafting: the pick still holds its place. |
| Yahoo/ESPN disconnects | Manual mode. Enter the picks you missed in order. |

## Numbers to remember
- Sample data is FAKE. Real projections must be loaded before you trust any name on the page.
- Tests: `pytest` should show everything passing before draft day.


## Own projection model (basketball)
`python build_own_model.py` (add `--blend 0.5` to pull toward `data/nba/players_third_party.csv`), then restart. Trades, rookies and
injury returns go in `data/nba/overrides.csv`. A last-minute change during the draft: open http://127.0.0.1:8000/docs, `PUT /players/override`
with `{"player": "Name", "games": 50}`; `DELETE /players/overrides` undoes all of them.


## Blend slider
Setup card > Projection blend. Left = own model only, right = the reference list only. The board, recommendations and your team score
recompute as you move it (about a second). Your what-if overrides stay on top. Appears only after `python build_own_model.py --blend 0.5`.


## Slow draft clicks?
The server prints `[slow] GET /recommend/plan took 2.3s` in its window for anything over 1 second (set `SLOW_REQUEST_SECONDS` in `.env` to change it).
Clicking Draft updates the player list at once; the recommendations follow once calculated. While drafting, start the server WITHOUT
`--reload` (`uvicorn app.main:app`), plug the laptop in, and set Windows power mode to "Best performance".

## Sort the player list
Next to the position filter: value over replacement (default) or overall rank.


## Picks are race-safe
Two tabs, or two people, can't draft the same player or take the same pick number. A repeated `Idempotency-Key` header (or `request_id` in the body)
on `POST /draft/pick` returns the first result instead of drafting again.

# data/

Put your own files here. Nothing in this folder is committed to git.

- `nba/raw/per_game_2024.csv`, `per_game_2025.csv`, `per_game_2026.csv` : Basketball-Reference per-game tables (season END year in the name)
- `nba/overrides.csv` : optional trades, rookies, injury returns (see the README)
- `nba/players.csv` : written by `python build_own_model.py`, or fake sample data from `python make_sample_sport.py nba`
- `draft_state_<sport>.json` : your saved draft (written automatically)
- `oauth2.json` : Yahoo credentials. Keep it private.

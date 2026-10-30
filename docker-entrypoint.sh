#!/bin/sh
# Starts the app. With no data mounted it generates FAKE sample projections so the app is usable immediately.
set -e
SPORT="${SPORT:-nba}"
export SPORT
mkdir -p data
case "$SPORT" in
  nba|nfl) [ -f "data/$SPORT/players.csv" ] || { echo "[docker] no data/$SPORT/players.csv: generating FAKE sample data"; python make_sample_sport.py "$SPORT"; } ;;
  mlb)     [ -f "data/hitters.csv" ] || { echo "[docker] no data/hitters.csv: generating FAKE sample data"; python make_sample_data.py; } ;;
esac
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"

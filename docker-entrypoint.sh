#!/bin/sh
# Starts the app. With no data mounted it generates FAKE sample projections so the app is usable immediately.
set -e
SPORT="${SPORT:-nba}"
export SPORT
mkdir -p data
SAMPLE=0
case "$SPORT" in
  nba|nfl) [ -f "data/$SPORT/players.csv" ] || { echo "[docker] no data/$SPORT/players.csv: generating FAKE sample data"; python make_sample_sport.py "$SPORT"; SAMPLE=1; } ;;
  mlb)     [ -f "data/hitters.csv" ] || { echo "[docker] no data/hitters.csv: generating FAKE sample data"; python make_sample_data.py; SAMPLE=1; } ;;
esac
if [ "$SAMPLE" = 1 ] && [ -z "${DEMO_NOTICE:-}" ]; then
  export DEMO_NOTICE="Sample data: the players and stats in this copy are invented, so use it to try the tool, not to draft."
fi
exec python -m uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"

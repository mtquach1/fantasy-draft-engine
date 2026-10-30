#!/usr/bin/env bash
# macOS / Linux: ./start_draft.sh   (first time: chmod +x start_draft.sh)
cd "$(dirname "$0")"
(sleep 4; (open http://127.0.0.1:8000/ || xdg-open http://127.0.0.1:8000/) >/dev/null 2>&1) &
python3 -m uvicorn app.main:app

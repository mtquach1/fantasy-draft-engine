@echo off
rem Double-click to start the Draft Room. Close this window to stop it.
cd /d "%~dp0"
echo Starting the Draft Room... your browser will open in a few seconds.
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://127.0.0.1:8000/"
python -m uvicorn app.main:app
if errorlevel 1 (
  echo.
  echo Something went wrong. Read the message above, then see CHEATSHEET.md.
  pause
)

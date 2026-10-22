@echo off
setlocal
set "PYTHONUTF8=1"
cd /d "%~dp0"
uv run --locked --extra neural xiangqi-train %*
if errorlevel 1 pause

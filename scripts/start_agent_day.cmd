@echo off
rem Autonomous Claude session (branch experiment/autonomous-agent; user design of
rem 2026-10-16): Claude Opus 5.5 (medium effort) runs the whole intraday session by
rem itself on a Rs 20 lakh paper wallet (up to 5x MIS leverage). It picks its own
rem stocks, strategy, sizes and exits from objective data and the web; there is no
rem scanner, estimate or rule from us. AI budget $30 a day. Paper only: no order
rem reaches Kite. After the session, Kite minute bars with open interest of the
rem live F&O contracts go to data\fno\kite_minute (until 18:15).
rem The one-time scheduled task "AI Trader agent run" starts this; it can also be
rem double-clicked. Data, .env and .venv come from the main checkout, the code from
rem this script's checkout. Extra arguments go to run_agent.
set "ROOT=%~dp0.."
if not exist "%ROOT%\.venv\Scripts\python.exe" set "ROOT=%~dp0..\..\.."
set "PYTHONPATH=%~dp0..\src"
cd /d "%ROOT%"
title AI trader autonomous agent
".venv\Scripts\python.exe" -m ai_trader.cli.run_agent --model claude-opus-5-5 --effort medium --budget 30 --wallet 2000000 --leverage 5 --fno-capture data\fno\kite_minute --login-wait 360 %*
set CODE=%ERRORLEVEL%
if "%AI_TRADER_NO_HOLD%"=="1" exit /b %CODE%
echo.
echo Agent run ended (exit code %CODE%). Full log: data\paper\logs
echo This window closes in 2 hours; press a key to close it now.
timeout /t 7200 >nul
exit /b %CODE%

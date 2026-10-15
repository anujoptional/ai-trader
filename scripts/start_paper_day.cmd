@echo off
rem Daily paper-trading run. The Windows scheduled task "AI Trader paper run"
rem starts this on weekdays at 08:30; it can also be double-clicked. It opens the
rem Kite sign-in in the browser and waits for it until 14:30. Weekends, NSE
rem holidays and a second copy exit at once. Extra arguments go to run_paper.
rem From 2026-10-14 (user design): +-1% movement scanner at 0.90, the AIs call
rem LONG / SHORT / NO_TRADE on the direction to the close, paper trades exit at
rem the target or 15:00. From 2026-10-15 (user): GPT-6.1 Sol medium (main) and
rem Claude Opus 5.5 medium get only each pick's symbol and company, research it
rem themselves (web search) and answer one word on the 15:00 price; target 0.5%;
rem no capital, position, per-stock or AI spend limits; state in
rem data\paper\research_tp05 (Oct 14: data\paper\intraday, 1% target).
cd /d "%~dp0.."
title AI trader paper run
".venv\Scripts\python.exe" -m ai_trader.cli.run_paper --main gpt-6.1-sol:medium --shadow claude-opus-5-5:medium --prompt research --take-profit 0.5 --no-limits --ai-budget inf --ai-timeout 150 --store data\paper\research_tp05\paper.sqlite --journal data\paper\research_tp05\decisions.sqlite --report-dir data\paper\research_tp05\report --login-wait 360 %*
set CODE=%ERRORLEVEL%
rem End-to-end check of a finished session's inputs, fills and exits (data\validation_YYYYMMDD\auto).
if "%CODE%"=="0" (
  echo.
  echo Validating today's session end to end...
  ".venv\Scripts\python.exe" -m scripts.validate_paper_inputs --journals data\paper\research_tp05 --independent > "data\paper\logs\validation_latest.log" 2>&1
  type "data\paper\logs\validation_latest.log"
)
if "%AI_TRADER_NO_HOLD%"=="1" exit /b %CODE%
echo.
echo Paper run ended (exit code %CODE%). Full log: data\paper\logs
echo This window closes in 2 hours; press a key to close it now.
timeout /t 7200 >nul
exit /b %CODE%

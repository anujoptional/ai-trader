@echo off
rem Daily paper-trading run. The Windows scheduled task "AI Trader paper run"
rem starts this on weekdays at 08:30; it can also be double-clicked. It opens the
rem Kite sign-in in the browser and waits for it until 14:30. Weekends, NSE
rem holidays and a second copy exit at once. Extra arguments go to run_paper.
cd /d "%~dp0.."
title AI trader paper run
".venv\Scripts\python.exe" -m ai_trader.cli.run_paper --shadow gpt-6.1-sol:low --shadow claude-sonnet-5-5:low --login-wait 360 %*
set CODE=%ERRORLEVEL%
if "%AI_TRADER_NO_HOLD%"=="1" exit /b %CODE%
echo.
echo Paper run ended (exit code %CODE%). Full log: data\paper\logs
echo This window closes in 2 hours; press a key to close it now.
timeout /t 7200 >nul
exit /b %CODE%

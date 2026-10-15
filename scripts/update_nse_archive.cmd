@echo off
rem Evening NSE archive top-up. The Windows scheduled task "AI Trader NSE archive"
rem starts this on weekdays at 20:45 (NSE publishes the day's files in the evening;
rem a missed run starts when the PC is next on). It keeps the last 15 days of NSE's
rem F&O and cash bhavcopies, participant OI/volume, ban lists, combined OI and the
rem all-index closes in data\nse, the day's FII/DII cash flows (NSE serves only the
rem latest day), then rebuilds the F&O-derived features in data\fno\features.
cd /d "%~dp0.."
title AI trader NSE archive
".venv\Scripts\python.exe" -m ai_trader.cli.fetch_nse_archives --update > "data\paper\logs\nse_archive_latest.log" 2>&1
".venv\Scripts\python.exe" -m ai_trader.cli.build_fno_dataset >> "data\paper\logs\nse_archive_latest.log" 2>&1
rem The Kite minute capture of F&O contracts against NSE's end-of-day file.
".venv\Scripts\python.exe" -m scripts.validate_fno_data --only capture --output data\fno\validation\latest >> "data\paper\logs\nse_archive_latest.log" 2>&1
exit /b %ERRORLEVEL%

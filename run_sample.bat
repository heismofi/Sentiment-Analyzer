@echo off
cd /d "%~dp0"
python sentiment_tool.py %*
pause

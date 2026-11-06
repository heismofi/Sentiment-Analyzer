@echo off
cd /d "%~dp0"
python sentiment_gui.py
if errorlevel 1 pause

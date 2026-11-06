@echo off
setlocal
cd /d "%~dp0"
echo ============================================================
echo  Building SentimentAnalyzer.exe   (takes a few minutes)
echo ============================================================

if not exist sentiment_gui.py goto missing
if not exist sentiment_tool.py goto missing
if not exist requirements.txt goto missing

set PY=python
%PY% --version >nul 2>nul
if errorlevel 1 set PY=py
%PY% --version >nul 2>nul
if errorlevel 1 goto nopython

if not exist .venv_build\Scripts\python.exe (
    echo Creating a clean build environment...
    %PY% -m venv .venv_build
    if errorlevel 1 goto fail
)
set VPY=.venv_build\Scripts\python.exe

"%VPY%" -m pip install --upgrade pip
if errorlevel 1 goto fail
"%VPY%" -m pip install -r requirements.txt pyinstaller
if errorlevel 1 goto fail

echo Preparing the sample data to bundle...
"%VPY%" -c "import sentiment_tool; sentiment_tool.ensure_sample_file()"
if errorlevel 1 goto fail
if not exist data\sample_social_media.csv goto fail

echo Downloading the VADER lexicon so it can be bundled inside the app...
"%VPY%" -c "import sys, nltk; sys.exit(0 if nltk.download('vader_lexicon', download_dir='nltk_data') else 1)"
if errorlevel 1 goto fail
if not exist nltk_data goto fail

"%VPY%" -m PyInstaller --noconfirm --clean --onefile --windowed --name SentimentAnalyzer ^
  --add-data "data;data" ^
  --add-data "nltk_data;nltk_data" ^
  --hidden-import nltk.sentiment.vader ^
  --hidden-import matplotlib.backends.backend_tkagg ^
  sentiment_gui.py
if errorlevel 1 goto fail

echo.
echo ============================================================
echo  SUCCESS.  Your application is:  dist\SentimentAnalyzer.exe
echo  It is a single file. Copy it anywhere and double-click it.
echo ============================================================
pause
exit /b 0

:missing
echo Some project files are missing. Put ALL of these in the same folder as build_exe.bat:
echo     sentiment_tool.py   sentiment_gui.py   requirements.txt
pause
exit /b 1

:nopython
echo Python was not found. Install it from https://www.python.org/downloads/
echo and tick "Add python.exe to PATH" in the installer, then run this again.
pause
exit /b 1

:fail
echo.
echo The build failed. Scroll up to read the error message.
pause
exit /b 1

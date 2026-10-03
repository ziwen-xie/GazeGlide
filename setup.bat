@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  py -3.13 -m venv .venv
  if errorlevel 1 goto failed
)
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe download_model.py
if errorlevel 1 goto failed
echo GazeGlide is ready. Double-click start.bat to launch.
pause
exit /b 0
:failed
echo Setup failed. Make sure Python 3.13 is installed and internet access is available.
pause
exit /b 1

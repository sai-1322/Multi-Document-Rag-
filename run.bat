@echo off
REM One-click launcher for Windows. Double-click this file.
cd /d "%~dp0"
if not exist .venv (
  echo Creating virtual environment...
  py -3.11 -m venv .venv 2>nul || python -m venv .venv
)
call .venv\Scripts\activate
echo Installing dependencies (first run takes a few minutes)...
python -m pip install -q -r requirements.txt
if "%OPENAI_API_KEY%"=="" set /p OPENAI_API_KEY=Paste your OpenAI API key and press Enter: 
echo.
echo Starting server. First start downloads models and can take a minute.
echo When you see "Application startup complete", open http://localhost:8000
start "" cmd /c "timeout /t 25 >nul & start http://localhost:8000"
uvicorn main:app --port 8000
pause

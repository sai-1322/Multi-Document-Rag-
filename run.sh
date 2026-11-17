#!/usr/bin/env bash
# One-command launcher for macOS/Linux:  ./run.sh
set -e
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
echo "Installing dependencies (first run takes a few minutes)..."
python -m pip install -q -r requirements.txt
if [ -z "$OPENAI_API_KEY" ]; then read -r -s -p "Paste your OpenAI API key: " OPENAI_API_KEY; echo; export OPENAI_API_KEY; fi
echo "Open http://localhost:8000 once you see 'Application startup complete'"
exec uvicorn main:app --port 8000

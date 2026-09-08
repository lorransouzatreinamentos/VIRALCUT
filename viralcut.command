#!/usr/bin/env bash
# VIRALCUT — launcher (macOS). Atualiza, sobe o servidor local e abre o app.
cd "$(dirname "$0")"
git pull --ff-only >/dev/null 2>&1 || true

export RESOLVE_SCRIPT_API="/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
export RESOLVE_SCRIPT_LIB="/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
export PYTHONPATH="$PYTHONPATH:/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules/"

( sleep 4; open "http://127.0.0.1:8756/ui/" ) &
exec .venv/bin/python -m uvicorn core.main:app --port 8756

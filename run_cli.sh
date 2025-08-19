#!/bin/bash
# PeptiDIA CLI Launcher Script

echo "🧬 Starting PeptiDIA Command Line Interface..."

# Select Python executable: prefer local venv, else current environment (supports Conda)
if [ -x "./peptidia_env/bin/python" ]; then
	PYTHON="./peptidia_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
	PYTHON="python3"
else
	PYTHON="python"
fi

# Ensure dependencies are installed (use locked requirements if available)
if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import pandas
PY
then
	echo "❌ Dependencies not found. Installing required packages into current environment..."
	REQ_FILE="requirements.txt"
	if [ -f "requirements-locked.txt" ]; then
		REQ_FILE="requirements-locked.txt"
	fi
	"$PYTHON" -m pip install -r "$REQ_FILE"
fi

# Run CLI
"$PYTHON" peptidia_cli.py
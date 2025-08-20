#!/bin/bash
# PeptiDIA vs DIA-NN Comparison Tool Launcher

echo "📊 Starting PeptiDIA vs DIA-NN Comparison Tool..."

# Check if virtual environment exists
if [ ! -d "peptidia_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "💡 Please run: python scripts/install.py"
    echo "   This will create the environment and install dependencies."
    exit 1
fi

# Select Python executable: prefer local venv, else current environment
if [ -x "./peptidia_env/bin/python" ]; then
	PYTHON="./peptidia_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
	PYTHON="python3"
else
	PYTHON="python"
fi

# Use selected Python
"$PYTHON" src/peptidia/analysis/diann_comparison.py
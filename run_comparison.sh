#!/bin/bash
# PeptiDIA vs DIA-NN Comparison Tool Launcher

echo "📊 Starting PeptiDIA vs DIA-NN Comparison Tool..."

# Check if virtual environment exists
if [ ! -d "peptidia_env" ]; then
    echo "❌ Virtual environment not found!"
    echo "💡 Please run: python install.py"
    echo "   This will create the environment and install dependencies."
    exit 1
fi

# Use Python from virtual environment
./peptidia_env/bin/python diann_comparison.py
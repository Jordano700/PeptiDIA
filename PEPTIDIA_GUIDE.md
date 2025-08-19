# PeptiDIA User Guide

## Windows (PowerShell) Quick Start

```powershell
# 1) Install Python 3.12 (if not already installed)
winget install Python.Python.3.12

# Restart PowerShell so PATH updates apply

# 2) Clone the repo and enter it
git clone https://github.com/Jordano700/PeptiDIA.git
cd PeptiDIA

# 3) Run the installer with Python 3.12
py -3.12 install.py

# 4) Launch the app
./start_peptidia.bat
# then open http://localhost:8501
```

## Linux/macOS Quick Start

```bash
git clone https://github.com/Jordano700/PeptiDIA.git
cd PeptiDIA

# Option A: Installer (requires Python 3.12.2 available)
python3.12 install.py
./start_peptidia.sh

# Option B: Conda (installs Python 3.12.2)
conda env create -f environment.yml
conda activate peptidia
./run_streamlit.sh
```

## Installing Python 3.12 on macOS/Linux

If your system doesn't have Python 3.12 yet, use one of the following methods:

- macOS (Homebrew):
```bash
brew install python@3.12
python3.12 --version
```

- Ubuntu/Debian (Deadsnakes PPA):
```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
python3.12 --version
```

- Fedora:
```bash
sudo dnf install -y python3.12
python3.12 --version
```

- Pyenv (works on macOS/Linux):
```bash
curl https://pyenv.run | bash
# restart your shell so pyenv is on PATH
pyenv install 3.12.2
pyenv local 3.12.2
python --version
```

- Conda (installs interpreter inside the environment):
```bash
conda create -n peptidia python=3.12.2 -y
conda activate peptidia
```

After installing Python 3.12, you can create a venv if you prefer:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

## Data Folder Structure

```
data/
  YourDataset/
    short_gradient/
      FDR_1/
      FDR_20/
      FDR_50/
    long_gradient/
      FDR_1/
```

## Notes
- Use Python 3.12.2 for best compatibility.
- `.streamlit/secrets.toml` should not be committed.
- For private repos, use SSH keys or a Personal Access Token when cloning.
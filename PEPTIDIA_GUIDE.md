# PeptiDIA User Guide (Whitepaper)

## Table of Contents
- [1. Overview](#1-overview)
- [2. System Requirements](#2-system-requirements)
- [3. Installation](#3-installation)
  - [3.1 Windows (PowerShell)](#31-windows-powershell)
  - [3.2 macOS/Linux Quick Start](#32-macoslinux-quick-start)
  - [3.3 Installing Python 3.12 on macOS/Linux](#33-installing-python-312-on-macoslinux)
  - [3.4 Verify the Installation](#34-verify-the-installation)
- [4. Dataset Preparation](#4-dataset-preparation)
- [5. Using PeptiDIA](#5-using-peptidia)
  - [5.1 Web Interface Workflow](#51-web-interface-workflow)
  - [5.2 Command Line Interface (CLI)](#52-command-line-interface-cli)
  - [5.3 Comparison Workflow](#53-comparison-workflow)
- [6. Results & Outputs](#6-results--outputs)
- [7. Troubleshooting](#7-troubleshooting)
- [8. Security Notes](#8-security-notes)
- [9. FAQ](#9-faq)
- [10. Support & License](#10-support--license)

---

## 1. Overview
PeptiDIA helps scientists discover more peptides in DIA-NN mass spectrometry data using machine learning. It provides:
- **Web interface**: no coding required
- **AI-powered**: enhanced peptide discovery at low error rates
- **Streamlined workflow**: setup → training → inference

---

## 2. System Requirements
- **Python**: 3.12.2 (recommended; 3.12.x required by installer)
- **OS**: Windows, macOS, or Linux
- **RAM**: 8 GB+ recommended (depends on dataset size)
- **Disk**: Sufficient space for datasets and results

Optional:
- **Conda**: for users who prefer environment management

---

## 3. Installation
Choose one of the methods below.

### 3.1 Windows (PowerShell)
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
# open http://localhost:8501
```

### 3.2 macOS/Linux Quick Start
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

### 3.3 Installing Python 3.12 on macOS/Linux
If your system doesn't have Python 3.12 yet, use one of the following:

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

- Pyenv (macOS/Linux):
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

After installing Python 3.12, you can use a venv if preferred:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3.4 Verify the Installation
- Web interface starts at: `http://localhost:8501`
- If port 8501 is in use, launch with another port, for example:
```bash
python -m streamlit run streamlit_app.py --server.port 8502
```

---

## 4. Dataset Preparation
Organize your DIA-NN analyzed `.parquet` files in the `data/` folder with the following structure and FDR levels:

```
data/
  YourDataset/
    short_gradient/
      FDR_1/      # DIA-NN results at 1% FDR (baseline)
      FDR_20/     # DIA-NN results at 20% FDR (training)
      FDR_50/     # DIA-NN results at 50% FDR (training)
    long_gradient/
      FDR_1/      # DIA-NN results at 1% FDR (ground truth)
```

Notes:
- Short gradient requires: 1%, 20%, 50% FDR
- Long gradient requires: 1% FDR

---

## 5. Using PeptiDIA

### 5.1 Web Interface Workflow
1. **Setup Mode**
   - Configure datasets and ground truth matching
2. **Training Mode**
   - Train models and monitor metrics
3. **Inference Mode**
   - Apply trained models, view and download results

Start the web UI:
```bash
./run_streamlit.sh   # macOS/Linux (or use the installer launcher)
# Windows users can double-click start_peptidia.bat
```

### 5.2 Command Line Interface (CLI)
```bash
./run_cli.sh
```

### 5.3 Comparison Workflow
```bash
./run_comparison.sh
```

---

## 6. Results & Outputs
Typical outputs (indicative):
- `results/` — analysis results, summaries, and artifacts
- `history/` — run history metadata
- Downloadable reports from the UI (e.g., CSV/JSON artifacts)

---

## 7. Troubleshooting
- **Port 8501 in use**: run Streamlit with `--server.port 8502`
- **Missing dependencies**: the launch scripts auto-install from `requirements-locked.txt` (if present) or `requirements.txt`
- **Private repo cloning**: use SSH keys or a GitHub Personal Access Token
- **DNS issues cloning GitHub**: restart DNS (`systemd-resolved`), set DNS to `1.1.1.1/8.8.8.8`, or try another network
- **Windows installer errors (encoding)**: the installer generates an ASCII-only `.bat` launcher

---

## 8. Security Notes
- Streamlit CORS/XSRF protections are disabled for local usage convenience; do not expose the app directly to the internet without proper hardening
- Do not commit secrets; keep them in `.streamlit/secrets.toml` (ignored by git)

---

## 9. FAQ
- **Which Python version do I need?** 3.12.x (3.12.2 recommended)
- **Conda vs venv?** Either is fine; Conda can install Python for you
- **Do I need admin rights?** Only for system-wide package managers (e.g., apt/brew)

---

## 10. Support & License
- Issues and questions: GitHub Issues on the project repository
- License: see `LICENSE` file
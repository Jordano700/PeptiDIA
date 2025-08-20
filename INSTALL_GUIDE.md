# PeptiDIA Installation Guide

Simple installation steps for **Windows** and **Mac/Linux** users.

## 🖥️ Windows Installation

### Step 1: Get PeptiDIA
```powershell
git clone https://github.com/Jordano700/PeptiDIA.git
cd PeptiDIA
```

### Step 2: Install
```powershell
python scripts/install.py
```

### Step 3: Run PeptiDIA
```powershell
start_peptidia.bat
```

Your browser will open automatically to `http://localhost:8501`

---

## 🍎 Mac/Linux Installation  

### Step 1: Get PeptiDIA
```bash
git clone https://github.com/Jordano700/PeptiDIA.git
cd PeptiDIA
```

### Step 2: Install
```bash
python scripts/install.py
```

### Step 3: Run PeptiDIA
```bash
./start_peptidia.sh
```

Your browser will open automatically to `http://localhost:8501`

---

## 🚨 Troubleshooting

### "Port 8501 is already in use" Error

If you see this error, try these solutions:

**Option 1: Kill existing process (Recommended)**
```bash
# Mac/Linux
lsof -ti:8501 | xargs kill -9

# Windows PowerShell  
netstat -ano | findstr :8501
taskkill /PID <PID_NUMBER> /F
```

**Option 2: Use a different port**

Edit the launcher file and change the port:

**Windows (`start_peptidia.bat`):**
```batch
streamlit run src/peptidia/web/streamlit_app.py --server.port=8502
```

**Mac/Linux (`start_peptidia.sh`):**  
```bash
streamlit run src/peptidia/web/streamlit_app.py --server.port=8502
```

Then open `http://localhost:8502` instead.

### Python Not Found

**Windows:**
```powershell
winget install Python.Python.3.12
```

**Mac:**
```bash
brew install python@3.12
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
```

### "pip not recognized" (Windows)

Try using:
```powershell
python -m pip install -r requirements.txt
```

Instead of just `pip install -r requirements.txt`

---

## That's It! 🎉

Once installed, you can:
- **🔧 Setup Mode**: Configure your datasets
- **🎯 Training Mode**: Train AI models  
- **🚀 Inference Mode**: Find new peptides

The web interface guides you through each step!
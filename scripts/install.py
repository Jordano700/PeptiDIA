#!/usr/bin/env python3
"""
PeptiDIA Universal Installer

"""

import sys
import subprocess
import platform
import os
import venv
from pathlib import Path

def print_header():
    """Print installation header."""
    print("🧬 PeptiDIA Universal Installer")
    print("=" * 50)
    print("Creating isolated environment with tested versions")
    print()

def check_python_version():
    """Support Python 3.10+ (3.12 is the tested reference). Newer versions use flexible deps."""
    version = sys.version_info
    ver_str = f"{version.major}.{version.minor}.{version.micro}"
    if version.major != 3 or version.minor < 10:
        print("❌ PeptiDIA needs Python 3.10 or newer (3.12 recommended)")
        print(f"   Current: Python {ver_str}")
        print()
        print("💡 Easiest fix - use Conda (creates a working 3.12 environment):")
        print("   conda env create -f config/environment.yml && conda activate peptidia")
        return False

    if version.minor == 12:
        print(f"✅ Python {ver_str} - Compatible!")
    elif version.minor < 12:
        print(f"✅ Python {ver_str} detected (3.12 is the tested reference). Continuing...")
    else:
        print(f"⚠️  Python {ver_str} is newer than the tested range - PeptiDIA will install with")
        print("    flexible dependency versions. If anything fails, use Conda:")
        print("    conda env create -f config/environment.yml")
    return True

def create_virtual_environment():
    """Create virtual environment."""
    venv_path = Path("peptidia_env")
    
    # Check if environment already exists and is working
    if venv_path.exists():
        python_exe = get_venv_python(venv_path)
        if python_exe.exists():
            print("✅ Virtual environment already exists - reusing")
            return venv_path
        else:
            print("🗑️  Removing corrupted environment...")
            import shutil
            shutil.rmtree(venv_path)
    
    print("🏗️  Creating fresh virtual environment...")
    try:
        venv.create(venv_path, with_pip=True)
        print("✅ Virtual environment created!")
        return venv_path
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        print("💡 Try running as administrator/sudo")
        return None

def get_venv_python(venv_path):
    """Get Python executable in virtual environment."""
    if platform.system() == "Windows":
        return venv_path / "Scripts" / "python.exe"
    else:
        return venv_path / "bin" / "python"

def check_packages_installed(python_exe):
    """Check if key packages are already installed to avoid reinstalling."""
    try:
        test_script = '''
import streamlit
import pandas
import numpy
import plotly
import sklearn
print("PACKAGES_OK")
'''
        result = subprocess.run([
            str(python_exe), "-c", test_script
        ], capture_output=True, text=True, timeout=10)
        
        return result.returncode == 0 and "PACKAGES_OK" in result.stdout
    except Exception:
        return False

def install_dependencies(python_exe):
    """Install dependencies from locked file if present, else requirements.txt."""
    print("📦 Checking dependencies...")
    
    # Check if packages are already installed
    if check_packages_installed(python_exe):
        print("✅ All key packages already installed - skipping installation")
        return True
    
    print("📦 Installing missing dependencies...")
    
    # Upgrade pip first (non-fatal if it fails)
    try:
        subprocess.check_call([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    except subprocess.CalledProcessError:
        print("⚠️  Could not upgrade pip; continuing with the existing version.")

    pinned = "config/requirements.txt"
    flexible = "config/requirements-flexible.txt"

    # On Python newer than the tested 3.12, the exact pins may lack wheels, so try
    # flexible first. Otherwise use the tested pins and fall back to flexible on failure.
    order = [flexible, pinned] if sys.version_info[:2] > (3, 12) else [pinned, flexible]

    for i, req_file in enumerate(order):
        if not os.path.exists(req_file):
            continue
        label = "tested versions" if req_file == pinned else "flexible versions"
        print(f"   Installing dependencies ({label}) from {req_file}")
        try:
            subprocess.check_call([str(python_exe), "-m", "pip", "install", "-r", req_file])
            print("✅ All dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            if i + 1 < len(order):
                print(f"⚠️  Install from {req_file} failed; retrying with {order[i + 1]} ...")
            else:
                print(f"❌ Installation failed: {e}")
                print("💡 Try Conda instead: conda env create -f config/environment.yml")
    return False

def test_installation(python_exe):
    """Test that PeptiDIA can start."""
    print("🧪 Testing installation...")
    
    test_script = '''
import streamlit
import pandas
import numpy
import plotly
import sklearn
print("SUCCESS")
'''
    
    try:
        result = subprocess.run([
            str(python_exe), "-c", test_script
        ], capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and "SUCCESS" in result.stdout:
            print("✅ Installation test passed!")
            return True
        else:
            print("❌ Installation test failed")
            print("💡 Some packages may be incompatible")
            return False
    except subprocess.TimeoutExpired:
        print("❌ Installation test timed out")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def create_launchers(venv_path):
    """Create both Unix (.sh) and Windows (.bat) launchers regardless of host OS.

    Launchers auto-detect the local venv first and fall back to system Python.
    """
    # Unix shell launcher (works on macOS/Linux)
    sh_content = '''#!/bin/bash
echo "Starting PeptiDIA..."
cd "$(dirname "$0")"

# Kill any existing Streamlit processes to free up ports
echo "Checking for existing Streamlit processes..."
pkill -f "streamlit" 2>/dev/null || true
sleep 2

echo "Waiting for ports to be available..."
sleep 1

if [ -x "./peptidia_env/bin/python" ]; then
    PY="./peptidia_env/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    PY="python"
fi

echo "Launching PeptiDIA on port 8501..."
"$PY" -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8501
if [ $? -ne 0 ]; then
    echo "ERROR: PeptiDIA failed to start on port 8501"
    echo "Trying alternative port 8502..."
    "$PY" -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8502
fi
'''
    sh_path = Path("start_peptidia.sh")
    sh_path.write_text(sh_content)
    try:
        os.chmod(sh_path, 0o755)
    except Exception:
        pass

    # Windows batch launcher
    bat_content = r'''@echo off
title PeptiDIA
echo Starting PeptiDIA...
cd /d "%~dp0"

REM Kill any existing Streamlit processes to free up ports
echo Checking for existing Streamlit processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq streamlit*" 2>nul
taskkill /F /IM python.exe /FI "COMMANDLINE eq *streamlit*" 2>nul
timeout /t 2 /nobreak >nul

REM Wait a moment for ports to be freed
echo Waiting for ports to be available...
timeout /t 3 /nobreak >nul

if exist "peptidia_env\\Scripts\\python.exe" (
  set "PY=peptidia_env\\Scripts\\python.exe"
) else (
  set "PY=python"
)

echo Launching PeptiDIA on port 8501...
"%PY%" -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8501
if errorlevel 1 (
  echo.
  echo ERROR: PeptiDIA failed to start on port 8501
  echo Trying alternative port 8502...
  "%PY%" -m streamlit run src/peptidia/web/streamlit_app.py --server.port 8502
  if errorlevel 1 (
    echo.
    echo ERROR: PeptiDIA failed to start on both ports
    echo TIP: Try closing all browser tabs and restarting
    pause
  )
)
'''
    bat_path = Path("start_peptidia.bat")
    bat_path.write_text(bat_content)

    print(f"✅ Launchers created: {sh_path} and {bat_path}")
    # Return platform-preferred path for convenience
    preferred = bat_path if platform.system() == "Windows" else sh_path
    return preferred, sh_path, bat_path

def print_success_message(preferred_launcher, sh_path, bat_path):
    """Print final success message with both launchers."""
    print()
    print("🎉 INSTALLATION COMPLETE!")
    print("=" * 50)
    print()
    print("🚀 To start PeptiDIA:")
    print(f"   📁 Unix/macOS: ./{sh_path}")
    print(f"   📁 Windows PowerShell: ./{bat_path}")
    print(f"   📁 Windows CMD: {bat_path}")
    
    # Show platform-specific preferred command
    current_os = platform.system()
    if current_os == "Windows":
        print(f"   ⭐ Recommended for PowerShell: ./{bat_path}")
    else:
        print(f"   ⭐ Preferred for this system: ./{preferred_launcher}")
    
    print()
    print("📱 Then open your browser to: http://localhost:8501")
    print()
    print("📂 Next steps:")
    print("   1. Put your DIA-NN .parquet files in the 'data/' folder")
    print("   2. Follow the web interface - it guides you through everything!")
    print()
    print("💡 Need help? Open an issue on GitHub")

def main():
    """Main installation function."""
    print_header()
    
    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Create virtual environment
    venv_path = create_virtual_environment()
    if not venv_path:
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Get Python executable
    python_exe = get_venv_python(venv_path)
    
    # Install dependencies
    if not install_dependencies(python_exe):
        input("Press Enter to exit...")
        sys.exit(1)
    
    # Test installation
    if not test_installation(python_exe):
        print("⚠️  Installation may have issues, but continuing...")
    
    # Create launchers
    preferred, sh_launcher, bat_launcher = create_launchers(venv_path)
    
    # Success message
    print_success_message(preferred, sh_launcher, bat_launcher)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        print("💡 Please report this issue on GitHub")
        input("Press Enter to exit...")
        sys.exit(1)
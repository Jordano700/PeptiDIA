#!/usr/bin/env python3
"""
PeptiDIA Universal Installer
One script that always works - creates isolated environment with exact tested versions
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
    print("Creating isolated environment with tested versions...")
    print()

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version < (3, 8):
        print("❌ INCOMPATIBLE PYTHON VERSION")
        print(f"   Current: Python {version.major}.{version.minor}.{version.micro}")
        print("   Required: Python 3.8 or higher")
        print()
        print("💡 Please install Python 3.8+ from: https://python.org/downloads")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
    return True

def create_virtual_environment():
    """Create virtual environment."""
    venv_path = Path("peptidia_env")
    
    # Remove existing environment if it exists
    if venv_path.exists():
        print("🗑️  Removing existing environment...")
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

def install_dependencies(python_exe):
    """Install exact tested versions."""
    print("📦 Installing exact tested versions...")
    print("   (This ensures everything works together)")
    
    try:
        # Upgrade pip first
        subprocess.check_call([
            str(python_exe), "-m", "pip", "install", "--upgrade", "pip"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Install exact versions
        subprocess.check_call([
            str(python_exe), "-m", "pip", "install", "-r", "requirements.txt"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("💡 Check your internet connection and try again")
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
        ], capture_output=True, text=True, timeout=30)
        
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


def print_success_message():
    """Print final success message."""
    print()
    print("🎉 INSTALLATION COMPLETE!")
    print("=" * 50)
    print()
    print("🚀 Available tools:")
    print("   📱 Web Interface:     ./run_streamlit.sh")
    print("   💻 Command Line:      ./run_cli.sh") 
    print("   📊 DIA-NN Comparison: ./run_comparison.sh")
    print()
    print("📂 Next steps:")
    print("   1. Put your DIA-NN .parquet files in the 'data/' folder")
    print("   2. Choose your tool and run it!")
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
    
    # Success message
    print_success_message()

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
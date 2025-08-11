<div align="center">
  <img src="peptidia_retro.png" alt="PeptiDIA - Pixel Art DNA Logo" width="500"/>
  
  # PeptiDIA
  **Find More Peptides in Your Data** 🧬
</div>

## What is PeptiDIA?

PeptiDIA helps scientists find **MORE peptides** in their DIA-NN mass spectrometry analyzed data using machine learning.

- 📊 **Web interface** - No coding required !
- 🤖 **AI-powered** - Finds additional peptides at low error rates
- 🔬 **Easy to use** - Upload data, get results

## Quick Start (3 steps!)

### Step 1: Get PeptiDIA
```bash
git clone https://github.com/Jordano700/PeptiDIA.git
cd PeptiDIA
pip install -r requirements.txt
```

### Step 2: Add Your Data
Put your `.parquet` files in the `data/` folder like this:
```
data/
  YourDataset/
    short_gradient/
      FDR_1/
        your_file.parquet
    long_gradient/  
      FDR_1/
        your_file.parquet
```

### Step 3: Run PeptiDIA
```bash
./run_streamlit.sh
```
Then open your web browser to `http://localhost:8501` 

## Interface Modes 🎛️

PeptiDIA has **3 simple modes** to guide you through the process:

### 1. 🔧 **Setup Mode**
- Configure your datasets
- Set up ground truth matching
- Tell PeptiDIA which files to compare

### 2. 🎯 **Training Mode** 
- AI learns from your data
- Creates smart models
- Shows training progress

### 3. 🚀 **Inference Mode**
- Apply trained models to find new peptides
- Get results instantly
- Download your discoveries

## That's it! 🎉

The interface walks you through each step - no guessing needed!

## Need Help?

- 🐛 Report issues on GitHub
- 💡 Questions? Open an issue!


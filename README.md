<div align="center">
  <img src="peptidia_retro.png" alt="PeptiDIA - Pixel Art DNA Logo" width="500"/>
  
  # PeptiDIA
  
  **Advanced DIA Proteomics Analysis Platform**
</div>

**PeptiDIA** is a powerful web-based application for machine learning-enhanced peptide validation in DIA mass spectrometry data. It uses advanced algorithms to identify additional valid peptides from higher FDR data by leveraging comprehensive ground truth datasets.

![PeptiDIA Interface](https://img.shields.io/badge/Interface-Streamlit-red) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Quick Start

### 1. Clone and Install
```bash
git clone https://github.com/username/PeptiDIA.git
cd PeptiDIA
pip install -r requirements.txt
```

### 2. Add Your Data
Place your DIA-NN processed `.parquet` files in the data directory following this structure:
```
data/
└── YourDataset/
    ├── short_gradient/     # Fast gradient data
    │   ├── FDR_1/         # 1% FDR files (baseline)
    │   ├── FDR_20/        # 20% FDR files (training)
    │   └── FDR_50/        # 50% FDR files (training)
    └── long_gradient/      # Slow gradient data
        └── FDR_1/         # 1% FDR files (ground truth)
```

### 3. Launch PeptiDIA
```bash
./run_streamlit.sh
```

The web interface will open automatically at `http://localhost:8501`

## 🎯 What PeptiDIA Does

PeptiDIA helps you **recover additional valid peptides** from fast gradient DIA-NN analyses by:

1. **Training ML models** on multiple datasets using higher FDR data (20% or 50%)
2. **Validating against ground truth** from comprehensive slow gradient analysis (1% FDR)
3. **Identifying which higher-FDR peptides** are actually true positives
4. **Achieving low FDR** (typically 1-5%) while discovering significantly more peptides

### Key Innovation
- **Ground Truth**: Slow gradient 1% FDR peptides (comprehensive analysis)
- **Baseline**: Fast gradient 1% FDR peptides (high-confidence detections)  
- **Target**: Additional fast gradient peptides validated by ground truth
- **ML Classification**: Distinguish true positives from false positives in higher-FDR data

## 📊 Features

### Interactive Web Interface
- **🎛️ Intuitive Configuration**: Point-and-click parameter selection
- **📈 Real-time Visualizations**: Professional Nature-style plots with Plotly
- **📋 Session Management**: Save and reload analysis configurations
- **💾 Export Results**: CSV, JSON, and high-resolution plot downloads

### Advanced ML Pipeline  
- **🤖 XGBoost Integration**: GPU-accelerated gradient boosting with optimal hyperparameters
- **🔄 Cross-method Validation**: Train on multiple methods, test on holdout method
- **📏 Feature Engineering**: Automatic log transforms, ratios, and DIA-NN score integration
- **⚖️ Model Calibration**: Probability calibration for reliable confidence scores

### Production-Ready Architecture
- **🔍 Automatic Dataset Discovery**: Detects any dataset following the folder structure
- **📁 Generic File Handling**: Works with any DIA-NN processed parquet files
- **🔗 Triplicate Support**: Automatic grouping and ensemble analysis
- **📊 Comprehensive Reporting**: Detailed FDR analysis and performance metrics

## 📁 Data Requirements

PeptiDIA works with **DIA-NN processed data** exported as `.parquet` files. You need:

- **Short gradient data** at 1%, 20%, and 50% FDR
- **Long gradient data** at 1% FDR (ground truth)
- Files should contain standard DIA-NN columns (Modified.Sequence, Q.Value, etc.)

## 🛠️ System Requirements

### Required
- Python 3.8+
- 8GB+ RAM (16GB+ recommended for large datasets)

### Optional (for optimal performance)
- NVIDIA GPU with CUDA support (XGBoost acceleration)
- 32GB+ RAM for very large multi-dataset analyses

## 🔬 Scientific Background

PeptiDIA implements the methodology described in our research on **machine learning-enhanced peptide recovery** from DIA mass spectrometry data. The approach:

1. Uses **slow gradient comprehensive analysis** as ground truth validation
2. **Trains on cross-method data** to avoid overfitting to specific conditions
3. **Focuses on additional peptide discovery** beyond standard DIA-NN cutoffs
4. **Achieves superior FDR control** compared to simple Q-value thresholds

## 📖 Documentation

### Basic Workflow
1. **Configure Analysis**: Select training methods and holdout test method
2. **Set Parameters**: Choose FDR levels and target thresholds  
3. **Run Analysis**: Monitor progress with real-time updates
4. **Review Results**: Interactive visualizations and detailed metrics
5. **Export Data**: Save results and publication-ready plots

### Advanced Features
- **Triplicate Grouping**: Toggle `🔗 Use Triplicate Grouping` for ensemble analysis
- **Custom Thresholds**: Fine-tune FDR targets for specific research needs
- **Method Comparison**: Cross-validate across multiple analytical methods
- **Feature Analysis**: SHAP-based model interpretability


## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Support

- **Issues**: Report bugs and feature requests via GitHub Issues

## 🏆 Citation

If you use PeptiDIA in your research, please cite:

```bibtex
@software{peptidia2024,
  title={PeptiDIA: Machine Learning-Enhanced Peptide Validation for DIA Mass Spetrometry Data},
  author={[Your Name]},
  year={2025},
  url={https://github.com/username/PeptiDIA}
}
```

---

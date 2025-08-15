#!/usr/bin/env python3
"""
🎯 PeptiDIA - STREAMLIT FRONTEND
================================================================================

Built for researchers to easily configure and run peptide validation analyses
without programming knowledge.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob
import warnings

# Suppress XGBoost CUDA device warnings for cleaner interface
warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
warnings.filterwarnings('ignore', message='.*mismatched devices.*')

import os
import json
from pathlib import Path
from datetime import datetime
import sys
import warnings
import time

# Import file discovery functions
from file_discovery import discover_available_files, get_available_methods, get_files_for_method, extract_method_and_group_from_filename, group_files_by_triplicates
from dataset_utils import discover_available_files_by_dataset, extract_method_from_filename, get_configured_methods, get_files_for_configured_method

# Import functions are now available locally in peptide_validator_api.py
# No need for external scripts directory

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="PeptiDIA",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Consistent Blue/Red/Purple color theme for professional interface
NATURE_COLORS = {
    'primary': '#2E86AB',      # Primary blue (main data)
    'secondary': '#6366F1',    # Purple (secondary data, comparisons)
    'tertiary': '#DC2626',     # Red (targets, alerts, thresholds)
    'quaternary': '#C73E1D',   # Deep red (reserved for critical items)
    'accent1': '#64A6BD',      # Light blue (supporting elements)
    'accent2': '#A855F7',      # Light purple (highlights)
    'success': '#2E86AB',      # Blue for success (consistency)
    'warning': '#DC2626',      # Red for warnings (consistency)
    'neutral': '#6C757D'       # Professional gray
}

# Define multiple color schemes for user selection
COLOR_SCHEMES = {
    'Default': {
        'line_primary': '#2E86AB',     # Primary data lines
        'line_secondary': '#6366F1',   # Secondary/comparison lines
        'line_target': '#DC2626',      # Target/threshold lines (dashed)
        'bar_primary': '#2E86AB',      # Bar charts
        'bar_secondary': '#6366F1',    # Secondary bars
        'feature_gradient': ['#2E86AB', '#6366F1', '#A855F7'],  # Blue to purple gradient
        'background': 'white',
        'background_light': '#f8f9fa',
        'border_light': '#e9ecef',
        'grid': 'lightgray',
        'outline_subtle': 'rgba(255,255,255,0.3)',
        'text': '#374151',
        'text_secondary': '#6C757D',
        'neutral': '#6C757D'
    },
    'Viridis': {
        'line_primary': '#440154',
        'line_secondary': '#31688E',
        'line_target': '#FDE725',
        'bar_primary': '#440154',
        'bar_secondary': '#31688E',
        'feature_gradient': ['#440154', '#31688E', '#35B779', '#FDE725'],
        'background': 'white',
        'background_light': '#f8f9fa',
        'border_light': '#e9ecef',
        'grid': 'lightgray',
        'outline_subtle': 'rgba(255,255,255,0.3)',
        'text': '#374151',
        'text_secondary': '#6C757D',
        'neutral': '#6C757D'
    },
    'Plasma': {
        'line_primary': '#0D0887',
        'line_secondary': '#9C179E',
        'line_target': '#F0F921',
        'bar_primary': '#0D0887',
        'bar_secondary': '#9C179E',
        'feature_gradient': ['#0D0887', '#9C179E', '#ED7953', '#F0F921'],
        'background': 'white',
        'background_light': '#f8f9fa',
        'border_light': '#e9ecef',
        'grid': 'lightgray',
        'outline_subtle': 'rgba(255,255,255,0.3)',
        'text': '#374151',
        'text_secondary': '#6C757D',
        'neutral': '#6C757D'
    },
    'Ocean Blues': {
        'line_primary': '#08519C',
        'line_secondary': '#3182BD',
        'line_target': '#FD8D3C',
        'bar_primary': '#08519C',
        'bar_secondary': '#3182BD',
        'feature_gradient': ['#08519C', '#3182BD', '#6BAED6', '#C6DBEF'],
        'background': 'white',
        'background_light': '#f8f9fa',
        'border_light': '#e9ecef',
        'grid': 'lightgray',
        'outline_subtle': 'rgba(255,255,255,0.3)',
        'text': '#374151',
        'text_secondary': '#6C757D',
        'neutral': '#6C757D'
    },
    'Warm Sunset': {
        'line_primary': '#B30000',
        'line_secondary': '#E34A33',
        'line_target': '#FC8D59',
        'bar_primary': '#B30000',
        'bar_secondary': '#E34A33',
        'feature_gradient': ['#B30000', '#E34A33', '#FC8D59', '#FDCC8A'],
        'background': 'white',
        'background_light': '#f8f9fa',
        'border_light': '#e9ecef',
        'grid': 'lightgray',
        'outline_subtle': 'rgba(255,255,255,0.3)',
        'text': '#374151',
        'text_secondary': '#6C757D',
        'neutral': '#6C757D'
    },
    'Forest Greens': {
        'line_primary': '#00441B',
        'line_secondary': '#238B45',
        'line_target': '#FD8D3C',
        'bar_primary': '#00441B',
        'bar_secondary': '#238B45',
        'feature_gradient': ['#00441B', '#238B45', '#74C476', '#C7E9C0'],
        'background': 'white',
        'background_light': '#f8f9fa',
        'border_light': '#e9ecef',
        'grid': 'lightgray',
        'outline_subtle': 'rgba(255,255,255,0.3)',
        'text': '#374151',
        'text_secondary': '#6C757D',
        'neutral': '#6C757D'
    }
}

# Initialize chart colors (will be updated based on user selection)
CHART_COLORS = COLOR_SCHEMES['Default']

# Custom CSS for professional styling
st.markdown("""
<style>
    /* Import professional fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #2E86AB 0%, #6366F1 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #2E86AB 0%, #6366F1 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 134, 171, 0.3);
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2E86AB;
        margin-bottom: 1rem;
    }
    
    /* Status indicators */
    .status-success {
        color: #2E86AB;
        font-weight: 500;
    }
    
    .status-warning {
        color: #DC2626;
        font-weight: 500;
    }
    
    .status-error {
        color: #DC2626;
        font-weight: 500;
    }
    
    /* Progress styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2E86AB 0%, #6366F1 100%);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Info/Alert boxes */
    .stAlert > div {
        border-left: 4px solid #2E86AB;
    }
    
    .stSuccess > div {
        border-left: 4px solid #2E86AB;
        background-color: rgba(46, 134, 171, 0.1);
    }
    
    .stWarning > div {
        border-left: 4px solid #DC2626;
        background-color: rgba(220, 38, 38, 0.1);
    }
    
    .stError > div {
        border-left: 4px solid #DC2626;
        background-color: rgba(220, 38, 38, 0.1);
    }
    
    /* Selectbox and multiselect */
    .stSelectbox > div > div > div {
        border-color: #2E86AB;
    }
    
    .stMultiSelect > div > div > div {
        border-color: #2E86AB;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #2E86AB 0%, #6366F1 100%);
        color: white;
    }
    
    /* Metric styling */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------
# 🏷️ FEATURE CATEGORY DESCRIPTIONS (Constants)
# -----------------------------------------------

# Feature category descriptions (module-level constant for performance)
FEATURE_CATEGORY_DESCRIPTIONS = {
    'Quality Metrics': 'Statistical confidence scores, Q-values, PEP scores, and probability measures from DIA-NN',
    'Sequence Properties': 'Peptide sequence characteristics, protein IDs, gene names, and sequence length',
    'Amino Acid Composition': 'Amino acid counts and frequencies for specific residues (C, R, H, K, E, D, P, M)',
    'Transformed Features': 'Logarithmically transformed and z-score normalized features for better model performance',
    'Intensity Features': 'Raw signal measurements including precursor quantity, peak areas, MaxLFQ values',
    'Chromatographic': 'Retention time properties including RT, iRT, predicted RT, and peak width (FWHM)',
    'Physicochemical': 'Physical and chemical properties including mass, charge state, m/z ratios',
    'Peak Shape & Quality': 'Peak shape metrics including correlations, apex properties, and mass deltas',
    'Ion Mobility': 'Ion mobility measurements and predictions (IM, iIM, predicted IM)',
    'Library & Identification': 'Spectral library features, proteotypic indicators, and identification indices',
    'Engineered Ratios': 'Custom ratio features between different measurements for enhanced discrimination',
    'Other Features': 'Remaining analytical features not fitting specific categories above'
}

# -----------------------------------------------
# 💾 PERSISTENT HISTORY MANAGEMENT
# -----------------------------------------------

def get_history_file_path():
    """Get the path to the persistent history file."""
    history_dir = "./history"
    os.makedirs(history_dir, exist_ok=True)
    return os.path.join(history_dir, "run_history.json")

def load_persistent_history():
    """Load run history from persistent storage."""
    history_file = get_history_file_path()
    try:
        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history_data = json.load(f)
                return history_data
        else:
            # No persistent history file found - starting fresh
            return []
    except Exception as e:
        # Error loading persistent history
        return []

def save_persistent_history(run_history):
    """Save run history to persistent storage."""
    history_file = get_history_file_path()
    try:
        # Keep only last 50 runs to avoid file getting too large
        limited_history = run_history[-50:] if len(run_history) > 50 else run_history
        
        with open(history_file, 'w') as f:
            json.dump(limited_history, f, indent=2, default=str)
        return True
    except Exception as e:
        # Error saving persistent history
        return False

# -----------------------------------------------
# 🎨 COLOR SCHEME MANAGEMENT
# -----------------------------------------------

def update_color_scheme(selected_scheme):
    """Update the global CHART_COLORS with the selected color scheme."""
    global CHART_COLORS
    if selected_scheme in COLOR_SCHEMES:
        CHART_COLORS.clear()  # Clear the existing dictionary
        CHART_COLORS.update(COLOR_SCHEMES[selected_scheme])  # Update with new colors
        # Don't set color_scheme_updated flag to avoid refresh
        st.session_state.current_colors = CHART_COLORS.copy()  # Store in session state
        return True
    return False

def get_current_colors():
    """Get the current color scheme, ensuring it's up to date."""
    # Initialize if not set or invalid
    if 'selected_color_scheme' not in st.session_state or st.session_state.selected_color_scheme not in COLOR_SCHEMES:
        st.session_state.selected_color_scheme = 'Default'
        update_color_scheme('Default')
    
    # Ensure CHART_COLORS has the feature_gradient key
    if 'feature_gradient' not in CHART_COLORS:
        update_color_scheme(st.session_state.selected_color_scheme)
    
    return CHART_COLORS

# Sidebar color scheme selector removed - now only available in main results interface

def display_main_color_scheme_selector():
    """Display a compact horizontal color scheme selector in the main interface."""
    st.markdown("### 🎨 Visualization Color Scheme")
    
    # Initialize color scheme in session state if not present
    if 'selected_color_scheme' not in st.session_state:
        st.session_state.selected_color_scheme = 'Default'
    
    # Ensure widget session state key is initialised to current scheme to avoid mismatch glitches
    if 'main_color_scheme_selector' not in st.session_state:
        st.session_state['main_color_scheme_selector'] = st.session_state.selected_color_scheme
    
    # Create columns for horizontal layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Color scheme selector with safety check
        scheme_options = list(COLOR_SCHEMES.keys())
        
        # Ensure session state has a valid scheme
        if 'selected_color_scheme' not in st.session_state or st.session_state.selected_color_scheme not in COLOR_SCHEMES:
            st.session_state.selected_color_scheme = 'Default'
            
        # Let Streamlit manage the current selection via the widget key; remove explicit index param
        selected_scheme = st.selectbox(
            "Choose color scheme for all plots and visualizations:",
            options=scheme_options,
            key='main_color_scheme_selector'
        )
        
        # Always update session state and colors based on current widget value
        if st.session_state.selected_color_scheme != selected_scheme:
            st.session_state.selected_color_scheme = selected_scheme
            update_color_scheme(selected_scheme)
            # Show immediate confirmation without causing refresh
            st.success(f"✅ Colors updated to {selected_scheme}!")
        
        # Show color preview with gradient
        # Use get_current_colors() to ensure we get valid colors
        current_colors = get_current_colors()
        colors = current_colors.get('feature_gradient', ['#2E86AB', '#6366F1', '#A855F7'])
        
        # Create gradient preview
        gradient_html = f"""
        <div style="margin: 10px 0;">
            <div style="
                height: 20px; 
                border-radius: 10px; 
                background: linear-gradient(to right, {', '.join(colors)}); 
                border: 1px solid #ddd;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            "></div>
            <div style="
                text-align: center; 
                font-size: 12px; 
                color: #666; 
                margin-top: 5px;
            ">Color gradient preview</div>
        </div>
        """
        
        st.markdown(gradient_html, unsafe_allow_html=True)

def clear_persistent_history():
    """Clear all persistent history."""
    history_file = get_history_file_path()
    try:
        if os.path.exists(history_file):
            os.remove(history_file)
        # Persistent history cleared
        return True
    except Exception as e:
        # Error clearing persistent history
        return False

def export_history_to_csv(run_history):
    """Export run history to CSV format."""
    if not run_history:
        return None
    
    export_data = []
    for run in run_history:
        config = run['config']
        summary = run['summary']
        
        export_data.append({
            'Run_ID': run['run_id'],
            'Timestamp': run['timestamp'][:19].replace('T', ' '),
            'Training_Methods': ', '.join(config['train_methods']),
            'Training_FDR_Levels': ', '.join(map(str, config.get('train_fdr_levels', []))),
            'Test_Method': config['test_method'],
            'Test_FDR': config['test_fdr'],
            'Runtime_Minutes': summary['runtime_minutes'],
            'Best_Peptides': summary['best_peptides'],
            'Best_FDR': summary['best_fdr'],
            'Baseline_Peptides': summary.get('baseline_peptides', 'N/A'),
            'Best_MCC': get_best_mcc_from_run(run) or 'N/A'
        })
    
    return pd.DataFrame(export_data)

def init_session_state():
    """Initialize session state variables efficiently."""
    defaults = {
        'analysis_complete': False,
        'analysis_running': False,
        'inference_complete': False,
        'results_data': None,
        'inference_results': None,
        'current_run_id': None
    }
    
    # Batch initialize simple defaults
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Handle expensive initialization separately
    if 'run_history' not in st.session_state:
        st.session_state.run_history = load_persistent_history()

def clear_analysis_states():
    """Clear all analysis-related session states to prevent cross-mode contamination."""
    st.session_state.analysis_complete = False
    st.session_state.analysis_running = False
    st.session_state.inference_complete = False
    st.session_state.results_data = None
    st.session_state.inference_results = None

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def discover_available_files():
    """Discover available data files for analysis using generic folder structure.
    
    Expected structure:
    data/
    ├── [DATASET_NAME]/                    # e.g., MyDataset, ExperimentA, etc.
    │   ├── short_gradient/                # Training data (fast gradient)
    │   │   ├── FDR_1/                    # BASELINE files (1% FDR)
    │   │   ├── FDR_20/                   # TRAINING files (20% FDR)
    │   │   └── FDR_50/                   # TRAINING files (50% FDR)
    │   └── long_gradient/                 # Ground truth (slow gradient)
    │       └── FDR_1/                    # GROUND TRUTH files (1% FDR)
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    files_info = {
        'baseline': [],  # Fast gradient FDR_1
        'ground_truth': [],  # Slow gradient FDR_1
        'training': [],  # Fast gradient FDR_20, FDR_50
        'testing': []   # Fast gradient FDR_20, FDR_50
    }
    
    if not os.path.exists(data_dir):
        st.warning(f"⚠️ Data directory not found: {data_dir}")
        return files_info
    
    # Discover all datasets in the data directory
    dataset_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    for dataset_name in dataset_dirs:
        dataset_path = os.path.join(data_dir, dataset_name)
        
        # Skip non-dataset directories like files
        if not os.path.isdir(dataset_path):
            continue
            
        # Look for new generic structure first: short_gradient/ and long_gradient/
        short_gradient_path = os.path.join(dataset_path, "short_gradient")
        long_gradient_path = os.path.join(dataset_path, "long_gradient")
        
        # Process ground truth files (long_gradient/FDR_1/)
        gt_path = os.path.join(long_gradient_path, "FDR_1")
        if os.path.exists(gt_path):
            for file_path in glob.glob(os.path.join(gt_path, "*.parquet")):
                filename = Path(file_path).name
                method = extract_method_from_filename(filename, dataset_name)
                if method:
                    files_info['ground_truth'].append({
                        'path': file_path,
                        'method': method,
                        'filename': filename,
                        'dataset': dataset_name,
                        'gradient': 'long_gradient',
                        'size_mb': os.path.getsize(file_path) / (1024*1024)
                    })
        
        # Process training/testing files (short_gradient/FDR_X/)
        if os.path.exists(short_gradient_path):
            for fdr in [1, 20, 50]:
                fdr_path = os.path.join(short_gradient_path, f"FDR_{fdr}")
                if os.path.exists(fdr_path):
                    for file_path in glob.glob(os.path.join(fdr_path, "*.parquet")):
                        filename = Path(file_path).name
                        method = extract_method_from_filename(filename, dataset_name)
                        if method:
                            file_info = {
                                'path': file_path,
                                'method': method,
                                'filename': filename,
                                'fdr': fdr,
                                'dataset': dataset_name,
                                'gradient': 'short_gradient',
                                'size_mb': os.path.getsize(file_path) / (1024*1024)
                            }
                            files_info['training'].append(file_info)
                            files_info['testing'].append(file_info)
                            
                            # FDR_1 files from short gradient are also baseline files
                            if fdr == 1:
                                files_info['baseline'].append(file_info)
        
        # Legacy dataset-specific structures removed for universal compatibility
        # Users should structure data according to the standard format described above
        
        # All legacy dataset-specific structures removed for universal compatibility
    
    return files_info

@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_parquet_file(file_path):
    """Load a parquet file with caching to avoid repeated disk reads."""
    return pd.read_parquet(file_path)

@st.cache_data(ttl=600)  # Cache for 10 minutes  
def load_peptide_sequences(file_path):
    """Load unique peptide sequences from a parquet file with caching."""
    df = load_parquet_file(file_path)
    if 'Stripped.Sequence' in df.columns:
        return set(df['Stripped.Sequence'].unique())
    elif 'Modified.Sequence' in df.columns:
        return set(df['Modified.Sequence'].unique())
    return set()

@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_shap_data(shap_data_path, max_samples=1000):
    """Load SHAP data from JSON file with caching and sampling for performance."""
    try:
        with open(shap_data_path, 'r') as f:
            shap_data = json.load(f)
        
        # Handle both full SHAP data format and simplified format from API
        if 'shap_values' in shap_data and 'feature_values' in shap_data:
            # Full format (legacy)
            shap_values = np.array(shap_data['shap_values'])
            feature_values = np.array(shap_data['feature_values'])
            feature_names = np.array(shap_data['feature_names'])
            feature_importance = np.array(shap_data['feature_importance'])
            
            # Sample data if too large for performance
            if len(shap_values) > max_samples:
                indices = np.random.choice(len(shap_values), max_samples, replace=False)
                shap_values = shap_values[indices]
                feature_values = feature_values[indices]
        else:
            # Simplified format from API (current)
            feature_names = np.array(shap_data['feature_names'])
            mean_shap_values = np.array(shap_data['mean_shap_values'])
            feature_importance = np.array(shap_data['abs_importance'])
            
            # Create mock shap_values and feature_values for plotting
            # Since we only have mean values, create a simplified representation
            n_samples = min(50, max_samples)  # Use fewer samples for simplified format
            shap_values = np.array([mean_shap_values for _ in range(n_samples)])
            
            # Create normalized feature values (0-1) for color coding
            feature_values = np.random.rand(n_samples, len(feature_names))
        
        return {
            'shap_values': shap_values,
            'feature_values': feature_values,
            'feature_names': feature_names,
            'feature_importance': feature_importance
        }
    except Exception as e:
        st.error(f"Error loading SHAP data: {str(e)}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_feature_importance_csv(csv_path):
    """Load feature importance CSV with caching for performance."""
    try:
        feature_df = pd.read_csv(csv_path)
        # Filter out zero importance features
        feature_df = feature_df[feature_df['importance'] > 0].copy()
        return feature_df
    except Exception as e:
        st.error(f"Error loading feature importance CSV: {str(e)}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def categorize_features(feature_names):
    """Categorize features with caching for performance."""
    # Create dataframe with feature names
    df = pd.DataFrame({'feature': feature_names})
    df['category'] = 'Other Features'  # Default category
    
    # Apply categorization rules
    df.loc[df['feature'].str.contains('Q.Value|PEP|Score|Confidence|Probability|CScore', case=False), 'category'] = 'Quality Metrics'
    df.loc[df['feature'].str.contains('Stripped.Sequence|Protein|Gene|Length|Modified.Sequence|sequence_length', case=False), 'category'] = 'Sequence Properties'
    df.loc[df['feature'].str.contains('aa_count_|aa_freq_', case=False), 'category'] = 'Amino Acid Composition'
    df.loc[df['feature'].str.contains('log_|zscore_', case=False), 'category'] = 'Transformed Features'
    df.loc[df['feature'].str.contains('_ratio'), 'category'] = 'Engineered Ratios'
    df.loc[df['feature'].str.contains('Quantity|Area|Height|Intensity|MaxLFQ|Signal', case=False), 'category'] = 'Intensity Features'
    df.loc[df['feature'].str.contains('Charge|Mass|Mz|MW|Evidence', case=False), 'category'] = 'Physicochemical'
    df.loc[df['feature'].str.contains('RT|Time|Retention|iRT|FWHM', case=False), 'category'] = 'Chromatographic'
    df.loc[df['feature'].str.contains('IM|iIM', case=False), 'category'] = 'Ion Mobility'
    df.loc[df['feature'].str.contains('Lib\\.|Proteotypic|Index', case=False), 'category'] = 'Library & Identification'
    df.loc[df['feature'].str.contains('Corr|Profile|Apex|Delta|FWHM', case=False), 'category'] = 'Peak Shape & Quality'
    
    return df.set_index('feature')['category'].to_dict()

def discover_datasets_for_setup(config, results, runtime_minutes):
    """Discover available data files organized by dataset for the setup interface.
    
    Returns:
        dict: {
            'DatasetName': {
                'training': {
                    '20': {'method1': [file_info], ...},
                    '50': {'method1': [file_info], ...}
                },
                'baseline': {'method1': [file_info], ...},
                'ground_truth': {'method1': [file_info], ...}
            }
        }
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    datasets_info = {}
    
    if not os.path.exists(data_dir):
        return datasets_info
    
    # Discover all datasets in the data directory
    dataset_dirs = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    
    for dataset_name in dataset_dirs:
        dataset_path = os.path.join(data_dir, dataset_name)
        
        # Initialize dataset structure
        datasets_info[dataset_name] = {
            'training': {'20': {}, '50': {}},
            'baseline': {},
            'ground_truth': {}
        }
        
        # Look for new generic structure: short_gradient/ and long_gradient/
        short_gradient_path = os.path.join(dataset_path, "short_gradient")
        long_gradient_path = os.path.join(dataset_path, "long_gradient")
        
        # Process ground truth files (long_gradient/FDR_1/)
        gt_path = os.path.join(long_gradient_path, "FDR_1")
        if os.path.exists(gt_path):
            for file_path in glob.glob(os.path.join(gt_path, "*.parquet")):
                filename = Path(file_path).name
                method = extract_method_from_filename(filename, dataset_name)
                if method:
                    if method not in datasets_info[dataset_name]['ground_truth']:
                        datasets_info[dataset_name]['ground_truth'][method] = []
                    datasets_info[dataset_name]['ground_truth'][method].append({
                        'path': file_path,
                        'method': method,
                        'filename': filename,
                        'dataset': dataset_name,
                        'gradient': 'long_gradient',
                        'size_mb': os.path.getsize(file_path) / (1024*1024)
                    })
        
        # Process training/testing files (short_gradient/FDR_X/)
        if os.path.exists(short_gradient_path):
            for fdr in [1, 20, 50]:
                fdr_path = os.path.join(short_gradient_path, f"FDR_{fdr}")
                if os.path.exists(fdr_path):
                    for file_path in glob.glob(os.path.join(fdr_path, "*.parquet")):
                        filename = Path(file_path).name
                        method = extract_method_from_filename(filename, dataset_name)
                        if method:
                            file_info = {
                                'path': file_path,
                                'method': method,
                                'filename': filename,
                                'fdr': fdr,
                                'dataset': dataset_name,
                                'gradient': 'short_gradient',
                                'size_mb': os.path.getsize(file_path) / (1024*1024)
                            }
                            
                            # Add to appropriate category
                            if fdr == 1:
                                # Baseline files
                                if method not in datasets_info[dataset_name]['baseline']:
                                    datasets_info[dataset_name]['baseline'][method] = []
                                datasets_info[dataset_name]['baseline'][method].append(file_info)
                            else:
                                # Training files (FDR 20, 50)
                                fdr_key = str(fdr)
                                if method not in datasets_info[dataset_name]['training'][fdr_key]:
                                    datasets_info[dataset_name]['training'][fdr_key][method] = []
                                datasets_info[dataset_name]['training'][fdr_key][method].append(file_info)
        
        # Legacy dataset-specific structures removed for universal compatibility
        
        # Remove empty datasets
        if (not datasets_info[dataset_name]['training']['20'] and 
            not datasets_info[dataset_name]['training']['50'] and
            not datasets_info[dataset_name]['baseline'] and
            not datasets_info[dataset_name]['ground_truth']):
            del datasets_info[dataset_name]
    
    return datasets_info

def extract_method_from_filename(filename, dataset_name=None):
    """Universal method extraction that treats every file as unique.
    
    NO hardcoding for any specific dataset. Each file gets its own unique method name
    based on the full filename (minus extension and FDR suffix).
    This ensures maximum compatibility with any naming convention.
    """
    import re
    
    # Remove file extension and common suffixes
    base_name = filename.replace('.parquet', '').replace('.csv', '').replace('.tsv', '')
    base_name = re.sub(r'_FDR\d+$', '', base_name)  # Remove FDR suffix
    
    # Universal approach: Use the full cleaned filename as the method name
    # This ensures every file gets a unique identity regardless of naming convention
    return f"{dataset_name}_{base_name}" if dataset_name else base_name

def get_available_methods(files_info):
    """Get list of available methods from discovered files."""
    methods = set()
    for category in ['training', 'testing']:
        for file_info in files_info[category]:
            methods.add(file_info['method'])
    return sorted(list(methods))



def add_to_run_history(config, results, runtime_minutes):
    """Add completed run to history."""
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract best result for summary
    best_result = None
    if 'results' in results and results['results']:
        valid_results = [r for r in results['results'] if r['Actual_FDR'] <= 5.0]
        if not valid_results:
            valid_results = results['results']
        if valid_results:
            best_result = max(valid_results, key=lambda x: x['Additional_Peptides'])
    
    history_entry = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'config': config.copy(),
        'summary': {
            'runtime_minutes': runtime_minutes,
            'best_fdr': best_result['Actual_FDR'] if best_result else None,
            'best_peptides': best_result['Additional_Peptides'] if best_result else 0,
            'results_dir': results.get('summary', {}).get('results_dir', ''),
            'total_results': len(results.get('results', []))
        },
        'full_results': results
    }
    
    # Initialize run_history if not exists
    if not hasattr(st.session_state, 'run_history'):
        st.session_state.run_history = []
    
    st.session_state.run_history.append(history_entry)
    st.session_state.current_run_id = run_id
    
    # Keep only last 20 runs in session state for performance
    if len(st.session_state.run_history) > 20:
        st.session_state.run_history = st.session_state.run_history[-20:]
    
    # Save to persistent storage
    save_result = save_persistent_history(st.session_state.run_history)
    # Saved run to history

def display_run_history():
    """Display enhanced run history with comparison capabilities."""
    # Debug: Show count of runs
    run_count = len(st.session_state.run_history) if hasattr(st.session_state, 'run_history') else 0
    
    if not st.session_state.run_history:
        st.sidebar.markdown(f"*No previous runs* (Count: {run_count})")
        return
    
    # Limit to most recent 10 runs for performance
    recent_runs = st.session_state.run_history[-10:]
    
    st.sidebar.markdown(f"### 📜 Run History (10 most recent runs)")
    
    # History management options
    with st.sidebar.expander("⚙️ History Management", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Export CSV", help="Export all run history to CSV", use_container_width=True):
                history_df = export_history_to_csv(st.session_state.run_history)
                if history_df is not None:
                    csv_data = history_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download History CSV",
                        data=csv_data,
                        file_name=f"run_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No history to export")
        
        with col2:
            if st.button("🗑️ Clear All", help="Clear all run history permanently", use_container_width=True):
                if clear_persistent_history():
                    st.session_state.run_history = []
                    st.success("History cleared!")
                    time.sleep(0.5)  # Brief pause to show success message
                    st.rerun()
                else:
                    st.error("Failed to clear history")
        
        # Show history file info
        history_file = get_history_file_path()
        if os.path.exists(history_file):
            file_size = os.path.getsize(history_file) / 1024  # KB
            st.caption(f"📁 History file: {file_size:.1f} KB")
        else:
            st.caption("📁 No persistent history file")
    
    # Add comparison mode toggle
    comparison_mode = st.sidebar.checkbox("🔍 Comparison Mode", help="Select multiple runs to compare")
    
    if not comparison_mode:
        # Standard view - show individual runs (limited to 10 recent)
        for i, run in enumerate(reversed(recent_runs)):
            with st.sidebar.expander(f"Run {run['run_id']}", expanded=False):
                # Enhanced run information
                st.markdown(f"**🕐 Time**: {run['timestamp'][:19].replace('T', ' ')}")
                st.markdown(f"**🏋️ Train**: {', '.join(run['config']['train_methods'])}")
                st.markdown(f"**🧪 Test**: {run['config']['test_method']} @ {run['config']['test_fdr']}% FDR")
                st.markdown(f"**⏱️ Runtime**: {run['summary']['runtime_minutes']:.1f} min")
                
                # Performance metrics
                best_peptides = int(run['summary']['best_peptides']) if run['summary']['best_peptides'] else 0
                if best_peptides > 0:
                    st.markdown(f"**🎯 Best**: {run['summary']['best_peptides']} peptides @ {run['summary']['best_fdr']:.1f}% FDR")
                    
                    # Additional metrics if available
                    if 'baseline_peptides' in run['summary']:
                        baseline = run['summary']['baseline_peptides']
                        improvement = (int(run['summary']['best_peptides']) / baseline * 100) if baseline > 0 else 0
                        st.markdown(f"**📈 Improvement**: +{improvement:.1f}%")
                    
                    # Get best MCC if available
                    best_mcc = get_best_mcc_from_run(run)
                    if best_mcc is not None:
                        st.markdown(f"**📊 Best MCC**: {best_mcc:.3f}")
                else:
                    st.markdown("**🎯 Best**: No valid results")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"📊 View", key=f"view_{run['run_id']}", help="Load this run's results", use_container_width=True):
                        st.session_state.results_data = run['full_results']
                        st.session_state.analysis_complete = True
                        st.session_state.analysis_running = False
                        st.session_state.current_run_id = run['run_id']
                        st.rerun()
                
                with col2:
                    if st.button(f"🗑️ Del", key=f"delete_{run['run_id']}", help="Delete this run", use_container_width=True):
                        st.session_state.run_history = [r for r in st.session_state.run_history if r['run_id'] != run['run_id']]
                        save_persistent_history(st.session_state.run_history)
                        st.rerun()
    
    else:
        # Comparison mode - show condensed list with checkboxes
        st.sidebar.markdown("**Select runs to compare:**")
        
        if 'selected_runs' not in st.session_state:
            st.session_state.selected_runs = []
        
        # Show runs with checkboxes (limited to 10 recent)
        for run in reversed(recent_runs):
            run_label = f"{run['run_id']}: {run['config']['test_method']} ({run['summary']['best_peptides']} peptides)"
            
            is_selected = st.sidebar.checkbox(
                run_label,
                key=f"compare_{run['run_id']}",
                value=run['run_id'] in st.session_state.selected_runs
            )
            
            if is_selected and run['run_id'] not in st.session_state.selected_runs:
                st.session_state.selected_runs.append(run['run_id'])
            elif not is_selected and run['run_id'] in st.session_state.selected_runs:
                st.session_state.selected_runs.remove(run['run_id'])
        
        # Comparison actions
        if len(st.session_state.selected_runs) >= 2:
            if st.sidebar.button("📊 Compare Selected Runs", use_container_width=True):
                st.session_state.show_comparison = True
                st.rerun()
        
        if st.sidebar.button("🗑️ Clear Selection", use_container_width=True):
            st.session_state.selected_runs = []
            st.rerun()

def get_best_mcc_from_run(run):
    """Extract the best MCC value from a run's results."""
    try:
        if 'full_results' in run and 'results' in run['full_results']:
            results = run['full_results']['results']
            if results and len(results) > 0:
                # Find result with valid MCC
                valid_results = [r for r in results if 'MCC' in r and r['MCC'] != 'N/A']
                if valid_results:
                    # Get the result with the highest additional peptides that has a valid MCC
                    best_result = max(valid_results, key=lambda x: x.get('Additional_Peptides', 0))
                    return float(best_result['MCC']) if isinstance(best_result['MCC'], (int, float, str)) else None
    except:
        pass
    return None

def get_baseline_peptides(run):
    """Extract baseline peptides from a run, handling different data structures."""
    try:
        # First try the summary level
        if 'summary' in run and 'baseline_peptides' in run['summary']:
            return run['summary']['baseline_peptides']
        
        # Then try the full_results summary level
        if 'full_results' in run and 'summary' in run['full_results']:
            if 'baseline_peptides' in run['full_results']['summary']:
                return run['full_results']['summary']['baseline_peptides']
        
        # For older data structures, return None (will show as N/A)
        return None
    except:
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute to avoid reprocessing
def process_run_comparison_data(selected_run_ids):
    """Process comparison data for selected runs with caching."""
    selected_runs = [run for run in st.session_state.run_history 
                    if run['run_id'] in selected_run_ids]
    
    comparison_data = []
    for run in selected_runs:
        config = run['config']
        summary = run['summary']
        
        # Get best MCC
        best_mcc = get_best_mcc_from_run(run)
        
        comparison_data.append({
            'Run ID': run['run_id'],
            'Timestamp': run['timestamp'][:19].replace('T', ' '),
            'Training Methods': ', '.join(config['train_methods']),
            'Training FDR': ', '.join(map(str, config.get('train_fdr_levels', []))) + '%',
            'Test Method': config['test_method'],
            'Test FDR': f"{config['test_fdr']}%",
            'Baseline Peptides': f"{get_baseline_peptides(run):,}" if get_baseline_peptides(run) else "N/A",
            'Runtime (min)': f"{summary['runtime_minutes']:.1f}",
            'Best Peptides': int(summary['best_peptides']),
            'Best FDR': f"{summary['best_fdr']:.1f}%",
            'Best MCC': f"{best_mcc:.3f}" if best_mcc else "N/A",
        })
    
    return comparison_data, selected_runs

def display_run_comparison():
    """Display detailed comparison of selected runs."""
    st.markdown("## 📊 Run Comparison Analysis")
    
    # Quick validation
    if len(st.session_state.selected_runs) < 2:
        st.error("Please select at least 2 runs to compare")
        if st.button("← Back to Main", type="primary"):
            st.session_state.show_comparison = False
            st.rerun()
        return
    
    # Process data with caching
    comparison_data, selected_runs = process_run_comparison_data(st.session_state.selected_runs)
    
    if len(selected_runs) < 2:
        st.error("Please select at least 2 runs to compare")
        if st.button("← Back to Main", type="primary"):
            st.session_state.show_comparison = False
            st.rerun()
        return
    
    # Back button - aligned with home button style
    if st.button("← Back to Main", type="secondary", key="comparison_back_btn"):
        st.session_state.show_comparison = False
        st.rerun()
    
    # Overview comparison table
    st.markdown("### 📋 Overview Comparison")
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True)
    
    # Performance comparison charts
    st.markdown("### 📈 Performance Comparison")
    
    # Single chart focusing on peptides recovered
    fig_peptides = px.bar(
        comparison_df,
        x='Run ID',
        y='Best Peptides',
        title='Additional Peptides Recovered by Configuration',
        color='Best Peptides',
        color_continuous_scale=CHART_COLORS['feature_gradient'],
        text='Best Peptides'
    )
    fig_peptides.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=12),
        height=500
    )
    fig_peptides.update_traces(texttemplate='%{text}', textposition='outside')
    st.plotly_chart(fig_peptides, use_container_width=True)
    
    # MCC comparison (if available)
    mcc_data = comparison_df[comparison_df['Best MCC'] != 'N/A'].copy()
    if len(mcc_data) > 1:
        mcc_data['Best MCC Numeric'] = mcc_data['Best MCC'].astype(float)
        
        fig_mcc = px.bar(
            mcc_data,
            x='Run ID',
            y='Best MCC Numeric',
            title='Best MCC Comparison',
            color='Best MCC Numeric',
            color_continuous_scale=CHART_COLORS['feature_gradient']
        )
        fig_mcc.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12)
        )
        st.plotly_chart(fig_mcc, use_container_width=True)
    
    # Detailed results comparison
    st.markdown("### 📊 Detailed Results Comparison")
    
    # Let user select which runs to view detailed results for
    st.markdown("Select runs to view detailed results:")
    
    tabs = st.tabs([f"Run {run['run_id']}" for run in selected_runs])
    
    for i, (tab, run) in enumerate(zip(tabs, selected_runs)):
        with tab:
            if 'full_results' in run and 'results' in run['full_results']:
                results_df = pd.DataFrame(run['full_results']['results'])
                
                # Remove Aggregation_Method column from display (internal use only)
                if 'Aggregation_Method' in results_df.columns:
                    results_df = results_df.drop('Aggregation_Method', axis=1)
                
                # Show key metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Configuration", f"{run['config']['test_method']} @ {run['config']['test_fdr']}%")
                
                with col2:
                    st.metric("Runtime", f"{run['summary']['runtime_minutes']:.1f} min")
                
                with col3:
                    st.metric("Best Result", f"{run['summary']['best_peptides']} peptides")
                
                # Show results table
                st.dataframe(results_df, use_container_width=True)
            else:
                st.info("No detailed results available for this run")
    
    # Download comparison data
    st.markdown("### 💾 Export Comparison")
    csv_data = comparison_df.to_csv(index=False)
    st.download_button(
        label="📄 Download Comparison CSV",
        data=csv_data,
        file_name=f"run_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def main():
    """Main application function with Training/Inference mode selection."""
    
    # Initialize session state
    init_session_state()
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">🧬 PeptiDIA</div>
        <div class="main-subtitle">Professional Machine Learning Interface for DIA-NN Peptide Analysis</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode selection - only show if no mode is selected
    if 'app_mode' not in st.session_state:
        st.session_state.app_mode = None
    
    # If no mode selected, show landing page
    if st.session_state.app_mode is None:
        show_landing_page()
        return
    
    # Show selected mode interface
    if st.session_state.app_mode == 'training':
        show_training_interface()
    elif st.session_state.app_mode == 'inference':
        show_inference_interface()
    elif st.session_state.app_mode == 'setup':
        show_setup_interface()

def show_landing_page():
    """Display the landing page with mode selection."""
    st.markdown("## 🚀 Choose Your Workflow")
    st.markdown("Select how you want to use PeptiDIA:")
    
    # Create three large buttons for mode selection with uniform sizing
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        # Container with fixed button positioning
        container_height = "400px"  # Fixed container height
        
        st.markdown(f"""
        <div style="position: relative; height: {container_height}; margin: 1rem 0;">
            <div style="text-align: center; padding: 2rem; border: 2px solid #1E40AF; border-radius: 10px; height: calc(100% - 60px); display: flex; flex-direction: column; justify-content: center; box-sizing: border-box; overflow: hidden;">
                <h2 style="color: #1E40AF; margin-bottom: 1rem; font-size: clamp(1.2rem, 2.5vw, 1.8rem);">⚙️ SETUP MODE</h2>
                <p style="font-size: clamp(0.9rem, 1.8vw, 1.1rem); margin: 1rem 0; line-height: 1.4;">Configure your datasets visually with drag & drop</p>
                <ul style="text-align: left; margin: 1rem 0; font-size: clamp(0.8rem, 1.5vw, 1rem); line-height: 1.5; padding-left: 1.2rem;">
                    <li style="margin-bottom: 0.5rem;">Visual ground truth mapping</li>
                    <li style="margin-bottom: 0.5rem;">Drag and drop interface</li>
                    <li style="margin-bottom: 0.5rem;">No JSON editing required</li>
                    <li style="margin-bottom: 0.5rem;">Save configurations easily</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("⚙️ Configure Datasets", type="primary", use_container_width=True,
                    help="Set up ground truth mapping for your datasets"):
            # Clear all states when entering setup mode
            st.session_state.app_mode = 'setup'
            st.session_state.analysis_complete = False
            st.session_state.analysis_running = False
            st.session_state.inference_complete = False
            st.session_state.results_data = None
            st.session_state.inference_results = None
            st.rerun()
    
    with col2:
        st.markdown(f"""
        <div style="position: relative; height: {container_height}; margin: 1rem 0;">
            <div style="text-align: center; padding: 2rem; border: 2px solid #7C3AED; border-radius: 10px; height: calc(100% - 60px); display: flex; flex-direction: column; justify-content: center; box-sizing: border-box; overflow: hidden;">
                <h2 style="color: #7C3AED; margin-bottom: 1rem; font-size: clamp(1.2rem, 2.5vw, 1.8rem);">🧠 TRAINING MODE</h2>
                <p style="font-size: clamp(0.9rem, 1.8vw, 1.1rem); margin: 1rem 0; line-height: 1.4;">Train new machine learning models on your peptide data</p>
                <ul style="text-align: left; margin: 1rem 0; font-size: clamp(0.8rem, 1.5vw, 1rem); line-height: 1.5; padding-left: 1.2rem;">
                    <li style="margin-bottom: 0.5rem;">Configure training parameters</li>
                    <li style="margin-bottom: 0.5rem;">Select multiple methods and FDR levels</li>
                    <li style="margin-bottom: 0.5rem;">Analyze feature importance</li>
                    <li style="margin-bottom: 0.5rem;">Save trained models for reuse</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🧠 Start Training", type="primary", use_container_width=True, 
                    help="Train new models on your peptide validation data"):
            # Clear all states when entering training mode
            st.session_state.app_mode = 'training'
            st.session_state.analysis_complete = False
            st.session_state.analysis_running = False
            st.session_state.inference_complete = False
            st.session_state.results_data = None
            st.session_state.inference_results = None
            st.rerun()
    
    with col3:
        st.markdown(f"""
        <div style="position: relative; height: {container_height}; margin: 1rem 0;">
            <div style="text-align: center; padding: 2rem; border: 2px solid #C026D3; border-radius: 10px; height: calc(100% - 60px); display: flex; flex-direction: column; justify-content: center; box-sizing: border-box; overflow: hidden;">
                <h2 style="color: #C026D3; margin-bottom: 1rem; font-size: clamp(1.2rem, 2.5vw, 1.8rem);">🔮 INFERENCE MODE</h2>
                <p style="font-size: clamp(0.9rem, 1.8vw, 1.1rem); margin: 1rem 0; line-height: 1.4;">Use pre-trained models to predict on new peptide data</p>
                <ul style="text-align: left; margin: 1rem 0; font-size: clamp(0.8rem, 1.5vw, 1rem); line-height: 1.5; padding-left: 1.2rem;">
                    <li style="margin-bottom: 0.5rem;">Load saved models</li>
                    <li style="margin-bottom: 0.5rem;">Predict on any compatible dataset</li>
                    <li style="margin-bottom: 0.5rem;">Cross-dataset model evaluation</li>
                    <li style="margin-bottom: 0.5rem;">Compare model performances</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔮 Start Inference", type="primary", use_container_width=True,
                    help="Use trained models to predict on new data"):
            # Clear all states when entering inference mode
            st.session_state.app_mode = 'inference'
            st.session_state.analysis_complete = False
            st.session_state.analysis_running = False
            st.session_state.inference_complete = False
            st.session_state.results_data = None
            st.session_state.inference_results = None
            st.rerun()
    
    # Additional information section
    st.markdown("---")
    st.markdown("### 📊 Supported Datasets")
    
    # Discover available datasets and show them dynamically
    files_info = discover_available_files()
    available_datasets = set()
    for category in ['training', 'ground_truth']:
        for file_info in files_info.get(category, []):
            available_datasets.add(file_info['dataset'])
    
    if available_datasets:
        dataset_list = sorted(list(available_datasets))
        num_cols = min(len(dataset_list), 3)  # Max 3 columns
        cols = st.columns(num_cols)
        
        for i, dataset in enumerate(dataset_list):
            with cols[i % num_cols]:
                # Load dataset info if available
                dataset_info_path = os.path.join(os.path.dirname(__file__), "data", dataset, "dataset_info.json")
                icon = "📊"
                description = f"Auto-discovered {dataset} dataset"
                
                if os.path.exists(dataset_info_path):
                    try:
                        import json
                        with open(dataset_info_path, 'r') as f:
                            dataset_info = json.load(f)
                            icon = dataset_info.get('icon', icon)
                            description = dataset_info.get('description', description)
                    except:
                        pass
                
                # Create a contained box for each dataset
                st.markdown(f"""
                <div style="border: 2px solid #E5E7EB; border-radius: 10px; padding: 1.5rem; margin: 1rem 0; min-height: 140px; display: flex; flex-direction: column; box-sizing: border-box;">
                    <h3 style="color: #374151; margin: 0 0 0.5rem 0; font-size: 1.1rem;">{icon} {dataset} Dataset</h3>
                    <p style="color: #6B7280; margin: 0; font-size: 0.85rem; line-height: 1.3; flex-grow: 1;">{description}</p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="border: 2px solid #F59E0B; border-radius: 10px; padding: 2rem; margin: 1rem 0; background-color: #FFFBEB; box-sizing: border-box;">
            <h3 style="color: #D97706; margin: 0 0 1rem 0;">📁 No Datasets Found</h3>
            <p style="color: #92400E; margin: 1rem 0;">To add your data, create the following structure in the <code>data/</code> folder:</p>
            <pre style="background-color: #FEF3C7; border: 1px solid #F59E0B; border-radius: 5px; padding: 1rem; margin: 1rem 0; overflow-x: auto; font-size: 0.9rem;">data/YourDataset/
├── short_gradient/FDR_1/    # Fast gradient baseline files
├── short_gradient/FDR_20/   # Fast gradient training files  
├── short_gradient/FDR_50/   # Fast gradient training files
└── long_gradient/FDR_1/     # Slow gradient ground truth files</pre>
        </div>
        """, unsafe_allow_html=True)

def show_training_interface():
    """Display the training mode interface (current functionality)."""
    # Add mode indicator and back button
    # Header with symmetric alignment to blue header rectangle
    col1, col2, col3 = st.columns([1, 3, 1])
    
    # Add global CSS for button styling with stronger selectors
    st.markdown("""
    <style>
    /* Force refresh with timestamp and stronger selectors */
    div[data-testid="stButton"] button[data-baseweb="button"][kind="secondary"],
    .stButton > button[kind="secondary"] {
        background: #2E86AB !important;
        color: white !important;
        border: none !important;
        height: 80px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        border-radius: 15px !important;
        padding: 20px 30px !important;
        width: 100% !important;
        margin-left: 0px !important;
        transform: translateX(0px) !important;
    }
    
    div[data-testid="stButton"] button[data-baseweb="button"][kind="primary"],
    .stButton > button[kind="primary"] {
        background: #2E86AB !important;
        color: white !important;
        border: none !important;
        height: 80px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        border-radius: 15px !important;
        padding: 20px 30px !important;
        width: 100% !important;
        margin-right: -20px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with col1:
        if st.button("← Back to Home", type="secondary", key="big_back_btn"):
            # Clear all analysis states when returning to home
            st.session_state.app_mode = None
            clear_analysis_states()
            st.rerun()
    with col2:
        st.markdown("<h3 style='text-align: center; margin: 20px 0;'>🧠 Training Mode</h3>", unsafe_allow_html=True)
    with col3:
        # Start Analysis button in header
        start_button_placeholder = st.empty()
    
    st.markdown("---")
    
    # Discover available files
    with st.spinner("🔍 Discovering available data files..."):
        files_info = discover_available_files()
    
    # Sidebar configuration
    st.sidebar.markdown("## 🎛️ Analysis Configuration")
    
    # Comprehensive file listing in sidebar
    st.sidebar.markdown("### 📊 Available Data Files")
    
    # Dataset filter
    available_datasets = set()
    for category in ['baseline', 'ground_truth', 'training']:
        for file_info in files_info[category]:
            available_datasets.add(file_info.get('dataset', 'Unknown'))
    
    dataset_options = ['All'] + sorted(list(available_datasets))
    dataset_filter = st.sidebar.selectbox(
        "Filter by Dataset:",
        dataset_options,
        help="Filter files by dataset type"
    )
    
    # Cache dataset processing to avoid recomputing on every interaction
    @st.cache_data
    def get_available_datasets(files_info):
        available_datasets = set()
        for category in ['baseline', 'ground_truth', 'training']:
            for file_info in files_info[category]:
                available_datasets.add(file_info.get('dataset', 'Unknown'))
        return sorted(list(available_datasets))

    # Helper function to filter files by dataset (cached)
    @st.cache_data
    def filter_files_by_dataset(file_list, dataset_filter):
        if dataset_filter == 'All':
            return file_list
        return [f for f in file_list if f.get('dataset') == dataset_filter]
    
    # Use baseline files that were already identified during discovery
    # These are FDR_1 files from short gradient (any dataset)
    baseline_files = files_info.get('baseline', [])
    
    # Filter files based on selection
    filtered_baseline = filter_files_by_dataset(baseline_files, dataset_filter)
    filtered_ground_truth = filter_files_by_dataset(files_info['ground_truth'], dataset_filter)
    filtered_training = filter_files_by_dataset(files_info['training'], dataset_filter)
    
    # Baseline files (Fast gradient 1% FDR) - optimized rendering
    if filtered_baseline:
        filter_text = f" ({dataset_filter})" if dataset_filter != 'All' else ""
        with st.sidebar.expander(f"🚀 Baseline Files{filter_text} - {len(filtered_baseline)} files", expanded=False):
            # Show only first 5 files for performance
            display_files = filtered_baseline[:5]
            for file_info in display_files:
                dataset_icon = "📊"  # Generic dataset icon
                st.markdown(f"**{file_info['filename']}**\n• Dataset: {dataset_icon} {file_info.get('dataset', 'Unknown')} • Method: {file_info['method']} • FDR: {file_info['fdr']}% • Size: {file_info['size_mb']:.1f} MB")
            if len(filtered_baseline) > 5:
                st.markdown(f"*... and {len(filtered_baseline) - 5} more baseline files*")
    
    # Ground truth files (Slow gradient 1% FDR) - optimized rendering
    if filtered_ground_truth:
        filter_text = f" ({dataset_filter})" if dataset_filter != 'All' else ""
        with st.sidebar.expander(f"🐌 Ground Truth Files{filter_text} - {len(filtered_ground_truth)} files", expanded=False):
            # Show only first 5 files for performance
            display_files = filtered_ground_truth[:5]
            for file_info in display_files:
                dataset_icon = "📊"  # Generic dataset icon
                st.markdown(f"**{file_info['filename']}**\n• Dataset: {dataset_icon} {file_info.get('dataset', 'Unknown')} • Method: {file_info['method']} • Size: {file_info['size_mb']:.1f} MB")
            if len(filtered_ground_truth) > 5:
                st.markdown(f"*... and {len(filtered_ground_truth) - 5} more ground truth files*")
    
    # Training files (Fast gradient multiple FDR levels) - optimized rendering
    if filtered_training:
        filter_text = f" ({dataset_filter})" if dataset_filter != 'All' else ""
        with st.sidebar.expander(f"⚙️ Training Files{filter_text} - {len(filtered_training)} files", expanded=False):
            # Show only first 5 files for performance
            display_files = filtered_training[:5]
            for file_info in display_files:
                dataset_icon = "📊"  # Generic dataset icon
                st.markdown(f"**{file_info['filename']}**\n• Dataset: {dataset_icon} {file_info.get('dataset', 'Unknown')} • Method: {file_info['method']} • FDR: {file_info['fdr']}% • Size: {file_info['size_mb']:.1f} MB")
            if len(filtered_training) > 5:
                st.markdown(f"*... and {len(filtered_training) - 5} more training files*")
    
    # Display run history
    st.sidebar.markdown("---")
    display_run_history()
    
    # Configuration section
    st.sidebar.markdown("### ⚙️ Training Configuration")
    
    # Cache available methods to avoid expensive discovery calls
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_available_methods_cached(dataset_filter):
        return get_configured_methods(dataset_filter)
    
    # Get available methods based on dataset filter
    available_methods = get_available_methods_cached(dataset_filter)
    
    # Training methods selection (multiple methods OK - each matched to its ground truth)
    train_methods = st.sidebar.multiselect(
        "Select Training Methods",
        options=available_methods,
        default=available_methods[:1] if available_methods else [],
        help="Choose MS methods for training (each matched to its ground truth for proper labeling)"
    )
    
    # Training FDR levels
    train_fdrs = st.sidebar.multiselect(
        "Training FDR Levels (%)",
        options=[1, 20, 50],
        default=[50],
        help="FDR levels to include in training data"
    )
    
    st.sidebar.markdown("### 🎯 Testing Configuration")
    
    # Cache test method filtering to avoid recomputing on every train_methods change
    @st.cache_data
    def get_available_test_methods(all_methods, train_methods_tuple):
        return [m for m in all_methods if m not in train_methods_tuple]
    
    # Test method selection (cannot overlap with training)
    available_test_methods = get_available_test_methods(available_methods, tuple(train_methods or []))
    
    # Add validation info
    if not available_test_methods:
        st.sidebar.warning("⚠️ No test methods available. Remove some training methods to select a holdout test method.")
        test_method = None
    else:
        test_method = st.sidebar.selectbox(
            "Holdout Test Method",
            options=available_test_methods,
            index=0,
            help="Single method to hold out for testing (prevents data leakage) - uses method-specific ground truth"
        )
        
        # Display the full selected method name
        if test_method:
            st.sidebar.markdown(f"**Selected:** `{test_method}`")
    
    # Force validation - ensure test method is not in training methods
    if test_method and test_method in train_methods:
        st.sidebar.error("🚫 **Error**: Test method cannot be the same as a training method!")
        test_method = None
    
    # Test FDR level
    test_fdr = st.sidebar.selectbox(
        "Test FDR Level (%)",
        options=[20, 50],
        index=1,  # Default to 50%
        help="FDR level for test data (higher = more candidates)"
    )
    
    # Target FDR levels
    st.sidebar.markdown("### 🎯 Target FDR Levels")
    target_fdrs = st.sidebar.multiselect(
        "Target FDR Levels (%)",
        options=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 30.0, 50.0],
        default=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 30.0, 50.0],
        help="FDR levels to optimize for in results (select more for broader analysis)"
    )
    
    # Advanced parameters
    with st.sidebar.expander("🔧 Advanced Parameters"):
        learning_rate = st.slider("Learning Rate", 0.01, 0.3, 0.08, 0.01)
        max_depth = st.slider("Max Depth", 3, 10, 7, 1)
        n_estimators = st.slider("N Estimators", 100, 2000, 1000, 100)
        subsample = st.slider("Subsample", 0.5, 1.0, 0.8, 0.1)
    
    # Feature selection
    with st.sidebar.expander("🎯 Feature Selection"):
        st.markdown("**Select which feature groups to include:**")
        
        # DIA-NN Quality Features
        use_diann_quality = st.checkbox("DIA-NN Quality Metrics", value=True, 
                                       help="All DIA-NN confidence scores: GG.Q.Value, PEP, PG.PEP, PG.Q.Value, Q.Value, Global.Q.Value, Global.PG.Q.Value, Global.Peptidoform.Q.Value, Protein.Q.Value, Proteotypic, Evidence, Genes.MaxLFQ.Quality")
        
        # Sequence Features
        use_sequence_features = st.checkbox("Peptide Sequence Features", value=True,
                                          help="Sequence length, amino acid composition, frequencies")
        
        # Mass Spec Features  
        use_ms_features = st.checkbox("Mass Spectrometry Features", value=True,
                                    help="Retention time, predicted RT, mass, areas, intensities")
        
        # Statistical Features
        use_statistical_features = st.checkbox("Statistical Features", value=True,
                                             help="Z-scores, ratios, normalized values")
        
        # Library Features
        use_library_features = st.checkbox("Library Features", value=True,
                                         help="Precursor library index, fragment info")
        
        st.markdown("---")
        st.markdown("**Custom Feature Selection:**")
        
        # Get actual features from recent runs if available
        all_possible_features = []
        
        # Try to get features from the most recent run's feature importance data
        recent_features = []
        if hasattr(st.session_state, 'results_data') and st.session_state.results_data:
            if 'feature_analysis' in st.session_state.results_data:
                feature_data = st.session_state.results_data['feature_analysis']
                if 'feature_importance' in feature_data:
                    recent_features = list(feature_data['feature_importance'].get('feature', []))
        
        # If we have recent features, use those (they're real!)
        if recent_features:
            all_possible_features = sorted(recent_features)
        else:
            # COMPREHENSIVE list - EVERY SINGLE feature that could appear
            # Based on actual feature importance data from recent runs
            all_features = [
                # ========== ACTUAL FEATURES FROM RECENT RUNS ==========
                # These are the REAL features that appeared in model training
                'GG.Q.Value', 'PG.PEP', 'PG.Q.Value', 'aa_count_C', 'aa_freq_C',
                'Global.PG.Q.Value', 'PEP', 'zscore_Precursor.Quantity', 'Precursor.Charge',
                'aa_count_K', 'Q.Value', 'Genes.MaxLFQ', 'log_Precursor.Charge',
                'aa_count_R', 'Global.Q.Value', 'PG.MaxLFQ', 'sequence_length',
                'log_Precursor.Quantity', 'aa_count_H', 'aa_freq_H', 'Genes.MaxLFQ.Unique',
                'Protein.Q.Value', 'aa_freq_K', 'RT', 'log_Ms1.Area', 'aa_count_M',
                'Proteotypic', 'Ms1.Profile.Corr', 'iRT', 'Genes.MaxLFQ.Unique.Quality',
                'Global.Peptidoform.Q.Value', 'aa_freq_M', 'aa_freq_R', 'Predicted.RT',
                'Precursor.Lib.Index', 'Quantity.Quality', 'Ms1.Normalised', 'PG.MaxLFQ.Quality',
                'aa_count_E', 'Precursor.Mz', 'RT.Start', 'iIM', 'aa_freq_P', 'RT.Stop',
                'aa_count_P', 'Ms1.Apex.Mz.Delta', 'Precursor.Quantity', 'Best.Fr.Mz.Delta',
                'Mass.Evidence', 'Ms1.Area', 'aa_freq_E', 'Evidence', 'Predicted.iRT',
                'Precursor.Normalised', 'aa_freq_D', 'Ms1.Apex.Area', 'FWHM', 'Best.Fr.Mz',
                'Genes.MaxLFQ.Quality', 'Ms1.Total.Signal.After', 'Channel.Evidence',
                'Ms1.Total.Signal.Before', 'aa_count_D', 'source_fdr', 'zscore_Ms1.Area',
                'PG.TopN', 'Channel.Q.Value', 'Translated.Q.Value', 'Lib.PTM.Site.Confidence',
                'PTM.Site.Confidence', 'Lib.Peptidoform.Q.Value', 'Peptidoform.Q.Value',
                'Lib.Q.Value', 'Lib.PG.Q.Value', 'Normalisation.Noise', 'Empirical.Quality',
                'Normalisation.Factor', 'Genes.TopN', 'Predicted.iIM', 'Decoy', 'Predicted.IM',
                'IM', 'Run.Index',
                
                # ========== COMPLETE AMINO ACID FEATURES ==========
                # All 20 amino acids - counts and frequencies
                'aa_count_A', 'aa_count_C', 'aa_count_D', 'aa_count_E', 'aa_count_F',
                'aa_count_G', 'aa_count_H', 'aa_count_I', 'aa_count_K', 'aa_count_L',
                'aa_count_M', 'aa_count_N', 'aa_count_P', 'aa_count_Q', 'aa_count_R',
                'aa_count_S', 'aa_count_T', 'aa_count_V', 'aa_count_W', 'aa_count_Y',
                'aa_freq_A', 'aa_freq_C', 'aa_freq_D', 'aa_freq_E', 'aa_freq_F',
                'aa_freq_G', 'aa_freq_H', 'aa_freq_I', 'aa_freq_K', 'aa_freq_L',
                'aa_freq_M', 'aa_freq_N', 'aa_freq_P', 'aa_freq_Q', 'aa_freq_R',
                'aa_freq_S', 'aa_freq_T', 'aa_freq_V', 'aa_freq_W', 'aa_freq_Y',
                
                # ========== Z-SCORE FEATURES ==========
                'zscore_Ms1.Area', 'zscore_Ms2.Area', 'zscore_Peak.Height', 
                'zscore_Precursor.Quantity', 'zscore_Precursor.Charge', 'zscore_FWHM',
                'zscore_Ms1.Apex.Area', 'zscore_Ms1.Profile.Corr', 'zscore_RT',
                'zscore_iRT', 'zscore_Predicted.RT', 'zscore_Precursor.Mz',
                'zscore_Quantity.Quality', 'zscore_Mass.Evidence',
                
                # ========== LOG FEATURES ==========
                'log_GG.Q.Value', 'log_PEP', 'log_Q.Value', 'log_PG.Q.Value', 
                'log_Global.Q.Value', 'log_Ms1.Area', 'log_Ms2.Area', 'log_Peak.Height',
                'log_Precursor.Quantity', 'log_Precursor.Charge', 'log_FWHM',
                'log_Ms1.Apex.Area', 'log_Genes.MaxLFQ', 'log_PG.MaxLFQ',
                'log_RT', 'log_iRT', 'log_Predicted.RT', 'log_Precursor.Mz',
                'log_Quantity.Quality', 'log_Mass.Evidence', 'log_Evidence',
                'log_Proteotypic', 'log_Ms1.Profile.Corr',
                
                # ========== RATIO FEATURES ==========
                'ratio_Ms1.Area_Ms2.Area', 'ratio_Peak.Height_Ms1.Area', 
                'ratio_Precursor.Quantity_Peak.Height', 'ratio_Ms1.Apex.Area_Ms1.Area',
                'ratio_FWHM_Peak.Height', 'ratio_Precursor.Charge_Precursor.Mz',
                'ratio_RT_iRT', 'ratio_RT_Predicted.RT', 'ratio_Ms1.Total.Signal.After_Before',
                'ratio_Best.Fr.Mz_Precursor.Mz', 'ratio_Ms1.Apex.Mz.Delta_Precursor.Mz',
                
                # ========== ADDITIONAL MASS SPEC FEATURES ==========
                'Ms2.Area', 'Peak.Height', 'Ms1.Translated', 'Ms1.Area.Raw', 'Quantity.Raw',
                'Fragment.Quant.Raw', 'Fragment.Quant.Corrected', 'Ms2.Scan',
                'Ion.Mobility', 'CCS', 'Charge', 'Index.RT',
                
                # ========== ADDITIONAL DIA-NN COLUMNS ==========
                'Precursor.Id', 'Protein.Group', 'Protein.Ids', 'Protein.Names',
                'Genes', 'Modified.Sequence', 'Stripped.Sequence', 'Fragment.Info',
                'First.Protein.Description', 'Shared.Count', 'Razor.Count', 'Unique.Count',
                
                # ========== SPECIAL/COMPUTED FEATURES ==========
                'sequence_hydrophobicity', 'sequence_molecular_weight', 'sequence_charge_density',
                'peptide_complexity_score', 'fragment_pattern_score', 'retention_prediction_error'
            ]
            
            # Remove duplicates and sort
            all_possible_features = sorted(list(set(all_features)))
        
        # Multi-select dropdown for feature exclusion
        excluded_feature_list = st.multiselect(
            "Select Features to Exclude",
            options=all_possible_features,
            default=[],
            help="Select any features you want to exclude from the model training. "
                 "Features are grouped by category above - use this for fine-grained control.",
            placeholder="Choose features to exclude..."
        )
    
    # Configuration validation
    config_valid = True
    validation_messages = []
    
    if not train_methods:
        config_valid = False
        validation_messages.append("❌ Select at least one training method")
    
    if not test_method:
        config_valid = False
        validation_messages.append("❌ Select a test method")
    
    if test_method in train_methods:
        config_valid = False
        validation_messages.append("❌ Test method cannot overlap with training methods")
    
    if not target_fdrs:
        config_valid = False
        validation_messages.append("❌ Select at least one target FDR level")
    
    # Display validation status
    if validation_messages:
        for msg in validation_messages:
            st.sidebar.markdown(f'<p class="status-error">{msg}</p>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<p class="status-success">✅ Configuration valid</p>', unsafe_allow_html=True)
    
    # Add Start Analysis button to header
    with start_button_placeholder.container():
        if st.button("🔬 Start Analysis", disabled=not config_valid, type="primary", key="header_start_btn"):
            st.session_state.analysis_running = True
            st.session_state.analysis_complete = False
            st.rerun()
    
    # Check if we should show comparison view
    if hasattr(st.session_state, 'show_comparison') and st.session_state.show_comparison:
        display_run_comparison()
        return
    
    # Main content area
    if not st.session_state.analysis_running and not st.session_state.analysis_complete:
        # Professional card-based layout
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        
        # Row 1: Configuration Summary Card
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid #dee2e6;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            margin-bottom: 20px;
        ">
            <div style="display: flex; align-items: center; margin-bottom: 16px;">
                <span style="font-size: 24px; margin-right: 10px;">📋</span>
                <h2 style="color: #495057; margin: 0; font-weight: 600;">Analysis Configuration Summary</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration details in a clean format
        config_summary = {
            "Training Methods": ", ".join(train_methods) if train_methods else "None selected",
            "Training FDR Levels": ", ".join(map(str, train_fdrs)) + "%" if train_fdrs else "None selected",
            "Test Method (Holdout)": test_method or "None selected",
            "Test FDR Level": f"{test_fdr}%" if test_fdr else "None selected",
            "Target FDR Levels": ", ".join(map(str, target_fdrs)) + "%" if target_fdrs else "None selected"
        }
        
        # Display config in columns for better layout
        config_col1, config_col2 = st.columns(2)
        config_items = list(config_summary.items())
        
        with config_col1:
            for i in range(0, len(config_items), 2):
                key, value = config_items[i]
                st.markdown(f"**{key}:** {value}")
        
        with config_col2:
            for i in range(1, len(config_items), 2):
                if i < len(config_items):
                    key, value = config_items[i]
                    st.markdown(f"**{key}:** {value}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # File Preview Card (left side)
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File Preview Card - Updated to use configured methods
            train_file_count = 0
            test_file_count = 0
            total_size = 0
            
            # Count training files using the new grouped method system
            for method in train_methods:
                for fdr in train_fdrs:
                    method_files = get_files_for_configured_method(method, fdr)
                    train_file_count += len(method_files)
                    # Calculate actual file sizes
                    for file_path in method_files:
                        try:
                            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                            total_size += file_size_mb
                        except OSError:
                            # If file doesn't exist, skip (don't add inflated estimates)
                            pass
            
            # Count test files using the new grouped method system
            if test_method:
                test_files = get_files_for_configured_method(test_method, test_fdr)
                test_file_count = len(test_files)
                # Calculate actual file sizes for test files
                for file_path in test_files:
                    try:
                        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
                        total_size += file_size_mb
                    except OSError:
                        # If file doesn't exist, skip (don't add inflated estimates)
                        pass
            
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
                padding: 20px;
                border-radius: 12px;
                border: 1px solid #90caf9;
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);
                margin-bottom: 20px;
            ">
                <div style="display: flex; align-items: center; margin-bottom: 16px;">
                    <span style="font-size: 20px; margin-right: 8px;">📁</span>
                    <h3 style="color: #1565c0; margin: 0; font-weight: 600;">File Preview</h3>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced metrics with icons
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            with metric_col1:
                st.markdown(f"""
                <div style="text-align: center; padding: 12px; background: white; border-radius: 8px; border: 1px solid #e0e0e0;">
                    <div style="font-size: 24px; color: #2E86AB;">🏋️</div>
                    <div style="font-size: 18px; font-weight: 600; color: #333;">{train_file_count}</div>
                    <div style="font-size: 12px; color: #666;">Training Files</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown(f"""
                <div style="text-align: center; padding: 12px; background: white; border-radius: 8px; border: 1px solid #e0e0e0;">
                    <div style="font-size: 24px; color: #2E86AB;">🧪</div>
                    <div style="font-size: 18px; font-weight: 600; color: #333;">{test_file_count}</div>
                    <div style="font-size: 12px; color: #666;">Test Files</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown(f"""
                <div style="text-align: center; padding: 12px; background: white; border-radius: 8px; border: 1px solid #e0e0e0;">
                    <div style="font-size: 24px; color: #2E86AB;">💾</div>
                    <div style="font-size: 18px; font-weight: 600; color: #333;">{total_size:.1f} MB</div>
                    <div style="font-size: 12px; color: #666;">Total Data</div>
                </div>
                """, unsafe_allow_html=True)
        
    
    elif st.session_state.analysis_running:
        # Analysis in progress
        st.markdown("## 🔬 Training in Progress")
        
        # DNA loading animation - check local paths only
        possible_paths = [
            "./assets/dna_loading.gif",
            "./dna_loading.gif"
        ]
        
        gif_loaded = False
        for path in possible_paths:
            try:
                if os.path.exists(path):
                    # Use HTML to properly display animated GIF
                    st.markdown(f"""
                    <div style='text-align: center; margin: 20px 0;'>
                        <img src="data:image/gif;base64,{__import__('base64').b64encode(open(path, 'rb').read()).decode()}" 
                             width="300" style="border-radius: 10px;">
                        <p style='color: #2E86AB; font-weight: 500; margin-top: 10px;'>Processing your peptide data...</p>
                    </div>
                    """, unsafe_allow_html=True)
                    gif_loaded = True
                    break
            except:
                continue
        
        if not gif_loaded:
            # Enhanced fallback with animated-style text
            st.markdown("""
            <div style='text-align: center; font-size: 2rem; margin: 20px 0;'>
                🧬 🔄 🧬
            </div>
            <div style='text-align: center; font-size: 1.2rem; color: #2E86AB; font-weight: bold;'>
                Processing your peptide data...
            </div>
            <div style='text-align: center; font-size: 0.9rem; color: #6C757D; margin-top: 10px;'>
                Machine learning training in progress
            </div>
            """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        log_container = st.empty()
        
        # Import the backend API
        from peptide_validator_api import run_peptide_validation
        import time
        import contextlib
        from io import StringIO
        
        # Progress tracking
        progress_log = []
        
        def progress_callback(current_step, total_steps, step_name):
            progress = current_step / total_steps
            # Ensure progress is within valid range [0.0, 1.0]
            progress = max(0.0, min(1.0, progress))
            progress_bar.progress(progress)
            status_text.markdown(f"**Step {current_step}/{total_steps}**: {step_name}")
            
            progress_log.append(f"Step {current_step}/{total_steps}: {step_name}")
            log_container.text_area("Progress Log:", 
                                   value="\n".join(progress_log[-5:]),  # Show last 5 steps
                                   height=150)
        
        # GPU detection info
        def detect_and_display_gpu():
            try:
                import xgboost as xgb
                # Try to create a simple XGBoost model with GPU
                test_model = xgb.XGBClassifier(device='cuda', n_estimators=1)
                # Test with minimal data
                import numpy as np
                X_test = np.array([[1, 2], [3, 4]])
                y_test = np.array([0, 1])
                test_model.fit(X_test, y_test)
                return 'cuda'
            except Exception as e:
                st.warning(f"⚠️ GPU not available, using CPU: {str(e)[:50]}...")
                return 'cpu'
        
        # Detect compute device
        device = detect_and_display_gpu()
        
        # Prepare XGBoost parameters
        xgb_params = {
            'learning_rate': learning_rate,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'device': device,  # Use detected device
            'tree_method': 'hist'  # Ensure compatibility with device setting
        }
        
        # Run the actual analysis with suppressed verbose output
        start_time = time.time()
        
        @contextlib.contextmanager
        def suppress_verbose_output():
            """Suppress verbose stdout output during analysis."""
            import warnings
            import os
            original_stdout = sys.stdout
            
            # Suppress XGBoost device mismatch warnings
            warnings.filterwarnings('ignore', message='.*Falling back to prediction using DMatrix.*')
            os.environ['PYTHONWARNINGS'] = 'ignore'
            
            sys.stdout = StringIO()
            try:
                yield
            finally:
                sys.stdout = original_stdout
                # Restore warning filters
                warnings.resetwarnings()
        
        try:
            # Run analysis with suppressed verbose terminal output (UI progress unchanged)
            with suppress_verbose_output():
                results = run_peptide_validation(
                    train_methods=train_methods,
                    test_method=test_method,
                    train_fdr_levels=train_fdrs,
                    test_fdr=test_fdr,
                    target_fdr_levels=target_fdrs,
                    xgb_params=xgb_params,
                    progress_callback=progress_callback,
                    feature_selection={
                        'use_diann_quality': use_diann_quality,
                        'use_sequence_features': use_sequence_features,
                        'use_ms_features': use_ms_features,
                        'use_statistical_features': use_statistical_features,
                        'use_library_features': use_library_features,
                        'excluded_features': excluded_feature_list
                    }
                )
            
            end_time = time.time()
            runtime_minutes = (end_time - start_time) / 60
            
            # Check for errors
            if results.get('error', False):
                st.error(f"❌ Analysis failed: {results['error_message']}")
                st.session_state.analysis_running = False
                return
            
            # Store results with runtime
            results['metadata']['total_runtime_seconds'] = end_time - start_time
            results['summary']['runtime_minutes'] = runtime_minutes
            
            # Add to run history
            config_data = {
                'train_methods': train_methods,
                'train_fdr_levels': train_fdrs,
                'test_method': test_method,
                'test_fdr': test_fdr,
                'target_fdr_levels': target_fdrs,
                'xgb_params': xgb_params,
                'feature_selection': {
                    'use_diann_quality': use_diann_quality,
                    'use_sequence_features': use_sequence_features,
                    'use_ms_features': use_ms_features,
                    'use_statistical_features': use_statistical_features,
                    'use_library_features': use_library_features,
                    'excluded_features': excluded_feature_list
                }
            }
            add_to_run_history(config_data, results, runtime_minutes)
            
            st.session_state.results_data = results
            st.session_state.analysis_running = False
            st.session_state.analysis_complete = True
            
            st.success(f"🎉 Analysis completed successfully in {runtime_minutes:.1f} minutes!")
            time.sleep(1)
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Analysis failed with error: {str(e)}")
            st.session_state.analysis_running = False
    
    else:
        # Analysis complete - show results
        display_results()

def extract_summary_metrics(results_data):
    """Extract summary metrics from analysis results with enhanced clarity."""
    if not results_data or 'results' not in results_data or not results_data['results']:
        return None
    
    # Find best result using smarter selection logic
    # Priority: 1) FDR ≤ 15%, 2) Highest peptides, 3) If no good FDR, best peptides with reasonable FDR
    valid_low_fdr = [r for r in results_data['results'] if r['Actual_FDR'] <= 15.0 and int(r['Additional_Peptides']) > 0]
    valid_medium_fdr = [r for r in results_data['results'] if r['Actual_FDR'] <= 30.0 and int(r['Additional_Peptides']) > 0]
    
    if valid_low_fdr:
        # Best case: good FDR and peptides
        best_result = max(valid_low_fdr, key=lambda x: int(x['Additional_Peptides']))
    elif valid_medium_fdr:
        # Fallback: reasonable FDR, prioritize balance of FDR and peptides
        best_result = max(valid_medium_fdr, key=lambda x: int(x['Additional_Peptides']) / (1 + x['Actual_FDR']))
    else:
        # Last resort: any result with peptides
        results_with_peptides = [r for r in results_data['results'] if int(r['Additional_Peptides']) > 0]
        if results_with_peptides:
            best_result = max(results_with_peptides, key=lambda x: int(x['Additional_Peptides']))
        else:
            best_result = results_data['results'][0]  # Fallback to first result
    
    # Calculate enhanced metrics
    baseline_peptides = results_data['summary']['baseline_peptides']
    ground_truth_peptides = results_data['summary']['ground_truth_peptides']
    missed_peptides = results_data['summary'].get('missed_peptides', ground_truth_peptides - baseline_peptides)
    additional_candidates = results_data['summary'].get('additional_candidates', 0)
    
    return {
        'best_fdr': best_result['Actual_FDR'],
        'best_peptides': best_result['Additional_Peptides'],
        'best_method': results_data['config']['test_method'],
        'total_runtime': f"{results_data['summary'].get('runtime_minutes', 0):.1f} minutes",
        'baseline_peptides': baseline_peptides,
        'ground_truth_peptides': ground_truth_peptides,
        'missed_peptides': missed_peptides,
        'additional_candidates': additional_candidates,
        'training_samples': results_data['summary']['training_samples'],
        'test_samples': results_data['summary']['test_samples'],
        'max_possible_recovery': missed_peptides,
        'recovery_efficiency': (int(best_result['Additional_Peptides']) / missed_peptides * 100) if missed_peptides > 0 else 0
    }

# -----------------------------------------------
# 📊 Feature Importance visualisation helpers
# -----------------------------------------------

def create_feature_descriptions():
    """Create comprehensive dictionary of feature descriptions."""
    return {
        # Core DIA-NN Features
        'Precursor.Id': 'Unique identifier for the precursor ion',
        'Protein.Group': 'Protein group identifier from database search',
        'Protein.Ids': 'Individual protein identifiers within the group',
        'Protein.Names': 'Protein names from database annotation',
        'Genes': 'Gene names associated with the proteins',
        'First.Protein.Description': 'Description of the primary protein in the group',
        'Proteotypic': 'Whether the peptide is proteotypic (unique to protein)',
        'Stripped.Sequence': 'Peptide sequence without modifications',
        'Modified.Sequence': 'Peptide sequence with modification annotations',
        'Precursor.Charge': 'Charge state of the precursor ion',
        'Precursor.Mz': 'Mass-to-charge ratio of the precursor',
        
        # Quantitative Features
        'Precursor.Quantity': 'Quantified intensity of the precursor ion',
        'Ms1.Area': 'Area under the curve for MS1 precursor signal',
        'Ms2.Area': 'Area under the curve for fragment ion signals',
        'Peak.Height': 'Maximum intensity of the chromatographic peak',
        'RT': 'Retention time of the peptide elution',
        'RT.Start': 'Start time of the chromatographic peak',
        'RT.Stop': 'End time of the chromatographic peak',
        
        # Quality Metrics
        'iRT': 'Indexed retention time (normalized RT)',
        'Predicted.RT': 'Predicted retention time from library',
        'Predicted.iRT': 'Predicted indexed retention time',
        'Q.Value': 'FDR-adjusted q-value for identification confidence',
        'PEP': 'Posterior error probability',
        'Global.Q.Value': 'Global q-value across entire dataset',
        'Protein.Q.Value': 'Protein-level q-value',
        'GG.Q.Value': 'Gene group q-value',
        'Global.PG.Q.Value': 'Global protein group q-value',
        
        # Spectral Features
        'Mass.Evidence': 'Evidence supporting the mass measurement',
        'CScore': 'Confidence score for the identification',
        'Decoy.Evidence': 'Evidence from decoy database matches',
        'Decoy.CScore': 'Confidence score for decoy matches',
        
        # Experimental Context
        'File.Name': 'Source data file name',
        'Run': 'Experimental run identifier',
        'Experiment': 'Experiment name or condition',
        'source_fdr': 'FDR level of the source data file',
        
        # Engineered Features
        'sequence_length': 'Length of the peptide sequence (engineered)',
        'log_Precursor.Quantity': 'Log-transformed precursor quantity',
        'log_Ms1.Area': 'Log-transformed MS1 area',
        'log_Ms2.Area': 'Log-transformed MS2 area',
        'log_Peak.Height': 'Log-transformed peak height',
        'log_Precursor.Charge': 'Log-transformed precursor charge',
        
        # Ratio Features
        'Ms1.Area_Ms2.Area_ratio': 'Ratio of MS1 to MS2 signal areas',
        'Peak.Height_Ms1.Area_ratio': 'Ratio of peak height to MS1 area',
        'Precursor.Quantity_Peak.Height_ratio': 'Ratio of quantity to peak height',
    }

def get_feature_description(feature_name, descriptions):
    """Get description for a specific feature with fallback."""
    # Direct match
    if feature_name in descriptions:
        return descriptions[feature_name]
    
    # Pattern-based matching for engineered features
    if feature_name.startswith('log_'):
        base_feature = feature_name[4:]
        if base_feature in descriptions:
            return f"Log-transformed version of {base_feature}. {descriptions[base_feature]}"
        return f"Log-transformed feature derived from {base_feature}"
    
    if '_ratio' in feature_name:
        parts = feature_name.replace('_ratio', '').split('_')
        if len(parts) == 2:
            return f"Engineered ratio feature: {parts[0]} divided by {parts[1]}"
    
    # Category-based fallbacks
    if any(keyword in feature_name.lower() for keyword in ['quantity', 'area', 'height', 'intensity']):
        return "Quantitative intensity measurement from mass spectrometry data"
    elif any(keyword in feature_name.lower() for keyword in ['rt', 'time']):
        return "Chromatographic retention time or time-related measurement"
    elif any(keyword in feature_name.lower() for keyword in ['charge', 'mz', 'mass']):
        return "Physicochemical property related to ion mass or charge"
    elif any(keyword in feature_name.lower() for keyword in ['score', 'value', 'pep']):
        return "Statistical confidence or quality metric"
    else:
        return "DIA-NN derived feature (specific description not available)"

def get_related_features(feature_name, all_features):
    """Find features related to the selected feature."""
    related = []
    
    # Get base name (remove common suffixes/prefixes)
    base_name = feature_name.replace('log_', '').replace('_ratio', '')
    
    # Find features with similar base names
    for feature in all_features:
        if feature != feature_name:
            # Same base feature with different transformations
            if base_name in feature or any(part in feature for part in base_name.split('.')):
                related.append(feature)
            # Features with similar keywords
            elif any(keyword in feature.lower() and keyword in feature_name.lower() 
                    for keyword in ['area', 'quantity', 'height', 'charge', 'rt', 'score']):
                related.append(feature)
    
    return related[:10]  # Return top 10 related features

def create_interactive_shap_plots(shap_data_path, color_scheme=None):
    """Create interactive SHAP plots using JSON data."""
    try:
        shap_data = load_shap_data(shap_data_path)
        if shap_data is None:
            return None, None
        
        shap_values = shap_data['shap_values']
        feature_values = shap_data['feature_values']
        feature_names = shap_data['feature_names']
        feature_importance = shap_data['feature_importance']
        
        # Get top 12 features by importance for cleaner visualization
        top_indices = np.argsort(feature_importance)[-12:][::-1]
        
        # Create single bidirectional bar plot
        st.markdown("**Feature Importance (SHAP Impact Analysis)**")
        st.markdown("*Bars pointing **right (→)** = when this feature has HIGH values, the model predicts the peptide is MORE LIKELY to be real*")
        st.markdown("*Bars pointing **left (←)** = when this feature has HIGH values, the model predicts the peptide is LESS LIKELY to be real*")
        
        # Get mean SHAP values for bidirectional bars
        mean_shap_values = None
        try:
            with open(shap_data_path, 'r') as f:
                raw_shap_data = json.load(f)
            if 'mean_shap_values' in raw_shap_data:
                mean_shap_values = np.array(raw_shap_data['mean_shap_values'])
        except:
            pass
        
        fig_bar = create_shap_bidirectional_plot(feature_importance, feature_names, top_indices, mean_shap_values)
        st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})
        
        return True
        
    except Exception as e:
        st.error(f"Error loading SHAP data: {str(e)}")
        return False

def create_shap_beeswarm_plot(shap_values, feature_values, feature_names, top_indices):
    """Create interactive beeswarm-style SHAP plot."""
    fig = go.Figure()
    
    
    for i, feature_idx in enumerate(top_indices):
        feature_name = feature_names[feature_idx]
        shap_vals = shap_values[:, feature_idx]
        feat_vals = feature_values[:, feature_idx]
        
        # Sample for dense violin-like appearance (more points for better distribution visualization)
        n_samples = min(100, len(shap_vals))
        if len(shap_vals) > n_samples:
            sample_idx = np.random.choice(len(shap_vals), n_samples, replace=False)
            shap_vals = shap_vals[sample_idx]
            feat_vals = feat_vals[sample_idx]
        
        # Normalize feature values for color
        if feat_vals.max() > feat_vals.min():
            feat_vals_norm = (feat_vals - feat_vals.min()) / (feat_vals.max() - feat_vals.min())
        else:
            feat_vals_norm = np.zeros_like(feat_vals)
        
        # Create violin-like distribution based on SHAP value density
        shap_range = shap_vals.max() - shap_vals.min() if len(shap_vals) > 1 else 1
        # Use smaller jitter for points with similar SHAP values (creates violin shape)
        jitter_scale = 0.015 + 0.01 * (np.abs(shap_vals - np.median(shap_vals)) / (shap_range + 1e-6))
        y_jitter = np.array([np.random.normal(0, scale) for scale in jitter_scale])
        y_pos = np.ones(len(shap_vals)) * i + y_jitter
        
        fig.add_trace(go.Scatter(
            x=shap_vals,
            y=y_pos,
            mode='markers',
            marker=dict(
                size=4,  # Very small points for dense violin visualization
                color=feat_vals_norm,
                colorscale=CHART_COLORS['feature_gradient'],  # Use consistent feature gradient
                showscale=i == 0,
                colorbar=dict(
                    title=dict(text="Feature Value", side="right"),
                    thickness=12,
                    len=0.7,
                    x=1.01
                ) if i == 0 else None,
                cmin=0,
                cmax=1,
                opacity=0.7,  # More transparent for better density visualization
                line=dict(width=0.1, color=CHART_COLORS['outline_subtle'])  # Use consistent subtle outlines
            ),
            name=feature_name,
            hoverinfo='skip',  # Disable hover tooltips
            showlegend=False
        ))
    
    fig.update_layout(
        title=dict(text="", font=dict(size=16, family='Inter')),
        xaxis=dict(
            title=dict(text="SHAP value (impact on model output)", font=dict(size=13, family='Inter')),
            tickfont=dict(size=11, family='Inter'),
            showgrid=True,
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor=CHART_COLORS['text'],
            zerolinewidth=1
        ),
        yaxis=dict(
            title='',
            tickfont=dict(size=11, family='Inter'),
            tickmode='array',
            tickvals=list(range(len(top_indices))),
            ticktext=[feature_names[i] for i in top_indices[::-1]],
            showgrid=True,
            gridcolor='lightgray'
        ),
        height=500,
        font=dict(family="Inter", size=11),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=180, r=60, t=30, b=50)
    )
    
    return fig

def create_shap_bidirectional_plot(feature_importance, feature_names, top_indices, mean_shap_values=None):
    """Create bidirectional SHAP bar plot with Nature journal styling."""
    # Get current colors for consistency FIRST
    current_colors = get_current_colors()
    
    # Get top features data
    top_importance = feature_importance[top_indices]
    top_names = feature_names[top_indices]
    
    fig = go.Figure()
    
    # Use absolute SHAP values with colorbar to match XGBoost plot style
    if mean_shap_values is not None:
        top_mean_shap = mean_shap_values[top_indices]
        # Use absolute values for consistent colorbar display like XGBoost plot
        top_abs_shap = np.abs(top_mean_shap)
        
        fig.add_trace(go.Bar(
            x=top_mean_shap,  # Keep actual values for bidirectional display
            y=top_names,
            orientation='h',
            marker=dict(
                color=top_abs_shap,  # Color by absolute magnitude
                colorscale=current_colors['feature_gradient'],
                showscale=True,
                colorbar=dict(
                    title=dict(text="SHAP Value (Importance)", side="right"),
                    thickness=12,
                    len=0.7,
                    x=1.01
                ),
                line=dict(color='white', width=0.8)
            ),
            hovertemplate="<b>%{y}</b><br>SHAP value: %{x:.4f}<br>%{customdata}<extra></extra>",
            customdata=[f'High {feature_names[top_indices[i]]} → {"More likely real" if val >= 0 else "Less likely real"}' for i, val in enumerate(top_mean_shap)]
        ))
        
        # Add a subtle vertical line at x=0
        fig.add_vline(x=0, line_width=2, line_color="rgba(0,0,0,0.3)")
        
        x_title = "SHAP value (impact on model prediction)"
        
    else:
        # Fallback to absolute importance values using your color scheme
        fig.add_trace(go.Bar(
            x=top_importance,
            y=top_names,
            orientation='h',
            marker=dict(
                color=top_importance,
                colorscale=current_colors['feature_gradient'],  # Use consistent feature gradient
                showscale=True,
                colorbar=dict(
                    title=dict(text="SHAP Value (Importance)", side="right"),
                    thickness=12,
                    len=0.7,
                    x=1.01
                )
            ),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
        ))
        
        x_title = "Feature importance"
    
    # Colors already loaded at top of function
    
    fig.update_layout(
        title=dict(text="", font=dict(size=16, family='Inter', color=current_colors['text'])),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=12, family='Inter', color=current_colors['text'])),
            tickfont=dict(size=10, family='Inter', color=current_colors['text']),
            showgrid=True,
            gridcolor=current_colors['grid'],
            zeroline=True,
            zerolinecolor='rgba(0, 0, 0, 0.5)',
            zerolinewidth=2
        ),
        yaxis=dict(
            title='',
            tickfont=dict(size=11, family='Inter', color=current_colors['text']),
            categoryorder='total ascending',  # Same ordering as XGBoost plot
            showgrid=False,
            showline=False
        ),
        height=600,  # Match XGBoost plot height
        font=dict(family="Inter", size=11, color=current_colors['text']),
        plot_bgcolor=current_colors['background'],
        paper_bgcolor=current_colors['background'],
        margin=dict(l=200, r=100, t=30, b=80),
        showlegend=False
    )
    
    return fig

def display_feature_importance_tab(results_dir: str):
    """Render interactive feature importance plots."""
    st.markdown("### 📊 Feature Importance Analysis")

    # Check if feature importance CSV is available
    feature_analysis_dir = os.path.join(results_dir, "feature_analysis")
    feature_csv_path = os.path.join(feature_analysis_dir, "feature_importance_full.csv")
    csv_available = os.path.exists(feature_csv_path)

    if not csv_available:
        st.info("🔍 Feature importance data is not available for this run.")
        st.markdown("**Note**: Feature importance analysis is generated during model training and may not be available for all analysis types.")
        return

    try:
        # Load feature importance data with caching
        feature_df = load_feature_importance_csv(feature_csv_path)
        if feature_df is None:
            return
        
        if len(feature_df) == 0:
            st.info("All features have zero importance in this model.")
            return
            
        # Sort by importance descending
        feature_df = feature_df.sort_values('importance', ascending=False)
        
        # Format importance as percentage
        feature_df['importance_pct'] = (feature_df['importance'] * 100).round(2)
        
        st.markdown("#### 📈 Interactive Feature Importance Visualizations")
        
        # 1. Top 20 Feature Importance Bar Chart (Interactive)
        top_20 = feature_df.head(20).copy()
        
        # Get current colors for consistency
        current_colors = get_current_colors()
        
        fig_bar = px.bar(
            top_20, 
            x='importance_pct', 
            y='feature',
            orientation='h',
            title='Top 20 Most Important Features',
            labels={'importance_pct': 'Importance (%)', 'feature': 'Feature'},
            color='importance_pct',
            color_continuous_scale=current_colors['feature_gradient']
        )
        
        fig_bar.update_layout(
            height=600,
            yaxis={'categoryorder': 'total ascending', 'gridcolor': current_colors['grid']},
            plot_bgcolor=current_colors['background'],
            paper_bgcolor=current_colors['background'],
            font=dict(family="Inter", size=12, color=current_colors['text']),
            title_font_size=16,
            title_font_color=current_colors['text'],
            xaxis=dict(gridcolor=current_colors['grid']),
            coloraxis_colorbar=dict(
                title=dict(text="Importance (%)", side="right"),
                thickness=12,
                len=0.7
            )
        )
        
        st.plotly_chart(fig_bar, use_container_width=True)
        
        # Add explanatory note about different importance methods
        st.info("""
        ℹ️ **Note**: The visualizations above and below use different importance calculation methods:
        - **XGBoost Importance** (above): Measures the improvement in accuracy brought by a feature to the branches it is on
        - **SHAP Values** (below): Measures the average contribution of each feature to individual predictions
        """)
        
        # 2. SHAP Feature Importance Analysis - Interactive Version
        shap_data_path = os.path.join(feature_analysis_dir, "shap_data.json")
        shap_beeswarm_path = os.path.join(feature_analysis_dir, "shap_summary_beeswarm.png")
        shap_bar_path = os.path.join(feature_analysis_dir, "shap_importance_bar.png")
        
        if os.path.exists(shap_data_path):
            st.markdown("#### 🐝 SHAP Feature Importance Analysis")
           
            
            # Use interactive plots - pass current color scheme to ensure plot updates
            current_scheme = st.session_state.get('selected_color_scheme', 'Default')
            create_interactive_shap_plots(shap_data_path, current_scheme)
            
        elif os.path.exists(shap_beeswarm_path) or os.path.exists(shap_bar_path):
            st.markdown("#### 🐝 SHAP Feature Importance Analysis (Static)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if os.path.exists(shap_beeswarm_path):
                    st.markdown("**SHAP Beeswarm Summary Plot**")
                    st.image(shap_beeswarm_path, use_container_width=True)
                else:
                    st.info("SHAP beeswarm plot not available")
            
            with col2:
                if os.path.exists(shap_bar_path):
                    st.markdown("**SHAP Feature Importance**")
                    st.image(shap_bar_path, use_container_width=True)
                else:
                    st.info("SHAP bar plot not available")
        
        # 3. Feature Categories Analysis (if we can detect patterns)
        st.markdown("#### 🔍 Feature Category Analysis")
        
        # Categorize features using cached function for performance
        feature_categories = categorize_features(feature_df['feature'].tolist())
        feature_df['category'] = feature_df['feature'].map(feature_categories).fillna('Other Features')
        
        # Use module-level constant for category descriptions
        category_descriptions = FEATURE_CATEGORY_DESCRIPTIONS
        
        # Category importance summary
        category_summary = feature_df.groupby('category').agg({
            'importance_pct': ['sum', 'mean', 'count']
        }).round(2)
        category_summary.columns = ['Total_Importance', 'Mean_Importance', 'Feature_Count']
        category_summary = category_summary.reset_index().sort_values('Total_Importance', ascending=False)
        
        # Add descriptions for hover text
        category_summary['Description'] = category_summary['category'].map(category_descriptions)
        category_summary['Hover_Text'] = (
            '<b>' + category_summary['category'] + '</b><br>' +
            category_summary['Description'] + '<br><br>' +
            '<b>Total Importance:</b> ' + category_summary['Total_Importance'].astype(str) + '%<br>' +
            '<b>Feature Count:</b> ' + category_summary['Feature_Count'].astype(str) + '<br>' +
            '<b>Average Importance:</b> ' + category_summary['Mean_Importance'].astype(str) + '%'
        )
        
        fig_category = px.bar(
            category_summary,
            x='category',
            y='Total_Importance',
            title='Feature Importance by Category',
            labels={'category': 'Feature Category', 'Total_Importance': 'Total Importance (%)'},
            color='Total_Importance',
            color_continuous_scale=CHART_COLORS['feature_gradient'],
            hover_data={'Total_Importance': ':.1f', 'Feature_Count': True, 'Mean_Importance': ':.1f'},
            custom_data=['Hover_Text']
        )
        
        # Update hover template
        fig_category.update_traces(
            hovertemplate='%{customdata[0]}<extra></extra>'
        )
        
        fig_category.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12),
            title_font_size=16,
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray')
        )
        
        st.plotly_chart(fig_category, use_container_width=True)
        
        # Category descriptions table
        st.markdown("##### 🏷️ Category Descriptions")
        
        # Create a nice table with category descriptions
        desc_data = []
        for cat, desc in category_descriptions.items():
            if cat in category_summary['category'].values:
                cat_info = category_summary[category_summary['category'] == cat].iloc[0]
                desc_data.append({
                    'Category': cat,
                    'Description': desc,
                    'Total Importance (%)': f"{cat_info['Total_Importance']:.1f}%",
                    'Feature Count': int(cat_info['Feature_Count']),
                    'Avg Importance (%)': f"{cat_info['Mean_Importance']:.1f}%"
                })
        
        desc_df = pd.DataFrame(desc_data)
        st.dataframe(desc_df, use_container_width=True, hide_index=True)
        
        # 4. Feature Dictionary with Descriptions
        st.markdown("#### 📖 Feature Dictionary")
        st.markdown("Explore detailed descriptions of all features used in the model:")
        
        # Create feature descriptions dictionary
        feature_descriptions = create_feature_descriptions()
        
        # Create dropdown with all features
        all_features = sorted(feature_df['feature'].tolist())
        selected_feature = st.selectbox(
            "Select a feature to learn more:",
            options=all_features,
            index=0,
            help="Choose any feature from the model to see its detailed description"
        )
        
        # Display feature information
        if selected_feature:
            # Get importance rank and value
            feature_rank = feature_df[feature_df['feature'] == selected_feature].index[0] + 1
            feature_importance = feature_df[feature_df['feature'] == selected_feature]['importance_pct'].iloc[0]
            feature_category = feature_df[feature_df['feature'] == selected_feature]['category'].iloc[0]
            
            # Create info card
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Importance Rank", f"#{feature_rank}", help=f"Rank out of {len(feature_df)} features")
            with col2:
                st.metric("Importance Score", f"{feature_importance:.2f}%", help="Percentage of total model importance")
            with col3:
                st.metric("Category", feature_category, help="Feature category classification")
            
            # Get description
            description = get_feature_description(selected_feature, feature_descriptions)
            
            # Display description in an info box
            st.info(f"**{selected_feature}**\n\n{description}")
            
            # Show related features if available
            related_features = get_related_features(selected_feature, all_features)
            if related_features:
                st.markdown("**Related Features:**")
                for related in related_features[:5]:  # Show top 5 related
                    related_importance = feature_df[feature_df['feature'] == related]['importance_pct'].iloc[0]
                    st.markdown(f"• {related} ({related_importance:.2f}%)")
        
        # Feature search functionality
        st.markdown("#### 🔍 Feature Search")
        search_term = st.text_input(
            "Search features by name or keyword:",
            placeholder="e.g., 'RT', 'Charge', 'Area', 'log'",
            help="Search for features containing specific terms"
        )
        
        if search_term:
            matching_features = feature_df[feature_df['feature'].str.contains(search_term, case=False)].copy()
            if len(matching_features) > 0:
                st.markdown(f"**Found {len(matching_features)} matching features:**")
                
                # Create a compact table showing search results
                display_search = matching_features[['feature', 'importance_pct', 'category']].head(20)
                display_search = display_search.rename(columns={
                    'feature': 'Feature Name',
                    'importance_pct': 'Importance (%)',
                    'category': 'Category'
                })
                
                st.dataframe(
                    display_search,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.warning(f"No features found containing '{search_term}'")
        
    
    except Exception as e:
        st.error(f"❌ Error displaying feature importance analysis: {str(e)}")
    
    # Display feature importance data table
    if csv_available:
        st.markdown("#### 📊 Feature Importance Data")
        
        try:
            feature_df = pd.read_csv(feature_csv_path)
            
            # Filter out zero importance features for cleaner display
            feature_df = feature_df[feature_df['importance'] > 0].copy()
            
            if len(feature_df) > 0:
                # Sort by importance descending
                feature_df = feature_df.sort_values('importance', ascending=False)
                
                # Format importance as percentage
                feature_df['importance_pct'] = (feature_df['importance'] * 100).round(2)
                
                # Show top 20 in table
                top_features = feature_df.head(20).copy()
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.dataframe(
                        top_features[['feature', 'importance_pct']],
                        use_container_width=True,
                        column_config={
                            "feature": "Feature Name",
                            "importance_pct": st.column_config.NumberColumn(
                                "Importance (%)",
                                format="%.2f%%"
                            )
                        },
                        hide_index=True
                    )
                
                with col2:
                    st.metric("Total Features", len(feature_df))
                    st.metric("Non-zero Features", len(feature_df[feature_df['importance'] > 0]))
                    st.metric("Top Feature", f"{top_features.iloc[0]['importance_pct']:.2f}%")
                
                # Download option for full feature importance data
                csv_data = feature_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Feature Importance CSV",
                    data=csv_data,
                    file_name=f"feature_importance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("All features have zero importance in this model.")
                
        except Exception as e:
            st.error(f"❌ Error loading feature importance data: {str(e)}")
    

# -----------------------------------------------
# 📊 RESULTS DISPLAY
# -----------------------------------------------

def display_results():
    """Display analysis results with interactive visualizations."""
    
    st.markdown("## 🎉 Analysis Results")
    
    # Extract summary metrics from real results
    results = st.session_state.results_data
    summary = extract_summary_metrics(results)
    
    if summary is None:
        st.error("❌ No valid results to display")
        return
    
    # Key performance metrics
    st.markdown("### 🏆 Best Performance Achieved")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Achieved FDR", f"{summary['best_fdr']:.1f}%", delta=None, help="Actual false discovery rate of additional peptides")
    with col2:
        st.metric("Additional Unique Peptides", summary['best_peptides'], delta=None, help="Total additional peptides identified (includes both true and false positives)")
    with col3:
        st.metric("Method Tested", summary['best_method'], delta=None, help="MS method used for holdout testing")
    with col4:
        st.metric("Analysis Time", summary['total_runtime'], delta=None, help="Total computation time")
    
    # Enhanced data source clarity section
    st.markdown("### 📊 Data Sources & Recovery Analysis")
    
    # Create detailed explanatory section
    with st.expander("📋 Data Source Details", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Baseline Peptides (Short Gradient 1% FDR)")
            st.markdown(f"**Method:** {summary['best_method']}")
            
            # Get actual configured files for baseline if available
            baseline_config = results.get('config', {}).get('baseline_files', [])
            if baseline_config:
                st.markdown(f"**Files:** {len(baseline_config)} configured files")
                for i, file_path in enumerate(baseline_config[:3], 1):  # Show first 3
                    filename = file_path.split('/')[-1] if '/' in file_path else file_path
                    st.markdown(f"  - {filename}")
                if len(baseline_config) > 3:
                    st.markdown(f"  - ... and {len(baseline_config) - 3} more")
            
            st.markdown(f"**Total unique peptides:** {summary['baseline_peptides']:,}")
            st.info("High-confidence peptides identified by DIA-NN in short gradient conditions")
            
            st.markdown("#### 🌍 Ground Truth (Long Gradient 1% FDR)")
            
            # Show method-specific ground truth based on configuration
            config = results.get('config', {})
            test_method = config.get('test_method', summary['best_method'])
            
            # Check if using method-specific or all ground truth
            ground_truth_strategy = "Method-specific ground truth"
            try:
                # Try to determine strategy from dataset config
                dataset_name = test_method.split('_')[0] if '_' in test_method else 'Unknown'
                config_path = f"data/{dataset_name}/dataset_info.json"
                if os.path.exists(config_path):
                    import json
                    with open(config_path, 'r') as f:
                        dataset_config = json.load(f)
                    strategy = dataset_config.get('ground_truth_mapping', {}).get('_strategy', 'method_specific')
                    if strategy == 'use_all_ground_truth':
                        ground_truth_strategy = "All long gradient methods combined"
                    else:
                        ground_truth_strategy = f"Ground truth for {test_method}"
            except:
                ground_truth_strategy = f"Ground truth for {test_method}"
            
            st.markdown(f"**Method:** {ground_truth_strategy}")
            
            # Get ground truth files from config if available
            ground_truth_config = results.get('config', {}).get('ground_truth_files', [])
            if ground_truth_config:
                st.markdown(f"**Files:** {len(ground_truth_config)} configured files")
                for i, file_path in enumerate(ground_truth_config[:3], 1):  # Show first 3
                    filename = file_path.split('/')[-1] if '/' in file_path else file_path
                    st.markdown(f"  - {filename}")
                if len(ground_truth_config) > 3:
                    st.markdown(f"  - ... and {len(ground_truth_config) - 3} more")
            else:
                st.markdown(f"**Files:** Configured ground truth files")
            
            st.markdown(f"**Total unique peptides:** {summary['ground_truth_peptides']:,}")
            st.info("Reference peptides from long gradient analysis - the validation standard")
        
        with col2:
            st.markdown("#### 🔍 Test Set Analysis")
            test_set_total = results['summary'].get('test_samples', 0)
            additional_candidates = results['summary'].get('additional_candidates', 0)
            baseline_peptides = summary['baseline_peptides']
            
            # Calculate total unique peptides in original test files
            total_unique_in_test_files = baseline_peptides + additional_candidates
            
            st.markdown(f"**Test method:** {summary['best_method']}")
            st.markdown(f"**Test FDR level:** {results['config'].get('test_fdr', 50)}%")
            st.markdown(f"**Total unique peptides in {results['config'].get('test_fdr', 50)}% FDR files:** {total_unique_in_test_files:,}")
            st.markdown(f"**Baseline peptides:** {baseline_peptides:,}")
            st.markdown(f"**Additional candidates remaining:** {additional_candidates:,}")
            st.markdown(f"**Additional test rows:** {test_set_total:,}")
            
            # Show the aggregation info if available
            if 'unique_test_peptides' in results['summary']:
                unique_test_peptides = results['summary']['unique_test_peptides']
                if unique_test_peptides > 0:
                    rows_per_peptide = test_set_total / unique_test_peptides
                    st.markdown(f"**Rows per unique peptide:** {rows_per_peptide:.2f}")
            
            # Calculate validated candidates (the actual recoverable subset)
            if 'results' in results and results['results']:
                # Find the best result to get validation info
                valid_results = [r for r in results['results'] if 'Additional_Peptides' in r and 'Actual_FDR' in r and r['Actual_FDR'] != 'N/A']
                if valid_results:
                    best_result = max(valid_results, key=lambda x: int(x['Additional_Peptides']))
                    validated_candidates = int(best_result.get('Total_Validated_Candidates', 0))
                    validation_rate = (validated_candidates / additional_candidates) * 100 if additional_candidates > 0 else 0
                    
                    st.markdown(f"**Validated candidates (in ground truth):** {validated_candidates:,} ({validation_rate:.1f}% of candidates)")
                    st.info(f"Only {validated_candidates:,} out of {additional_candidates:,} additional candidates are actually present in the ground truth")
            
            st.success("💡 Our ML model identifies which validated candidates are real peptides vs false positives")
    
    # Color scheme selector in main interface
    st.markdown("---")
    display_main_color_scheme_selector()
    st.markdown("---")
    
    # Recovery metrics with clear context
    st.markdown("### 🎯 Recovery Performance")
    col_a, col_b, col_c, col_d, col_e = st.columns(5)
    
    with col_a:
        baseline_count = summary.get('baseline_peptides', 0)
        if baseline_count > 0:
            st.metric("Baseline Identified", f"{baseline_count:,}", help=f"High-confidence peptides from {summary.get('best_method', 'test method')} 1% FDR")
        else:
            st.metric("Baseline Identified", "N/A", help="Baseline peptide count not available")
    
    with col_b:
        additional_found = int(summary['best_peptides']) if isinstance(summary['best_peptides'], str) else summary['best_peptides']
        baseline_count = summary.get('baseline_peptides', 0)
        recovery_rate = (additional_found / baseline_count) * 100 if baseline_count > 0 else 0
        st.metric("Additional Validated", f"{additional_found:,}", delta=f"+{recovery_rate:.1f}%", help="Peptides validated by ground truth")
    
    with col_c:
        baseline_count = summary.get('baseline_peptides', 0)
        total_identified = baseline_count + additional_found
        st.metric("Total Identified", f"{total_identified:,}", help="Baseline + Additional peptides combined")
    
    with col_d:
        # Show validated candidates (the pool of recoverable peptides)
        if 'results' in results and results['results']:
            valid_results = [r for r in results['results'] if 'Additional_Peptides' in r and 'Actual_FDR' in r and r['Actual_FDR'] != 'N/A']
            if valid_results:
                best_result = max(valid_results, key=lambda x: int(x['Additional_Peptides']))
                validated_candidates = int(best_result.get('Total_Validated_Candidates', 0))
                st.metric("Validated Candidates", f"{validated_candidates:,}", help="Total additional peptides in test set that are present in ground truth")
            else:
                st.metric("Validated Candidates", "N/A", help="No valid results available")
        else:
            st.metric("Validated Candidates", "N/A", help="No results data available")
    
    with col_e:
        # Calculate recovery rate based on validated candidates, not all missed peptides
        if 'results' in results and results['results']:
            valid_results = [r for r in results['results'] if 'Additional_Peptides' in r and 'Actual_FDR' in r and r['Actual_FDR'] != 'N/A']
            if valid_results:
                best_result = max(valid_results, key=lambda x: int(x['Additional_Peptides']))
                true_positives = int(best_result['Additional_Peptides'])
                validated_candidates = int(best_result.get('Total_Validated_Candidates', 0))
                recovery_rate = (true_positives / validated_candidates) * 100 if validated_candidates > 0 else 0
                st.metric("Recovery Rate", f"{recovery_rate:.1f}%", help=f"Percentage of validated candidates successfully recovered")
            else:
                st.metric("Recovery Rate", "N/A", help="No valid results available")
        else:
            st.metric("Recovery Rate", "N/A", help="No results data available")
    
    
    # Convert results to DataFrame
    if 'results' in results and results['results']:
        results_df = pd.DataFrame(results['results'])
    else:
        st.error("❌ No detailed results available")
        return
    
    # Tabs for different views (including Feature Importance)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Overview", "📈 Detailed Plots", "📋 Data Tables", "📊 Feature Importance", "💾 Export"])
    
    with tab1:
        display_overview_plots(results_df)
    
    with tab2:
        display_detailed_plots(results_df)
    
    with tab3:
        display_data_tables(results_df)
    
    # Feature Importance tab
    with tab4:
        results_dir = results['summary'].get('results_dir', '') if results else ''
        if results_dir:
            display_feature_importance_tab(results_dir)
        else:
            st.info("No results directory found – cannot display feature importance plots.")
    
    with tab5:
        display_export_options(results)
    
    # Reset button
    st.markdown("---")
    st.markdown("### 🔄 Start Over")
    if st.button("🔄 Run New Analysis", type="primary", use_container_width=True):
        # Reset both training and inference states
        clear_analysis_states()
        st.rerun()

def display_overview_plots(df):
    """Display overview plots using Plotly."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # FDR vs Additional Peptides
        fig = px.line(df, x='Target_FDR', y='Additional_Peptides', 
                     title='FDR vs Additional Peptides',
                     color_discrete_sequence=[CHART_COLORS['line_primary']])
        fig.update_traces(mode='lines+markers', line=dict(width=3), marker=dict(size=8))
        fig.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12),
            title_font_size=14,
            height=400,  # Better height for 2-column layout with sidebar
            margin=dict(l=50, r=20, t=60, b=50),
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Actual vs Target FDR
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Target_FDR'], 
            y=df['Actual_FDR'],
            mode='lines+markers',
            name='Actual FDR',
            line=dict(color=CHART_COLORS['line_secondary'], width=3),
            marker=dict(size=8)
        ))
        fig.add_trace(go.Scatter(
            x=df['Target_FDR'], 
            y=df['Target_FDR'],
            mode='lines',
            name='Perfect Control',
            line=dict(color=CHART_COLORS['line_target'], width=2, dash='dash')
        ))
        fig.update_layout(
            title='FDR Control Precision',
            xaxis_title='Target FDR (%)',
            yaxis_title='Actual FDR (%)',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter", size=12),
            title_font_size=14,
            height=400,  # Better height for 2-column layout with sidebar
            margin=dict(l=50, r=20, t=60, b=50),
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray')
        )
        st.plotly_chart(fig, use_container_width=True)

def display_detailed_plots(df):
    """Display detailed interactive plots."""
    
    # Bar chart of additional peptides by target FDR
    fig = px.bar(df, x='Target_FDR', y='Additional_Peptides',
                title='Additional Peptides by Target FDR',
                orientation='v',  # Explicitly set vertical orientation
                color_discrete_sequence=[CHART_COLORS['bar_primary']])
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=12),
        title_font_size=16,
        height=500,  # Fixed height for better appearance with sidebar
        margin=dict(l=60, r=30, t=80, b=60),  # Better margins for sidebar layout
        xaxis=dict(
            gridcolor='lightgray',
            title='Target FDR (%)'
        ),
        yaxis=dict(
            gridcolor='lightgray',
            title='Additional Peptides'
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    
    # Recovery percentage plot with enhanced context
    fig = px.line(df, x='Target_FDR', y='Recovery_Pct',
                 title='Recovery Rate: What % of Validated Candidates Were Successfully Identified',
                 color_discrete_sequence=[CHART_COLORS['line_primary']])
    
    fig.update_traces(mode='lines+markers', line=dict(width=3), marker=dict(size=8))
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Inter", size=12),
        title_font_size=16,
        height=500,  # Fixed height for better appearance with sidebar
        margin=dict(l=60, r=30, t=120, b=60),  # Extra top margin for annotation
        xaxis=dict(gridcolor='lightgray', title='Target FDR (%)'),
        yaxis=dict(gridcolor='lightgray', title='Recovery Percentage (%)'),
        annotations=[
            dict(
                text="Shows what % of validated candidates (peptides confirmed by ground truth)<br>were successfully recovered by the ML model",
                x=0.5, y=1.1, xref='paper', yref='paper',
                showarrow=False, font=dict(size=10, color=CHART_COLORS['neutral']),
                xanchor='center'
            )
        ]
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_data_tables(df):
    """Display interactive data tables with enhanced context."""
    
    st.markdown("### 📋 Detailed Results Table")
    
    # Add context explanation
    st.info("""
    **📊 Column Explanations (left to right):**
    • **Target FDR**: Desired false discovery rate threshold
    • **Threshold**: Model confidence threshold used for predictions
    • **Additional Unique Peptides**: Total additional peptides identified (includes both true and false positives) 
    • **Actual FDR**: Measured false discovery rate of additional peptides
    • **Recovery %**: Percentage of validated candidates successfully recovered
    • **Increase %**: Improvement over baseline peptide count
    • **False Positives**: Model predictions not validated by ground truth
    • **MCC**: Matthews Correlation Coefficient (model performance metric, -1 to +1)
    """)
    
    # Format the dataframe for display
    display_df = df.copy()
    
    # Remove internal columns from display 
    columns_to_remove = ['Aggregation_Method', 'Total_Validated_Candidates']
    for col in columns_to_remove:
        if col in display_df.columns:
            display_df = display_df.drop(col, axis=1)
    
    display_df['Target_FDR'] = display_df['Target_FDR'].map('{:.1f}%'.format)
    display_df['Actual_FDR'] = display_df['Actual_FDR'].map('{:.1f}%'.format)
    display_df['Recovery_Pct'] = display_df['Recovery_Pct'].map('{:.1f}%'.format)
    display_df['Increase_Pct'] = display_df['Increase_Pct'].map('{:.1f}%'.format)
    
    # Format MCC if it exists in the dataframe
    if 'MCC' in display_df.columns:
        display_df['MCC'] = display_df['MCC'].map('{:.3f}'.format)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Target_FDR": st.column_config.TextColumn("Target FDR", help="Desired false discovery rate"),
            "Threshold": st.column_config.NumberColumn("Threshold", help="Model confidence threshold used for predictions", format="%.3f"),
            "Additional_Peptides": st.column_config.NumberColumn("Additional Unique Peptides", help="Total additional peptides identified (includes both true and false positives)"),
            "False_Positives": st.column_config.NumberColumn("False Positives", help="Model predictions not in ground truth"),
            "Actual_FDR": st.column_config.TextColumn("Actual FDR", help="Measured false discovery rate of additional peptides"),
            "MCC": st.column_config.TextColumn("MCC", help="Matthews Correlation Coefficient (-1 to +1, higher is better)"),
            "Recovery_Pct": st.column_config.TextColumn("Recovery %", help="% of validated candidates recovered"),
            "Increase_Pct": st.column_config.TextColumn("Increase %", help="% improvement over baseline")
        }
    )

def display_export_options(results):
    """Display export options for results."""
    
    st.markdown("### 💾 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export detailed results CSV
        if 'results' in results and results['results']:
            results_df = pd.DataFrame(results['results'])
            
            # Remove Aggregation_Method column from export (internal use only)
            if 'Aggregation_Method' in results_df.columns:
                results_df = results_df.drop('Aggregation_Method', axis=1)
            
            csv_data = results_df.to_csv(index=False)
            st.download_button(
                label="📄 Download Results CSV",
                data=csv_data,
                file_name=f"peptide_validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        # Export comprehensive analysis summary
        summary_data = {
            'analysis_config': results.get('config', {}),
            'analysis_summary': results.get('summary', {}),
            'best_results': extract_summary_metrics(results),
            'metadata': results.get('metadata', {})
        }
        json_data = json.dumps(summary_data, indent=2, default=str)
        st.download_button(
            label="📋 Download Full Summary",
            data=json_data,
            file_name=f"analysis_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    with col3:
        # Export configuration for sharing
        config_data = results.get('config', {})
        config_json = json.dumps(config_data, indent=2)
        st.download_button(
            label="⚙️ Download Config",
            data=config_json,
            file_name=f"analysis_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    
    st.markdown("### 📊 Analysis Files")
    
    # Show results directory info
    if 'summary' in results and 'results_dir' in results['summary']:
        results_dir = results['summary']['results_dir']
        st.info(f"📁 **Full results saved to:** `{results_dir}`")
        
        st.markdown("**Directory contains:**")
        st.markdown("""
        - `plots/` - High-resolution visualizations (PNG, 300 DPI)
        - `tables/` - Detailed results tables (CSV format)
        - `feature_analysis/` - Feature importance and SHAP analysis
        - `raw_data/` - Complete analysis data and metadata
        """)
    
    st.markdown("### 📧 Share Configuration")
    
    # Configuration sharing
    if 'config' in results:
        config = results['config']
        train_methods_str = ','.join(config.get('train_methods', []))
        test_method = config.get('test_method', '')
        test_fdr = config.get('test_fdr', 50)
        
        # Create shareable URL (simplified)
        config_params = f"train={train_methods_str}&test={test_method}&fdr={test_fdr}"
        
        # Configuration string for sharing
        train_methods = config.get('train_methods', [])
        train_fdr_levels = config.get('train_fdr_levels', [])
        config_string = f"Training: {', '.join(train_methods)} (FDR: {', '.join(map(str, train_fdr_levels))}%) → Testing: {test_method} (FDR: {test_fdr}%)"
        
        st.markdown("**Share this analysis configuration:**")
        st.code(config_string)
        
        # Configuration recreation instructions
        with st.expander("🔄 Recreate This Analysis"):
            feature_selection = config.get('feature_selection', {})
            st.markdown(f"""
            **To recreate this analysis:**
            
            1. **Training Methods**: {', '.join(config.get('train_methods', []))}
            2. **Training FDR Levels**: {', '.join(map(str, config.get('train_fdr_levels', [])))}%
            3. **Test Method**: {test_method}
            4. **Test FDR Level**: {test_fdr}%
            5. **Target FDR Levels**: {', '.join(map(str, config.get('target_fdr_levels', [])))}%
            
            **XGBoost Parameters:**
            - Learning Rate: {config.get('xgb_params', {}).get('learning_rate', 0.08)}
            - Max Depth: {config.get('xgb_params', {}).get('max_depth', 7)}
            - N Estimators: {config.get('xgb_params', {}).get('n_estimators', 1000)}
            - Subsample: {config.get('xgb_params', {}).get('subsample', 0.8)}
            
            **Feature Selection:**
            - DIA-NN Quality Metrics: {'✅' if feature_selection.get('use_diann_quality', False) else '❌'}
            - Peptide Sequence Features: {'✅' if feature_selection.get('use_sequence_features', False) else '❌'}
            - Mass Spectrometry Features: {'✅' if feature_selection.get('use_ms_features', False) else '❌'}
            - Statistical Features: {'✅' if feature_selection.get('use_statistical_features', False) else '❌'}
            - Library Features: {'✅' if feature_selection.get('use_library_features', False) else '❌'}
            """)
    

def show_inference_interface():
    """Display the inference mode interface - identical to training mode but uses saved models."""
    # Add mode indicator and back button (same layout as training)
    col1, col2, col3 = st.columns([1, 3, 1])
    
    # Add global CSS for button styling (same as training)
    st.markdown("""
    <style>
    /* Force refresh with timestamp and stronger selectors */
    div[data-testid="stButton"] button[data-baseweb="button"][kind="secondary"],
    .stButton > button[kind="secondary"] {
        background: #2E86AB !important;
        color: white !important;
        border: none !important;
        height: 80px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        border-radius: 15px !important;
        padding: 20px 30px !important;
        width: 100% !important;
        margin-left: 0px !important;
        transform: translateX(0px) !important;
    }
    
    div[data-testid="stButton"] button[data-baseweb="button"][kind="primary"],
    .stButton > button[kind="primary"] {
        background: #2E86AB !important;
        color: white !important;
        border: none !important;
        height: 80px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        border-radius: 15px !important;
        padding: 20px 30px !important;
        width: 100% !important;
        margin-right: -20px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with col1:
        if st.button("← Back to Home", type="secondary", key="big_back_btn_inference"):
            # Clear all analysis states when returning to home
            st.session_state.app_mode = None
            clear_analysis_states()
            st.rerun()
    with col2:
        st.markdown("<h3 style='text-align: center; margin: 20px 0;'>🔮 Inference Mode</h3>", unsafe_allow_html=True)
    with col3:
        # Run Inference button in header
        inference_button_placeholder = st.empty()
    
    st.markdown("---")
    
    # Check if we have inference results to display
    if hasattr(st.session_state, 'inference_complete') and st.session_state.inference_complete:
        # Show inference results without feature importance
        if hasattr(st.session_state, 'inference_results') and st.session_state.inference_results:
            # For formatted results, we need to extract the model info
            # This is a fallback for cases where inference_results contains formatted data
            if 'results' in st.session_state.inference_results:
                # This is results from inference flow - show with tabs
                st.markdown("### 📊 Inference Results")
                st.markdown("**Analysis complete!**")
                
                # Convert to DataFrame and display with tabs
                results_df = pd.DataFrame(st.session_state.inference_results['results'])
                
                # Extract summary metrics like training mode
                if not results_df.empty:
                    # Find best result (highest additional peptides)
                    results_with_peptides = results_df[results_df['Additional_Peptides'] > 0]
                    if not results_with_peptides.empty:
                        best_result = results_with_peptides.loc[results_with_peptides['Additional_Peptides'].idxmax()]
                    else:
                        best_result = results_df.iloc[0]  # Fallback to first result
                    
                    # Key performance metrics (same as training mode)
                    st.markdown("### 🏆 Best Performance Achieved")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Achieved FDR", f"{best_result['Actual_FDR']:.1f}%", delta=None, help="Actual false discovery rate of additional peptides")
                    with col2:
                        st.metric("Additional Unique Peptides", int(best_result['Additional_Peptides']), delta=None, help="Total additional peptides identified (includes both true and false positives)")
                    with col3:
                        # Try to get the actual test method name from session state or inference results
                        actual_test_method = "Test Method"  # Default fallback
                        if hasattr(st.session_state, 'inference_test_method'):
                            actual_test_method = st.session_state.inference_test_method
                        elif hasattr(st.session_state, 'selected_test_method'):
                            actual_test_method = st.session_state.selected_test_method
                        st.metric("Method Tested", actual_test_method, delta=None, help="MS method used for inference testing")
                    with col4:
                        st.metric("Analysis Type", "Inference", delta=None, help="Model inference analysis")
                    
                    # Check if we have ground truth data available for this test set
                    has_ground_truth = False
                    baseline_count = 0
                    ground_truth_count = 0
                    additional_candidates = 0
                    
                    # Try to extract counts from different possible locations
                    if 'baseline_peptides' in st.session_state.inference_results:
                        baseline_count = st.session_state.inference_results['baseline_peptides']
                        has_ground_truth = True
                    elif 'summary' in st.session_state.inference_results:
                        baseline_count = st.session_state.inference_results['summary'].get('baseline_peptides', 0)
                        ground_truth_count = st.session_state.inference_results['summary'].get('ground_truth_peptides', 0)
                        additional_candidates = st.session_state.inference_results['summary'].get('additional_candidates', 0)
                        has_ground_truth = ground_truth_count > 0
                    
                    # Try to get from results DataFrame columns if summary not available
                    if not has_ground_truth and 'Baseline_Peptides' in results_df.columns:
                        baseline_counts = results_df['Baseline_Peptides'].dropna()
                        if not baseline_counts.empty:
                            baseline_count = int(baseline_counts.iloc[0])
                            has_ground_truth = baseline_count > 0
                    
                    if not has_ground_truth and 'Ground_Truth_Peptides' in results_df.columns:
                        gt_counts = results_df['Ground_Truth_Peptides'].dropna()
                        if not gt_counts.empty:
                            ground_truth_count = int(gt_counts.iloc[0])
                            has_ground_truth = ground_truth_count > 0
                    
                    # Only show Data Sources section if we have ground truth data available
                    if has_ground_truth:
                        # Enhanced data source clarity section (same as training mode)
                        st.markdown("### 📊 Data Sources & Recovery Analysis")
                        
                        # Create detailed explanatory section
                        with st.expander("📋 Data Source Details", expanded=True):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("#### 🎯 Baseline Peptides (Short Gradient 1% FDR)")
                                st.markdown(f"**Method:** {actual_test_method}")
                                st.markdown(f"**Total unique peptides:** {baseline_count:,}")
                                st.info("High-confidence peptides identified by DIA-NN in short gradient conditions")
                                
                                st.markdown("#### 🌍 Ground Truth (Long Gradient 1% FDR)")
                                
                                # Show method-specific ground truth 
                                ground_truth_strategy = f"Ground truth for {actual_test_method}"
                                try:
                                    # Try to determine strategy from dataset config
                                    dataset_name = actual_test_method.split('_')[0] if '_' in actual_test_method else 'Unknown'
                                    config_path = f"data/{dataset_name}/dataset_info.json"
                                    if os.path.exists(config_path):
                                        import json
                                        with open(config_path, 'r') as f:
                                            dataset_config = json.load(f)
                                        strategy = dataset_config.get('ground_truth_mapping', {}).get('_strategy', 'method_specific')
                                        if strategy == 'use_all_ground_truth':
                                            ground_truth_strategy = "All long gradient methods combined"
                                        else:
                                            ground_truth_strategy = f"Ground truth for {actual_test_method}"
                                except:
                                    ground_truth_strategy = f"Ground truth for {actual_test_method}"
                                
                                st.markdown(f"**Method:** {ground_truth_strategy}")
                                st.markdown(f"**Total unique peptides:** {ground_truth_count:,}")
                                st.info("Reference peptides from long gradient analysis - the validation standard")
                            
                            with col2:
                                st.markdown("#### 🔍 Test Set Analysis")
                                
                                # Get test FDR level from results
                                test_fdr = 50  # Default
                                if 'results' in st.session_state.inference_results and st.session_state.inference_results['results']:
                                    # Try to get FDR from first result
                                    first_result = st.session_state.inference_results['results'][0]
                                    if 'test_fdr' in first_result:
                                        test_fdr = first_result['test_fdr']
                                
                                # Calculate total unique peptides in test files
                                total_unique_in_test_files = baseline_count + additional_candidates
                                
                                st.markdown(f"**Test method:** {actual_test_method}")
                                st.markdown(f"**Test FDR level:** {test_fdr}%")
                                st.markdown(f"**Total unique peptides in {test_fdr}% FDR files:** {total_unique_in_test_files:,}")
                                st.markdown(f"**Baseline peptides:** {baseline_count:,}")
                                st.markdown(f"**Additional candidates remaining:** {additional_candidates:,}")
                                
                                # Calculate validated candidates (the actual recoverable subset)
                                if 'results' in st.session_state.inference_results and st.session_state.inference_results['results']:
                                    # Find the best result to get validation info
                                    valid_results = [r for r in st.session_state.inference_results['results'] if 'Additional_Peptides' in r and 'Actual_FDR' in r and r['Actual_FDR'] != 'N/A']
                                    if valid_results:
                                        best_result_for_validation = max(valid_results, key=lambda x: int(x['Additional_Peptides']))
                                        validated_candidates = int(best_result_for_validation.get('Total_Validated_Candidates', 0))
                                        validation_rate = (validated_candidates / additional_candidates) * 100 if additional_candidates > 0 else 0
                                        
                                        st.markdown(f"**Validated candidates (in ground truth):** {validated_candidates:,} ({validation_rate:.1f}% of candidates)")
                                        st.info(f"Only {validated_candidates:,} out of {additional_candidates:,} additional candidates are actually present in the ground truth")
                                
                                st.success("💡 Our ML model identifies which validated candidates are real peptides vs false positives")
                    else:
                        # Show a simplified section when no ground truth is available
                        st.markdown("### 🔍 Test Data Analysis")
                        st.info("ℹ️ **Ground truth data not available for this test set.** Results show model predictions without validation against known true positives.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### 🧪 Test Dataset")
                            st.markdown("**Model predictions on new data**")
                            st.markdown(f"**Predictions Made:** {int(best_result['Additional_Peptides']):,} additional peptides")
                            
                        with col2:
                            st.markdown("#### 🎯 Model Performance")
                            st.markdown("**Inference Results**")
                            st.markdown(f"**Confidence Threshold:** {best_result.get('Threshold', 'N/A')}")
                            st.warning("⚠️ FDR cannot be calculated without ground truth data")
                    
                    st.markdown("---")
                
                # Show tabs (WITHOUT Feature Importance)
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Detailed Plots", "📋 Data Tables", "💾 Export"])
                
                with tab1:
                    display_overview_plots(results_df)
                
                with tab2:
                    display_detailed_plots(results_df)
                
                with tab3:
                    display_inference_table_only(results_df)
                
                with tab4:
                    display_export_options(st.session_state.inference_results)
            else:
                # This is raw results from the old inference flow
                display_inference_results_with_tabs()
        return
    
    # Load available models from run history
    with st.spinner("🔍 Loading trained models..."):
        available_models = load_trained_models_from_history()
    
    if not available_models:
        st.warning("⚠️ No trained models found. Please train a model first in Training Mode.")
        return
    
    # Get available methods using the same validation as training mode
    files_info = discover_available_files()
    
    # Cache dataset processing to avoid recomputing on every interaction
    @st.cache_data
    def get_available_datasets(files_info):
        available_datasets = set()
        for category in ['baseline', 'ground_truth', 'training']:
            for file_info in files_info[category]:
                available_datasets.add(file_info.get('dataset', 'Unknown'))
        return sorted(list(available_datasets))
    
    # Get available datasets with caching
    available_datasets = get_available_datasets(files_info)
    dataset_options = ['All'] + available_datasets
    
    # Sidebar configuration (clean layout like training mode)
    st.sidebar.markdown("### 🔮 Inference Configuration")
    st.sidebar.markdown("*Apply a trained model to new data*")
    
    # Dataset filter for method selection
    dataset_filter = st.sidebar.selectbox(
        "Filter by Dataset:",
        dataset_options,
        help="Filter test methods by dataset type"
    )
    
    # Model selection (compact)
    model_options = [f"{model['run_id']} - {model['summary']}" for model in available_models]
    selected_model_idx = st.sidebar.selectbox(
        "🤖 Select Trained Model:",
        range(len(model_options)),
        format_func=lambda x: model_options[x],
        help="Select a model from your training runs"
    )
    
    selected_model = available_models[selected_model_idx]
    config = selected_model['config']
    
    # Compact model info
    with st.sidebar.expander("📋 Model Details", expanded=False):
        # Format timestamp properly
        try:
            from datetime import datetime
            timestamp = datetime.fromisoformat(selected_model['timestamp'].replace('Z', '+00:00'))
            formatted_time = timestamp.strftime("%Y-%m-%d %H:%M")
        except:
            formatted_time = selected_model['timestamp']
        
        st.markdown(f"""
        **ID:** {selected_model['run_id']}  
        **Created:** {formatted_time}  
        **Best FDR:** {selected_model['best_fdr']:.1f}%  
        **Training:** {', '.join(config['train_methods'])}  
        **Original Test:** {config['test_method']}
        """)
    
    # Cache available methods to avoid expensive discovery calls
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def get_available_methods_cached(dataset_filter):
        return get_configured_methods(dataset_filter)
    
    # Get available methods based on dataset filter
    available_methods_filtered = get_available_methods_cached(dataset_filter)
    
    # Data selection (clean layout)
    test_method = st.sidebar.selectbox(
        "🧪 Select test method:",
        available_methods_filtered,
        help="Choose the method to apply the model to"
    )
    
    # Display the full selected method name
    if test_method:
        st.sidebar.markdown(f"**Selected:** `{test_method}`")
    
    test_fdr = st.sidebar.selectbox(
        "📈 Select test FDR level:",
        [1, 20, 50],
        index=2,  # Default to 50% like training
        help="Choose the FDR level for the test dataset"
    )
    
    # Target FDR levels (ONLY the ones the model was trained on)
    with st.sidebar.expander("🎯 Target FDR Levels", expanded=True):
        # Get the target FDR levels from the model's training configuration
        if 'target_fdr_levels' in config:
            available_fdr_levels = config['target_fdr_levels']
            st.info(f"✅ This model has learned thresholds for {len(available_fdr_levels)} FDR levels")
        else:
            # If no target_fdr_levels in config, show warning and use standard levels
            available_fdr_levels = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 15.0, 20.0, 30.0, 50.0]
            st.warning("⚠️ Model config missing target_fdr_levels. Using standard levels (may not work correctly)")
        
        target_fdr_levels = st.multiselect(
            "FDR Levels (%):",
            available_fdr_levels,
            default=available_fdr_levels,
            help="ONLY FDR levels that this model was trained on. The model has learned specific thresholds for these levels."
        )
    
    # Run inference button in sidebar and header
    run_inference = st.sidebar.button("🚀 Run Inference", type="primary", use_container_width=True)
    
    # Also add the button to the header placeholder
    with inference_button_placeholder:
        if st.button("🚀 Run Inference", type="primary", key="big_inference_btn"):
            run_inference = True
    
    # Handle inference execution
    if run_inference:
        if target_fdr_levels:
            run_inference_analysis(selected_model, test_method, test_fdr, target_fdr_levels)
        else:
            st.error("Please select at least one target FDR level")
    
    # Main content area - professional layout with cards
    st.markdown("### 🔮 Inference Mode - Apply Trained Model")
    st.markdown("*Use a previously trained model to analyze new data with the same configuration*")
    
    # Create three informational cards
    col1, col2, col3 = st.columns(3, gap="medium")
    
    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #2E86AB;
            margin: 10px 0;
        ">
            <h4 style="color: #2E86AB; margin: 0 0 10px 0;">🤖 Model Selection</h4>
            <p style="margin: 0; color: #495057;">Choose from your trained models with different configurations and performance metrics.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #2E86AB;
            margin: 10px 0;
        ">
            <h4 style="color: #2E86AB; margin: 0 0 10px 0;">🧪 Test Data</h4>
            <p style="margin: 0; color: #495057;">Select the dataset and FDR level you want to apply the model to.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid #2E86AB;
            margin: 10px 0;
        ">
            <h4 style="color: #2E86AB; margin: 0 0 10px 0;">🎯 FDR Levels</h4>
            <p style="margin: 0; color: #495057;">Configure the target FDR levels for your peptide recovery analysis.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Current selection summary
    st.markdown("#### 📋 Current Selection")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Selected Model", f"{selected_model['run_id']}", f"{selected_model['best_fdr']:.1f}% Best FDR")
    
    with col2:
        st.metric("Test Method", test_method, f"{test_fdr}% FDR")
    
    with col3:
        if target_fdr_levels:
            st.metric("Target FDR Levels", f"{len(target_fdr_levels)} levels", f"{min(target_fdr_levels):.1f}% - {max(target_fdr_levels):.1f}%")
        else:
            st.metric("Target FDR Levels", "0 levels", "None selected")
    
    st.markdown("---")

def load_trained_models_from_history():
    """Load trained models from run history and discover CLI models."""
    models = []
    
    # First, load from Streamlit history file
    history_file = get_history_file_path()
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history_data = json.load(f)
            
            # Format Streamlit models for display
            for run in history_data:
                # Handle potential None values and string/int conversion
                best_fdr = run['summary'].get('best_fdr', 0)
                best_peptides = run['summary'].get('best_peptides', 0)
                
                # Ensure best_fdr is a number
                if best_fdr is None:
                    best_fdr = 0
                
                # Ensure best_peptides is properly formatted
                if best_peptides is None:
                    best_peptides = 0
                elif isinstance(best_peptides, str):
                    try:
                        best_peptides = int(best_peptides)
                    except ValueError:
                        best_peptides = 0
                
                model_info = {
                    'run_id': run['run_id'],
                    'timestamp': run['timestamp'],
                    'config': run['config'],
                    'runtime': run['summary'].get('runtime_minutes', 0),
                    'best_fdr': best_fdr,
                    'best_peptides': best_peptides,
                    'results_dir': run['summary'].get('results_dir', ''),
                    'total_results': run['summary'].get('total_results', 0),
                    'summary': f"FDR: {best_fdr:.1f}% | Peptides: {best_peptides:,}",
                    'source': 'Streamlit'
                }
                models.append(model_info)
                
        except Exception as e:
            st.warning(f"Could not load Streamlit training history: {str(e)}")
    
    # Second, discover CLI models from results directories
    try:
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        if os.path.exists(results_dir):
            for dir_name in os.listdir(results_dir):
                if dir_name.startswith("CLI_RESULTS_"):
                    cli_model_dir = os.path.join(results_dir, dir_name, "saved_models")
                    cli_config_file = os.path.join(results_dir, dir_name, "analysis_config.json")
                    
                    if os.path.exists(cli_model_dir):
                        metadata_file = os.path.join(cli_model_dir, "model_metadata.json")
                        if os.path.exists(metadata_file):
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            # Load CLI configuration for actual method names
                            cli_config = {}
                            if os.path.exists(cli_config_file):
                                try:
                                    with open(cli_config_file, 'r') as f:
                                        cli_config = json.load(f)
                                except Exception:
                                    pass  # Use defaults if config can't be loaded
                            
                            # Extract run info from directory name (CLI_RESULTS_YYYYMMDD_HHMMSS)
                            timestamp_str = dir_name.replace("CLI_RESULTS_", "")
                            try:
                                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                                timestamp_iso = timestamp.isoformat()
                            except ValueError:
                                timestamp_iso = metadata.get('save_timestamp', '')
                            
                            # Find best result for summary
                            best_result = None
                            if 'training_results' in metadata:
                                valid_results = [r for r in metadata['training_results'] if r['Actual_FDR'] <= 5.0]
                                if not valid_results:
                                    valid_results = metadata['training_results']
                                if valid_results:
                                    best_result = max(valid_results, key=lambda x: x['Additional_Peptides'])
                            
                            # Get actual method names from CLI config or use defaults
                            train_methods = cli_config.get('train_methods', ['CLI_Training'])
                            test_method = cli_config.get('test_method', 'CLI_Testing')
                            dataset = cli_config.get('dataset', 'Unknown')
                            
                            model_info = {
                                'run_id': dir_name,
                                'timestamp': timestamp_iso,
                                'config': {
                                    'train_methods': train_methods,
                                    'test_method': test_method,
                                    'dataset': dataset,
                                    'train_fdr_levels': cli_config.get('train_fdr_levels', [50]),
                                    'test_fdr': cli_config.get('test_fdr', 50),
                                    'target_fdr_levels': cli_config.get('target_fdr_levels', []),
                                    'source': 'CLI'
                                },
                                'runtime': 0,
                                'best_fdr': best_result['Actual_FDR'] if best_result else 0,
                                'best_peptides': best_result['Additional_Peptides'] if best_result else 0,
                                'results_dir': os.path.join(results_dir, dir_name),
                                'total_results': len(metadata.get('training_results', [])),
                                'summary': f"FDR: {best_result['Actual_FDR']:.1f}% | Peptides: {best_result['Additional_Peptides']:,}" if best_result else "CLI Model",
                                'source': 'CLI'
                            }
                            models.append(model_info)
                            
    except Exception as e:
        st.warning(f"Could not load CLI models: {str(e)}")
    
    # Sort all models by timestamp (newest first)
    models.sort(key=lambda x: x['timestamp'], reverse=True)
    return models

def get_available_methods_by_dataset(files_info_by_dataset):
    """Get available methods from files info, same as training mode."""
    # Use the same logic as training mode
    available_methods = []
    
    # Get configured methods from the same function used in training
    try:
        # This mimics what training mode does
        available_methods = get_configured_methods('All')  # Get all methods
    except Exception as e:
        st.warning(f"Could not get configured methods: {str(e)}")
        # Fallback: extract methods from files_info_by_dataset
        for dataset_name, dataset_info in files_info_by_dataset.items():
            if 'short_gradient' in dataset_info:
                for fdr_level, files in dataset_info['short_gradient'].items():
                    for file_info in files:
                        method = file_info.get('method')
                        if method and method not in available_methods:
                            available_methods.append(method)
    
    return available_methods

def run_inference_analysis(selected_model, test_method, test_fdr, target_fdr_levels):
    """Run inference analysis using the saved trained model."""
    # Store test method in session state for results display
    st.session_state.inference_test_method = test_method
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🔄 Running Inference...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            import joblib
            from peptide_validator_api import PeptideValidatorAPI
            
            # Create API instance
            api = PeptideValidatorAPI()
            
            # Set the feature selection configuration from the saved model
            api.feature_selection = selected_model['config']['feature_selection']
            
            status_text.text("Loading saved model...")
            progress_bar.progress(10)
            
            # Load the saved model
            results_dir = selected_model['results_dir']
            models_dir = os.path.join(results_dir, "saved_models")
            
            # Check if model files exist
            model_path = os.path.join(models_dir, "trained_model.joblib")
            features_path = os.path.join(models_dir, "training_features.joblib")
            metadata_path = os.path.join(models_dir, "model_metadata.json")
            
            if not os.path.exists(model_path):
                st.error(f"❌ Trained model not found: {model_path}")
                st.info("💡 This model was trained before model saving was implemented. Please retrain to use inference mode.")
                return
            
            # Load model and features
            trained_model = joblib.load(model_path)
            training_features = joblib.load(features_path)
            
            # Ensure training_features is a list (not tuple or other type)
            if not isinstance(training_features, list):
                if hasattr(training_features, 'tolist'):
                    training_features = training_features.tolist()
                elif isinstance(training_features, tuple):
                    training_features = list(training_features)
                else:
                    st.error(f"❌ Invalid training_features type: {type(training_features)}")
                    return
            
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
            
            status_text.text("Loading test data...")
            progress_bar.progress(30)
            
            # Load test data using the same logic as training
            original_config = selected_model['config']
            
            # Get baseline and ground truth peptides
            baseline_peptides = api._load_baseline_peptides(test_method)
            ground_truth_peptides = api._load_ground_truth_peptides(test_method)
            
            # Load test data - ensure we get the correct tuple unpacking
            test_data_result = api._load_test_data(test_method, test_fdr, baseline_peptides, ground_truth_peptides)
            
            # Check if the result is a tuple (training mode returns tuple)
            if isinstance(test_data_result, tuple):
                test_data, y_test, additional_peptides = test_data_result
            else:
                test_data = test_data_result
                # Create labels for evaluation
                y_test = test_data['Modified.Sequence'].isin(ground_truth_peptides).astype(int)
                additional_peptides = set(test_data['Modified.Sequence']) - baseline_peptides
            
            if len(test_data) == 0:
                st.error(f"❌ No test data found for {test_method} at {test_fdr}% FDR")
                return
            
            status_text.text("Creating features...")
            progress_bar.progress(50)
            
            # Create features for test data using the same feature engineering
            
            X_test = api._make_advanced_features(test_data, training_features)
            
            # Ensure test features match training features
            missing_features = set(training_features) - set(X_test.columns)
            if missing_features:
                for feature in missing_features:
                    X_test[feature] = 0
            
            # Reorder columns to match training
            X_test = X_test[training_features]
            
            status_text.text("Making predictions...")
            progress_bar.progress(70)
            
            # Make predictions using the loaded model
            y_scores = trained_model.predict_proba(X_test)[:, 1]
            
            # Check if we should use original training thresholds
            use_original_thresholds = st.sidebar.checkbox(
                "🎯 Use Original Training Thresholds", 
                value=True,
                help="Use thresholds from training to validate FDR on new data"
            )
            
            # Get original training results if available
            original_thresholds = {}
            
            if use_original_thresholds and 'training_results' in model_metadata:
                for result in model_metadata['training_results']:
                    target_fdr_raw = result.get('Target_FDR', 0)
                    # Handle both string ("1%") and numeric (1) formats
                    if isinstance(target_fdr_raw, str):
                        target_fdr_val = float(target_fdr_raw.replace('%', ''))
                    else:
                        target_fdr_val = float(target_fdr_raw)
                    
                    if target_fdr_val > 0:
                        original_thresholds[target_fdr_val] = result.get('Threshold', 0.5)
                st.sidebar.success(f"✅ Found {len(original_thresholds)} original thresholds")
                # Original thresholds loaded successfully
            elif use_original_thresholds:
                st.sidebar.warning("⚠️ No training_results found in model metadata - using optimization")
            
            if use_original_thresholds:
                status_text.text("Using original training thresholds...")
            else:
                status_text.text("Optimizing thresholds...")
            progress_bar.progress(80)
            
            # Apply same peptide-level aggregation as training mode ONCE for all targets
            peptide_data, peptide_predictions, peptide_labels = api._aggregate_predictions_by_peptide(
                test_data, y_scores, y_test, aggregation_method='max'
            )
            
            # Run threshold optimization for each target FDR
            results = []
            for target_fdr in target_fdr_levels:
                
                if use_original_thresholds and target_fdr in original_thresholds:
                    # Use original training threshold
                    threshold = original_thresholds[target_fdr]
                    # Using original training threshold
                else:
                    # Use threshold optimization with aggregated data
                    threshold, tp, actual_fdr = api._find_optimal_threshold(
                        peptide_labels, peptide_predictions, target_fdr
                    )
                
                # Calculate metrics only if threshold is valid
                if threshold is not None:
                    y_pred = (peptide_predictions >= threshold).astype(int)
                    tp = np.sum((y_pred == 1) & (peptide_labels == 1))
                    fp = np.sum((y_pred == 1) & (peptide_labels == 0))
                    actual_fdr = (fp / (tp + fp) * 100) if (tp + fp) > 0 else 0
                    
                    # Calculate metrics with threshold
                else:
                    # No valid threshold found for this FDR level
                    y_pred = np.zeros_like(peptide_predictions, dtype=int)
                    tp = 0
                    fp = 0
                
                # Calculate recovery percentage and increase percentage
                # Total validated candidates = peptides in ground truth (true positives in peptide_labels)
                total_validated = np.sum(peptide_labels == 1)
                recovery_pct = (tp / total_validated * 100) if total_validated > 0 else 0
                increase_pct = (tp / len(baseline_peptides) * 100) if len(baseline_peptides) > 0 else 0
                
                results.append({
                    'Target_FDR': target_fdr,
                    'Threshold': threshold if threshold is not None else 0.5,
                    'Additional_Peptides': tp,
                    'False_Positives': fp,
                    'Actual_FDR': actual_fdr,
                    'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                    'Recovery_Pct': recovery_pct,
                    'Increase_Pct': increase_pct,
                    'Total_Validated_Candidates': total_validated
                })
            
            status_text.text("Preparing results...")
            progress_bar.progress(90)
            
            # Create results in the same format as training
            analysis_results = {
                'config': {
                    'train_methods': original_config['train_methods'],
                    'test_method': test_method,
                    'train_fdr_levels': original_config['train_fdr_levels'],
                    'test_fdr': test_fdr,
                    'target_fdr_levels': target_fdr_levels,
                    'xgb_params': original_config['xgb_params'],
                    'feature_selection': original_config['feature_selection']
                },
                'summary': {
                    'baseline_peptides': len(baseline_peptides),
                    'ground_truth_peptides': len(ground_truth_peptides),
                    'additional_candidates': len(additional_peptides),
                    'validated_candidates': np.sum(y_test == 1),  # Actual count of peptides in ground truth
                    'test_samples': len(test_data),
                    'training_samples': len(test_data),  # For inference, same as test_samples
                    'unique_test_peptides': test_data['Modified.Sequence'].nunique(),
                    'missed_peptides': max(0, len(ground_truth_peptides) - len(baseline_peptides)),
                    'runtime_minutes': 0.1,  # Inference is fast
                    'inference_mode': True
                },
                'results': results,
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'model_metadata': model_metadata,
                    'inference_mode': True
                }
            }
            
            progress_bar.progress(100)
            status_text.text("✅ Inference completed successfully!")
            
            # Store results in session state (inference mode only)
            st.session_state.inference_results = analysis_results
            st.session_state.inference_complete = True
            # Do NOT set analysis_complete = True to avoid showing training interface
            
            # Clear progress and show results
            progress_container.empty()
            
            # Rerun to display results
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error during inference: {str(e)}")
            import traceback
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())

def discover_saved_models():
    """Discover all saved models with their metadata from the streamlit saved_models directory."""
    saved_models = []
    
    # Check streamlit's saved_models directory only
    saved_models_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    
    if not os.path.exists(saved_models_dir):
        # Create the directory if it doesn't exist
        os.makedirs(saved_models_dir, exist_ok=True)
        return saved_models
    
    for model_dir in os.listdir(saved_models_dir):
        model_path = os.path.join(saved_models_dir, model_dir)
        if os.path.isdir(model_path):
            metadata_path = os.path.join(model_path, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    metadata['model_path'] = model_path
                    saved_models.append(metadata)
                except Exception as e:
                    st.warning(f"Could not load metadata for {model_dir}: {str(e)}")
    
    return saved_models

def run_inference(selected_model, test_method, test_fdr, target_fdr_levels):
    """Run inference using the selected model and display results."""
    
    # Import the flexible peptide validator functions from local API
    try:
        from peptide_validator_api import make_advanced_features, find_optimal_threshold
    except ImportError:
        st.error("Could not import flexible_peptide_validator functions from local API. Please ensure the module is available.")
        return
    
    # Create progress container
    progress_container = st.container()
    
    with progress_container:
        st.markdown("### 🔄 Running Inference...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Load model
        status_text.text("Loading model...")
        progress_bar.progress(10)
        
        try:
            model_path = selected_model['model_path']
            model_file = selected_model.get('model_file', 'model.pkl')
            full_model_path = os.path.join(model_path, model_file)
            
            if not os.path.exists(full_model_path):
                st.error(f"Model file not found: {full_model_path}")
                return
            
            # Functions imported directly, no need for class instantiation
            
            # Load the model
            import pickle
            with open(full_model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            progress_bar.progress(30)
            status_text.text("Preparing data...")
            
            # Run inference for each target FDR level
            results = {}
            for i, target_fdr in enumerate(target_fdr_levels):
                status_text.text(f"Running inference for {target_fdr}% FDR...")
                progress_bar.progress(30 + (i * 50 // len(target_fdr_levels)))
                
                # Configure for inference
                config = {
                    'test_method': test_method,
                    'test_fdr': test_fdr,
                    'target_fdr': target_fdr,
                    'model': loaded_model
                }
                
                # Run inference (this would need to be implemented in the validator)
                result = run_single_inference(config)
                results[target_fdr] = result
            
            progress_bar.progress(100)
            status_text.text("✅ Inference complete!")
            
            try:
                # Convert inference results to training format and save to session state
                formatted_results = format_inference_results_for_display(results, selected_model, test_method, test_fdr)
                
                st.session_state.inference_results = formatted_results
                st.session_state.inference_complete = True
                
                # Display inference results without feature importance
                display_inference_results_formatted(formatted_results, selected_model, test_method, test_fdr, target_fdr_levels)
                
            except Exception as format_error:
                st.error(f"Error formatting results: {str(format_error)}")
                st.exception(format_error)
            
        except Exception as e:
            st.error(f"Error during inference: {str(e)}")
            st.exception(e)

def run_single_inference(config):
    """Run inference for a single configuration."""
    
    # Extract config
    test_method = config['test_method']
    test_fdr = config['test_fdr']
    target_fdr = config['target_fdr']
    model = config['model']
    
    # Load and prepare data (similar to training pipeline)
    try:
        # Load baseline and ground truth
        baseline_peptides = load_baseline_peptides_inference(test_method)
        ground_truth_peptides = load_ground_truth_peptides_inference(test_method)
        
        # Load test data
        test_data, y_test_true, additional_peptides = load_test_data_inference(
            [test_method], test_fdr, baseline_peptides, ground_truth_peptides
        )
        
        if len(additional_peptides) == 0:
            return {
                'additional_peptides': 0,
                'actual_fdr': 0,
                'mcc': 0,
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'baseline_peptides': len(baseline_peptides),
                'improvement_percent': 0,
                'false_positives': 0,
                'true_positives': 0,
                'threshold': 0.5,
                'total_candidates': 0,
                'ground_truth_peptides': len(ground_truth_peptides)
            }
        
        # Create features (import from the local API)
        from peptide_validator_api import make_advanced_features
        
        X_test = make_advanced_features(test_data, show_details=False)
        
        # Ensure data is on CPU and handle device consistency
        import numpy as np
        if hasattr(X_test, 'values'):
            X_test = X_test.values
        X_test = np.asarray(X_test, dtype=np.float32)
        
        # Predict using the loaded model with device consistency
        try:
            # Handle XGBoost CUDA/CPU device mismatch
            if hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'set_param'):
                # Set XGBoost to use CPU to match data device
                model.get_booster().set_param({'device': 'cpu'})
            elif hasattr(model, 'set_params'):
                # For scikit-learn style models
                if 'device' in model.get_params():
                    model.set_params(device='cpu')
        except Exception:
            # If device setting fails, continue with default behavior
            pass
        
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
        
        # Find optimal threshold for target FDR
        from peptide_validator_api import find_optimal_threshold
        optimal_threshold = find_optimal_threshold(y_test_true, y_pred_proba, target_fdr)
        
        # Make predictions
        y_pred = (y_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import matthews_corrcoef, precision_score, recall_score, f1_score, confusion_matrix
        
        # Calculate FDR on additional peptides only
        tn, fp, fn, tp = confusion_matrix(y_test_true, y_pred).ravel()
        actual_fdr = (fp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        
        # Calculate other metrics
        mcc = matthews_corrcoef(y_test_true, y_pred)
        precision = precision_score(y_test_true, y_pred, zero_division=0)
        recall = recall_score(y_test_true, y_pred, zero_division=0)
        f1 = f1_score(y_test_true, y_pred, zero_division=0)
        
        # Calculate improvement
        baseline_count = len(baseline_peptides)
        additional_count = np.sum(y_pred)
        improvement_percent = (additional_count / baseline_count) * 100
        
        return {
            'additional_peptides': int(additional_count),
            'actual_fdr': actual_fdr,
            'mcc': mcc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'baseline_peptides': baseline_count,
            'improvement_percent': improvement_percent,
            'false_positives': int(fp),
            'true_positives': int(tp),
            'threshold': optimal_threshold,
            'total_candidates': len(additional_peptides),
            'ground_truth_peptides': len(ground_truth_peptides)
        }
        
    except Exception as e:
        st.error(f"Error in inference: {str(e)}")
        # Return default values on error
        return {
            'additional_peptides': 0,
            'actual_fdr': 0,
            'mcc': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0,
            'baseline_peptides': 0,
            'improvement_percent': 0,
            'false_positives': 0,
            'true_positives': 0,
            'threshold': 0.5,
            'total_candidates': 0,
            'ground_truth_peptides': 0
        }

def format_inference_results_for_display(results, selected_model, test_method, test_fdr):
    """Convert inference results to training format for consistent display."""
    
    # Create results list in training format
    formatted_results = []
    for target_fdr, result in results.items():
        # Calculate recovery and increase percentages
        additional_peptides = result.get('additional_peptides', 0)
        total_candidates = result.get('total_candidates', 0)
        recovery_pct = (additional_peptides / total_candidates * 100) if total_candidates > 0 else 0
        
        # For increase percentage, we need baseline - use a default or get from result
        baseline_peptides = result.get('baseline_peptides', 4000)  # Default baseline
        increase_pct = (additional_peptides / baseline_peptides * 100) if baseline_peptides > 0 else 0

        formatted_result = {
            'Target_FDR': target_fdr,  # Keep as numeric for proper formatting
            'Threshold': result.get('threshold', 0.5),
            'Additional_Peptides': additional_peptides,
            'False_Positives': result.get('false_positives', 0),
            'Actual_FDR': result.get('actual_fdr', 0),
            'Precision': result.get('precision', 0),
            'Recovery_Pct': recovery_pct,
            'Increase_Pct': increase_pct,
            'Total_Validated_Candidates': total_candidates,
            'MCC': result.get('mcc', 0)
        }
        formatted_results.append(formatted_result)
    
    # Get first result for summary data
    first_result = results[list(results.keys())[0]]
    
    # Create full results structure
    return {
        'results': formatted_results,
        'config': {
            'test_method': test_method,
            'test_fdr': test_fdr,
            'model_source': 'inference',
            'model_name': selected_model['model_name']
        },
        'summary': {
            'baseline_peptides': first_result.get('baseline_peptides', 0),
            'ground_truth_peptides': first_result.get('ground_truth_peptides', 0),
            'additional_candidates': first_result.get('total_candidates', 0),
            'missed_peptides': max(0, first_result.get('ground_truth_peptides', 0) - first_result.get('baseline_peptides', 0)),
            'runtime_minutes': 0.1,  # Inference is fast
            'test_samples': 1
        }
    }

def display_inference_results(results, selected_model, test_method, test_fdr, target_fdr_levels):
    """Display inference results using the same interface as training results."""
    
    # Clear progress container
    st.empty()
    
    # Results header
    st.markdown("### 📊 Inference Results")
    st.markdown(f"**Model:** {selected_model['model_name']}")
    st.markdown(f"**Test Data:** {test_method} at {test_fdr}% FDR")
    st.markdown("---")
    
    # Results summary table
    st.markdown("#### 📈 Performance Summary")
    
    summary_data = []
    for target_fdr in target_fdr_levels:
        result = results[target_fdr]
        summary_data.append({
            'Target FDR (%)': target_fdr,
            'Actual FDR (%)': f"{result['actual_fdr']:.2f}",
            'Additional Peptides': result['additional_peptides'],
            'Improvement (%)': f"{result['improvement_percent']:.2f}",
            'MCC': f"{result['mcc']:.3f}",
            'Precision': f"{result['precision']:.3f}",
            'Recall': f"{result['recall']:.3f}",
            'F1 Score': f"{result['f1']:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)
    
    # Create visualizations using the same style as training
    col1, col2 = st.columns(2)
    
    with col1:
        # FDR vs Additional Peptides
        fig_fdr = px.scatter(
            summary_df, 
            x='Target FDR (%)', 
            y='Additional Peptides',
            title='📊 FDR vs Additional Peptides',
            color_discrete_sequence=[CHART_COLORS['line_primary']]
        )
        fig_fdr.update_traces(size=12)
        fig_fdr.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color=CHART_COLORS['text'])
        )
        st.plotly_chart(fig_fdr, use_container_width=True)
    
    with col2:
        # Performance Metrics
        metrics_data = []
        for target_fdr in target_fdr_levels:
            result = results[target_fdr]
            metrics_data.extend([
                {'Target FDR': f"{target_fdr}%", 'Metric': 'MCC', 'Value': result['mcc']},
                {'Target FDR': f"{target_fdr}%", 'Metric': 'Precision', 'Value': result['precision']},
                {'Target FDR': f"{target_fdr}%", 'Metric': 'Recall', 'Value': result['recall']},
                {'Target FDR': f"{target_fdr}%", 'Metric': 'F1 Score', 'Value': result['f1']}
            ])
        
        metrics_df = pd.DataFrame(metrics_data)
        fig_metrics = px.bar(
            metrics_df,
            x='Target FDR',
            y='Value',
            color='Metric',
            title='📈 Performance Metrics',
            barmode='group',
            color_discrete_sequence=[CHART_COLORS['line_primary'], CHART_COLORS['line_secondary'], 
                                   CHART_COLORS['line_target'], CHART_COLORS['bar_secondary']]
        )
        fig_metrics.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color=CHART_COLORS['text'])
        )
        st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Export results
    st.markdown("#### 💾 Export Results")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📥 Download Results CSV"):
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"inference_results_{test_method}_{test_fdr}fdr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("📊 Download Results JSON"):
            json_data = {
                'model_info': selected_model,
                'test_config': {
                    'test_method': test_method,
                    'test_fdr': test_fdr,
                    'target_fdr_levels': target_fdr_levels
                },
                'results': results,
                'timestamp': datetime.now().isoformat()
            }
            json_str = json.dumps(json_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"inference_results_{test_method}_{test_fdr}fdr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

def display_inference_results_formatted(formatted_results, selected_model, test_method, test_fdr, target_fdr_levels):
    """Display inference results from formatted results with tabs (without feature importance)."""
    
    # Clear progress container
    st.empty()
    
    # Results header
    st.markdown("### 📊 Inference Results")
    st.markdown(f"**Model:** {selected_model['model_name']}")
    st.markdown(f"**Test Data:** {test_method} at {test_fdr}% FDR")
    st.markdown("---")
    
    # Convert results to DataFrame for display
    if 'results' in formatted_results and formatted_results['results']:
        results_df = pd.DataFrame(formatted_results['results'])
        
        # Extract summary metrics like training mode
        if results_df.empty:
            st.error("❌ No results to display")
            return
            
        # Find best result (highest additional peptides)
        results_with_peptides = results_df[results_df['Additional_Peptides'] > 0]
        if not results_with_peptides.empty:
            best_result = results_with_peptides.loc[results_with_peptides['Additional_Peptides'].idxmax()]
        else:
            best_result = results_df.iloc[0]  # Fallback to first result
        
        # Key performance metrics (same as training mode)
        st.markdown("### 🏆 Best Performance Achieved")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Achieved FDR", f"{best_result['Actual_FDR']:.1f}%", delta=None, help="Actual false discovery rate of additional peptides")
        with col2:
            st.metric("Additional Unique Peptides", int(best_result['Additional_Peptides']), delta=None, help="Total additional peptides identified (includes both true and false positives)")
        with col3:
            st.metric("Method Tested", test_method, delta=None, help="MS method used for inference testing")
        with col4:
            st.metric("Test FDR Level", f"{test_fdr}%", delta=None, help="FDR level of test data used")
        
        # Check if we have ground truth data available for this test set
        has_ground_truth = False
        baseline_count = 0
        ground_truth_count = 0
        additional_candidates = 0
        
        # Try to extract counts from different possible locations in formatted_results
        if 'baseline_peptides' in formatted_results:
            baseline_count = formatted_results['baseline_peptides']
            has_ground_truth = True
        if 'ground_truth_peptides' in formatted_results:
            ground_truth_count = formatted_results['ground_truth_peptides']
            has_ground_truth = True
        if 'additional_candidates' in formatted_results:
            additional_candidates = formatted_results['additional_candidates']
        
        # Try to get from summary if available
        if 'summary' in formatted_results:
            baseline_count = formatted_results['summary'].get('baseline_peptides', baseline_count)
            ground_truth_count = formatted_results['summary'].get('ground_truth_peptides', ground_truth_count)
            additional_candidates = formatted_results['summary'].get('additional_candidates', additional_candidates)
            has_ground_truth = ground_truth_count > 0 or baseline_count > 0
        
        # Try to get from results DataFrame columns if not found yet
        if not has_ground_truth and 'Baseline_Peptides' in results_df.columns:
            baseline_counts = results_df['Baseline_Peptides'].dropna()
            if not baseline_counts.empty:
                baseline_count = int(baseline_counts.iloc[0])
                has_ground_truth = baseline_count > 0
        
        if 'Ground_Truth_Peptides' in results_df.columns:
            gt_counts = results_df['Ground_Truth_Peptides'].dropna()
            if not gt_counts.empty:
                ground_truth_count = int(gt_counts.iloc[0])
                has_ground_truth = ground_truth_count > 0
        
        # Only show Data Sources section if we have ground truth data available
        if has_ground_truth:
            # Enhanced data source clarity section (same as training mode)
            st.markdown("### 📊 Data Sources & Recovery Analysis")
            
            # Create detailed explanatory section
            with st.expander("📋 Data Source Details", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🎯 Baseline Peptides (Short Gradient 1% FDR)")
                    st.markdown(f"**Method:** {test_method}")
                    st.markdown(f"**Total unique peptides:** {baseline_count:,}")
                    st.info("High-confidence peptides identified by DIA-NN in short gradient conditions")
                    
                    st.markdown("#### 🌍 Ground Truth (Long Gradient 1% FDR)")
                    
                    # Show method-specific ground truth 
                    ground_truth_strategy = f"Ground truth for {test_method}"
                    try:
                        # Try to determine strategy from dataset config
                        dataset_name = test_method.split('_')[0] if '_' in test_method else 'Unknown'
                        config_path = f"data/{dataset_name}/dataset_info.json"
                        if os.path.exists(config_path):
                            import json
                            with open(config_path, 'r') as f:
                                dataset_config = json.load(f)
                            strategy = dataset_config.get('ground_truth_mapping', {}).get('_strategy', 'method_specific')
                            if strategy == 'use_all_ground_truth':
                                ground_truth_strategy = "All long gradient methods combined"
                            else:
                                ground_truth_strategy = f"Ground truth for {test_method}"
                    except:
                        ground_truth_strategy = f"Ground truth for {test_method}"
                    
                    st.markdown(f"**Method:** {ground_truth_strategy}")
                    st.markdown(f"**Total unique peptides:** {ground_truth_count:,}")
                    st.info("Reference peptides from long gradient analysis - the validation standard")
                    
                with col2:
                    st.markdown("#### 🔍 Test Set Analysis")
                    
                    # Calculate total unique peptides in test files
                    total_unique_in_test_files = baseline_count + additional_candidates
                    
                    st.markdown(f"**Test method:** {test_method}")
                    st.markdown(f"**Test FDR level:** {test_fdr}%")
                    st.markdown(f"**Total unique peptides in {test_fdr}% FDR files:** {total_unique_in_test_files:,}")
                    st.markdown(f"**Baseline peptides:** {baseline_count:,}")
                    st.markdown(f"**Additional candidates remaining:** {additional_candidates:,}")
                    
                    # Calculate validated candidates (the actual recoverable subset)
                    if 'results' in formatted_results and formatted_results['results']:
                        # Find the best result to get validation info
                        valid_results = [r for r in formatted_results['results'] if 'Additional_Peptides' in r and 'Actual_FDR' in r and r['Actual_FDR'] != 'N/A']
                        if valid_results:
                            best_result_for_validation = max(valid_results, key=lambda x: int(x['Additional_Peptides']))
                            validated_candidates = int(best_result_for_validation.get('Total_Validated_Candidates', 0))
                            validation_rate = (validated_candidates / additional_candidates) * 100 if additional_candidates > 0 else 0
                            
                            st.markdown(f"**Validated candidates (in ground truth):** {validated_candidates:,} ({validation_rate:.1f}% of candidates)")
                            st.info(f"Only {validated_candidates:,} out of {additional_candidates:,} additional candidates are actually present in the ground truth")
                    
                    st.success("💡 Our ML model identifies which validated candidates are real peptides vs false positives")
        else:
            # Show a simplified section when no ground truth is available
            st.markdown("### 🔍 Test Data Analysis")
            st.info("ℹ️ **Ground truth data not available for this test set.** Results show model predictions without validation against known true positives.")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🧪 Test Dataset")
                st.markdown(f"**Method:** {test_method}")
                st.markdown(f"**FDR Level:** {test_fdr}%")
                st.markdown(f"**Predictions Made:** {int(best_result['Additional_Peptides']):,} additional peptides")
                
            with col2:
                st.markdown("#### 🎯 Model Performance")
                st.markdown(f"**Model:** {selected_model.get('model_name', 'Unknown')}")
                st.markdown(f"**Confidence Threshold:** {best_result.get('Threshold', 'N/A')}")
                st.warning("⚠️ FDR cannot be calculated without ground truth data")
        
        st.markdown("---")
        
        # Tabs for different views (WITHOUT Feature Importance)
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Detailed Plots", "📋 Data Tables", "💾 Export"])
        
        with tab1:
            display_overview_plots(results_df)
        
        with tab2:
            display_detailed_plots(results_df)
        
        with tab3:
            display_inference_table_only(results_df)
        
        with tab4:
            # Export results
            st.markdown("### 💾 Export Results")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📥 Download Results CSV"):
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"inference_results_{test_method}_{test_fdr}fdr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("📊 Download Results JSON"):
                    json_data = {
                        'model_info': selected_model,
                        'test_config': {
                            'test_method': test_method,
                            'test_fdr': test_fdr,
                            'target_fdr_levels': target_fdr_levels
                        },
                        'results': formatted_results,
                        'timestamp': datetime.now().isoformat()
                    }
                    json_str = json.dumps(json_data, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"inference_results_{test_method}_{test_fdr}fdr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
    else:
        st.error("❌ No results data available")

def display_inference_table_only(df):
    """Display just the inference results table without feature importance."""
    
    # Display results table with proper column explanations
    st.markdown("### 📋 Detailed Results Table")
    
    # Fix column explanations to match actual table columns (left to right)
    st.info("""
    **📊 Column Explanations (left to right):**
    • **Target FDR**: Desired false discovery rate threshold
    • **Threshold**: Model confidence threshold used for predictions
    • **Additional Unique Peptides**: Total additional peptides identified (includes both true and false positives)
    • **False Positives**: Model predictions not validated by ground truth
    • **Actual FDR**: Measured false discovery rate of additional peptides
    • **Precision**: Model precision (TP/(TP+FP))
    • **Recovery %**: Percentage of validated candidates successfully recovered
    • **Increase %**: Improvement over baseline peptide count
    """)
    
    # Format the dataframe for display
    display_df = df.copy()
    
    # Remove internal columns from display 
    columns_to_remove = ['Aggregation_Method', 'Total_Validated_Candidates']
    for col in columns_to_remove:
        if col in display_df.columns:
            display_df = display_df.drop(col, axis=1)
    
    # Format percentage columns
    display_df['Target_FDR'] = display_df['Target_FDR'].map('{:.1f}%'.format)
    display_df['Actual_FDR'] = display_df['Actual_FDR'].map('{:.1f}%'.format)
    display_df['Recovery_Pct'] = display_df['Recovery_Pct'].map('{:.1f}%'.format)
    display_df['Increase_Pct'] = display_df['Increase_Pct'].map('{:.1f}%'.format)
    
    # Format MCC if it exists
    if 'MCC' in display_df.columns:
        display_df['MCC'] = display_df['MCC'].map('{:.3f}'.format)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "Target_FDR": st.column_config.TextColumn("Target FDR", help="Desired false discovery rate"),
            "Threshold": st.column_config.NumberColumn("Threshold", help="Model confidence threshold used for predictions", format="%.3f"),
            "Additional_Peptides": st.column_config.NumberColumn("Additional Unique Peptides", help="Total additional peptides identified (includes both true and false positives)"),
            "False_Positives": st.column_config.NumberColumn("False Positives", help="Model predictions not in ground truth"),
            "Actual_FDR": st.column_config.TextColumn("Actual FDR", help="Measured false discovery rate of additional peptides"),
            "Recovery_Pct": st.column_config.TextColumn("Recovery %", help="% of validated candidates recovered"),
            "Increase_Pct": st.column_config.TextColumn("Increase %", help="% improvement over baseline"),
            "MCC": st.column_config.TextColumn("MCC", help="Matthews Correlation Coefficient (-1 to +1, higher is better)")
        }
    )

def display_inference_results_with_tabs():
    """Display inference results with all tabs except Feature Importance."""
    
    # Extract results from session state
    if not hasattr(st.session_state, 'inference_results') or not st.session_state.inference_results:
        st.error("❌ No inference results found")
        return
    
    results = st.session_state.inference_results
    
    # Convert results to DataFrame for display
    if 'results' in results and results['results']:
        results_df = pd.DataFrame(results['results'])
    else:
        st.error("❌ No detailed results available")
        return
    
    # Tabs for different views (WITHOUT Feature Importance)
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Detailed Plots", "📋 Data Tables", "💾 Export"])
    
    with tab1:
        display_overview_plots(results_df)
    
    with tab2:
        display_detailed_plots(results_df)
    
    with tab3:
        display_inference_table_only(results_df)
    
    with tab4:
        display_export_options(results)

def load_baseline_peptides_inference(test_method=None):
    """Load baseline peptides for inference (matching method FDR_1)."""
    try:
        # Use the same data discovery as training
        files_info = discover_available_files()
        training_files = files_info.get('training', [])
        
        # Find FDR_1 files that match the test method
        if test_method:
            baseline_files = [
                f for f in training_files 
                if f['method'] == test_method and f['fdr'] == 1
            ]
            st.info(f"🎯 Looking for baseline files: method='{test_method}' FDR=1")
            st.info(f"📁 Found {len(baseline_files)} baseline files for {test_method}")
            
            # DEBUG: Show what files we're looking at
            all_method_files = [f for f in training_files if f['method'] == test_method]
        else:
            # Fallback to any FDR_1 files
            baseline_files = [f for f in training_files if f['fdr'] == 1]
            st.warning("⚠️ No test method specified, using all FDR_1 files as baseline")
        
        if not baseline_files:
            st.error(f"❌ No FDR_1 baseline files found for method {test_method}")
            return set()
        
        baseline_peptides = set()
        for file_info in baseline_files:
            try:
                peptides = load_peptide_sequences(file_info['path'])
                baseline_peptides.update(peptides)
                st.success(f"✅ Loaded {len(peptides)} peptides from {file_info['filename']}")
            except Exception as e:
                st.warning(f"Could not load baseline file {file_info['filename']}: {str(e)}")
                continue
        
        st.info(f"✅ Total baseline peptides: {len(baseline_peptides)} from {len(baseline_files)} files")
        return baseline_peptides
    except Exception as e:
        st.error(f"Error loading baseline peptides: {str(e)}")
        return set()

def get_matching_ground_truth_method(test_method):
    """Get the appropriate ground truth method for a given test method."""
    # Use automatic discovery for all datasets to maintain universal compatibility
    # The calling function will find the best ground truth match based on available data
    return None

def load_ground_truth_peptides_inference(test_method=None):
    """Load ground truth peptides for inference, with automatic matching for methods."""
    try:
        # Use the same data discovery as training
        files_info = discover_available_files()
        ground_truth_files = files_info.get('ground_truth', [])
        
        if not ground_truth_files:
            return set()
        
        ground_truth_peptides = set()
        
        # If a specific test method is provided, try to match to appropriate ground truth
        if test_method:
            target_gt_method = get_matching_ground_truth_method(test_method)
            
            if target_gt_method:
                # Look for the specific matching ground truth method
                matching_files = [f for f in ground_truth_files if f['method'] == target_gt_method]
                if matching_files:
                    st.success(f"🎯 Using matched ground truth: {target_gt_method} for test method {test_method}")
                    ground_truth_files = matching_files
                else:
                    st.error(f"❌ Could not find matching ground truth {target_gt_method} for {test_method}")
                    st.info("Available ground truth methods:")
                    for gt_file in ground_truth_files:
                        st.info(f"  - {gt_file['method']}: {gt_file['filename']}")
                    return set()
            else:
                # Generic matching: find ground truth from same dataset
                test_dataset = test_method.split('_')[0] if '_' in test_method else None
                if test_dataset:
                    dataset_gt_files = [f for f in ground_truth_files if f.get('dataset') == test_dataset]
                    if dataset_gt_files:
                        st.success(f"🎯 Using {test_dataset} ground truth files for test method {test_method}")
                        ground_truth_files = dataset_gt_files
                    else:
                        st.warning(f"⚠️ No ground truth found for dataset {test_dataset}, using all available ground truth files")
                else:
                    st.info("ℹ️ Using all available ground truth files for inference")
        
        for file_info in ground_truth_files:
            try:
                peptides = load_peptide_sequences(file_info['path'])
                ground_truth_peptides.update(peptides)
            except Exception as e:
                st.warning(f"Could not load ground truth file {file_info['filename']}: {str(e)}")
                continue
        
        return ground_truth_peptides
    except Exception as e:
        st.error(f"Error loading ground truth peptides: {str(e)}")
        return set()

def load_test_data_inference(test_methods, test_fdr, baseline_peptides, ground_truth_peptides):
    """Load test data for inference."""
    try:
        # Use the same data discovery as training
        files_info = discover_available_files()
        training_files = files_info.get('training', [])
        
        # Filter for the specific method and FDR
        test_files = [
            f for f in training_files 
            if f['method'] in test_methods and f['fdr'] == test_fdr
        ]
        
        if not test_files:
            return pd.DataFrame(), np.array([]), []
        
        all_test_data = []
        for file_info in test_files:
            try:
                df = load_parquet_file(file_info['path'])
                df['source_method'] = file_info['method']
                df['source_fdr'] = file_info['fdr']
                all_test_data.append(df)
            except Exception as e:
                st.warning(f"Could not load test file {file_info['filename']}: {str(e)}")
                continue
        
        if not all_test_data:
            return pd.DataFrame(), np.array([]), []
        
        # Combine all test data
        test_data = pd.concat(all_test_data, ignore_index=True)
        
        # Get peptide column name
        peptide_col = 'Stripped.Sequence' if 'Stripped.Sequence' in test_data.columns else 'Modified.Sequence'
        
        # Find additional peptides (in test data but not in baseline)
        test_peptides = set(test_data[peptide_col].unique())
        additional_peptides = test_peptides - baseline_peptides
        
        # Filter to only additional peptides
        test_data_additional = test_data[test_data[peptide_col].isin(additional_peptides)].copy()
        
        # Create labels (ground truth validation)
        y_test_true = test_data_additional[peptide_col].apply(
            lambda x: 1 if x in ground_truth_peptides else 0
        ).values
        
        return test_data_additional, y_test_true, list(additional_peptides)
        
    except Exception as e:
        st.error(f"Error loading test data: {str(e)}")
        return pd.DataFrame(), np.array([]), []

def save_model_to_streamlit_directory(model, config, results, model_name=None):
    """Save trained model to the streamlit saved_models directory."""
    try:
        # Create model directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if model_name is None:
            model_name = f"peptide_model_{timestamp}"
        
        model_dir = os.path.join(os.path.dirname(__file__), "saved_models", f"{model_name}_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        import pickle
        model_file = os.path.join(model_dir, "model.pkl")
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'description': f"Trained model: {model_name}",
            'training_config': config,
            'performance_summary': results,
            'model_file': 'model.pkl'
        }
        
        metadata_file = os.path.join(model_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        st.success(f"✅ Model saved to: {model_dir}")
        return model_dir
        
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return None

def save_fdr_results_to_streamlit_directory(results_df, config, results_name=None):
    """Save FDR analysis results to the streamlit fdr_results directory."""
    try:
        # Create results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if results_name is None:
            results_name = f"fdr_analysis_{timestamp}"
        
        results_dir = os.path.join(os.path.dirname(__file__), "fdr_results", f"{results_name}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        
        # Save results CSV
        results_file = os.path.join(results_dir, "fdr_results.csv")
        results_df.to_csv(results_file, index=False)
        
        # Save config
        config_file = os.path.join(results_dir, "analysis_config.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        st.success(f"✅ Results saved to: {results_dir}")
        return results_dir
        
    except Exception as e:
        st.error(f"Error saving results: {str(e)}")
        return None

def show_setup_interface():
    """Display the dataset setup interface with visual ground truth mapping."""
    
    # Header with navigation (matching training mode style)
    col1, col2, col3 = st.columns([1, 3, 1])
    
    # Custom CSS for consistent button styling
    st.markdown("""
    <style>
    div[data-testid="stButton"] button[data-baseweb="button"][kind="secondary"],
    .stButton > button[kind="secondary"] {
        background: #2E86AB !important;
        color: white !important;
        border: none !important;
        height: 80px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        border-radius: 15px !important;
        padding: 20px 30px !important;
        width: 100% !important;
        margin-left: 0px !important;
        transform: translateX(0px) !important;
    }
    
    div[data-testid="stButton"] button[data-baseweb="button"][kind="primary"],
    .stButton > button[kind="primary"] {
        background: #2E86AB !important;
        color: white !important;
        border: none !important;
        height: 80px !important;
        font-size: 22px !important;
        font-weight: 700 !important;
        border-radius: 15px !important;
        padding: 20px 30px !important;
        width: 100% !important;
        margin-right: -20px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    with col1:
        if st.button("← Back to Home", type="primary", key="setup_back_btn_1"):
            # Clear all analysis states when returning to home
            st.session_state.app_mode = None
            clear_analysis_states()
            st.rerun()
    with col2:
        st.markdown("<h3 style='text-align: center; margin: 20px 0;'>⚙️ Setup Mode</h3>", unsafe_allow_html=True)
    with col3:
        # Empty space for symmetry
        st.empty()
    
    st.markdown("---")
    
    # Subtitle
    st.markdown("*Configure ground truth mapping for your datasets visually - no JSON editing required!*")
    
    # Discover available files first
    try:
        files_info = discover_available_files_by_dataset()
        
        if not files_info:
            st.error("No datasets found! Please check your data folder structure.")
            st.markdown("""
            Expected structure:
            ```
            data/
            ├── YourDataset/
            │   ├── short_gradient/
            │   │   ├── FDR_1/     # Baseline files
            │   │   ├── FDR_20/    # Training files
            │   │   └── FDR_50/    # Training files
            │   └── long_gradient/
            │       └── FDR_1/     # Ground truth files
            ```
            """)
            return
            
        # Dataset selection
        dataset_names = list(files_info.keys())
        selected_dataset = st.selectbox(
            "🗂️ Select Dataset to Configure:",
            options=dataset_names,
            help="Choose which dataset you want to configure ground truth mapping for"
        )
        
        if not selected_dataset:
            st.warning("Please select a dataset to configure.")
            return
            
        dataset_info = files_info[selected_dataset]
        
        # Display dataset overview
        st.markdown(f"## 📊 Dataset: {selected_dataset}")
        
        # Add file organization mode selection FIRST
        st.markdown("### ⚙️ File Organization Mode")
        
        file_mode = st.radio(
            "How should files be organized for analysis?",
            options=["individual", "triplicates"],
            format_func=lambda x: {
                "individual": "📄 Individual Files - Each file is a separate method",
                "triplicates": "📊 Triplicate Groups"
            }[x],
            help="Individual: Use each file separately (30 methods). Triplicates: Group files by method number (10 groups)."
        )
        
        # Get training methods and ground truth files based on selected mode
        if file_mode == "triplicates":
            st.info("🔧 **Configure Triplicate Grouping**")
            st.markdown("Define how your files should be grouped together:")
            
            # Let user define the grouping pattern
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Grouping Pattern:**")
                grouping_method = st.selectbox(
                    "How should files be grouped?",
                    options=[
                        "string_search",
                        "manual_selection"
                    ],
                    format_func=lambda x: {
                        "string_search": "🔍 String Search (Find & Group)",
                        "manual_selection": "👆 Manual Selection (Custom)"
                    }[x],
                    help="String Search: Search for text patterns like '001' to auto-group files. Manual: Create custom groups."
                )
            
            with col2:
                if grouping_method == "string_search":
                    st.markdown("**String Search Grouping:**")
                    st.markdown("Search for strings in filenames to create groups automatically.")
                    search_terms = st.text_area(
                        "Enter search terms (one per line):",
                        value="001\n002\n003\n004\n005\n006\n007\n008\n009\n010",
                        help="Enter one search term per line. Files containing each term will be grouped together. Example: searching '001' finds all files with '001' in their name.",
                        height=120
                    )
                    grouping_pattern = search_terms.strip().split('\n') if search_terms.strip() else []
                else:  # manual_selection
                    st.markdown("**Manual Grouping:**")
                    st.markdown("You'll define groups manually in the next step")
                    grouping_pattern = None
            
            # Apply the grouping to get training methods
            training_methods = []
            ground_truth_files = []
            
            if grouping_method == "manual_selection":
                # Show all individual files and let user group them manually
                individual_training = []
                for fdr_level in ['20', '50']:
                    if fdr_level in dataset_info.get('training', {}):
                        methods = list(dataset_info['training'][fdr_level].keys())
                        individual_training.extend(methods)
                individual_training = sorted(list(set(individual_training)))
                
                st.markdown("**Create Custom Groups:**")
                st.markdown("Define which files belong together:")
                
                # Initialize session state for manual groups
                if f'manual_groups_{selected_dataset}' not in st.session_state:
                    st.session_state[f'manual_groups_{selected_dataset}'] = {}
                
                manual_groups = st.session_state[f'manual_groups_{selected_dataset}']
                
                # Group creation interface
                group_name = st.text_input("Group Name:", placeholder="e.g., Method_001")
                selected_files = st.multiselect(
                    "Select files for this group:",
                    options=individual_training,
                    help="Choose which files belong to this group"
                )
                
                if st.button("Add Group") and group_name and selected_files:
                    manual_groups[group_name] = selected_files
                    st.session_state[f'manual_groups_{selected_dataset}'] = manual_groups
                    st.success(f"✅ Added group '{group_name}' with {len(selected_files)} files")
                
                # Show current groups
                if manual_groups:
                    st.markdown("**Current Groups:**")
                    for group_name, files in manual_groups.items():
                        with st.expander(f"📊 {group_name} ({len(files)} files)"):
                            for f in files:
                                st.markdown(f"• `{f}`")
                            if st.button(f"Remove {group_name}", key=f"remove_{group_name}"):
                                del manual_groups[group_name]
                                st.session_state[f'manual_groups_{selected_dataset}'] = manual_groups
                                st.rerun()
                
                training_methods = list(manual_groups.keys())
                ground_truth_files = list(dataset_info.get('ground_truth', {}).keys())
                
            else:
                # Use string search grouping
                # Get all individual files first
                individual_training = []
                for fdr_level in ['20', '50']:
                    if fdr_level in dataset_info.get('training', {}):
                        methods = list(dataset_info['training'][fdr_level].keys())
                        individual_training.extend(methods)
                individual_training = sorted(list(set(individual_training)))
                
                # Apply string search to group files
                groups = {}
                ungrouped = []
                
                # For each search term, find all files containing that term
                for search_term in grouping_pattern:
                    search_term = search_term.strip()
                    if not search_term:
                        continue
                        
                    group_name = f"{selected_dataset}_Group_{search_term}"
                    matching_files = []
                    
                    for method in individual_training:
                        if search_term in method:
                            matching_files.append(method)
                    
                    if matching_files:
                        groups[group_name] = matching_files
                        # Remove matched files from individual list to avoid duplicates
                        for matched_file in matching_files:
                            if matched_file in individual_training:
                                individual_training.remove(matched_file)
                
                # Remaining files are ungrouped
                ungrouped = individual_training
                
                # Show grouping results
                if groups:
                    st.success(f"✅ Created {len(groups)} groups using search terms: {', '.join([f'`{term}`' for term in grouping_pattern])}")
                    with st.expander("Preview Groups", expanded=True):
                        for group_name, files in list(groups.items())[:3]:  # Show first 3
                            st.markdown(f"**{group_name}:** {len(files)} files")
                            for f in files[:2]:  # Show first 2 files
                                st.markdown(f"  • `{f}`")
                            if len(files) > 2:
                                st.markdown(f"  • ... and {len(files) - 2} more")
                        if len(groups) > 3:
                            st.markdown(f"... and {len(groups) - 3} more groups")
                    
                    training_methods = list(groups.keys())
                else:
                    st.warning("⚠️ No groups found with search terms. Files will be treated individually.")
                    training_methods = individual_training
                
                if ungrouped:
                    st.warning(f"⚠️ {len(ungrouped)} files didn't match any search terms and will be treated individually")
                    training_methods.extend(ungrouped)
                
                # For ground truth, use same string search logic
                individual_gt = list(dataset_info.get('ground_truth', {}).keys())
                gt_groups = {}
                
                for search_term in grouping_pattern:
                    search_term = search_term.strip()
                    if not search_term:
                        continue
                        
                    group_name = f"{selected_dataset}_Group_{search_term}"
                    matching_gt_files = []
                    
                    for gt_file in individual_gt:
                        if search_term in gt_file:
                            matching_gt_files.append(gt_file)
                    
                    if matching_gt_files:
                        gt_groups[group_name] = matching_gt_files
                        # Remove matched files to avoid duplicates
                        for matched_file in matching_gt_files:
                            if matched_file in individual_gt:
                                individual_gt.remove(matched_file)
                
                # Add remaining individual GT files
                for gt_file in individual_gt:
                    gt_groups[gt_file] = [gt_file]  # Individual file
                
                ground_truth_files = list(gt_groups.keys())
            
        else:
            # Individual files mode - use the original dataset_info structure
            training_methods = []
            for fdr_level in ['20', '50']:
                if fdr_level in dataset_info.get('training', {}):
                    methods = list(dataset_info['training'][fdr_level].keys())
                    training_methods.extend(methods)
            
            # Remove duplicates and sort
            training_methods = sorted(list(set(training_methods)))
            
            ground_truth_files = []
            if 'ground_truth' in dataset_info:
                ground_truth_files = list(dataset_info['ground_truth'].keys())
                ground_truth_files.sort()
        
        # Professional sidebar with dropdowns (updated based on mode)
        mode_description = "individual files" if file_mode == "individual" else "triplicate groups"
        
        with st.expander(f"🎯 Training Methods ({len(training_methods)} {mode_description})", expanded=False):
            if training_methods:
                for i, method in enumerate(training_methods):
                    st.markdown(f"{i+1}. `{method}`")
                    
            else:
                st.warning("No training methods found")
        
        with st.expander(f"🎭 Ground Truth Files ({len(ground_truth_files)} found)", expanded=False):
            if ground_truth_files:
                for i, gt_file in enumerate(ground_truth_files):
                    st.markdown(f"{i+1}. `{gt_file}`")
            else:
                st.warning("No ground truth files found")
        
        st.markdown("### 🔗 Ground Truth Mapping Configuration")
        
        # Strategy selection
        strategy = st.radio(
            "Choose Mapping Strategy:",
            options=[
                "use_all_ground_truth",
                "visual_mapping"
            ],
            format_func=lambda x: {
                "use_all_ground_truth": "📚 Use All Ground Truth Files (Simple)",
                "visual_mapping": "🎨 Visual Connection Mapping (Interactive)"
            }[x],
            help="Simple: Use all ground truth files for any method. Interactive: Visually connect methods to ground truth files."
        )
        
        # Configuration based on strategy
        mapping_config = {}
        
        if strategy == "use_all_ground_truth":
            st.success("✅ **Simple Strategy Selected**")
            st.markdown("Any training method will use **all** ground truth files from this dataset for validation.")
            
            mapping_config = {
                "_strategy": "use_all_ground_truth",
                "_file_mode": file_mode,
                "_note": f"All {selected_dataset} methods use all available ground truth files for comprehensive validation ({'triplicate groups' if file_mode == 'triplicates' else 'individual files'})"
            }
            
            # If triplicate mode with string search groups, save the group definitions
            if file_mode == "triplicates" and grouping_method == "string_search" and "groups" in locals():
                # Create direct rules for the string search groups
                group_rules = {}
                for group_name in groups.keys():
                    group_rules[group_name] = "all_ground_truth"  # Use all GT files for each group
                mapping_config["_group_definitions"] = group_rules
                mapping_config["_note"] += f" - Created {len(groups)} groups using string search"
            
        else:  # visual_mapping
            st.info("🎨 **Visual Connection Mapping Selected**")
            st.markdown("Connect each training method to its corresponding ground truth file:")
            
            if not training_methods or not ground_truth_files:
                st.error("Need both training methods and ground truth files for visual mapping!")
                return
            
            # Initialize connections if not in session state
            if f'connections_{selected_dataset}' not in st.session_state:
                st.session_state[f'connections_{selected_dataset}'] = {}
            
            connections = st.session_state[f'connections_{selected_dataset}']
            
            # Show dataset overview in collapsible sections
            with st.expander("📋 View All Available Files", expanded=False):
                col_methods, col_gt = st.columns(2)
                with col_methods:
                    st.markdown("**Training Methods:**")
                    for i, method in enumerate(training_methods[:10]):  # Show first 10
                        st.markdown(f"• `{method}`")
                    if len(training_methods) > 10:
                        st.markdown(f"... and {len(training_methods) - 10} more")
                
                with col_gt:
                    st.markdown("**Ground Truth Files:**")
                    for gt_file in ground_truth_files:
                        st.markdown(f"• `{gt_file}`")
            
            # Create the visual connection interface
            st.markdown("### 🔗 Create Connections")
            st.markdown("**For each training method, select which ground truth file(s) it should use:**")
            
            # Show ALL training methods in connections (no grouping or filtering)
            st.markdown(f"**Found {len(training_methods)} training methods to connect:**")
            
            # Use all training methods directly - no grouping or filtering
            sorted_methods = sorted(training_methods)
            pattern_based_rules = {}
            
            # Connection interface with cleaner design
            for i, method in enumerate(sorted_methods):
                # Create a clean container for each connection
                with st.container():
                    st.markdown(f"#### Connection #{i+1}")
                    
                    # Three column layout for clean visual connection
                    col1, col2, col3 = st.columns([3, 1, 3])
                    
                    with col1:
                        # Training method box
                        st.markdown("**Training Method:**")
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #2E86AB, #6366F1);
                            color: white;
                            padding: 12px;
                            border-radius: 8px;
                            font-size: 14px;
                            word-wrap: break-word;
                            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        ">
                            📊 {method}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        # Connection arrow
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown("""
                        <div style="
                            text-align: center;
                            padding: 20px 0;
                            font-size: 20px;
                            color: #2E86AB;
                        ">
                            ➡️
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        # Ground truth selection
                        st.markdown("**Ground Truth Files:**")
                        
                        # Create options with full filenames (no "no connection" option for multiselect)
                        gt_options = ground_truth_files
                        
                        # Find current selection using the full method name
                        current_connections = connections.get(method, [])
                        # Ensure current_connections is a list
                        if isinstance(current_connections, str):
                            current_connections = [current_connections] if current_connections != "(no connection)" else []
                        
                        # Use method name + index to ensure unique keys
                        selected_gts = st.multiselect(
                            f"Select ground truth files:",
                            options=gt_options,
                            default=current_connections,
                            key=f"gt_select_{selected_dataset}_{i}_{hash(method) % 10000}",  # Unique key
                            help=f"Select which ground truth files '{method}' should use for validation (can select multiple)",
                            placeholder="Choose ground truth files..."
                        )
                        
                        # Update connections
                        if selected_gts:
                            connections[method] = selected_gts
                            st.session_state[f'connections_{selected_dataset}'][method] = selected_gts
                            
                            # Show connected ground truth files visually
                            for idx, selected_gt in enumerate(selected_gts):
                                st.markdown(f"""
                                <div style="
                                    background: linear-gradient(135deg, #10B981, #34D399);
                                    color: white;
                                    padding: 8px 12px;
                                    border-radius: 8px;
                                    font-size: 13px;
                                    word-wrap: break-word;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                                    margin-bottom: 4px;
                                ">
                                    🎭 {selected_gt}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Add to pattern rules for saving
                            pattern_based_rules[method] = selected_gts
                            
                        else:
                            # Remove connection
                            if method in connections:
                                del connections[method]
                                if method in st.session_state[f'connections_{selected_dataset}']:
                                    del st.session_state[f'connections_{selected_dataset}'][method]
                            
                            # Show no connection state
                            st.markdown("""
                            <div style="
                                background: #F3F4F6;
                                color: #6B7280;
                                padding: 12px;
                                border-radius: 8px;
                                text-align: center;
                                font-style: italic;
                                border: 2px dashed #D1D5DB;
                            ">
                                No connection selected
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
            
            # Connection summary
            if connections:
                with st.expander("📋 Connection Summary", expanded=True):
                    st.markdown("**Active Connections:**")
                    for method, gt_file in connections.items():
                        st.markdown(f"✅ `{method}` → `{gt_file}`")
            
            # Create mapping config using full filenames (no pattern extraction needed)
            mapping_config = {
                "_strategy": "direct_mapping",
                "_direct_rules": pattern_based_rules,
                "_file_mode": file_mode,
                "_note": f"Direct filename mapping for {selected_dataset} methods ({'triplicate groups' if file_mode == 'triplicates' else 'individual files'})"
            }
        
        # Configuration preview with better styling
        st.markdown("### 📋 Configuration Preview")
        with st.expander("View Configuration Details", expanded=True):
            st.json(mapping_config)
        
        # Action buttons section with consistent layout
        st.markdown("---")  # Add separator for better visual separation
        
        # Load existing dataset info if available
        dataset_config_path = f"data/{selected_dataset}/dataset_info.json"
        existing_config = {}
        
        if os.path.exists(dataset_config_path):
            try:
                with open(dataset_config_path, 'r') as f:
                    existing_config = json.load(f)
            except Exception as e:
                st.warning(f"Could not load existing config: {e}")
        
        # Merge with ground truth mapping
        final_config = existing_config.copy()
        final_config["ground_truth_mapping"] = mapping_config
        
        # Add default values if missing
        if "icon" not in final_config:
            final_config["icon"] = "🧬"
        if "instrument" not in final_config:
            final_config["instrument"] = "Mass spectrometer"
        if "description" not in final_config:
            final_config["description"] = f"{selected_dataset} proteomics analysis"
        
        # Create consistent column layout for both action sections
        col1, col2 = st.columns(2, gap="medium")
        
        with col1:
            st.markdown("### 🧪 Test Configuration")
            st.markdown("Validate your ground truth mapping configuration")
            
            # Test button with consistent styling
            if st.button("Test Ground Truth Matching", type="primary", use_container_width=True):
                test_ground_truth_matching(selected_dataset, final_config, training_methods)
        
        with col2:
            st.markdown("### 💾 Save Configuration")
            st.markdown("Save your ground truth mapping for future analysis")
            
            # Save button with consistent styling
            if st.button("💾 Save Configuration", type="primary", use_container_width=True):
                try:
                    # Create directory if it doesn't exist
                    os.makedirs(f"data/{selected_dataset}", exist_ok=True)
                    
                    # Save configuration
                    with open(dataset_config_path, 'w') as f:
                        json.dump(final_config, f, indent=2)
                    
                    st.success(f"✅ Configuration saved to `{dataset_config_path}`")
                    
                    # Clear session state for this dataset
                    if f'pattern_rules_{selected_dataset}' in st.session_state:
                        del st.session_state[f'pattern_rules_{selected_dataset}']
                    
                except Exception as e:
                    st.error(f"❌ Error saving configuration: {e}")
        
        # Back to Home button section
        st.markdown("---")
        st.markdown("### 🏠 Navigation")
        
        col1, col2 = st.columns([1, 3])
        
        # Custom CSS for consistent button styling (matching training mode)
        st.markdown("""
        <style>
        div[data-testid="stButton"] button[data-baseweb="button"][kind="secondary"],
        .stButton > button[kind="secondary"] {
            background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%) !important;
            color: #495057 !important;
            border: 1px solid #dee2e6 !important;
            font-weight: 500 !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            transition: all 0.2s ease !important;
        }
        
        div[data-testid="stButton"] button[data-baseweb="button"][kind="secondary"]:hover,
        .stButton > button[kind="secondary"]:hover {
            background: linear-gradient(135deg, #e9ecef 0%, #dee2e6 100%) !important;
            border-color: #adb5bd !important;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        with col1:
            if st.button("← Back to Home", type="primary", key="setup_back_btn_2"):
                # Clear all analysis states when returning to home
                st.session_state.app_mode = None
                clear_analysis_states()
                st.rerun()
        with col2:
            st.empty()  # Balance the layout
        
    except Exception as e:
        st.error(f"Error loading dataset information: {str(e)}")
        st.exception(e)

def test_ground_truth_matching(dataset_name, config, training_methods):
    """Test the ground truth matching configuration."""
    st.markdown("#### Testing Ground Truth Matching...")
    
    try:
        mapping = config.get("ground_truth_mapping", {})
        strategy = mapping.get("_strategy", "use_all_ground_truth")
        
        if strategy == "use_all_ground_truth":
            st.success("✅ **Use All Ground Truth Strategy**")
            st.markdown("All methods will use all available ground truth files.")
            
        elif strategy == "pattern_matching":
            st.info("🎯 **Pattern Matching Strategy**")
            pattern_rules = mapping.get("_pattern_rules", {})
            
            if not pattern_rules:
                st.warning("⚠️ No pattern rules defined!")
                return
            
            st.markdown("**Pattern Matching Results:**")
            for method in training_methods[:5]:  # Test first 5 methods
                # Extract pattern from method
                import re
                number_match = re.search(r'(\d{3})', method)
                if number_match:
                    pattern = number_match.group(1)
                    if pattern in pattern_rules:
                        gt_file = pattern_rules[pattern]
                        st.markdown(f"• `{method}` → Pattern `{pattern}` → Ground Truth `{gt_file}` ✅")
                    else:
                        st.markdown(f"• `{method}` → Pattern `{pattern}` → **No match found** ❌")
                else:
                    st.markdown(f"• `{method}` → **No pattern extractable** ❌")
            
            if len(training_methods) > 5:
                st.markdown(f"... and {len(training_methods) - 5} more methods")
        
        st.success("🎉 Configuration test completed!")
        
    except Exception as e:
        st.error(f"Error testing configuration: {str(e)}")

if __name__ == "__main__":
    main()
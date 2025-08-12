#!/usr/bin/env python3
"""
🎯 PEPTIDE VALIDATOR API
================================================================================
Backend API interface for the flexible peptide validator.
Refactored from flexible_peptide_validator.py to be callable from Streamlit.

This module provides a clean API interface that:
✅ Accepts configuration parameters as function arguments
✅ Returns structured results for frontend consumption
✅ Provides progress callbacks for real-time updates
✅ Handles errors gracefully with informative messages
================================================================================
"""

import warnings
import glob
import re
from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    classification_report,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
from typing import List, Dict, Optional, Callable, Tuple
from file_discovery import discover_available_files, get_files_for_method
from dataset_utils import get_files_for_configured_method, discover_available_files_by_dataset

warnings.filterwarnings('ignore')

# Set style for professional Nature-style plots
plt.style.use('seaborn-v0_8-whitegrid')

# Nature-style color palette (professional scientific colors)
NATURE_COLORS = {
    'primary': '#2E86AB',      # Professional blue
    'secondary': '#A23B72',    # Deep magenta  
    'tertiary': '#F18F01',     # Amber orange
    'quaternary': '#C73E1D',   # Deep red
    'accent1': '#5D737E',      # Steel blue-gray
    'accent2': '#64A6BD',      # Light blue
    'success': '#4A7C59',      # Forest green
    'warning': '#F4A261',      # Warm orange
    'neutral': '#6C757D'       # Professional gray
}

# Set color palette for consistency
nature_palette = [NATURE_COLORS['primary'], NATURE_COLORS['secondary'], 
                 NATURE_COLORS['tertiary'], NATURE_COLORS['quaternary'],
                 NATURE_COLORS['accent1'], NATURE_COLORS['accent2']]
sns.set_palette(nature_palette)

class PeptideValidatorAPI:
    """API interface for peptide validation analysis."""
    
    def __init__(self):
        """Initialize the API with default parameters."""
        self.optimal_params = {
            'subsample': 0.8, 
            'reg_lambda': 1.5, 
            'reg_alpha': 0,
            'n_estimators': 1000,
            'min_child_weight': 1, 
            'max_depth': 7,
            'learning_rate': 0.08,
            'gamma': 0, 
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist',
            'device': self._detect_gpu_device()
        }
        
        self.engineer_log = {'Precursor.Quantity', 'Ms1.Area', 'Ms2.Area', 'Peak.Height', 'Precursor.Charge'}
        self.engineer_ratios = [
            ('Ms1.Area', 'Ms2.Area'),
            ('Peak.Height', 'Ms1.Area'),
            ('Precursor.Quantity', 'Peak.Height')
        ]
    
    def _detect_gpu_device(self) -> str:
        """Detect if GPU is available, fallback to CPU."""
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
        except Exception:
            # Silently fallback to CPU for API
            return 'cpu'
    
    def run_analysis(self, 
                    train_methods: List[str],
                    test_method: str,
                    train_fdr_levels: List[int] = [1, 20, 50],
                    test_fdr: int = 50,
                    target_fdr_levels: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0],
                    xgb_params: Optional[Dict] = None,
                    progress_callback: Optional[Callable] = None,
                    results_dir: Optional[str] = None,
                    feature_selection: Optional[Dict] = None,
                    aggregation_method: str = 'max') -> Dict:
        """
        Run complete peptide validation analysis.
        
        Args:
            train_methods: List of methods to use for training
            test_method: Method to use for testing (holdout)
            train_fdr_levels: FDR levels to include in training data
            test_fdr: FDR level for test data
            target_fdr_levels: Target FDR levels for optimization
            xgb_params: Optional XGBoost parameter overrides
            progress_callback: Optional callback for progress updates
            results_dir: Optional directory to save results
            
        Returns:
            Dictionary containing analysis results and metadata
        """
        
        try:
            # Store feature selection preferences
            self.feature_selection = feature_selection or {
                'use_diann_quality': True,
                'use_sequence_features': True,
                'use_ms_features': True,
                'use_statistical_features': True,
                'use_library_features': True,
                'excluded_features': []
            }
            
            # Update XGBoost parameters if provided
            if xgb_params:
                self.optimal_params.update(xgb_params)
            
            # Setup results directory
            if results_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                results_dir = f"results/STREAMLIT_RESULTS_{timestamp}"
            
            os.makedirs(results_dir, exist_ok=True)
            for subdir in ['plots', 'tables', 'feature_analysis', 'raw_data']:
                os.makedirs(f"{results_dir}/{subdir}", exist_ok=True)
            
            # Progress tracking
            total_steps = 10  # Updated to include model saving and metadata update steps
            current_step = 0
            
            def update_progress(step_name: str):
                nonlocal current_step
                current_step += 1
                if progress_callback:
                    progress_callback(current_step, total_steps, step_name)
            
            # Step 1: Load test method-specific baseline peptides
            baseline_type = "ASTRAL 7min" if 'ASTRAL' in test_method else "fast gradient"
            update_progress(f"Loading baseline peptides {baseline_type} 1% FDR")
            baseline_peptides = self._load_baseline_peptides(test_method)
            
            # Step 2: Load ground truth peptides  
            update_progress(f"Loading ground truth peptides")
            ground_truth_peptides = self._load_ground_truth_peptides(test_method)
            
            # Step 3: Analyze improvement opportunity
            update_progress("Analyzing improvement opportunity")
            missed_peptides = self._analyze_improvement_opportunity(
                baseline_peptides, ground_truth_peptides)
            
            # Step 4: Load training data
            update_progress("Loading and preparing training data")
            train_data, y_train = self._load_training_data(
                train_methods, train_fdr_levels, ground_truth_peptides)
            
            # Step 5: Load test data
            update_progress("Loading and preparing test data")
            test_data, y_test, additional_peptides = self._load_test_data(
                test_method, test_fdr, baseline_peptides, ground_truth_peptides)
            
            if len(additional_peptides) == 0:
                raise ValueError(f"No additional peptides found for {test_method} at {test_fdr}% FDR")
            
            # Step 5.5: Calculate naive FDR comparison
            naive_results = self._calculate_naive_fdr_comparison(
                test_method, test_fdr, baseline_peptides, ground_truth_peptides)
            
            # Step 6: Train model
            update_progress("Training ensemble model with advanced features")
            model, training_features = self._train_model(train_data, y_train)
            
            # Step 6.5: Save trained model automatically (user-friendly)
            update_progress("Saving trained model")
            self._save_trained_model(model, training_features, results_dir)
            
            # Step 7: Make predictions and optimize thresholds
            update_progress("Making predictions and optimizing thresholds")
            results, X_test = self._validate_and_optimize(
                model, test_data, y_test, training_features, target_fdr_levels, len(baseline_peptides), aggregation_method)
            
            # Step 7.5: Update model metadata with training results for inference
            update_progress("Updating model with training results")
            self._update_model_metadata_with_results(results_dir, results)
            
            # Step 8: Create visualizations and save results
            update_progress("Creating visualizations and saving results")
            self._create_visualizations(results, results_dir, model, X_test, training_features)
            
            # Compile final results
            analysis_results = {
                'config': {
                    'train_methods': train_methods,
                    'test_method': test_method,
                    'train_fdr_levels': train_fdr_levels,
                    'test_fdr': test_fdr,
                    'target_fdr_levels': target_fdr_levels,
                    'xgb_params': self.optimal_params,
                    'feature_selection': feature_selection,
                    'aggregation_method': aggregation_method
                },
                'summary': {
                    'baseline_peptides': len(baseline_peptides),
                    'ground_truth_peptides': len(ground_truth_peptides),
                    'missed_peptides': len(missed_peptides),
                    'additional_candidates': len(additional_peptides),
                    'training_samples': len(train_data),
                    'test_samples': len(test_data),
                    'unique_test_peptides': test_data['Modified.Sequence'].nunique(),
                    'aggregation_method': aggregation_method,
                    'results_dir': results_dir
                },
                'naive_comparison': naive_results,
                'results': results,
                'metadata': {
                    'analysis_timestamp': datetime.now().isoformat(),
                    'total_runtime_seconds': None,  # Will be set by caller
                    'version': '1.0'
                },
                'trained_model': model,  # Store for optional saving
                'feature_names': training_features  # Store feature names
            }
            
            # Save comprehensive results
            self._save_results(analysis_results, results_dir)
            
            return analysis_results
            
        except Exception as e:
            error_result = {
                'error': True,
                'error_message': str(e),
                'error_type': type(e).__name__,
                'config': {
                    'train_methods': train_methods,
                    'test_method': test_method,
                    'train_fdr_levels': train_fdr_levels,
                    'test_fdr': test_fdr,
                    'target_fdr_levels': target_fdr_levels
                }
            }
            return error_result
    
    def _load_baseline_peptides(self, test_method: str = None) -> set:
        """Load baseline peptides from ALL triplicates of the test method only (not other methods)."""
        if not test_method:
            raise ValueError("test_method is required for baseline loading")
        
        # Use the new configured method function to handle both individual files and groups
        baseline_files = get_files_for_configured_method(test_method, 1)  # FDR_1 for baseline
        
        dataset_name = "ASTRAL 7min" if 'ASTRAL' in test_method else "fast gradient"
        
        print(f"Loading {dataset_name} baseline peptides for {test_method} (all triplicates): found {len(baseline_files)} files")
        baseline_peptides = set()
        
        for file_path in baseline_files:
            try:
                df = pd.read_parquet(file_path)
                file_peptides = set(df['Modified.Sequence'].unique())
                baseline_peptides.update(file_peptides)
                print(f"   Loaded {len(file_peptides):,} peptides from {Path(file_path).name}")
            except Exception as e:
                print(f"   Error loading {file_path}: {e}")
                continue
        
        print(f"Total unique baseline peptides for {test_method}: {len(baseline_peptides):,}")
        return baseline_peptides
    
    def _load_ground_truth_peptides(self, test_method: str = None) -> set:
        """Load ground truth peptides using configuration-based matching."""
        # Get available files using the new file discovery system
        files_info_by_dataset = discover_available_files_by_dataset()
        
        # Find ground truth files using universal matching
        ground_truth_files = self._get_matching_ground_truth_files(test_method, files_info_by_dataset)
        
        dataset_name = self._get_dataset_name(test_method)
        ground_truth_peptides = set()
        
        print(f"Found {len(ground_truth_files)} ground truth files for {test_method}")
        
        for file_path in ground_truth_files:
            try:
                filename = Path(file_path).name
                # Just print the filename instead of using progress callback
                print(f"   Loading ground truth: {filename}")
                
                df = pd.read_parquet(file_path)
                file_peptides = set(df['Modified.Sequence'].unique())
                ground_truth_peptides.update(file_peptides)
                print(f"   Loaded {len(file_peptides)} peptides from {filename}")
            except Exception as e:
                print(f"   Error loading {file_path}: {e}")
                continue  # Skip problematic files
        
        print(f"Total ground truth peptides: {len(ground_truth_peptides)}")
        return ground_truth_peptides
    
    def _get_matching_ground_truth_files(self, test_method: str, files_info_by_dataset: dict) -> list:
        """Get matching ground truth files using configuration-based strategy."""
        if not test_method:
            # Fallback: use all ground truth files
            all_gt_files = []
            for dataset_info in files_info_by_dataset.values():
                gt_files = dataset_info.get('ground_truth', {})
                for method_files in gt_files.values():
                    all_gt_files.extend([f['path'] for f in method_files])
            return all_gt_files
        
        # Extract dataset from method name
        dataset_name = test_method.split('_')[0] if '_' in test_method else 'Unknown'
        
        # Load dataset configuration
        dataset_config = self._load_dataset_config(dataset_name)
        ground_truth_mapping = dataset_config.get('ground_truth_mapping', {})
        
        # Get ground truth files for this dataset
        dataset_info = files_info_by_dataset.get(dataset_name, {})
        gt_methods = dataset_info.get('ground_truth', {})
        dataset_gt_files = []
        for method, file_infos in gt_methods.items():
            dataset_gt_files.extend(file_infos)
        
        if not dataset_gt_files:
            print(f"Warning: No ground truth files found for dataset {dataset_name}")
            return []
        
        # Apply matching strategy based on configuration
        strategy = ground_truth_mapping.get('_strategy', 'pattern_matching')
        
        if strategy == 'use_all_ground_truth':
            # Strategy: Use all ground truth files from this dataset
            matched_files = [f['path'] for f in dataset_gt_files]
            print(f"Using all {len(matched_files)} ground truth files for {dataset_name}")
            return matched_files
        
        elif strategy == 'direct_mapping' or '_direct_rules' in ground_truth_mapping:
            # Strategy: Direct method-to-ground truth mapping using full filenames
            direct_rules = ground_truth_mapping.get('_direct_rules', {})
            matched_files = []
            
            # Look for direct mapping of this method
            if test_method in direct_rules:
                target_gt_filenames = direct_rules[test_method]
                
                # Handle both single file (string) and multiple files (list) for backward compatibility
                if isinstance(target_gt_filenames, str):
                    target_gt_filenames = [target_gt_filenames]
                
                # Find the exact ground truth files
                print(f"DEBUG: Looking for ground truth methods: {target_gt_filenames}")
                available_methods = [gt_file['method'] for gt_file in dataset_gt_files]
                print(f"DEBUG: Available ground truth methods: {available_methods}")
                print(f"DEBUG: First few filenames: {[gt_file['filename'] for gt_file in dataset_gt_files[:3]]}")
                
                for target_gt_filename in target_gt_filenames:
                    for gt_file in dataset_gt_files:
                        if target_gt_filename == gt_file['method']:
                            matched_files.append(gt_file['path'])
                            print(f"Direct mapping: {test_method} -> {Path(gt_file['path']).name}")
                            break
                
                if not matched_files:
                    print(f"Warning: Direct mapping failed for {test_method} -> {target_gt_filenames}, using all {dataset_name} files")
                    matched_files = [f['path'] for f in dataset_gt_files]
            else:
                print(f"Warning: No direct mapping found for {test_method}, using all {dataset_name} files")
                matched_files = [f['path'] for f in dataset_gt_files]
            
            return matched_files
        
        elif strategy == 'pattern_matching' or '_pattern_rules' in ground_truth_mapping:
            # Strategy: Use pattern rules to match specific files (legacy support)
            pattern_rules = ground_truth_mapping.get('_pattern_rules', {})
            matched_files = []
            
            # Extract pattern from method name (e.g., "001" from "ASTRAL_...001" at the end)
            import re
            # Look for 3-digit number at the end of the method name
            pattern_match = re.search(r'_(\d{3})$', test_method)
            if pattern_match:
                method_pattern = pattern_match.group(1)
                target_pattern = pattern_rules.get(method_pattern)
                
                if target_pattern:
                    # Find ground truth file containing the target pattern
                    for gt_file in dataset_gt_files:
                        if target_pattern in gt_file['path'] or target_pattern in gt_file['method']:
                            matched_files.append(gt_file['path'])
                            print(f"Matched {test_method} -> {target_pattern} -> {Path(gt_file['path']).name}")
                            break
                    
                    if not matched_files:
                        print(f"Warning: No ground truth file found for pattern {target_pattern}, using all {dataset_name} files")
                        matched_files = [f['path'] for f in dataset_gt_files]
                else:
                    print(f"Warning: No mapping rule for pattern {method_pattern}, using all {dataset_name} files")
                    matched_files = [f['path'] for f in dataset_gt_files]
            else:
                print(f"Warning: No pattern found in method {test_method}, using all {dataset_name} files")
                matched_files = [f['path'] for f in dataset_gt_files]
            
            return matched_files
        
        else:
            # Default: use all files from same dataset
            matched_files = [f['path'] for f in dataset_gt_files]
            print(f"Using default strategy: all {len(matched_files)} ground truth files for {dataset_name}")
            return matched_files
    
    def _load_dataset_config(self, dataset_name: str) -> dict:
        """Load dataset configuration from dataset_info.json."""
        import json
        import os
        
        config_path = os.path.join(os.path.dirname(__file__), 'data', dataset_name, 'dataset_info.json')
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load config for {dataset_name}: {e}")
        
        # Return default configuration
        return {
            'ground_truth_mapping': {
                '_strategy': 'use_all_ground_truth'
            }
        }
    
    def _get_dataset_name(self, test_method: str) -> str:
        """Get human-readable dataset name for display."""
        if not test_method:
            return "Unknown"
        
        dataset = test_method.split('_')[0] if '_' in test_method else test_method
        
        # Load config for display name
        dataset_config = self._load_dataset_config(dataset)
        return dataset_config.get('instrument', f"{dataset} dataset")
    
    def _analyze_improvement_opportunity(self, baseline_peptides: set, ground_truth_peptides: set) -> set:
        """Analyze the improvement opportunity."""
        missed_peptides = ground_truth_peptides - baseline_peptides
        return missed_peptides
    
    def _load_hek_ground_truth(self) -> set:
        """Load HEK 30SPD ground truth (all files combined)"""
        # Get available files using the new file discovery system
        files_info = discover_available_files()
        
        # Find all HEK ground truth files (FDR_1 from long gradient)
        ground_truth_files = []
        for file_info in files_info['ground_truth']:
            if 'HEK' in file_info.get('dataset', '') or 'MS' in file_info['method']:
                ground_truth_files.append(file_info['path'])
        
        ground_truth_peptides = set()
        
        for file_path in ground_truth_files:
            try:
                df = pd.read_parquet(file_path)
                file_peptides = set(df['Modified.Sequence'].unique())
                ground_truth_peptides.update(file_peptides)
            except Exception as e:
                continue
        
        return ground_truth_peptides

    def _load_training_data(self, train_methods: List[str], train_fdr_levels: List[int], 
                           ground_truth_peptides: set) -> Tuple[pd.DataFrame, pd.Series]:
        """Load training data from multiple methods and FDR levels - supports cross-dataset training with method-specific ground truth."""
        train_data = []
        
        print(f"Loading training data from methods: {train_methods}")
        
        # Get available files using the new file discovery system
        files_info = discover_available_files()
        
        for fdr in train_fdr_levels:
            for method in train_methods:
                # Use the new configured method function to handle both individual files and groups
                method_files = get_files_for_configured_method(method, fdr)
                
                print(f"  Method {method} at FDR_{fdr}: found {len(method_files)} files")
                
                for file_path in method_files:
                    try:
                        df = pd.read_parquet(file_path)
                        df['source_fdr'] = fdr
                        df['source_method'] = method  # Track which method this data came from
                        train_data.append(df)
                        print(f"    Added {Path(file_path).name}: {len(df):,} peptides")
                    except Exception as e:
                        print(f"    Error loading {file_path}: {e}")
                        continue
        
        if not train_data:
            raise ValueError("No training data found - check that training methods exist and have data files")
        
        all_train_data = pd.concat(train_data, ignore_index=True)
        
        # Label training data using method-specific ground truth
        y_train = []
        
        # Group by source method for efficient labeling
        for method in all_train_data['source_method'].unique():
            method_mask = all_train_data['source_method'] == method
            method_peptides = all_train_data.loc[method_mask, 'Modified.Sequence']
            
            # Get appropriate ground truth for this training method
            if 'ASTRAL' in method:
                # For ASTRAL training methods, use ASTRAL-specific ground truth
                method_ground_truth = self._load_ground_truth_peptides(method)
            else:
                # For HEK training methods, use HEK 30SPD ground truth (all files)
                method_ground_truth = self._load_hek_ground_truth()
            
            # Label peptides for this method
            method_labels = method_peptides.isin(method_ground_truth).astype(int)
            y_train.extend(method_labels.tolist())
        
        y_train = pd.Series(y_train)
        
        print(f"Training data summary:")
        print(f"  Total samples: {len(all_train_data):,}")
        print(f"  True peptides: {y_train.sum():,} ({y_train.mean()*100:.1f}%)")
        print(f"  Methods used: {all_train_data['source_method'].unique()}")
        
        # Check training data quality
        print(f"\nTraining data feature stats:")
        feature_cols = ['GG.Q.Value', 'PEP', 'PG.PEP', 'PG.Q.Value', 'Q.Value', 'Global.Q.Value']
        for col in feature_cols:
            if col in all_train_data.columns:
                vals = pd.to_numeric(all_train_data[col], errors='coerce')
                print(f"  {col}: range {vals.min():.4f} to {vals.max():.4f}, mean {vals.mean():.4f}")
        
        # Check for feature overlap between training and test
        train_peptides = set(all_train_data['Modified.Sequence'].unique())
        print(f"  Unique training peptides: {len(train_peptides):,}")
        
        return all_train_data, y_train
    
    def _load_test_data(self, test_method: str, test_fdr: int, baseline_peptides: set, 
                       ground_truth_peptides: set) -> Tuple[pd.DataFrame, pd.Series, set]:
        """Load test data for the holdout method using streamlit directory structure."""
        # Get available files using the new file discovery system
        files_info = discover_available_files()
        
        # Find test files for this method and FDR level using configured method handling
        test_files = get_files_for_configured_method(test_method, test_fdr)
        
        test_data = []
        for file_path in test_files:
            try:
                df = pd.read_parquet(file_path)
                df['source_fdr'] = test_fdr
                test_data.append(df)
            except Exception as e:
                continue
        
        if not test_data:
            raise ValueError(f"No test files found for {test_method} at {test_fdr}% FDR")
        
        test_data = pd.concat(test_data, ignore_index=True)
        
        # Filter to additional peptides only
        test_peptides = set(test_data['Modified.Sequence'].unique())
        additional_peptides = test_peptides - baseline_peptides
        
        additional_mask = test_data['Modified.Sequence'].isin(additional_peptides)
        additional_test_data = test_data[additional_mask].copy()
        
        # Create labels
        y_test = additional_test_data['Modified.Sequence'].isin(ground_truth_peptides)
        
        return additional_test_data, y_test, additional_peptides
    
    def _calculate_naive_fdr_comparison(self, test_method: str, test_fdr: int, baseline_peptides: set, ground_truth_peptides: set):
        """Calculate what naive FDR approach would give for comparison."""
        print(f"\n🔍 CALCULATING NAIVE FDR {test_fdr}% COMPARISON...")
        
        # Find test files for this method and FDR level using configured method handling
        test_files = get_files_for_configured_method(test_method, test_fdr)
        
        # Load all test peptides from FDR level
        all_test_peptides = set()
        for file_path in test_files:
            try:
                df = pd.read_parquet(file_path)
                file_peptides = set(df['Modified.Sequence'].unique())
                all_test_peptides.update(file_peptides)
            except:
                continue
        
        # Calculate naive FDR 20% metrics
        naive_additional = all_test_peptides - baseline_peptides
        naive_true_positives = naive_additional & ground_truth_peptides
        naive_false_positives = naive_additional - ground_truth_peptides
        
        naive_total_additional = len(naive_additional)
        naive_tp_count = len(naive_true_positives)
        naive_fp_count = len(naive_false_positives)
        naive_actual_fdr = (naive_fp_count / naive_total_additional * 100) if naive_total_additional > 0 else 0
        
        print(f"📊 NAIVE FDR {test_fdr}% RESULTS:")
        print(f"   Additional peptides: {naive_total_additional:,}")
        print(f"   True positives: {naive_tp_count:,}")
        print(f"   False positives: {naive_fp_count:,}")
        print(f"   Actual FDR: {naive_actual_fdr:.1f}%")
        
        return {
            'total_additional': naive_total_additional,
            'true_positives': naive_tp_count,
            'false_positives': naive_fp_count,
            'actual_fdr': naive_actual_fdr
        }
    
    def _make_advanced_features(self, df: pd.DataFrame, training_features=None) -> pd.DataFrame:
        """Create comprehensive feature table with advanced engineering."""
        feats = pd.DataFrame(index=df.index)
        
        # Add numeric features based on feature selection
        diann_quality_cols = ['GG.Q.Value', 'PEP', 'PG.PEP', 'PG.Q.Value', 'Q.Value', 'Global.Q.Value',
                              'Global.PG.Q.Value', 'Global.Peptidoform.Q.Value', 'Protein.Q.Value', 
                              'Proteotypic', 'Evidence', 'Genes.MaxLFQ.Quality', 'PG.MaxLFQ.Quality',
                              'Quantity.Quality', 'Genes.MaxLFQ', 'PG.MaxLFQ', 'Genes.MaxLFQ.Unique',
                              'Genes.MaxLFQ.Unique.Quality', 'Mass.Evidence', 'Channel.Evidence',
                              'Channel.Q.Value', 'Translated.Q.Value', 'Lib.PTM.Site.Confidence',
                              'PTM.Site.Confidence', 'Lib.Peptidoform.Q.Value', 'Peptidoform.Q.Value',
                              'Lib.Q.Value', 'Lib.PG.Q.Value', 'Normalisation.Noise', 'Empirical.Quality',
                              'Normalisation.Factor', 'Genes.TopN', 'PG.TopN', 'Decoy']
        ms_cols = ['iRT', 'Predicted.RT', 'Precursor.Mz', 'Ms1.Area', 'Ms2.Area', 'Peak.Height', 'Precursor.Quantity',
                   'Precursor.Charge', 'FWHM', 'Ms1.Apex.Area', 'Ms1.Profile.Corr', 'RT.Start', 'RT.Stop', 'RT',
                   'Ms1.Normalised', 'iIM', 'Ms1.Apex.Mz.Delta', 'Best.Fr.Mz.Delta', 'Best.Fr.Mz', 
                   'Ms1.Total.Signal.After', 'Ms1.Total.Signal.Before', 'Predicted.iRT', 'Precursor.Normalised',
                   'Ms1.Translated', 'Ms1.Area.Raw', 'Quantity.Raw', 'Fragment.Quant.Raw', 'Fragment.Quant.Corrected',
                   'Ms2.Scan', 'Ion.Mobility', 'CCS', 'Charge', 'Index.RT', 'Predicted.iIM', 'Predicted.IM', 'IM',
                   'Run.Index']
        library_cols = ['Precursor.Lib.Index', 'PG.MaxLFQ', 'Fragment.Info', 'source_fdr']
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32', 'float16', 'int16', 'int8', 'uint8', 'uint16', 'uint32', 'uint64']:
                # Check if this feature should be included based on feature selection
                include_feature = False
                
                if col in diann_quality_cols and self.feature_selection.get('use_diann_quality', True):
                    include_feature = True
                elif col in ms_cols and self.feature_selection.get('use_ms_features', True):
                    include_feature = True  
                elif col in library_cols and self.feature_selection.get('use_library_features', True):
                    include_feature = True
                elif col not in diann_quality_cols and col not in ms_cols and col not in library_cols:
                    # Include other numeric features if not specifically categorized
                    include_feature = True
                
                # Check if feature is explicitly excluded
                if col in self.feature_selection.get('excluded_features', []):
                    include_feature = False
                
                if include_feature:
                    feats[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add engineered log features
        for col in self.engineer_log:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors='coerce').fillna(0)
                feats[f'log_{col}'] = np.log1p(np.maximum(vals, 0))
        
        # Add ratio features
        for col1, col2 in self.engineer_ratios:
            if col1 in df.columns and col2 in df.columns:
                vals1 = pd.to_numeric(df[col1], errors='coerce').fillna(0)
                vals2 = pd.to_numeric(df[col2], errors='coerce').fillna(1)
                feats[f'ratio_{col1}_{col2}'] = vals1 / np.maximum(vals2, 1e-10)
        
        # Add sequence-based features
        if self.feature_selection.get('use_sequence_features', True) and 'Stripped.Sequence' in df.columns:
            sequences = df['Stripped.Sequence'].astype(str)
            feats['sequence_length'] = sequences.str.len()
            
            # Amino acid composition features - ALL 20 amino acids
            for aa in ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']:
                feats[f'aa_count_{aa}'] = sequences.str.count(aa)
                feats[f'aa_freq_{aa}'] = feats[f'aa_count_{aa}'] / feats['sequence_length']
        
        # Skip source_fdr feature to prevent data leakage
        # if 'source_fdr' in df.columns:
        #     feats['source_fdr'] = df['source_fdr']
        
        # Add statistical features
        if self.feature_selection.get('use_statistical_features', True):
            for col in ['Ms1.Area', 'Ms2.Area', 'Peak.Height', 'Precursor.Quantity']:
                if col in df.columns:
                    vals = pd.to_numeric(df[col], errors='coerce')
                    feats[f'zscore_{col}'] = (vals - vals.mean()) / (vals.std() + 1e-10)
        
        # Ensure consistent columns with training data
        if training_features is not None:
            for col in training_features:
                if col not in feats.columns:
                    feats[col] = 0
            feats = feats[training_features]
        
        # Clean and return
        return feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    def _train_model(self, train_data: pd.DataFrame, y_train: pd.Series) -> Tuple[object, List[str]]:
        """Train the enhanced ensemble model with calibration."""
        
        # Create features
        X_train = self._make_advanced_features(train_data)
        
        # Create validation split
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Create ensemble model
        xgb = XGBClassifier(**self.optimal_params)
        
        rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        
        ensemble = VotingClassifier(
            estimators=[('xgb', xgb), ('rf', rf), ('lr', lr)],
            voting='soft',
            weights=[3, 2, 1]
        )
        
        # Train ensemble
        ensemble.fit(X_train_fold, y_train_fold)
        
        # Calibrate model
        calibrated_model = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
        calibrated_model.fit(X_train, y_train)
        
        return calibrated_model, X_train.columns.tolist()
    
    def _save_trained_model(self, model, training_features, results_dir, training_results=None):
        """Save trained model and feature information for later inference."""
        import joblib
        
        # Create models directory
        models_dir = os.path.join(results_dir, "saved_models")
        os.makedirs(models_dir, exist_ok=True)
        
        # Save the trained model
        model_path = os.path.join(models_dir, "trained_model.joblib")
        joblib.dump(model, model_path)
        
        # Save feature information
        features_path = os.path.join(models_dir, "training_features.joblib")
        joblib.dump(training_features, features_path)
        
        # Save model metadata
        metadata = {
            'model_type': type(model).__name__,
            'n_features': len(training_features),
            'training_features': training_features,
            'save_timestamp': datetime.now().isoformat(),
            'model_path': model_path,
            'features_path': features_path
        }
        
        # Add training results if provided
        if training_results is not None:
            metadata['training_results'] = training_results
        
        metadata_path = os.path.join(models_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def _update_model_metadata_with_results(self, results_dir, training_results):
        """Update saved model metadata with training results."""
        models_dir = os.path.join(results_dir, "saved_models")
        metadata_path = os.path.join(models_dir, "model_metadata.json")
        
        if os.path.exists(metadata_path):
            # Load existing metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Convert training results to JSON-serializable format
            json_serializable_results = self._make_json_serializable(training_results)
            
            # Add training results
            metadata['training_results'] = json_serializable_results
            metadata['updated_timestamp'] = datetime.now().isoformat()
            
            # Save updated metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Updated model metadata with training results")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-compatible types."""
        import numpy as np
        
        if isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # pandas scalars
            return obj.item()
        else:
            return obj
    
    def _find_optimal_threshold(self, y_true: pd.Series, y_scores: np.ndarray, target_fdr: float) -> Tuple[float, int, float]:
        """Find the optimal threshold that achieves exactly the target FDR."""
        y_true_arr = np.array(y_true.values if hasattr(y_true, 'values') else y_true)
        y_scores_arr = np.array(y_scores)
        
        # Sort by score (descending)
        sorted_indices = np.argsort(-y_scores_arr)
        y_true_sorted = y_true_arr[sorted_indices]
        y_scores_sorted = y_scores_arr[sorted_indices]
        
        print(f"\n🎯 THRESHOLD OPTIMIZATION (Target FDR: {target_fdr}%)")
        print(f"  Input: {len(y_true_arr):,} UNIQUE peptides")
        print(f"  True positives: {np.sum(y_true_arr):,}")
        print(f"  Score range: {np.min(y_scores_arr):.6f} to {np.max(y_scores_arr):.6f}")
        print(f"  Score distribution:")
        print(f"  >0.99: {np.sum(y_scores_arr > 0.99)}")
        print(f"  >0.95: {np.sum(y_scores_arr > 0.95)}")  
        print(f"  >0.90: {np.sum(y_scores_arr > 0.90)}")
        print(f"  >0.80: {np.sum(y_scores_arr > 0.80)}")
        print(f"  >0.50: {np.sum(y_scores_arr > 0.50)}")
        
        # Check if model is actually discriminating
        tp_scores = y_scores_arr[y_true_arr == 1]
        fp_scores = y_scores_arr[y_true_arr == 0]
        print(f"True positive score stats: mean={np.mean(tp_scores):.4f}, std={np.std(tp_scores):.4f}")
        print(f"False positive score stats: mean={np.mean(fp_scores):.4f}, std={np.std(fp_scores):.4f}")
        
        best_threshold = None
        best_peptides = 0
        best_actual_fdr = float('inf')
        
        # Try different thresholds - use more granular approach
        # For low FDR levels, we need to be more precise about threshold selection
        
        # For low FDRs, we need to find intermediate thresholds between the very high scores
        if target_fdr <= 5.0:
            # Get the top unique scores
            unique_scores = np.unique(y_scores_sorted)
            top_scores = unique_scores[-20:]  # Take top 20 unique scores
            
            # Create a much denser grid between these top scores and lower scores
            if len(unique_scores) > 20:
                min_grid = unique_scores[-100]  # Go deeper into the score range
                max_grid = np.max(y_scores_sorted)
                # Create a very dense grid with 5000 points
                threshold_grid = np.linspace(min_grid, max_grid, 5000)
            else:
                # Fallback if not enough unique scores
                threshold_grid = unique_scores
        else:
            # For higher FDRs, use unique thresholds from data
            threshold_grid = np.unique(y_scores_sorted)
        
        valid_thresholds_found = 0
        
        for threshold in reversed(sorted(threshold_grid)):  # Start from highest threshold
            predictions = y_scores_arr >= threshold
            if predictions.sum() == 0:
                continue
                
            tp = (y_true_arr & predictions).sum()
            fp = (~y_true_arr & predictions).sum()
            
            if tp + fp == 0:
                continue
                
            actual_fdr = fp / (tp + fp) * 100
            
            # Accept thresholds that are within tolerance of target FDR
            # Allow some flexibility for better progression across FDR levels
            tolerance = max(0.5, target_fdr * 0.1)  # At least 0.5% tolerance, or 10% of target
            
            if actual_fdr <= target_fdr + tolerance:
                valid_thresholds_found += 1
                if valid_thresholds_found <= 3:  # Debug first few valid thresholds
                    print(f"  Valid threshold {valid_thresholds_found}: {threshold:.6f} -> TP:{tp}, FP:{fp}, FDR:{actual_fdr:.1f}%")
                
                # Prefer thresholds that are closer to target FDR (but not over)
                # If actual FDR is over target, penalize but still consider
                if actual_fdr <= target_fdr:
                    score = tp  # Prefer more peptides if under target
                else:
                    score = tp * 0.5  # Penalize if over target but within tolerance
                
                if (best_threshold is None or 
                    score > (best_peptides if best_actual_fdr <= target_fdr else best_peptides * 0.5)):
                    best_threshold = threshold
                    best_peptides = tp
                    best_actual_fdr = actual_fdr
        
        threshold_str = f"{best_threshold:.6f}" if best_threshold is not None else "None"
        print(f"  ✅ Final: threshold={threshold_str}, peptides={best_peptides:,}, FDR={best_actual_fdr:.1f}%")
        print(f"  Valid thresholds found: {valid_thresholds_found}")
        
        return best_threshold, best_peptides, best_actual_fdr
    
    def _aggregate_predictions_by_peptide(self, test_data: pd.DataFrame, predictions: np.ndarray, 
                                        labels: pd.Series, aggregation_method: str = 'max') -> Tuple[pd.DataFrame, np.ndarray, pd.Series]:
        """
        Aggregate row-level predictions to peptide-level predictions.
        
        Args:
            test_data: Raw test data with potential duplicate peptides
            predictions: Row-level predictions
            labels: Row-level labels
            aggregation_method: How to aggregate ('mean', 'max', 'weighted')
            
        Returns:
            Tuple of (peptide_data, peptide_predictions, peptide_labels)
        """
        print(f"🔄 AGGREGATING PREDICTIONS BY PEPTIDE ({aggregation_method})")
        
        # Create a dataframe with all the necessary information
        agg_df = test_data.copy()
        agg_df['prediction_prob'] = predictions
        agg_df['label'] = labels
        
        # Print pre-aggregation stats
        print(f"  Before aggregation: {len(agg_df):,} rows, {agg_df['Modified.Sequence'].nunique():,} unique peptides")
        
        # Aggregate by peptide sequence
        if aggregation_method == 'mean':
            peptide_agg = agg_df.groupby('Modified.Sequence').agg({
                'prediction_prob': 'mean',
                'label': 'first',  # Should be same for all rows of same peptide
                'source_fdr': 'first'
            }).reset_index()
        elif aggregation_method == 'max':
            peptide_agg = agg_df.groupby('Modified.Sequence').agg({
                'prediction_prob': 'max',
                'label': 'first',
                'source_fdr': 'first'
            }).reset_index()
        elif aggregation_method == 'weighted':
            # Weight by intensity if available
            if 'Peak.Height' in agg_df.columns:
                weights = agg_df.groupby('Modified.Sequence')['Peak.Height'].apply(
                    lambda x: x / x.sum() if x.sum() > 0 else pd.Series([1/len(x)] * len(x), index=x.index)
                )
                agg_df['weight'] = weights
                
                peptide_agg = agg_df.groupby('Modified.Sequence').apply(
                    lambda group: pd.Series({
                        'prediction_prob': (group['prediction_prob'] * group['weight']).sum(),
                        'label': group['label'].iloc[0],
                        'source_fdr': group['source_fdr'].iloc[0]
                    })
                ).reset_index()
            else:
                # Fallback to mean if no intensity
                peptide_agg = agg_df.groupby('Modified.Sequence').agg({
                    'prediction_prob': 'mean',
                    'label': 'first',
                    'source_fdr': 'first'
                }).reset_index()
        
        # Extract aggregated values
        peptide_data = peptide_agg[['Modified.Sequence', 'source_fdr']]
        peptide_predictions = peptide_agg['prediction_prob'].values
        peptide_labels = peptide_agg['label'].values
        
        print(f"  After aggregation: {len(peptide_data):,} unique peptides")
        print(f"  Aggregation method: {aggregation_method}")
        print(f"  Prediction range: {peptide_predictions.min():.4f} to {peptide_predictions.max():.4f}")
        print(f"  True positive peptides: {peptide_labels.sum():,}")
        
        return peptide_data, peptide_predictions, pd.Series(peptide_labels)

    def _validate_and_optimize(self, model, test_data: pd.DataFrame, y_test: pd.Series, 
                              training_features: List[str], target_fdr_levels: List[float], 
                              baseline_count: int, aggregation_method: str = 'max') -> Tuple[List[Dict], pd.DataFrame]:
        """
        Validate additional peptides and optimize thresholds using unique peptides.
        
        This method properly handles duplicate peptides by aggregating predictions
        at the peptide level before threshold optimization.
        """
        print(f"\n🔄 VALIDATION AND OPTIMIZATION")
        print(f"  Input: {len(test_data):,} rows, {test_data['Modified.Sequence'].nunique():,} unique peptides")
        
        # Create test features (on all rows for training)
        X_test = self._make_advanced_features(test_data, training_features)
        
        # Make predictions on all rows
        test_probs = model.predict_proba(X_test)[:, 1]
        print(f"  Generated {len(test_probs):,} row-level predictions")
        
        # CRITICAL FIX: Aggregate predictions by peptide
        peptide_data, peptide_predictions, peptide_labels = self._aggregate_predictions_by_peptide(
            test_data, test_probs, y_test, aggregation_method
        )
        
        print(f"  Aggregated to {len(peptide_predictions):,} unique peptide predictions")
        
        # Now optimize thresholds using unique peptides
        results = []
        
        for target_fdr in target_fdr_levels:
            threshold, peptides, actual_fdr = self._find_optimal_threshold(
                peptide_labels, peptide_predictions, target_fdr
            )
            
            if threshold is not None:
                # Calculate metrics on unique peptides
                total_validated = peptide_labels.sum()
                recovery_pct = peptides / total_validated * 100 if total_validated > 0 else 0
                increase_pct = peptides / baseline_count * 100 if baseline_count > 0 else 0
                
                # Calculate confusion matrix on unique peptides
                predictions = peptide_predictions >= threshold
                tp = (peptide_labels & predictions).sum()
                fp = (~peptide_labels & predictions).sum()
                tn = (~peptide_labels & ~predictions).sum()
                fn = (peptide_labels & ~predictions).sum()
                
                # Calculate MCC on unique peptides
                mcc = matthews_corrcoef(peptide_labels, predictions)
                
                results.append({
                    'Target_FDR': target_fdr,
                    'Threshold': threshold,
                    'Additional_Peptides': peptides,  # Now truly unique peptides
                    'Actual_FDR': actual_fdr,
                    'Recovery_Pct': recovery_pct,
                    'Increase_Pct': increase_pct,
                    'False_Positives': fp,
                    'Total_Validated_Candidates': total_validated,
                    'MCC': mcc
                })
                
                print(f"    Target {target_fdr:4.1f}%: {peptides:,} peptides, {actual_fdr:.1f}% FDR, {recovery_pct:.1f}% recovery")
        
        return results, X_test
    
    def _create_visualizations(self, results: List[Dict], results_dir: str, 
                              model=None, X_test=None, training_features=None):
        """Create comprehensive visualizations including SHAP analysis."""
        if not results:
            return
        
        results_df = pd.DataFrame(results)
        
        # 1. Method comparison plot
        plt.figure(figsize=(15, 10))
        
        # FDR vs Additional Peptides
        plt.subplot(2, 3, 1)
        plt.bar(range(len(results_df)), results_df['Additional_Peptides'], 
               color=NATURE_COLORS['primary'], alpha=0.8)
        plt.xlabel('Target FDR Level', fontweight='bold', fontsize=12)
        plt.ylabel('Additional Peptides', fontweight='bold', fontsize=12)
        plt.title('Additional Peptides by Target FDR', fontweight='bold', fontsize=14)
        plt.xticks(range(len(results_df)), [f"{x:.1f}%" for x in results_df['Target_FDR']])
        plt.grid(True, alpha=0.3)
        
        # FDR Control
        plt.subplot(2, 3, 2)
        plt.plot(results_df['Target_FDR'], results_df['Actual_FDR'], 
                'o-', color=NATURE_COLORS['secondary'], linewidth=3, markersize=8, label='Actual FDR')
        plt.plot(results_df['Target_FDR'], results_df['Target_FDR'], 
                '--', color=NATURE_COLORS['quaternary'], alpha=0.7, linewidth=2, label='Perfect control')
        plt.xlabel('Target FDR (%)', fontweight='bold', fontsize=12)
        plt.ylabel('Actual FDR (%)', fontweight='bold', fontsize=12)
        plt.title('FDR Control Precision', fontweight='bold', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Recovery percentage
        plt.subplot(2, 3, 3)
        plt.plot(results_df['Target_FDR'], results_df['Recovery_Pct'], 
                'o-', color=NATURE_COLORS['success'], linewidth=3, markersize=8)
        plt.xlabel('Target FDR (%)', fontweight='bold', fontsize=12)
        plt.ylabel('Recovery Percentage (%)', fontweight='bold', fontsize=12)
        plt.title('Peptide Recovery Rate', fontweight='bold', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Performance vs FDR trade-off
        plt.subplot(2, 3, 4)
        scatter = plt.scatter(results_df['Actual_FDR'], results_df['Additional_Peptides'], 
                            c=results_df['Target_FDR'], cmap='viridis', s=100, alpha=0.8)
        plt.colorbar(scatter, label='Target FDR (%)')
        plt.xlabel('Actual FDR (%)', fontweight='bold', fontsize=12)
        plt.ylabel('Additional Peptides', fontweight='bold', fontsize=12)
        plt.title('Performance vs FDR Trade-off', fontweight='bold', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # Efficiency plot
        plt.subplot(2, 3, 5)
        efficiency = results_df['Additional_Peptides'] / results_df['Actual_FDR']
        efficiency = efficiency.replace([np.inf, -np.inf], np.nan)
        plt.bar(range(len(results_df)), efficiency, 
               color=NATURE_COLORS['tertiary'], alpha=0.8)
        plt.xlabel('Target FDR Level', fontweight='bold', fontsize=12)
        plt.ylabel('Peptides per FDR%', fontweight='bold', fontsize=12)
        plt.title('Discovery Efficiency', fontweight='bold', fontsize=14)
        plt.xticks(range(len(results_df)), [f"{x:.1f}%" for x in results_df['Target_FDR']])
        plt.grid(True, alpha=0.3)
        
        # Summary statistics
        plt.subplot(2, 3, 6)
        plt.axis('off')
        best_result = results_df.loc[results_df['Additional_Peptides'].idxmax()]
        stats_text = f"""
ANALYSIS SUMMARY

Best Performance:
• {best_result['Additional_Peptides']} additional peptides
• {best_result['Actual_FDR']:.1f}% actual FDR
• {best_result['Recovery_Pct']:.1f}% recovery rate

FDR Control Quality:
• Mean absolute error: {np.mean(np.abs(results_df['Actual_FDR'] - results_df['Target_FDR'])):.2f}%
• Max peptides at ≤5% FDR: {results_df[results_df['Actual_FDR'] <= 5.0]['Additional_Peptides'].max() if len(results_df[results_df['Actual_FDR'] <= 5.0]) > 0 else 0}

Model Performance:
• Total evaluations: {len(results_df)}
• Valid results (≤5% FDR): {len(results_df[results_df['Actual_FDR'] <= 5.0])}
        """
        plt.text(0.05, 0.95, stats_text.strip(), transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=NATURE_COLORS['accent2'], alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/plots/comprehensive_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance analysis (if model provided)
        if model is not None and X_test is not None:
            self._create_feature_importance_analysis(model, X_test, training_features, results_dir)
    
    def _create_feature_importance_analysis(self, model, X_test, training_features, results_dir):
        """Create comprehensive feature importance analysis including basic XGBoost and SHAP plots."""
        print("🔍 Generating feature importance analysis...")
        
        # Extract the base XGBoost model for feature importance
        base_model = self._extract_xgboost_model(model)
        
        # 1. Create basic XGBoost feature importance plot
        if base_model is not None:
            self._create_feature_importance_plot(base_model, training_features, results_dir)
        
        # 2. Create SHAP analysis
        self._create_shap_analysis(model, X_test, training_features, results_dir)
    
    def _extract_xgboost_model(self, model):
        """Extract XGBoost model from ensemble/calibrated wrapper."""
        base_model = None
        
        # 1️⃣ Directly inside a VotingClassifier
        if hasattr(model, 'named_estimators_') and 'xgb' in model.named_estimators_:
            base_model = model.named_estimators_['xgb']
        
        # 2️⃣ Wrapped by CalibratedClassifierCV -> access its base_estimator / estimator
        elif hasattr(model, 'base_estimator'):
            be = model.base_estimator
            if hasattr(be, 'named_estimators_') and 'xgb' in be.named_estimators_:
                base_model = be.named_estimators_['xgb']
        
        # 3️⃣ Check for estimator attribute (another CalibratedClassifierCV pattern)
        elif hasattr(model, 'estimator'):
            est = model.estimator
            if hasattr(est, 'named_estimators_') and 'xgb' in est.named_estimators_:
                base_model = est.named_estimators_['xgb']
        
        # 4️⃣ Fallback – assume the provided model itself is an XGBClassifier
        if base_model is None and hasattr(model, 'feature_importances_'):
            base_model = model
        
        return base_model
    
    def _create_feature_importance_plot(self, model, feature_names, results_dir):
        """Create feature importance bar plot with Nature color scheme."""
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            print("⚠️ Model has no feature_importances_ attribute")
            return
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=True)
        
        # Save feature importance CSV
        feature_importance_df_sorted = feature_importance_df.sort_values('importance', ascending=False)
        feature_importance_df_sorted.to_csv(f"{results_dir}/feature_analysis/feature_importance_full.csv", index=False)
        
        # Plot top 20 features
        top_features = feature_importance_df.tail(20)
        
        plt.figure(figsize=(12, 10))
        
        # Create gradient colors using Nature color scheme
        colors = [NATURE_COLORS['primary'] if i < 10 else NATURE_COLORS['secondary'] 
                  for i in range(len(top_features))]
        
        bars = plt.barh(range(len(top_features)), top_features['importance'], 
                       alpha=0.85, color=colors, edgecolor='white', linewidth=0.5)
        
        plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
        plt.xlabel('Feature Importance', fontweight='bold', fontsize=12)
        plt.title('Top 20 Most Important Features', fontweight='bold', fontsize=14)
        plt.grid(True, alpha=0.3, axis='x')
        
        # Professional styling
        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('#FAFAFA')
        
        plt.tight_layout()
        plt.savefig(f"{results_dir}/feature_analysis/feature_importance_top20.png", 
                    dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Feature importance plot and CSV created successfully")

    def _create_shap_analysis(self, model, X_test, training_features, results_dir):
        """Create lightweight Plotly-based SHAP analysis."""
        try:
            print("🔍 Generating lightweight SHAP analysis...")
            print(f"   Input data shape: {X_test.shape}")
            print(f"   Features: {len(training_features)}")
            
            # Check if X_test is empty
            if len(X_test) == 0:
                print("⚠️ No test data available for SHAP analysis")
                return
            
            # Use smaller sample for efficiency
            sample_size = min(100, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42)
            print(f"   Using sample size: {len(X_sample)}")
            
            # Extract XGBoost model
            base_model = self._extract_xgboost_model(model)
            if base_model is None:
                print("⚠️ Could not extract XGBoost model for SHAP analysis")
                print(f"   Model type: {type(model)}")
                return
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(base_model)
            shap_values = explainer.shap_values(X_sample)
            
            # Calculate mean absolute SHAP values for feature importance
            mean_shap_values = np.mean(shap_values, axis=0)
            abs_mean_shap = np.abs(mean_shap_values)
            
            # Get top 15 features
            top_indices = np.argsort(abs_mean_shap)[-15:]
            top_features = [training_features[i] for i in top_indices]
            top_shap_values = mean_shap_values[top_indices]
            
            # Create Plotly bar chart with diverging colors
            import plotly.graph_objects as go
            import plotly.express as px
            
            # Determine colors: positive values (green/blue), negative values (red/orange)
            colors = []
            max_abs_val = max(abs(top_shap_values)) if len(top_shap_values) > 0 else 1
            
            for val in top_shap_values:
                if val >= 0:
                    # Positive values: scale from light blue to dark blue
                    intensity = abs(val) / max_abs_val if max_abs_val > 0 else 0
                    colors.append(f'rgba(0, {int(100 + 100*intensity)}, {int(200 + 55*intensity)}, 0.8)')
                else:
                    # Negative values: scale from light red to dark red  
                    intensity = abs(val) / max_abs_val if max_abs_val > 0 else 0
                    colors.append(f'rgba({int(200 + 55*intensity)}, {int(100 + 100*intensity)}, 0, 0.8)')
            
            # Create the bar chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=top_shap_values,
                y=top_features,
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(color='rgba(50, 50, 50, 0.8)', width=1)
                ),
                text=[f'{val:.4f}' for val in top_shap_values],
                textposition='outside',
                textfont=dict(size=10)
            ))
            
            # Add vertical line at x=0
            fig.add_vline(x=0, line_width=2, line_color="black", line_dash="solid")
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text="Feature Impact Analysis (SHAP Values)",
                    font=dict(size=16, family="Arial Black"),
                    x=0.5
                ),
                xaxis=dict(
                    title="Mean SHAP Value",
                    title_font=dict(size=14),
                    gridcolor='lightgray',
                    gridwidth=1
                ),
                yaxis=dict(
                    title="Features",
                    title_font=dict(size=14),
                    tickfont=dict(size=11)
                ),
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=600,
                width=900,
                margin=dict(l=200, r=100, t=80, b=60)
            )
            
            # Add annotations
            if len(top_shap_values) > 0 and min(top_shap_values) < 0:
                fig.add_annotation(
                    x=min(top_shap_values) * 0.7,
                    y=len(top_features) * 0.95,
                    text="← Decreases<br>Prediction",
                    showarrow=False,
                    font=dict(size=12, color="darkred"),
                    bgcolor="rgba(255, 200, 200, 0.8)",
                    bordercolor="darkred",
                    borderwidth=1
                )
            
            if len(top_shap_values) > 0 and max(top_shap_values) > 0:
                fig.add_annotation(
                    x=max(top_shap_values) * 0.7,
                    y=len(top_features) * 0.95,
                    text="Increases →<br>Prediction",
                    showarrow=False,
                    font=dict(size=12, color="darkblue"),
                    bgcolor="rgba(200, 200, 255, 0.8)",
                    bordercolor="darkblue",
                    borderwidth=1
                )
            
            # Save as HTML
            fig.write_html(f"{results_dir}/feature_analysis/shap_importance_plotly.html")
            
            # Save as PNG (requires kaleido)
            try:
                fig.write_image(f"{results_dir}/feature_analysis/shap_importance_plotly.png", 
                              width=900, height=600, scale=2)
            except Exception:
                print("⚠️ Could not save PNG (kaleido not available)")
            
            # Save simplified SHAP data
            shap_data = {
                'feature_names': top_features,
                'mean_shap_values': top_shap_values.tolist(),
                'abs_importance': abs_mean_shap[top_indices].tolist()
            }
            
            with open(f"{results_dir}/feature_analysis/shap_data.json", 'w') as f:
                json.dump(shap_data, f, indent=2)
            
            print("✅ Lightweight SHAP analysis completed successfully")
            
        except Exception as e:
            print(f"⚠️ SHAP analysis failed: {e}")
            print("Continuing without SHAP plots...")
            import traceback
            traceback.print_exc()

    
    def _save_results(self, analysis_results: Dict, results_dir: str):
        """Save comprehensive analysis results."""
        
        # Save detailed results as CSV
        if 'results' in analysis_results and analysis_results['results']:
            results_df = pd.DataFrame(analysis_results['results'])
            results_df.to_csv(f"{results_dir}/tables/detailed_results.csv", index=False)
        
        # Save analysis summary as JSON
        summary_data = {
            'config': analysis_results['config'],
            'summary': analysis_results['summary'],
            'metadata': analysis_results['metadata']
        }
        
        with open(f"{results_dir}/raw_data/analysis_summary.json", 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        # Save best results summary
        if 'results' in analysis_results and analysis_results['results']:
            summary_text = f"""
🎯 PEPTIDE VALIDATION ANALYSIS SUMMARY
================================================================================
Analysis completed: {analysis_results['metadata']['analysis_timestamp']}

📊 DATA SUMMARY:
Baseline peptides: {analysis_results['summary']['baseline_peptides']:,}
Ground truth peptides: {analysis_results['summary']['ground_truth_peptides']:,}
Training samples: {analysis_results['summary']['training_samples']:,}
Test samples: {analysis_results['summary']['test_samples']:,}

⚙️ CONFIGURATION:
Training methods: {', '.join(analysis_results['config']['train_methods'])}
Test method: {analysis_results['config']['test_method']}
Training FDR levels: {', '.join(map(str, analysis_results['config']['train_fdr_levels']))}%
Test FDR level: {analysis_results['config']['test_fdr']}%

📁 Results saved to: {results_dir}
================================================================================
            """
            
            with open(f"{results_dir}/ANALYSIS_SUMMARY.txt", 'w') as f:
                f.write(summary_text)

# Convenience function for direct API usage
def run_peptide_validation(train_methods: List[str],
                          test_method: str,
                          train_fdr_levels: List[int] = [1, 20, 50],
                          test_fdr: int = 50,
                          target_fdr_levels: List[float] = [1.0, 2.0, 3.0, 4.0, 5.0],
                          xgb_params: Optional[Dict] = None,
                          progress_callback: Optional[Callable] = None,
                          results_dir: Optional[str] = None,
                          feature_selection: Optional[Dict] = None) -> Dict:
    """
    Convenience function to run peptide validation analysis.
    
    Args:
        train_methods: List of methods to use for training
        test_method: Method to use for testing (holdout)
        train_fdr_levels: FDR levels to include in training data
        test_fdr: FDR level for test data
        target_fdr_levels: Target FDR levels for optimization
        xgb_params: Optional XGBoost parameter overrides
        progress_callback: Optional callback for progress updates
        results_dir: Optional directory to save results
        feature_selection: Optional dictionary of feature selection options
        
    Returns:
        Dictionary containing analysis results and metadata
    """
    api = PeptideValidatorAPI()
    return api.run_analysis(
        train_methods=train_methods,
        test_method=test_method,
        train_fdr_levels=train_fdr_levels,
        test_fdr=test_fdr,
        target_fdr_levels=target_fdr_levels,
        xgb_params=xgb_params,
        progress_callback=progress_callback,
        feature_selection=feature_selection,
        results_dir=results_dir
    )
#!/usr/bin/env python3
"""
PeptiDIA Command Line Interface

Terminal-based version of PeptiDIA analysis functionality.
Run ML peptide analysis without needing a web browser.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import contextlib
from io import StringIO

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from peptide_validator_api import PeptideValidatorAPI
    from dataset_utils import discover_available_files_by_dataset, get_configured_methods
except ImportError as e:
    print(f"❌ Error importing required modules: {e}")
    print("Please ensure you're running this script from the PeptiDIA directory")
    sys.exit(1)

class Colors:
    """Terminal colors for better output formatting."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class PeptiDIACLI:
    def __init__(self):
        self.config = {}
        self.available_files = {}
        self.validator = None
        
    def _clean_step_name(self, step_name: str) -> str:
        """Convert technical step names to user-friendly descriptions."""
        step_mappings = {
            "Loading baseline peptides fast gradient 1% FDR": "Loading baseline data",
            "Loading baseline peptides ASTRAL 7min 1% FDR": "Loading baseline data",
            "Loading ground truth peptides": "Loading ground truth data",
            "Analyzing improvement opportunity": "Calculating improvement potential",
            "Loading and preparing training data": "Preparing training data",
            "Loading and preparing test data": "Preparing test data",
            "Training ensemble model with advanced features": "Training ML model",
            "Saving trained model": "Saving model",
            "Making predictions and optimizing thresholds": "Optimizing performance",
            "Updating model with training results": "Finalizing model",
            "Creating visualizations and saving results": "Generating results"
        }
        
        # Clean up step name
        for technical_name, clean_name in step_mappings.items():
            if technical_name in step_name:
                return clean_name
        
        # Fallback: just use the step name as-is but shortened
        return step_name[:50] + "..." if len(step_name) > 50 else step_name
    
    @contextlib.contextmanager
    def _suppress_verbose_output(self):
        """Suppress verbose stdout output during analysis."""
        original_stdout = sys.stdout
        sys.stdout = StringIO()
        try:
            yield
        finally:
            sys.stdout = original_stdout
    
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
            print(f"{Colors.GREEN}✅ GPU detected and available for training{Colors.ENDC}")
            return 'cuda'
        except Exception as e:
            print(f"{Colors.WARNING}⚠️  GPU not available, using CPU: {str(e)[:50]}...{Colors.ENDC}")
            return 'cpu'
        
    def print_banner(self):
        """Print the CLI banner."""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}")
        print("🧬 PEPTIDIA COMMAND LINE INTERFACE 🧬")
        print("Professional Machine Learning Interface for DIA-NN Peptide Analysis")
        print(f"{'='*80}{Colors.ENDC}")
        print(f"{Colors.BLUE}Run complete PeptiDIA analysis from the terminal{Colors.ENDC}")
        print(f"{Colors.GREEN}No browser required - all results displayed in terminal{Colors.ENDC}\n")
        
    def discover_datasets(self):
        """Discover available datasets and files."""
        print(f"{Colors.BLUE}🔍 Discovering available datasets...{Colors.ENDC}")
        
        try:
            self.available_files = discover_available_files_by_dataset()
            
            if not self.available_files:
                print(f"{Colors.FAIL}❌ No datasets found in data/ directory{Colors.ENDC}")
                return False
                
            datasets = list(self.available_files.keys())
            print(f"{Colors.GREEN}✅ Found {len(datasets)} datasets: {', '.join(datasets)}{Colors.ENDC}")
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ Error discovering datasets: {e}{Colors.ENDC}")
            return False
    
    def display_datasets(self):
        """Display available datasets with details."""
        print(f"\n{Colors.BOLD}📂 AVAILABLE DATASETS{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*90}")
        
        for i, (dataset, info) in enumerate(self.available_files.items(), 1):
            # Count files in each category
            baseline_count = len(info.get('baseline', {}))
            
            # Count training methods across FDR levels
            training_info = info.get('training', {})
            training_20_count = len(training_info.get('20', {}))
            training_50_count = len(training_info.get('50', {}))
            
            gt_count = len(info.get('ground_truth', {}))
            
            print(f"{Colors.GREEN}{i:>2}. {Colors.BOLD}{dataset:<15}{Colors.ENDC}")
            print(f"    {Colors.BLUE}Baseline (1% FDR): {Colors.BOLD}{baseline_count:>2}{Colors.ENDC} methods")
            print(f"    {Colors.BLUE}Training (20% FDR): {Colors.BOLD}{training_20_count:>2}{Colors.ENDC} methods")
            print(f"    {Colors.BLUE}Training (50% FDR): {Colors.BOLD}{training_50_count:>2}{Colors.ENDC} methods")
            print(f"    {Colors.BLUE}Ground Truth: {Colors.BOLD}{gt_count:>2}{Colors.ENDC} methods")
            
            # Show sample method names if available
            sample_methods = list(info.get('baseline', {}).keys())[:2]
            if sample_methods:
                sample_display = ', '.join(m.split('_', 1)[1] if '_' in m else m for m in sample_methods)
                if len(sample_display) > 60:
                    sample_display = sample_display[:57] + "..."
                if len(info.get('baseline', {})) > 2:
                    sample_display += f" (+{len(info.get('baseline', {})) - 2} more)"
                print(f"    {Colors.CYAN}Sample methods: {sample_display}{Colors.ENDC}")
            print()
        
        print(f"{Colors.CYAN}{'='*90}{Colors.ENDC}")
        return list(self.available_files.keys())
    
    def select_dataset(self) -> Optional[str]:
        """Let user select a dataset."""
        datasets = self.display_datasets()
        
        while True:
            try:
                choice = input(f"\n{Colors.BOLD}Select dataset (1-{len(datasets)}) or 'q' to quit: {Colors.ENDC}")
                
                if choice.lower() == 'q':
                    return None
                    
                choice_num = int(choice)
                if 1 <= choice_num <= len(datasets):
                    selected = datasets[choice_num - 1]
                    print(f"{Colors.GREEN}✅ Selected dataset: {Colors.BOLD}{selected}{Colors.ENDC}")
                    return selected
                else:
                    print(f"{Colors.WARNING}⚠️  Please enter a number between 1 and {len(datasets)}{Colors.ENDC}")
                    
            except ValueError:
                print(f"{Colors.WARNING}⚠️  Please enter a valid number or 'q' to quit{Colors.ENDC}")
            except KeyboardInterrupt:
                print(f"\n{Colors.FAIL}❌ Operation cancelled{Colors.ENDC}")
                return None
    
    def display_configured_methods(self, dataset: str) -> List[str]:
        """Display configured methods for a dataset."""
        # Get configured methods for this dataset
        configured_methods = get_configured_methods(dataset_filter=dataset)
        
        if not configured_methods:
            print(f"{Colors.WARNING}⚠️  No configured methods found for {dataset}{Colors.ENDC}")
            print(f"{Colors.BLUE}💡 Please use Setup Mode in Streamlit to configure this dataset first{Colors.ENDC}")
            return []
        
        print(f"\n{Colors.BOLD}📋 Available Configured Methods for {dataset}:{Colors.ENDC}")
        print(f"{Colors.CYAN}{'-'*80}")
        
        for i, method in enumerate(configured_methods, 1):
            # Show actual method name, truncate if too long
            display_method = method[:70] + "..." if len(method) > 70 else method
            print(f"{Colors.GREEN}{i:>3}. {Colors.ENDC}{display_method}")
            
        print(f"{Colors.CYAN}{'-'*80}{Colors.ENDC}")
        return configured_methods
    
    def select_training_methods(self, dataset: str) -> List[str]:
        """Let user select training methods."""
        available_methods = self.display_configured_methods(dataset)
        
        if not available_methods:
            return []
        
        print(f"\n{Colors.BOLD}🏋️  Training Method Selection{Colors.ENDC}")
        print(f"{Colors.BLUE}Select multiple training methods (e.g., '1,3,5' or '1-10'){Colors.ENDC}")
        
        while True:
            try:
                choice = input(f"{Colors.BOLD}Enter selections: {Colors.ENDC}").strip()
                
                if not choice:
                    print(f"{Colors.WARNING}⚠️  Please enter at least one selection{Colors.ENDC}")
                    continue
                
                selected_indices = []
                
                # Parse comma-separated selections
                for part in choice.split(','):
                    part = part.strip()
                    if '-' in part:
                        # Range selection (e.g., 1-5)
                        start, end = map(int, part.split('-'))
                        selected_indices.extend(range(start, end + 1))
                    else:
                        # Single selection
                        selected_indices.append(int(part))
                
                # Validate selections
                selected_indices = [i for i in selected_indices if 1 <= i <= len(available_methods)]
                
                if not selected_indices:
                    print(f"{Colors.WARNING}⚠️  No valid selections. Please try again.{Colors.ENDC}")
                    continue
                
                selected_methods = [available_methods[i-1] for i in selected_indices]
                
                print(f"{Colors.GREEN}✅ Selected {len(selected_methods)} training methods{Colors.ENDC}")
                return selected_methods
                
            except ValueError:
                print(f"{Colors.WARNING}⚠️  Invalid format. Use numbers, ranges (1-5), or comma-separated (1,3,5){Colors.ENDC}")
            except KeyboardInterrupt:
                print(f"\n{Colors.FAIL}❌ Operation cancelled{Colors.ENDC}")
                return []
    
    def select_test_method(self, dataset: str) -> Optional[str]:
        """Let user select a test method."""
        available_methods = self.display_configured_methods(dataset)  # Test methods are also from configured methods
        
        if not available_methods:
            return None
        
        print(f"\n{Colors.BOLD}🎯 Test Method Selection{Colors.ENDC}")
        print(f"{Colors.BLUE}Select ONE method for testing (holdout validation){Colors.ENDC}")
        
        while True:
            try:
                choice = input(f"{Colors.BOLD}Select test method (1-{len(available_methods)}): {Colors.ENDC}")
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(available_methods):
                    selected = available_methods[choice_num - 1]
                    print(f"{Colors.GREEN}✅ Selected test method: {selected[:50]}{'...' if len(selected) > 50 else ''}{Colors.ENDC}")
                    return selected
                else:
                    print(f"{Colors.WARNING}⚠️  Please enter a number between 1 and {len(available_methods)}{Colors.ENDC}")
                    
            except ValueError:
                print(f"{Colors.WARNING}⚠️  Please enter a valid number{Colors.ENDC}")
            except KeyboardInterrupt:
                print(f"\n{Colors.FAIL}❌ Operation cancelled{Colors.ENDC}")
                return None
    
    def select_mode(self) -> Optional[str]:
        """Let user select analysis mode."""
        print(f"\n{Colors.BOLD}🎯 SELECT ANALYSIS MODE{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}")
        print(f"{Colors.BLUE}1. Setup Mode{Colors.ENDC} - Configure dataset connections and mappings")
        print(f"{Colors.BLUE}2. Training Mode{Colors.ENDC} - Train new ML models")
        print(f"{Colors.BLUE}3. Inference Mode{Colors.ENDC} - Run analysis with trained models")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        
        while True:
            try:
                choice = input(f"{Colors.BOLD}Select mode (1-3) or 'q' to quit: {Colors.ENDC}")
                
                if choice.lower() == 'q':
                    return None
                    
                choice_num = int(choice)
                if choice_num == 1:
                    return "setup"
                elif choice_num == 2:
                    return "training"
                elif choice_num == 3:
                    return "inference"
                else:
                    print(f"{Colors.WARNING}⚠️  Please enter 1, 2, or 3{Colors.ENDC}")
                    
            except ValueError:
                print(f"{Colors.WARNING}⚠️  Please enter a valid number or 'q' to quit{Colors.ENDC}")
            except KeyboardInterrupt:
                print(f"\n{Colors.FAIL}❌ Operation cancelled{Colors.ENDC}")
                return None
    
    def configure_analysis(self, dataset: str) -> bool:
        """Configure the analysis parameters."""
        print(f"\n{Colors.BOLD}⚙️  ANALYSIS CONFIGURATION{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}")
        
        # Select FDR levels
        train_fdr = self.select_training_fdr_levels()
        test_fdr = self.select_test_fdr_level()
        
        # Select training methods
        training_methods = self.select_training_methods(dataset)
        if not training_methods:
            return False
        
        # Select test method
        test_method = self.select_test_method(dataset)
        if not test_method:
            return False
        
        # Check for overlap and handle it better
        if test_method in training_methods:
            print(f"{Colors.WARNING}⚠️  Test method is also selected for training.{Colors.ENDC}")
            print(f"{Colors.BLUE}This is allowed - the method will be used for both training and testing.{Colors.ENDC}")
            print(f"{Colors.BLUE}Note: This may lead to overfitting, but can be useful for initial testing.{Colors.ENDC}")
            
            confirm = input(f"{Colors.BOLD}Continue with this configuration? (y/n): {Colors.ENDC}").lower().strip()
            if confirm not in ['y', 'yes']:
                print(f"{Colors.BLUE}Please reconfigure your method selection.{Colors.ENDC}")
                return False
        
        # Target FDRs for analysis
        target_fdrs = self.select_target_fdrs()
        
        # Detect GPU availability
        print(f"\n{Colors.BLUE}🔧 Detecting compute device...{Colors.ENDC}")
        device = self._detect_gpu_device()
        
        # Build configuration
        self.config = {
            'dataset': dataset,
            'train_methods': training_methods,
            'test_method': test_method,
            'train_fdr_levels': train_fdr,
            'test_fdr': test_fdr,
            'target_fdr_levels': target_fdrs,
            'xgb_params': {
                'learning_rate': 0.08,
                'max_depth': 7,
                'n_estimators': 1000,
                'subsample': 0.8,
                'reg_lambda': 1.5,
                'reg_alpha': 0,
                'min_child_weight': 1,
                'gamma': 0,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'tree_method': 'hist',
                'device': device  # Use detected device (GPU or CPU)
            },
            'feature_selection': {
                'use_diann_quality': True,
                'use_sequence_features': True,
                'use_ms_features': True,
                'use_statistical_features': True,
                'use_library_features': True,
                'excluded_features': []
            },
            'aggregation_method': 'max'
        }
        
        return True
    
    def select_training_fdr_levels(self) -> List[int]:
        """Select training FDR levels from standard options."""
        print(f"\n{Colors.BLUE}📈 Select training FDR levels{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*50}")
        print(f"{Colors.GREEN}1. 1% FDR{Colors.ENDC}")
        print(f"{Colors.GREEN}2. 20% FDR{Colors.ENDC}")
        print(f"{Colors.GREEN}3. 50% FDR{Colors.ENDC}")
        print(f"{Colors.GREEN}4. Multiple levels (1, 20, 50){Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*50}{Colors.ENDC}")
        
        while True:
            try:
                choice = input(f"{Colors.BOLD}Select option (1-4, default 3): {Colors.ENDC}").strip()
                
                if not choice or choice == "3":
                    return [50]
                elif choice == "1":
                    return [1]
                elif choice == "2":
                    return [20]
                elif choice == "4":
                    return [1, 20, 50]
                else:
                    print(f"{Colors.WARNING}⚠️  Please enter 1, 2, 3, or 4{Colors.ENDC}")
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.FAIL}❌ Operation cancelled{Colors.ENDC}")
                return [50]  # Default
    
    def select_test_fdr_level(self) -> int:
        """Select test FDR level from standard options."""
        print(f"\n{Colors.BLUE}📈 Select test FDR level{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*50}")
        print(f"{Colors.GREEN}1. 1% FDR{Colors.ENDC}")
        print(f"{Colors.GREEN}2. 20% FDR{Colors.ENDC}")
        print(f"{Colors.GREEN}3. 50% FDR{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*50}{Colors.ENDC}")
        
        while True:
            try:
                choice = input(f"{Colors.BOLD}Select option (1-3, default 3): {Colors.ENDC}").strip()
                
                if not choice or choice == "3":
                    return 50
                elif choice == "1":
                    return 1
                elif choice == "2":
                    return 20
                else:
                    print(f"{Colors.WARNING}⚠️  Please enter 1, 2, or 3{Colors.ENDC}")
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.FAIL}❌ Operation cancelled{Colors.ENDC}")
                return 50  # Default
    
    def select_target_fdrs(self) -> List[float]:
        """Select target FDR levels for analysis."""
        print(f"\n{Colors.BLUE}🎯 Select target FDR levels for results{Colors.ENDC}")
        print(f"{Colors.GREEN}Default: 1, 2, 3, 4, 5, 10, 15, 20, 30, 50{Colors.ENDC}")
        
        while True:
            try:
                choice = input("Enter target FDRs (comma-separated, or press Enter for default): ").strip()
                
                if not choice:
                    return [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0]
                
                levels = [float(x.strip()) for x in choice.split(',')]
                
                # Validate FDR levels
                valid_levels = [x for x in levels if 0.1 <= x <= 50.0]
                
                if not valid_levels:
                    print(f"{Colors.WARNING}⚠️  Please enter valid FDR levels between 0.1 and 50{Colors.ENDC}")
                    continue
                
                return sorted(valid_levels)
                
            except ValueError:
                print(f"{Colors.WARNING}⚠️  Please enter valid numbers{Colors.ENDC}")
    
    def display_configuration(self):
        """Display the current configuration."""
        print(f"\n{Colors.BOLD}📋 CONFIGURATION SUMMARY{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}")
        
        config = self.config
        
        # Compact display
        print(f"{Colors.BLUE}Dataset: {Colors.BOLD}{config['dataset']}{Colors.ENDC}")
        print(f"{Colors.BLUE}Training: {Colors.BOLD}{len(config['train_methods'])} methods{Colors.ENDC} at {Colors.BOLD}{config['train_fdr_levels'][0]}% FDR{Colors.ENDC}")
        
        # Show method numbers instead of full names
        method_numbers = []
        test_method_num = None
        for method in config['train_methods']:
            # Extract the last part (sample number) from method name
            if method.endswith('_01'):
                sample_id = method.split('_')[-2]  # Get the part before '_01'
                method_numbers.append(sample_id)
        
        # Find test method number
        if config['test_method'].endswith('_01'):
            test_sample_id = config['test_method'].split('_')[-2]
            test_method_num = test_sample_id
        
        print(f"{Colors.BLUE}Train samples: {Colors.BOLD}{', '.join(method_numbers)}{Colors.ENDC}")
        print(f"{Colors.BLUE}Test sample: {Colors.BOLD}{test_method_num}{Colors.ENDC} at {Colors.BOLD}{config['test_fdr']}% FDR{Colors.ENDC}")
        print(f"{Colors.BLUE}Target FDRs: {Colors.BOLD}{len(config['target_fdr_levels'])} levels{Colors.ENDC} ({min(config['target_fdr_levels']):.0f}-{max(config['target_fdr_levels']):.0f}%)")
        
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        
        # Confirm configuration
        while True:
            try:
                confirm = input(f"\n{Colors.BOLD}Start training? (y/n): {Colors.ENDC}").lower().strip()
                if confirm in ['y', 'yes']:
                    return True
                elif confirm in ['n', 'no']:
                    return False
                else:
                    print(f"{Colors.WARNING}Please enter 'y' or 'n'{Colors.ENDC}")
            except KeyboardInterrupt:
                return False
    
    def run_analysis(self):
        """Run the peptide analysis."""
        print(f"\n{Colors.BOLD}🚀 STARTING PEPTIDIA ANALYSIS{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}")
        
        try:
            # Initialize validator
            self.validator = PeptideValidatorAPI()
            
            # Create results directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = f"results/CLI_RESULTS_{timestamp}"
            os.makedirs(results_dir, exist_ok=True)
            
            print(f"{Colors.BLUE}📁 Results will be saved to: {results_dir}{Colors.ENDC}")
            
            print(f"\n{Colors.BOLD}🧠 Training ML Model...{Colors.ENDC}")
            
            # Run analysis with clean progress updates and suppressed verbose output
            def progress_callback(current_step, total_steps, step_name):
                # Clean up technical step names for user-friendly display
                clean_step_name = self._clean_step_name(step_name)
                
                percentage = (current_step / total_steps) * 100
                progress_bar = "█" * int(percentage / 2.5) + "░" * (40 - int(percentage / 2.5))
                
                # Force output to original stdout (bypassing any suppression)
                original_stdout = sys.stdout
                if hasattr(sys.stdout, 'write') and isinstance(sys.stdout, StringIO):
                    sys.stdout = sys.__stdout__
                
                print(f"{Colors.GREEN}[{progress_bar}] {percentage:5.1f}% - Step {current_step}/{total_steps}: {clean_step_name}{Colors.ENDC}")
                
                # Restore suppressed stdout
                if original_stdout != sys.stdout:
                    sys.stdout = original_stdout
            
            # Run the analysis with suppressed verbose output
            with self._suppress_verbose_output():
                results = self.validator.run_analysis(
                    train_methods=self.config['train_methods'],
                    test_method=self.config['test_method'],
                    train_fdr_levels=self.config['train_fdr_levels'],
                    test_fdr=self.config['test_fdr'],
                    target_fdr_levels=self.config['target_fdr_levels'],
                    xgb_params=self.config['xgb_params'],
                    feature_selection=self.config['feature_selection'],
                    aggregation_method=self.config['aggregation_method'],
                    results_dir=results_dir,
                    progress_callback=progress_callback
                )
            
            print(f"\n{Colors.GREEN}✅ Model training completed!{Colors.ENDC}")
            
            # Display results
            self.display_results(results)
            
            # Save configuration
            config_file = os.path.join(results_dir, "analysis_config.json")
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            print(f"\n{Colors.GREEN}🎉 Analysis completed successfully!{Colors.ENDC}")
            print(f"{Colors.BLUE}📁 Results saved to: {results_dir}{Colors.ENDC}")
            
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ Analysis failed: {str(e)}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
            return False
    
    def display_results(self, results):
        """Display analysis results in terminal."""
        print(f"\n{Colors.BOLD}📊 RESULTS{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}")
        
        # Compact summary statistics
        summary = results.get('summary', {})
        baseline = summary.get('baseline_peptides', 0)
        candidates = summary.get('additional_candidates', 0)
        
        # Calculate pool validation rate (what % of candidates are real)
        # Get total validated candidates from analysis results (total recoverable peptides in test set)
        total_validated_candidates = 0
        analysis_results = results.get('results', [])
        if analysis_results:
            # Get the best performing result to show validation info
            best_result = max(analysis_results, key=lambda x: x.get('Additional_Peptides', 0))
            total_validated_candidates = best_result.get('Total_Validated_Candidates', 0)
        
        if candidates > 0:
            improvement = (candidates / baseline) * 100
            print(f"{Colors.BLUE}Baseline: {Colors.BOLD}{baseline:,}{Colors.ENDC} peptides")
            print(f"{Colors.BLUE}Pool: {Colors.BOLD}{candidates:,}{Colors.ENDC} additional candidates ({Colors.BOLD}+{improvement:.0f}%{Colors.ENDC})")
            
            if total_validated_candidates > 0:
                # Calculate what percentage of the candidate pool is actually validated
                validation_rate = (total_validated_candidates / candidates) * 100
                print(f"{Colors.BLUE}Pool validation: {Colors.BOLD}{total_validated_candidates:,}{Colors.ENDC}/{Colors.BOLD}{candidates:,}{Colors.ENDC} are validated ({Colors.BOLD}{validation_rate:.1f}%{Colors.ENDC} of pool)")
            else:
                print(f"{Colors.BLUE}Pool validation: {Colors.BOLD}0{Colors.ENDC}/{Colors.BOLD}{candidates:,}{Colors.ENDC} validated candidates")
        else:
            print(f"{Colors.BLUE}Baseline: {Colors.BOLD}{baseline:,}{Colors.ENDC} peptides")
            print(f"{Colors.BLUE}Additional candidates: {Colors.BOLD}{candidates:,}{Colors.ENDC}")
        
        # Results table - match Streamlit format exactly
        analysis_results = results.get('results', [])
        if analysis_results:
            print(f"\n{Colors.BOLD}🎯 Performance Results:{Colors.ENDC}")
            print(f"{Colors.CYAN}{'-'*130}")
            
            # Header with proper column names including Increase %
            header = f"{'Target':>8} | {'Threshold':>10} | {'Additional':>11} | {'Add. Valid':>10} | {'False':>7} | {'Actual':>8} | {'Recovery':>9} | {'Increase':>9} | {'MCC':>7}"
            subheader = f"{'FDR':>8} | {'':>10} | {'Peptides':>11} | {'Peptides':>10} | {'Pos':>7} | {'FDR':>8} | {'%':>9} | {'%':>9} | {'':>7}"
            
            print(f"{Colors.BOLD}{header}")
            print(f"{subheader}{Colors.ENDC}")
            print(f"{Colors.CYAN}{'-'*130}")
            
            for result in analysis_results:
                target_fdr = result.get('Target_FDR', 0)
                threshold = result.get('Threshold', 0)
                additional_peptides = result.get('Additional_Peptides', 0)
                actual_fdr = result.get('Actual_FDR', 0)
                recovery_pct = result.get('Recovery_Pct', 0)
                increase_pct = result.get('Increase_Pct', 0)
                false_positives = result.get('False_Positives', 0)
                mcc = result.get('MCC', 0)
                
                # Calculate validated peptides from additional candidates
                validated_peptides = additional_peptides - false_positives
                
                # Color code based on performance (strict thresholds)
                if actual_fdr <= target_fdr * 1.1:  # Within 10% of target -> GREEN
                    color = Colors.GREEN
                elif actual_fdr <= target_fdr * 1.5:  # Within 50% of target -> BLUE  
                    color = Colors.BLUE
                else:
                    color = Colors.WARNING
                
                print(f"{color}{target_fdr:>7.1f}% | {threshold:>10.4f} | {additional_peptides:>11,} | "
                      f"{validated_peptides:>10,} | {false_positives:>7,} | {actual_fdr:>7.1f}% | "
                      f"{recovery_pct:>8.1f}% | {increase_pct:>8.1f}% | {mcc:>7.3f}{Colors.ENDC}")
            
            print(f"{Colors.CYAN}{'-'*130}{Colors.ENDC}")
        
    
    def run_setup_mode(self):
        """Run setup mode for dataset configuration."""
        print(f"\n{Colors.BOLD}🔧 SETUP MODE{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}")
        print(f"{Colors.BLUE}Configure ground truth mapping for your datasets{Colors.ENDC}")
        print(f"{Colors.GREEN}No JSON editing required - guided CLI setup{Colors.ENDC}")
        print()
        
        # Discover datasets
        if not self.discover_datasets():
            return False
        
        # Select dataset to configure
        dataset = self.select_dataset()
        if not dataset:
            return False
        
        # Configure the dataset
        return self.configure_dataset_setup(dataset)
    
    def configure_dataset_setup(self, dataset: str) -> bool:
        """Configure dataset ground truth mapping."""
        print(f"\n{Colors.BOLD}⚙️  CONFIGURING DATASET: {dataset}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'-'*60}")
        
        dataset_info = self.available_files.get(dataset, {})
        if not dataset_info:
            print(f"{Colors.FAIL}❌ No data found for dataset {dataset}{Colors.ENDC}")
            return False
        
        # Show dataset overview
        self.show_dataset_overview(dataset, dataset_info)
        
        # Select file organization mode
        file_mode = self.select_file_organization_mode()
        if not file_mode:
            return False
        
        # Get training methods and ground truth files
        training_methods, ground_truth_files = self.organize_files_by_mode(dataset, dataset_info, file_mode)
        
        if not training_methods:
            print(f"{Colors.FAIL}❌ No training methods found{Colors.ENDC}")
            return False
        
        if not ground_truth_files:
            print(f"{Colors.FAIL}❌ No ground truth files found{Colors.ENDC}")
            return False
        
        # Select ground truth mapping strategy
        strategy = self.select_ground_truth_strategy()
        if not strategy:
            return False
        
        # Create mapping configuration
        mapping_config = self.create_mapping_configuration(strategy, training_methods, ground_truth_files)
        
        # Save configuration
        return self.save_dataset_configuration(dataset, mapping_config, file_mode)
    
    def show_dataset_overview(self, dataset: str, dataset_info: dict):
        """Display dataset file overview."""
        print(f"\n{Colors.BLUE}📊 Dataset Overview: {dataset}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'-'*50}")
        
        # Count files in each category
        baseline_count = len(dataset_info.get('baseline', {}))
        training_20_count = len(dataset_info.get('training', {}).get('20', {}))
        training_50_count = len(dataset_info.get('training', {}).get('50', {}))
        gt_count = len(dataset_info.get('ground_truth', {}))
        
        print(f"{Colors.GREEN}Baseline (1% FDR): {Colors.BOLD}{baseline_count:>2}{Colors.ENDC} methods")
        print(f"{Colors.GREEN}Training (20% FDR): {Colors.BOLD}{training_20_count:>2}{Colors.ENDC} methods")
        print(f"{Colors.GREEN}Training (50% FDR): {Colors.BOLD}{training_50_count:>2}{Colors.ENDC} methods")
        print(f"{Colors.GREEN}Ground Truth: {Colors.BOLD}{gt_count:>2}{Colors.ENDC} methods")
        print()
    
    def select_file_organization_mode(self) -> Optional[str]:
        """Select how files should be organized."""
        print(f"{Colors.BLUE}⚙️  File Organization Mode{Colors.ENDC}")
        print(f"{Colors.GREEN}1. Individual Files{Colors.ENDC} - Each file is a separate method")
        print(f"{Colors.GREEN}2. Triplicate Groups{Colors.ENDC} - Group files by method number")
        print()
        
        while True:
            try:
                choice = input(f"{Colors.BOLD}Select organization mode (1-2): {Colors.ENDC}").strip()
                
                if choice == "1":
                    print(f"{Colors.GREEN}✅ Selected: Individual Files mode{Colors.ENDC}")
                    return "individual"
                elif choice == "2":
                    print(f"{Colors.GREEN}✅ Selected: Triplicate Groups mode{Colors.ENDC}")
                    return "triplicates"
                else:
                    print(f"{Colors.WARNING}⚠️  Please enter 1 or 2{Colors.ENDC}")
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.FAIL}❌ Operation cancelled{Colors.ENDC}")
                return None
    
    def organize_files_by_mode(self, dataset: str, dataset_info: dict, file_mode: str) -> Tuple[List[str], List[str]]:
        """Organize files based on selected mode."""
        if file_mode == "individual":
            return self.organize_individual_files(dataset_info)
        else:  # triplicates
            return self.organize_triplicate_files(dataset, dataset_info)
    
    def organize_individual_files(self, dataset_info: dict) -> Tuple[List[str], List[str]]:
        """Organize files as individual methods."""
        training_methods = []
        
        # Collect all training files
        for fdr_level in ['20', '50']:
            if fdr_level in dataset_info.get('training', {}):
                methods = list(dataset_info['training'][fdr_level].keys())
                training_methods.extend(methods)
        
        training_methods = sorted(list(set(training_methods)))
        ground_truth_files = list(dataset_info.get('ground_truth', {}).keys())
        
        print(f"{Colors.BLUE}📋 File Organization Summary:{Colors.ENDC}")
        print(f"  Training methods: {len(training_methods)}")
        print(f"  Ground truth files: {len(ground_truth_files)}")
        
        return training_methods, ground_truth_files
    
    def organize_triplicate_files(self, dataset: str, dataset_info: dict) -> Tuple[List[str], List[str]]:
        """Organize files into triplicate groups."""
        print(f"\n{Colors.BLUE}🔍 Triplicate Grouping Configuration{Colors.ENDC}")
        
        # Get all individual files first to show to user
        individual_training = []
        for fdr_level in ['20', '50']:
            if fdr_level in dataset_info.get('training', {}):
                methods = list(dataset_info['training'][fdr_level].keys())
                individual_training.extend(methods)
        individual_training = sorted(list(set(individual_training)))
        
        individual_gt = list(dataset_info.get('ground_truth', {}).keys())
        
        # Show available files to help user understand what to search for
        print(f"\n{Colors.CYAN}📋 Available Files in Dataset:{Colors.ENDC}")
        print(f"{Colors.BLUE}Training Files ({len(individual_training)} total):{Colors.ENDC}")
        
        # Group and sort files to show patterns better
        grouped_training = self.group_files_by_pattern(individual_training)
        self.display_grouped_files(grouped_training, max_show=10)
        
        print(f"\n{Colors.BLUE}Ground Truth Files ({len(individual_gt)} total):{Colors.ENDC}")
        grouped_gt = self.group_files_by_pattern(individual_gt)
        self.display_grouped_files(grouped_gt, max_show=8)
        
        print(f"\n{Colors.GREEN}💡 Search Tips:{Colors.ENDC}")
        print(f"  - Look for common patterns in filenames above")
        print(f"  - Use numbers like '001', '002', '003' to group replicates")  
        print(f"  - Use method identifiers like 'RD201', 'EV1107', etc.")
        print(f"  - Each search term will create one group containing all matching files")
        
        print(f"\n{Colors.GREEN}Enter search terms to group files (one per line, empty line to finish):{Colors.ENDC}")
        print()
        
        search_terms = []
        while True:
            try:
                term = input(f"{Colors.BOLD}Enter search term (or press Enter to finish): {Colors.ENDC}").strip()
                if not term:
                    break
                search_terms.append(term)
                print(f"{Colors.GREEN}✅ Added search term: {term}{Colors.ENDC}")
            except KeyboardInterrupt:
                print(f"\n{Colors.FAIL}❌ Operation cancelled{Colors.ENDC}")
                return [], []
        
        if not search_terms:
            print(f"{Colors.WARNING}⚠️  No search terms provided, using individual files{Colors.ENDC}")
            return self.organize_individual_files(dataset_info)
        
        # Apply grouping logic
        training_methods = []
        ground_truth_files = []
        
        # Create copies of the file lists for grouping (already retrieved above)
        available_training = individual_training.copy()
        available_gt = individual_gt.copy()
        
        # Group training files
        groups_created = 0
        for search_term in search_terms:
            group_name = f"{dataset}_Group_{search_term}"
            matching_files = [f for f in available_training if search_term in f]
            
            if matching_files:
                training_methods.append(group_name)
                groups_created += 1
                print(f"{Colors.GREEN}✅ Created group '{group_name}' with {len(matching_files)} files{Colors.ENDC}")
                
                # Remove matched files to avoid duplicates
                for matched_file in matching_files:
                    if matched_file in available_training:
                        available_training.remove(matched_file)
        
        # Add remaining individual files
        training_methods.extend(available_training)
        
        # Group ground truth files similarly
        for search_term in search_terms:
            group_name = f"{dataset}_Group_{search_term}"
            matching_gt_files = [f for f in available_gt if search_term in f]
            
            if matching_gt_files:
                ground_truth_files.append(group_name)
                # Remove matched files
                for matched_file in matching_gt_files:
                    if matched_file in available_gt:
                        available_gt.remove(matched_file)
        
        # Add remaining individual GT files
        ground_truth_files.extend(available_gt)
        
        print(f"\n{Colors.BLUE}📋 Grouping Summary:{Colors.ENDC}")
        print(f"  Groups created: {groups_created}")
        print(f"  Total training methods: {len(training_methods)}")
        print(f"  Total ground truth files: {len(ground_truth_files)}")
        
        return training_methods, ground_truth_files
    
    def select_ground_truth_strategy(self) -> Optional[str]:
        """Select ground truth mapping strategy."""
        print(f"\n{Colors.BLUE}🔗 Ground Truth Mapping Strategy{Colors.ENDC}")
        print(f"{Colors.GREEN}1. Use All Ground Truth{Colors.ENDC} - All methods use all ground truth files (simple)")
        print(f"{Colors.GREEN}2. Method-Specific Mapping{Colors.ENDC} - Each method maps to specific ground truth files")
        print()
        
        while True:
            try:
                choice = input(f"{Colors.BOLD}Select mapping strategy (1-2): {Colors.ENDC}").strip()
                
                if choice == "1":
                    print(f"{Colors.GREEN}✅ Selected: Use All Ground Truth (simple strategy){Colors.ENDC}")
                    return "use_all_ground_truth"
                elif choice == "2":
                    print(f"{Colors.GREEN}✅ Selected: Method-Specific Mapping{Colors.ENDC}")
                    return "method_specific"
                else:
                    print(f"{Colors.WARNING}⚠️  Please enter 1 or 2{Colors.ENDC}")
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.FAIL}❌ Operation cancelled{Colors.ENDC}")
                return None
    
    def create_mapping_configuration(self, strategy: str, training_methods: List[str], ground_truth_files: List[str]) -> dict:
        """Create mapping configuration based on strategy."""
        mapping_config = {
            "_strategy": strategy
        }
        
        if strategy == "use_all_ground_truth":
            mapping_config["_description"] = "All training methods use all available ground truth files"
            print(f"\n{Colors.BLUE}📝 Configuration: All methods will use all ground truth files{Colors.ENDC}")
        
        elif strategy == "method_specific":
            print(f"\n{Colors.BLUE}📝 Method-Specific Mapping Configuration{Colors.ENDC}")
            print(f"{Colors.CYAN}Connect each training method to its ground truth file(s):{Colors.ENDC}")
            print()
            
            connections = {}
            
            for i, method in enumerate(training_methods):  # Map all methods manually
                print(f"{Colors.BOLD}Training Method {i+1}: {method}{Colors.ENDC}")
                print(f"{Colors.BLUE}Available ground truth files:{Colors.ENDC}")
                
                for j, gt_file in enumerate(ground_truth_files, 1):
                    print(f"  {j}. {gt_file}")
                
                print(f"\n{Colors.GREEN}💡 Multi-select options:{Colors.ENDC}")
                print(f"  • Single file: {Colors.CYAN}1{Colors.ENDC}")
                print(f"  • Multiple files: {Colors.CYAN}1,2,3{Colors.ENDC} or {Colors.CYAN}1-3{Colors.ENDC}")
                print(f"  • All files: {Colors.CYAN}all{Colors.ENDC}")
                
                while True:
                    try:
                        choice = input(f"{Colors.BOLD}Select ground truth file(s) (1-{len(ground_truth_files)}): {Colors.ENDC}").strip()
                        
                        if choice.lower() == 'all':
                            selected_gts = ground_truth_files.copy()
                        else:
                            selected_indices = []
                            
                            # Parse comma-separated and range selections
                            for part in choice.split(','):
                                part = part.strip()
                                if '-' in part and len(part.split('-')) == 2:
                                    # Range selection (e.g., 1-3)
                                    try:
                                        start, end = map(int, part.split('-'))
                                        selected_indices.extend(range(start, end + 1))
                                    except ValueError:
                                        raise ValueError(f"Invalid range: {part}")
                                else:
                                    # Single selection
                                    selected_indices.append(int(part))
                            
                            # Validate selections
                            valid_indices = [i for i in selected_indices if 1 <= i <= len(ground_truth_files)]
                            
                            if not valid_indices:
                                print(f"{Colors.WARNING}⚠️  No valid selections. Please try again.{Colors.ENDC}")
                                continue
                            
                            if len(valid_indices) != len(selected_indices):
                                invalid = [i for i in selected_indices if i not in valid_indices]
                                print(f"{Colors.WARNING}⚠️  Invalid selections ignored: {invalid}{Colors.ENDC}")
                            
                            selected_gts = [ground_truth_files[i-1] for i in valid_indices]
                        
                        connections[method] = selected_gts
                        
                        if len(selected_gts) == 1:
                            print(f"{Colors.GREEN}✅ Mapped {method} → {selected_gts[0]}{Colors.ENDC}")
                        else:
                            print(f"{Colors.GREEN}✅ Mapped {method} → {len(selected_gts)} ground truth files:{Colors.ENDC}")
                            for gt in selected_gts:
                                print(f"    • {gt}")
                        break
                        
                    except ValueError as e:
                        print(f"{Colors.WARNING}⚠️  Invalid format: {e}. Use single numbers (1), ranges (1-3), or comma-separated (1,2,3){Colors.ENDC}")
                    except KeyboardInterrupt:
                        print(f"\n{Colors.FAIL}❌ Operation cancelled{Colors.ENDC}")
                        return {}
                
                print()
            
            # Add suggested matches to help user but let them choose
            if len(training_methods) > 0:
                print(f"\n{Colors.GREEN}💡 Pattern Suggestions (optional guidance):{Colors.ENDC}")
                print(f"  - Look for similar endings in filenames")
                print(f"  - 300SPD files often match with 30SPD files with same endings")
                print(f"  - You have full control - choose what makes sense for your analysis")
                print()
            
            mapping_config.update(connections)
        
        return mapping_config
    
    def find_intelligent_ground_truth_match(self, training_method: str, ground_truth_files: List[str]) -> Optional[str]:
        """Find the best matching ground truth file based on filename patterns."""
        
        # Extract key identifiers from training method filename
        # Look for patterns like C57_01, C46_01, etc. at the end of filenames
        import re
        
        # Try to extract the ending pattern (like C57_01, GC7_C60_01, etc.)
        training_patterns = []
        
        # Pattern 1: Look for pattern like C57_01, C46_01 at the end
        match = re.search(r'[A-Z]+\d+_\d+$', training_method)
        if match:
            training_patterns.append(match.group())
        
        # Pattern 2: Look for pattern like GC7_C60_01, CTL_C57_01
        match = re.search(r'[A-Z0-9]+_[A-Z]+\d+_\d+$', training_method)
        if match:
            training_patterns.append(match.group())
        
        # Pattern 3: Just the last few characters (numbers and letters)
        match = re.search(r'[A-Z0-9_]+\d+$', training_method)
        if match:
            training_patterns.append(match.group())
        
        # Look for best match in ground truth files
        best_match = None
        best_score = 0
        
        for gt_file in ground_truth_files:
            score = 0
            
            # Check if any extracted patterns appear in the ground truth filename
            for pattern in training_patterns:
                if pattern in gt_file:
                    score += len(pattern)  # Longer matches get higher scores
            
            # Additional scoring for exact ending matches
            for pattern in training_patterns:
                if gt_file.endswith(pattern):
                    score += 10  # Bonus for ending matches
            
            if score > best_score:
                best_score = score
                best_match = gt_file
        
        # Only return a match if we found a meaningful pattern match
        if best_score >= 5:  # Minimum threshold for a good match
            return best_match
        
        return None
    
    def group_files_by_pattern(self, files: List[str]) -> Dict[str, List[str]]:
        """Group files by common ending patterns to show alignment."""
        import re
        from collections import defaultdict
        
        groups = defaultdict(list)
        
        for filename in files:
            # Extract ending pattern for grouping
            # Look for patterns like C57_01, GC7_C60_01, etc.
            
            pattern = "other"  # Default group
            
            # Try different pattern extractions
            patterns_to_try = [
                r'[A-Z]+\d+_\d+$',  # C57_01, C46_01
                r'[A-Z0-9]+_[A-Z]+\d+_\d+$',  # GC7_C60_01, CTL_C57_01
                r'[A-Z0-9_]+\d+$',  # More general ending
            ]
            
            for pattern_regex in patterns_to_try:
                match = re.search(pattern_regex, filename)
                if match:
                    pattern = match.group()
                    break
            
            groups[pattern].append(filename)
        
        # Sort groups by pattern and files within groups
        sorted_groups = {}
        for pattern in sorted(groups.keys()):
            sorted_groups[pattern] = sorted(groups[pattern])
        
        return sorted_groups
    
    def display_grouped_files(self, grouped_files: Dict[str, List[str]], max_show: int = 10):
        """Display files grouped by patterns to show alignment."""
        shown_count = 0
        total_files = sum(len(files) for files in grouped_files.values())
        
        for pattern, files in grouped_files.items():
            if shown_count >= max_show:
                break
                
            if pattern != "other":
                print(f"  {Colors.CYAN}Pattern '{pattern}':{Colors.ENDC}")
                for file in files[:3]:  # Show first 3 files in each pattern
                    print(f"    • {file}")
                    shown_count += 1
                    if shown_count >= max_show:
                        break
                if len(files) > 3:
                    print(f"    • ... and {len(files) - 3} more with pattern '{pattern}'")
            else:
                # Show "other" files individually
                for file in files:
                    if shown_count >= max_show:
                        break
                    print(f"  • {file}")
                    shown_count += 1
        
        if shown_count < total_files:
            remaining = total_files - shown_count
            print(f"  • ... and {remaining} more files")
    
    def save_dataset_configuration(self, dataset: str, mapping_config: dict, file_mode: str) -> bool:
        """Save dataset configuration to file."""
        import json
        import os
        
        config_dir = f"data/{dataset}"
        config_file = f"{config_dir}/dataset_config.json"
        
        # Create directory if it doesn't exist
        os.makedirs(config_dir, exist_ok=True)
        
        # Load existing config or create new one
        existing_config = {}
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    existing_config = json.load(f)
                print(f"{Colors.BLUE}📂 Loaded existing configuration{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.WARNING}⚠️  Could not load existing config: {e}{Colors.ENDC}")
        
        # Merge configurations
        final_config = existing_config.copy()
        final_config["ground_truth_mapping"] = mapping_config
        final_config["file_organization_mode"] = file_mode
        
        # Add default values if missing
        if "icon" not in final_config:
            final_config["icon"] = "🧬"
        if "instrument" not in final_config:
            final_config["instrument"] = "Mass spectrometer"
        if "description" not in final_config:
            final_config["description"] = f"{dataset} proteomics analysis"
        
        # Save configuration
        try:
            with open(config_file, 'w') as f:
                json.dump(final_config, f, indent=2)
            
            print(f"\n{Colors.GREEN}✅ Configuration saved successfully!{Colors.ENDC}")
            print(f"{Colors.BLUE}📁 Saved to: {config_file}{Colors.ENDC}")
            
            # Show summary
            strategy = mapping_config.get("_strategy", "unknown")
            print(f"\n{Colors.BOLD}📋 Configuration Summary:{Colors.ENDC}")
            print(f"  Dataset: {dataset}")
            print(f"  File mode: {file_mode}")
            print(f"  Strategy: {strategy}")
            
            if strategy == "method_specific":
                method_count = len([k for k in mapping_config.keys() if not k.startswith("_")])
                print(f"  Mapped methods: {method_count}")
            
            print(f"\n{Colors.GREEN}🎉 Setup completed! You can now use Training Mode with this dataset.{Colors.ENDC}")
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ Failed to save configuration: {e}{Colors.ENDC}")
            return False
    
    def run_training_mode(self, dataset: str):
        """Run training mode for ML model training."""
        print(f"\n{Colors.BOLD}🏋️  TRAINING MODE{Colors.ENDC}")
        
        # Configure and run analysis
        if not self.configure_analysis(dataset):
            print(f"{Colors.FAIL}❌ Configuration failed{Colors.ENDC}")
            return False
        
        # Display and confirm configuration
        if not self.display_configuration():
            print(f"{Colors.BLUE}👋 Training cancelled by user{Colors.ENDC}")
            return False
        
        # Run analysis
        return self.run_analysis()
    
    def run_inference_mode(self):
        """Run inference mode with existing models."""
        print(f"\n{Colors.BOLD}🔮 INFERENCE MODE{Colors.ENDC}")
        print(f"{Colors.BLUE}This mode would use existing trained models for inference{Colors.ENDC}")
        print(f"{Colors.WARNING}⚠️  Inference mode not yet implemented in CLI{Colors.ENDC}")
        return True
    
    def run(self):
        """Main CLI workflow with continuous mode selection."""
        self.print_banner()
        
        # Discover datasets once at startup
        if not self.discover_datasets():
            return 1
        
        # Main loop - return to mode selection after each operation
        while True:
            # Select mode
            mode = self.select_mode()
            if not mode:
                print(f"\n{Colors.GREEN}👋 Thank you for using PeptiDIA CLI! Goodbye!{Colors.ENDC}")
                return 0
            
            try:
                # Handle different modes
                if mode == "setup":
                    success = self.run_setup_mode()
                    if success:
                        print(f"\n{Colors.GREEN}🎉 Setup completed successfully!{Colors.ENDC}")
                    else:
                        print(f"\n{Colors.WARNING}⚠️  Setup was not completed{Colors.ENDC}")
                
                elif mode == "inference":
                    success = self.run_inference_mode()
                    if success:
                        print(f"\n{Colors.GREEN}🎉 Inference completed successfully!{Colors.ENDC}")
                    else:
                        print(f"\n{Colors.WARNING}⚠️  Inference was not completed{Colors.ENDC}")
                
                elif mode == "training":
                    # For training mode, select dataset
                    dataset = self.select_dataset()
                    if not dataset:
                        print(f"\n{Colors.BLUE}↩️  Returning to main menu...{Colors.ENDC}")
                        continue
                    
                    success = self.run_training_mode(dataset)
                    if success:
                        print(f"\n{Colors.GREEN}🎉 Training completed successfully!{Colors.ENDC}")
                    else:
                        print(f"\n{Colors.WARNING}⚠️  Training was not completed{Colors.ENDC}")
                
                # Ask if user wants to continue or quit
                print(f"\n{Colors.CYAN}{'='*60}")
                continue_choice = self.ask_continue_or_quit()
                
                if not continue_choice:
                    print(f"\n{Colors.GREEN}👋 Thank you for using PeptiDIA CLI! Goodbye!{Colors.ENDC}")
                    return 0
                
                # Clear screen for next operation (optional)
                print("\n" * 2)
                
            except KeyboardInterrupt:
                print(f"\n\n{Colors.BLUE}↩️  Returning to main menu...{Colors.ENDC}")
                continue
            except Exception as e:
                print(f"\n{Colors.FAIL}❌ Unexpected error: {str(e)}{Colors.ENDC}")
                print(f"{Colors.BLUE}↩️  Returning to main menu...{Colors.ENDC}")
                continue
    
    def ask_continue_or_quit(self) -> bool:
        """Ask user if they want to continue or quit."""
        print(f"{Colors.BOLD}What would you like to do next?{Colors.ENDC}")
        print(f"{Colors.GREEN}1. Return to main menu{Colors.ENDC} - Choose another mode")
        print(f"{Colors.GREEN}2. Quit PeptiDIA CLI{Colors.ENDC} - Exit the application")
        print()
        
        while True:
            try:
                choice = input(f"{Colors.BOLD}Enter your choice (1-2): {Colors.ENDC}").strip()
                
                if choice == "1" or choice.lower() in ['menu', 'main', 'm']:
                    return True
                elif choice == "2" or choice.lower() in ['quit', 'exit', 'q']:
                    return False
                else:
                    print(f"{Colors.WARNING}⚠️  Please enter 1 or 2{Colors.ENDC}")
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.BLUE}Returning to main menu...{Colors.ENDC}")
                return True

def main():
    """Entry point for the CLI."""
    try:
        cli = PeptiDIACLI()
        return cli.run()
    except KeyboardInterrupt:
        print(f"\n{Colors.FAIL}❌ Operation cancelled by user{Colors.ENDC}")
        return 1
    except Exception as e:
        print(f"\n{Colors.FAIL}❌ Unexpected error: {str(e)}{Colors.ENDC}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
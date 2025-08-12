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
        print(f"{Colors.BLUE}This mode would configure dataset connections and ground truth mappings{Colors.ENDC}")
        print(f"{Colors.WARNING}⚠️  Setup mode not yet implemented in CLI - use Streamlit interface{Colors.ENDC}")
        return True
    
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
        """Main CLI workflow."""
        self.print_banner()
        
        # Discover datasets
        if not self.discover_datasets():
            return 1
        
        # Select mode
        mode = self.select_mode()
        if not mode:
            print(f"{Colors.BLUE}👋 Goodbye!{Colors.ENDC}")
            return 0
        
        # Handle setup and inference modes
        if mode == "setup":
            if not self.run_setup_mode():
                return 1
            print(f"\n{Colors.GREEN}✨ Thank you for using PeptiDIA CLI!{Colors.ENDC}")
            return 0
        elif mode == "inference":
            if not self.run_inference_mode():
                return 1
            print(f"\n{Colors.GREEN}✨ Thank you for using PeptiDIA CLI!{Colors.ENDC}")
            return 0
        
        # For training mode, continue with dataset selection
        dataset = self.select_dataset()
        if not dataset:
            print(f"{Colors.BLUE}👋 Goodbye!{Colors.ENDC}")
            return 0
        
        # Run training mode
        if not self.run_training_mode(dataset):
            return 1
        
        print(f"\n{Colors.GREEN}✨ Thank you for using PeptiDIA CLI!{Colors.ENDC}")
        return 0

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
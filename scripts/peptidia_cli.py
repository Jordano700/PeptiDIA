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
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BLUE}Use existing trained models for inference on new data{Colors.ENDC}")
        
        # Step 1: Discover available models
        available_models = self._discover_trained_models()
        if not available_models:
            print(f"{Colors.FAIL}❌ No trained models found{Colors.ENDC}")
            print(f"{Colors.WARNING}💡 Train a model first using Training Mode{Colors.ENDC}")
            return False
        
        # Step 2: Select model
        selected_model = self._select_trained_model(available_models)
        if not selected_model:
            return False
        
        # Step 3: Select test method and dataset
        test_config = self._configure_inference_test(selected_model)
        if not test_config:
            return False
        
        # Step 4: Run inference analysis
        success = self._run_inference_analysis(selected_model, test_config)
        return success
    
    def _discover_trained_models(self) -> List[Dict]:
        """Discover all available trained models (CLI and Streamlit)."""
        models = []
        
        # Look for CLI models in results directory
        results_dir = Path("results")
        if results_dir.exists():
            for cli_dir in results_dir.glob("CLI_RESULTS_*"):
                model_dir = cli_dir / "saved_models"
                config_file = cli_dir / "analysis_config.json"
                
                if model_dir.exists() and config_file.exists():
                    try:
                        # Load CLI configuration
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        # Load model metadata
                        metadata_file = model_dir / "model_metadata.json"
                        if metadata_file.exists():
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            
                            # Extract best result for display
                            best_result = None
                            if 'training_results' in metadata:
                                valid_results = [r for r in metadata['training_results'] if r['Actual_FDR'] <= 5.0]
                                if not valid_results:
                                    valid_results = metadata['training_results']
                                if valid_results:
                                    best_result = max(valid_results, key=lambda x: x['Additional_Peptides'])
                            
                            model_info = {
                                'id': cli_dir.name,
                                'type': 'CLI',
                                'timestamp': cli_dir.name.replace('CLI_RESULTS_', ''),
                                'config': config,
                                'metadata': metadata,
                                'model_dir': str(model_dir),
                                'best_result': best_result,
                                'train_methods': config.get('train_methods', []),
                                'test_method': config.get('test_method', ''),
                                'dataset': config.get('dataset', 'Unknown')
                            }
                            models.append(model_info)
                    except Exception as e:
                        print(f"{Colors.WARNING}⚠️  Could not load CLI model {cli_dir.name}: {e}{Colors.ENDC}")
        
        # Look for Streamlit models in results directory
        for streamlit_dir in results_dir.glob("STREAMLIT_RESULTS_*"):
            model_dir = streamlit_dir / "saved_models"
            if model_dir.exists():
                try:
                    # Check for model files
                    model_file = model_dir / "trained_model.joblib"
                    metadata_file = model_dir / "model_metadata.json"
                    
                    if model_file.exists() and metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        # Try to load actual method information from analysis summary
                        train_methods = ['Streamlit Training']
                        test_method = 'Streamlit Testing'
                        dataset = 'Streamlit'
                        
                        summary_file = streamlit_dir / "ANALYSIS_SUMMARY.txt"
                        if summary_file.exists():
                            try:
                                with open(summary_file, 'r') as f:
                                    summary_content = f.read()
                                
                                # Extract training methods
                                for line in summary_content.split('\n'):
                                    if 'Training methods:' in line:
                                        methods_str = line.split('Training methods:')[1].strip()
                                        train_methods = [m.strip() for m in methods_str.split(',')]
                                    elif 'Test method:' in line:
                                        test_method = line.split('Test method:')[1].strip()
                                
                                # Extract dataset from first training method
                                if train_methods and train_methods[0] != 'Streamlit Training':
                                    dataset = train_methods[0].split('_')[0]  # e.g., ASTRAL_Group_001 -> ASTRAL
                                    
                            except Exception as e:
                                print(f"{Colors.WARNING}⚠️  Could not parse summary for {streamlit_dir.name}: {e}{Colors.ENDC}")
                        
                        model_info = {
                            'id': streamlit_dir.name,
                            'type': 'Streamlit',
                            'timestamp': streamlit_dir.name.replace('STREAMLIT_RESULTS_', ''),
                            'model_dir': str(model_dir),
                            'metadata': metadata,
                            'train_methods': train_methods,
                            'test_method': test_method,
                            'dataset': dataset
                        }
                        models.append(model_info)
                except Exception as e:
                    print(f"{Colors.WARNING}⚠️  Could not load Streamlit model {streamlit_dir.name}: {e}{Colors.ENDC}")
        
        # Sort by timestamp (newest first)
        models.sort(key=lambda x: x['timestamp'], reverse=True)
        return models
    
    def _select_trained_model(self, available_models: List[Dict]) -> Optional[Dict]:
        """Allow user to select from available trained models."""
        print(f"\n{Colors.BOLD}📋 Available Trained Models:{Colors.ENDC}")
        print(f"{Colors.CYAN}{'-'*80}{Colors.ENDC}")
        
        for i, model in enumerate(available_models, 1):
            source_icon = "🖥️" if model['type'] == 'CLI' else "🌐"
            timestamp_display = model['timestamp'][:8] + '_' + model['timestamp'][8:]  # Format: YYYYMMDD_HHMMSS
            
            # Show basic info
            print(f"{Colors.BOLD}{i:2d}. {source_icon} {model['id']}{Colors.ENDC}")
            print(f"     📅 {timestamp_display} | 🏗️  {model['type']} | 📊 {model['dataset']}")
            
            # Show training methods
            train_methods = ', '.join(model['train_methods'][:2])  # Show first 2 methods
            if len(model['train_methods']) > 2:
                train_methods += f" (+ {len(model['train_methods']) - 2} more)"
            print(f"     🏋️  Train: {train_methods}")
            print(f"     🧪 Test: {model['test_method']}")
            
            # Show performance summary
            if model['type'] == 'CLI' and model.get('best_result'):
                best = model['best_result']
                print(f"     🎯 Best: {best['Additional_Peptides']} peptides @ {best['Actual_FDR']:.1f}% FDR")
            elif model['type'] == 'Streamlit' and 'metadata' in model and 'training_results' in model['metadata']:
                # Show Streamlit model performance
                training_results = model['metadata']['training_results']
                if training_results:
                    best = max(training_results, key=lambda x: x.get('Additional_Peptides', 0))
                    print(f"     🎯 Best: {best.get('Additional_Peptides', 0)} peptides @ {best.get('Actual_FDR', 0):.1f}% FDR")
            
            print()
        
        # Get user selection
        while True:
            try:
                choice = input(f"{Colors.BLUE}Select model (1-{len(available_models)}), 'd' for details, or 'q' to quit: {Colors.ENDC}")
                if choice.lower() == 'q':
                    return None
                elif choice.lower() == 'd':
                    # Show detailed training results
                    model_choice = input(f"{Colors.BLUE}Enter model number to view training details: {Colors.ENDC}")
                    try:
                        model_idx = int(model_choice) - 1
                        if 0 <= model_idx < len(available_models):
                            # Show details and handle navigation - both 'c' and 'b' return to model selection
                            self._display_model_training_details(available_models[model_idx])
                        else:
                            print(f"{Colors.WARNING}⚠️  Invalid model number{Colors.ENDC}")
                    except ValueError:
                        print(f"{Colors.WARNING}⚠️  Please enter a valid number{Colors.ENDC}")
                    # Always continue to show model selection again after viewing details
                    continue
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_models):
                    selected = available_models[choice_idx]
                    
                    # Show model details and handle user navigation
                    print(f"\n{Colors.GREEN}✅ Selected: {selected['id']}{Colors.ENDC}")
                    
                    # Show training details and get user choice
                    wants_to_continue = self._display_model_training_details(selected)
                    
                    if wants_to_continue:
                        # User pressed 'c' to continue - they want to use this model
                        return selected
                    else:
                        # User pressed 'b' to go back from training details, return to model list
                        continue
                else:
                    print(f"{Colors.FAIL}❌ Invalid selection. Please choose 1-{len(available_models)}{Colors.ENDC}")
            except ValueError:
                print(f"{Colors.FAIL}❌ Please enter a valid number{Colors.ENDC}")
            except KeyboardInterrupt:
                return None
    
    def _display_model_training_details(self, model: Dict) -> bool:
        """Display detailed training results for a model, like in Streamlit.
        
        Returns:
            bool: True if user wants to continue with this model, False if they want to go back
        """
        print(f"\n{Colors.BOLD}📊 Training Details: {model['id']}{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*100}{Colors.ENDC}")
        
        # Show model info
        print(f"{Colors.BOLD}Model Type:{Colors.ENDC} {model['type']}")
        print(f"{Colors.BOLD}Dataset:{Colors.ENDC} {model.get('dataset', 'Unknown')}")
        print(f"{Colors.BOLD}Training Methods:{Colors.ENDC} {', '.join(model['train_methods'])}")
        print(f"{Colors.BOLD}Test Method:{Colors.ENDC} {model['test_method']}")
        
        # Get training results
        training_results = []
        if model['type'] == 'CLI' and 'metadata' in model and 'training_results' in model['metadata']:
            training_results = model['metadata']['training_results']
        elif model['type'] == 'Streamlit' and 'metadata' in model and 'training_results' in model['metadata']:
            training_results = model['metadata']['training_results']
        
        if training_results:
            print(f"\n{Colors.BOLD}Training Results:{Colors.ENDC}")
            print(f"{Colors.CYAN}{'-'*140}{Colors.ENDC}")
            header = f"{'Target FDR':<12} {'Threshold':<12} {'Additional':<12} {'Actual FDR':<12} {'Recovery %':<12} {'Increase %':<12} {'False Positives':<10} {'MCC':<10}"
            print(f"{Colors.BOLD}{header}{Colors.ENDC}")
            print(f"{Colors.CYAN}{'-'*140}{Colors.ENDC}")
            
            for result in training_results:
                actual_fdr = f"{result.get('Actual_FDR', 0):.1f}"
                recovery = f"{result.get('Recovery_Pct', 0):.1f}"
                increase = f"{result.get('Increase_Pct', 0):.1f}"
                false_pos = result.get('False_Positives', 0)
                mcc = f"{result.get('MCC', 0):.3f}"
                
                print(f"{result['Target_FDR']:<12.1f} {result['Threshold']:<12.4f} {result['Additional_Peptides']:<12,d} "
                      f"{actual_fdr:<12} {recovery:<12} {increase:<12} {false_pos:<10,d} {mcc:<10}")
            
            print(f"{Colors.CYAN}{'-'*140}{Colors.ENDC}")
        else:
            print(f"\n{Colors.WARNING}⚠️  No training results available for this model{Colors.ENDC}")
        
        while True:
            choice = input(f"\n{Colors.BLUE}Press 'c' to continue or 'b' to go back: {Colors.ENDC}")
            if choice.lower() in ['c', 'continue']:
                return True  # User wants to continue with this model
            elif choice.lower() in ['b', 'back']:
                return False  # User wants to go back to model selection
            else:
                print(f"{Colors.WARNING}⚠️  Please enter 'c' for continue or 'b' for go back{Colors.ENDC}")
    
    def _configure_inference_test(self, selected_model: Dict) -> Optional[Dict]:
        """Configure test settings for inference."""
        print(f"\n{Colors.BOLD}🔧 Configure Inference Test:{Colors.ENDC}")
        print(f"{Colors.CYAN}{'-'*50}{Colors.ENDC}")
        
        # Step 1: Select dataset for inference
        dataset = self.select_dataset()
        if not dataset:
            print(f"\n{Colors.BLUE}↩️  Returning to model selection...{Colors.ENDC}")
            return None
        
        # Step 2: Display configured methods for the selected dataset
        available_methods = self.display_configured_methods(dataset)
        if not available_methods:
            return None
        
        # Step 3: Select test method (use configured methods as-is, whether individual or groups)
        print(f"\n{Colors.BOLD}🧪 Test Method Selection{Colors.ENDC}")
        print(f"{Colors.BLUE}Select ONE test method/group for inference{Colors.ENDC}")
        
        # Check if methods are groups or individual
        has_groups = any('Group' in method or len([m for m in available_methods if method.split('_')[0] in m]) > 1 
                        for method in available_methods)
        
        if has_groups:
            print(f"{Colors.CYAN}💡 This dataset uses pre-configured groups (including triplicates){Colors.ENDC}")
        else:
            print(f"{Colors.CYAN}💡 This dataset uses individual methods{Colors.ENDC}")
        
        while True:
            try:
                choice = input(f"{Colors.BOLD}Select test method (1-{len(available_methods)}): {Colors.ENDC}")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_methods):
                    test_method = available_methods[choice_idx]
                    print(f"{Colors.GREEN}✅ Selected: {test_method}{Colors.ENDC}")
                    break
                else:
                    print(f"{Colors.WARNING}⚠️  Please enter a number between 1 and {len(available_methods)}{Colors.ENDC}")
                    
            except ValueError:
                print(f"{Colors.WARNING}⚠️  Please enter a valid number{Colors.ENDC}")
            except KeyboardInterrupt:
                print(f"\n{Colors.FAIL}❌ Operation cancelled{Colors.ENDC}")
                return None
        
        # Select test FDR
        fdr_options = [20, 50]
        print(f"\n{Colors.BOLD}Test FDR Level:{Colors.ENDC}")
        for i, fdr in enumerate(fdr_options, 1):
            print(f"{i}. {fdr}%")
        
        while True:
            try:
                choice = input(f"{Colors.BLUE}Select FDR level (1-{len(fdr_options)}): {Colors.ENDC}")
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(fdr_options):
                    test_fdr = fdr_options[choice_idx]
                    break
                else:
                    print(f"{Colors.FAIL}❌ Invalid selection{Colors.ENDC}")
            except (ValueError, KeyboardInterrupt):
                return None
        
        # Get target FDR levels from the trained model
        model_target_fdrs = []
        if 'config' in selected_model and 'target_fdr_levels' in selected_model['config']:
            model_target_fdrs = selected_model['config']['target_fdr_levels']
        elif 'metadata' in selected_model and 'training_results' in selected_model['metadata']:
            # Extract from training results
            model_target_fdrs = [result['Target_FDR'] for result in selected_model['metadata']['training_results']]
        
        if model_target_fdrs:
            print(f"\n{Colors.BOLD}Target FDR Levels for Optimization:{Colors.ENDC}")
            print(f"{Colors.GREEN}Using model's training FDRs: {', '.join(map(str, model_target_fdrs))}%{Colors.ENDC}")
            target_fdrs = model_target_fdrs
        else:
            # Fallback to default if no model FDRs found
            default_targets = [1.0, 2.0, 3.0, 4.0, 5.0]
            print(f"\n{Colors.BOLD}Target FDR Levels for Optimization:{Colors.ENDC}")
            print(f"{Colors.WARNING}Model FDRs not found, using default: {', '.join(map(str, default_targets))}%{Colors.ENDC}")
            target_fdrs = default_targets
        
        config = {
            'test_method': test_method,  # Single test method (can be individual or group)
            'test_fdr': test_fdr,
            'target_fdr_levels': target_fdrs,
            'dataset': dataset
        }
        
        print(f"\n{Colors.GREEN}✅ Test Configuration:{Colors.ENDC}")
        print(f"   🧪 Test Method: {test_method}")
        print(f"   📊 Test FDR: {test_fdr}%")
        print(f"   🎯 Target FDRs: {', '.join(map(str, target_fdrs))}%")
        
        return config
    
    def _run_inference_analysis(self, selected_model: Dict, test_config: Dict) -> bool:
        """Run the actual inference analysis."""
        print(f"\n{Colors.BOLD}🔮 Running Inference Analysis{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*60}{Colors.ENDC}")
        
        try:
            # Import required modules
            import joblib
            
            # Load the trained model configuration
            model_dir = Path(selected_model['model_dir'])
            model_file = model_dir / "trained_model.joblib"
            features_file = model_dir / "training_features.joblib"
            
            # Get feature selection configuration from the model
            feature_selection = {}
            if 'config' in selected_model:
                feature_selection = selected_model['config'].get('feature_selection', {})
            elif 'metadata' in selected_model:
                # For Streamlit models, use default feature selection
                feature_selection = {
                    'use_diann_quality': True,
                    'use_sequence_features': True,
                    'use_ms_features': True,
                    'use_statistical_features': True,
                    'use_library_features': True,
                    'excluded_features': []
                }
            
            # Initialize API with feature selection
            api = PeptideValidatorAPI()
            api.feature_selection = feature_selection
            
            if not model_file.exists():
                print(f"{Colors.FAIL}❌ Model file not found: {model_file}{Colors.ENDC}")
                return False
            
            if not features_file.exists():
                print(f"{Colors.FAIL}❌ Features file not found: {features_file}{Colors.ENDC}")
                return False
            
            print(f"{Colors.BLUE}📁 Loading model: {model_file.name}{Colors.ENDC}")
            trained_model = joblib.load(model_file)
            training_features = joblib.load(features_file)
            
            # Load test data
            print(f"{Colors.BLUE}📊 Loading test data: {test_config['test_method']}{Colors.ENDC}")
            
            # Get test data files
            from dataset_utils import get_files_for_configured_method
            test_files = get_files_for_configured_method(test_config['test_method'], test_config['test_fdr'])
            
            if not test_files:
                print(f"{Colors.FAIL}❌ No test data files found for {test_config['test_method']} at {test_config['test_fdr']}% FDR{Colors.ENDC}")
                return False
            
            # Load and combine test data
            test_data_frames = []
            for file_path in test_files:
                try:
                    df = pd.read_parquet(file_path)
                    df['source_fdr'] = test_config['test_fdr']
                    test_data_frames.append(df)
                    print(f"   📄 Loaded: {Path(file_path).name} ({len(df):,} rows)")
                except Exception as e:
                    print(f"{Colors.WARNING}⚠️  Could not load {file_path}: {e}{Colors.ENDC}")
            
            if not test_data_frames:
                print(f"{Colors.FAIL}❌ No valid test data loaded{Colors.ENDC}")
                return False
            
            test_data = pd.concat(test_data_frames, ignore_index=True)
            print(f"{Colors.GREEN}✅ Combined test data: {len(test_data):,} rows{Colors.ENDC}")
            
            # Try to load ground truth for proper statistics
            ground_truth = None
            try:
                ground_truth_data = api._load_ground_truth_peptides(test_config['test_method'])
                if ground_truth_data is not None and len(ground_truth_data) > 0:
                    ground_truth = ground_truth_data
                    print(f"{Colors.GREEN}✅ Loaded ground truth: {len(ground_truth):,} validated peptides{Colors.ENDC}")
                else:
                    print(f"{Colors.WARNING}⚠️  No ground truth available for {test_config['test_method']}{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.WARNING}⚠️  Could not load ground truth: {e}{Colors.ENDC}")
            
            # Load baseline peptides for proper Increase % calculation
            baseline_peptides = None
            baseline_count = 0
            try:
                baseline_peptides = api._load_baseline_peptides(test_config['test_method'])
                if baseline_peptides is not None:
                    baseline_count = len(baseline_peptides)
                    print(f"{Colors.GREEN}✅ Loaded baseline peptides: {baseline_count:,} peptides{Colors.ENDC}")
                else:
                    print(f"{Colors.WARNING}⚠️  No baseline peptides available{Colors.ENDC}")
            except Exception as e:
                print(f"{Colors.WARNING}⚠️  Could not load baseline peptides: {e}{Colors.ENDC}")
                # Use ground truth size as fallback
                if ground_truth is not None:
                    baseline_count = len(ground_truth)
                    print(f"{Colors.WARNING}⚠️  Using ground truth size as baseline: {baseline_count:,}{Colors.ENDC}")
            
            # Calculate additional candidates pool for summary
            additional_candidates_pool = 0
            pool_validation_rate = 0.0
            validated_additional_count = 0
            if baseline_peptides is not None:
                # Get unique test peptides
                test_peptides = set(test_data['Stripped.Sequence'].unique())
                additional_candidates = test_peptides - baseline_peptides
                additional_candidates_pool = len(additional_candidates)
                
                # Calculate validation rate if we have ground truth
                if ground_truth is not None:
                    validated_additional = additional_candidates & ground_truth
                    validated_additional_count = len(validated_additional)
                    pool_validation_rate = (validated_additional_count / additional_candidates_pool * 100) if additional_candidates_pool > 0 else 0
                    print(f"{Colors.GREEN}✅ Additional candidates pool: {additional_candidates_pool:,} peptides{Colors.ENDC}")
                    print(f"{Colors.GREEN}✅ Pool validation: {validated_additional_count:,}/{additional_candidates_pool:,} ({pool_validation_rate:.1f}%) are validated{Colors.ENDC}")
            
            # Create features for test data
            print(f"{Colors.BLUE}🔧 Creating features for inference...{Colors.ENDC}")
            X_test = api._make_advanced_features(test_data, training_features)
            
            # Ensure features match training
            if isinstance(training_features, list):
                training_features_list = training_features
            else:
                training_features_list = training_features.tolist()
            
            missing_features = set(training_features_list) - set(X_test.columns)
            if missing_features:
                print(f"{Colors.WARNING}⚠️  Adding missing features: {len(missing_features)} features{Colors.ENDC}")
                for feature in missing_features:
                    X_test[feature] = 0
            
            # Reorder columns to match training
            X_test = X_test[training_features_list]
            
            print(f"{Colors.GREEN}✅ Feature matrix ready: {X_test.shape[0]:,} samples × {X_test.shape[1]} features{Colors.ENDC}")
            
            # Make predictions
            print(f"{Colors.BLUE}🎯 Making predictions...{Colors.ENDC}")
            y_pred = trained_model.predict_proba(X_test)[:, 1]
            
            # Run optimization for different FDR targets (suppress verbose output)
            print(f"{Colors.BLUE}📈 Optimizing thresholds for target FDRs...{Colors.ENDC}")
            
            # Calculate results with proper statistics
            results = []
            for target_fdr in test_config['target_fdr_levels']:
                try:
                    if ground_truth is not None:
                        # Use ground truth for proper statistics calculation (like Streamlit)
                        # Create peptide-level aggregation like in the main API
                        aggregation_method = 'max'  # Default
                        if 'config' in selected_model and 'aggregation_method' in selected_model['config']:
                            aggregation_method = selected_model['config']['aggregation_method']
                        
                        # Create peptide predictions using the same process as training
                        # Create labels for test data based on ground truth
                        test_peptides = test_data['Stripped.Sequence'].values
                        peptide_labels = pd.Series([peptide in ground_truth for peptide in test_peptides])
                        
                        # Aggregate predictions by peptide
                        peptide_df, peptide_predictions, peptide_labels = api._aggregate_predictions_by_peptide(
                            test_data, y_pred, peptide_labels, aggregation_method
                        )
                        
                        # Filter to only additional candidates (not baseline) for Streamlit-like behavior
                        if baseline_peptides is not None:
                            # Determine which sequence column exists in the aggregated data
                            seq_column = 'Modified.Sequence' if 'Modified.Sequence' in peptide_df.columns else 'Stripped.Sequence'
                            additional_candidates_mask = ~peptide_df[seq_column].isin(baseline_peptides)
                            peptide_df = peptide_df[additional_candidates_mask].reset_index(drop=True)
                            peptide_predictions = peptide_predictions[additional_candidates_mask]
                            peptide_labels = peptide_labels[additional_candidates_mask].reset_index(drop=True)
                        
                        # Use the same threshold optimization as training
                        threshold, additional_peptides, actual_fdr = api._find_optimal_threshold(
                            peptide_labels, peptide_predictions, target_fdr, verbose=False
                        )
                        
                        if threshold is not None:
                            # Calculate metrics like in training
                            total_validated = peptide_labels.sum()
                            recovery_pct = additional_peptides / total_validated * 100 if total_validated > 0 else 0
                            increase_pct = additional_peptides / baseline_count * 100 if baseline_count > 0 else 0
                            
                            # Calculate MCC
                            predictions = peptide_predictions >= threshold
                            tp = (peptide_labels & predictions).sum()
                            fp = (~peptide_labels & predictions).sum()
                            
                            # MCC calculation
                            from sklearn.metrics import matthews_corrcoef
                            mcc = matthews_corrcoef(peptide_labels, predictions)
                            
                            result = {
                                'Target_FDR': target_fdr,
                                'Threshold': threshold,
                                'Additional_Peptides': additional_peptides,
                                'Actual_FDR': actual_fdr,
                                'Recovery_Pct': recovery_pct,
                                'Increase_Pct': increase_pct,
                                'False_Positives': fp,
                                'Total_Validated_Candidates': total_validated,
                                'MCC': mcc,
                                'Aggregation_Method': aggregation_method
                            }
                            results.append(result)
                        else:
                            # Threshold not found
                            result = {
                                'Target_FDR': target_fdr,
                                'Threshold': None,
                                'Additional_Peptides': 0,
                                'Actual_FDR': None,
                                'Recovery_Pct': 0,
                                'Increase_Pct': 0,
                                'False_Positives': 0,
                                'Total_Validated_Candidates': peptide_labels.sum() if peptide_labels is not None else 0,
                                'MCC': 0
                            }
                            results.append(result)
                    else:
                        # Fallback for when no ground truth is available
                        sorted_indices = np.argsort(y_pred)[::-1]
                        sorted_scores = y_pred[sorted_indices]
                        
                        # Conservative threshold estimation
                        num_predictions = len(y_pred)
                        target_false_positives = int(num_predictions * target_fdr / 100)
                        
                        if target_false_positives < len(sorted_scores):
                            threshold = sorted_scores[target_false_positives] if target_false_positives > 0 else sorted_scores[0]
                        else:
                            threshold = sorted_scores[-1]
                        
                        additional_peptides = np.sum(y_pred >= threshold)
                        
                        result = {
                            'Target_FDR': target_fdr,
                            'Threshold': threshold,
                            'Additional_Peptides': additional_peptides,
                            'Actual_FDR': None,  # Unknown without ground truth
                            'Recovery_Pct': None,
                            'Increase_Pct': None,
                            'False_Positives': None,
                            'Total_Validated_Candidates': None,
                            'MCC': None,
                            'Prediction_Score_Range': f"{y_pred.min():.4f} - {y_pred.max():.4f}",
                            'High_Confidence': np.sum(y_pred >= 0.8),
                            'Medium_Confidence': np.sum((y_pred >= 0.5) & (y_pred < 0.8)),
                            'Low_Confidence': np.sum(y_pred < 0.5)
                        }
                        results.append(result)
                        
                except Exception as e:
                    print(f"{Colors.WARNING}⚠️  Could not optimize for {target_fdr}% FDR: {e}{Colors.ENDC}")
            
            # Display results
            self._display_inference_results(results, selected_model, test_config, len(test_data), baseline_count, additional_candidates_pool, validated_additional_count, pool_validation_rate)
            
            # Save results
            self._save_inference_results(results, selected_model, test_config)
            
            return True
            
        except Exception as e:
            print(f"{Colors.FAIL}❌ Inference analysis failed: {str(e)}{Colors.ENDC}")
            import traceback
            print(f"{Colors.WARNING}Debug info:{Colors.ENDC}")
            traceback.print_exc()
            return False
    
    def _display_inference_results(self, results: List[Dict], model_info: Dict, test_config: Dict, total_samples: int, baseline_count: int = 0, additional_candidates_pool: int = 0, validated_additional_count: int = 0, pool_validation_rate: float = 0.0):
        """Display inference results in a formatted table matching Streamlit format."""
        print(f"\n{Colors.BOLD}🎉 INFERENCE RESULTS{Colors.ENDC}")
        print(f"{Colors.CYAN}{'='*120}{Colors.ENDC}")
        
        # Model summary
        print(f"{Colors.BOLD}Model Used:{Colors.ENDC} {model_info['id']} ({model_info['type']})")
        print(f"{Colors.BOLD}Test Method:{Colors.ENDC} {test_config['test_method']}")
        print(f"{Colors.BOLD}Test Data:{Colors.ENDC} {total_samples:,} samples at {test_config['test_fdr']}% FDR")
        
        # Add baseline peptides information if available
        if baseline_count > 0:
            print(f"{Colors.BOLD}Baseline Peptides:{Colors.ENDC} {baseline_count:,} peptides at 1% FDR")
        
        # Add additional candidates pool information
        if additional_candidates_pool > 0:
            print(f"{Colors.BOLD}Additional Candidates Pool:{Colors.ENDC} {additional_candidates_pool:,} peptides available for validation")
            if validated_additional_count > 0:
                print(f"{Colors.BOLD}Pool Validation Rate:{Colors.ENDC} {validated_additional_count:,}/{additional_candidates_pool:,} ({pool_validation_rate:.1f}%) of additional candidates are validated")
            
        print()
        
        # Check if we have ground truth results
        has_ground_truth = results and 'Actual_FDR' in results[0] and results[0]['Actual_FDR'] is not None
        
        # Results table - match training format exactly
        if has_ground_truth:
            # Match training results format exactly
            print(f"{Colors.BOLD}Target FDR Optimization Results:{Colors.ENDC}")
            print(f"{Colors.CYAN}{'-'*140}{Colors.ENDC}")
            header = f"{'Target FDR':<12} {'Threshold':<12} {'Additional':<12} {'Actual FDR':<12} {'Recovery %':<12} {'Increase %':<12} {'False Positives':<10} {'MCC':<10}"
            print(f"{Colors.BOLD}{header}{Colors.ENDC}")
            print(f"{Colors.CYAN}{'-'*140}{Colors.ENDC}")
            
            for result in results:
                actual_fdr = f"{result.get('Actual_FDR', 0):.1f}"
                recovery = f"{result.get('Recovery_Pct', 0):.1f}"
                increase = f"{result.get('Increase_Pct', 0):.1f}"
                false_pos = result.get('False_Positives', 0)
                mcc = f"{result.get('MCC', 0):.3f}"
                
                print(f"{result['Target_FDR']:<12.1f} {result['Threshold']:<12.4f} {result['Additional_Peptides']:<12,d} "
                      f"{actual_fdr:<12} {recovery:<12} {increase:<12} {false_pos:<10,d} {mcc:<10}")
            
            print(f"{Colors.CYAN}{'-'*140}{Colors.ENDC}")
        else:
            # Simplified table without ground truth (confidence-based)
            print(f"{Colors.CYAN}{'-'*120}{Colors.ENDC}")
            header = f"{'Target FDR':<12} {'Threshold':<12} {'Additional':<12} {'Confidence Distribution':<35} {'Score Range':<20}"
            print(f"{Colors.BOLD}{header}{Colors.ENDC}")
            subheader = f"{'(%)':<12} {'Score':<12} {'Peptides':<12} {'High(>80%) Med(50-80%) Low(<50%)':<35} {'Min - Max':<20}"
            print(f"{Colors.CYAN}{subheader}{Colors.ENDC}")
            print(f"{Colors.CYAN}{'-'*120}{Colors.ENDC}")
            
            for result in results:
                if 'High_Confidence' in result:
                    confidence_dist = f"{result['High_Confidence']:,}     {result['Medium_Confidence']:,}      {result['Low_Confidence']:,}"
                    score_range = result.get('Prediction_Score_Range', 'N/A')
                    print(f"{result['Target_FDR']:<12.1f} {result['Threshold']:<12.4f} {result['Additional_Peptides']:<12,d} "
                          f"{confidence_dist:<35} {score_range:<20}")
                else:
                    # Ground truth results without confidence breakdown
                    print(f"{result['Target_FDR']:<12.1f} {result['Threshold']:<12.4f} {result['Additional_Peptides']:<12,d}")
            
            print(f"{Colors.CYAN}{'-'*120}{Colors.ENDC}")
        
        
        # Update note based on ground truth availability
        if has_ground_truth:
            print(f"\n{Colors.GREEN}💡 Note: Results calculated with ground truth validation for accurate statistics{Colors.ENDC}")
        else:
            print(f"\n{Colors.BLUE}💡 Note: These are predictions on new data without ground truth validation{Colors.ENDC}")
    
    def _save_inference_results(self, results: List[Dict], model_info: Dict, test_config: Dict):
        """Save inference results to file."""
        try:
            # Create inference results directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_dir = Path(f"results/CLI_INFERENCE_{timestamp}")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results summary
            summary = {
                'timestamp': datetime.now().isoformat(),
                'model_used': {
                    'id': model_info['id'],
                    'type': model_info['type'],
                    'model_dir': model_info['model_dir']
                },
                'test_configuration': test_config,
                'results': results,
                'inference_type': 'CLI_Inference'
            }
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                elif hasattr(obj, 'item'):  # numpy scalars
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy arrays
                    return obj.tolist()
                else:
                    return obj
            
            summary_serializable = convert_numpy_types(summary)
            
            summary_file = results_dir / "inference_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_serializable, f, indent=2)
            
            print(f"\n{Colors.GREEN}💾 Results saved to: {results_dir}{Colors.ENDC}")
            print(f"   📄 Summary: {summary_file}")
            
        except Exception as e:
            print(f"{Colors.WARNING}⚠️  Could not save results: {e}{Colors.ENDC}")
    
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
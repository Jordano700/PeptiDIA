#!/usr/bin/env python3
"""
PeptiDIA vs DIA-NN Performance Comparison Tool

Interactive script to compare your ML model results against DIA-NN's native 
Q-value thresholds with clear terminal output and run selection.
"""

import pandas as pd
import numpy as np
import glob
import os
import json
from pathlib import Path
from datetime import datetime
import sys

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

def print_banner():
    """Print the tool banner."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}")
    print("🚀 PEPTIDIA vs DIA-NN PERFORMANCE COMPARISON TOOL 🚀")
    print(f"{'='*70}{Colors.ENDC}")
    print(f"{Colors.BLUE}Compare your ML model performance against DIA-NN's Q-value thresholds{Colors.ENDC}")
    print(f"{Colors.GREEN}💡 Tips: Use 'info <number>' to see training details before analysis{Colors.ENDC}\n")

def find_all_results(results_dir="results"):
    """Find all result directories with detailed information."""
    result_dirs = glob.glob(f"{results_dir}/STREAMLIT_RESULTS_*")
    
    results_info = []
    
    for results_dir in result_dirs:
        dir_name = Path(results_dir).name
        
        # Check if this directory has the required files
        csv_path = f"{results_dir}/tables/detailed_results.csv"
        json_path = f"{results_dir}/raw_data/analysis_summary.json"
        
        if os.path.exists(csv_path) and os.path.exists(json_path):
            try:
                # Load config to get test method and timestamp
                with open(json_path, 'r') as f:
                    config = json.load(f)
                
                test_method = config['config']['test_method']
                train_methods = config['config'].get('train_methods', [])
                test_fdr = config['config'].get('test_fdr', 50)
                train_fdr_levels = config['config'].get('train_fdr_levels', [])
                timestamp = config.get('metadata', {}).get('analysis_timestamp', '')
                
                # Extract dataset name
                dataset = test_method.split('_')[0]
                
                # Parse timestamp for sorting
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        sort_time = dt
                        display_time = dt.strftime('%Y-%m-%d %H:%M')
                    except:
                        sort_time = datetime.min
                        display_time = "Unknown"
                else:
                    sort_time = datetime.min
                    display_time = "Unknown"
                
                results_info.append({
                    'dir': results_dir,
                    'dir_name': dir_name,
                    'test_method': test_method,
                    'train_methods': train_methods,
                    'test_fdr': test_fdr,
                    'train_fdr_levels': train_fdr_levels,
                    'dataset': dataset,
                    'timestamp': display_time,
                    'sort_time': sort_time,
                    'csv_path': csv_path,
                    'json_path': json_path
                })
                
            except Exception as e:
                continue
    
    # Sort by timestamp (most recent first)
    results_info.sort(key=lambda x: x['sort_time'], reverse=True)
    
    return results_info

def display_available_runs(results_info):
    """Display available runs in a formatted table."""
    if not results_info:
        print(f"{Colors.FAIL}❌ No valid results found in results/ directory{Colors.ENDC}")
        return False
    
    print(f"{Colors.BOLD}📂 Available Analysis Runs:{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*90}")
    print(f"{'#':>3} | {'Date & Time':>16} | {'Dataset':>8} | {'Method Description'}")
    print(f"{'='*90}{Colors.ENDC}")
    
    for i, info in enumerate(results_info, 1):
        # Truncate long method names for display
        method_display = info['test_method']
        if len(method_display) > 50:
            method_display = method_display[:47] + "..."
        
        print(f"{Colors.GREEN}{i:>3}{Colors.ENDC} | {info['timestamp']:>16} | {Colors.BOLD}{info['dataset']:>8}{Colors.ENDC} | {method_display}")
    
    print(f"{Colors.CYAN}{'='*90}{Colors.ENDC}\n")
    return True

def display_run_details(run_info):
    """Display detailed information about the selected run."""
    print(f"\n{Colors.BOLD}📋 SELECTED RUN DETAILS{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*70}")
    
    # Basic info
    print(f"{Colors.BLUE}   Run ID: {Colors.BOLD}{run_info['dir_name']}{Colors.ENDC}")
    print(f"{Colors.BLUE}   Date: {Colors.BOLD}{run_info['timestamp']}{Colors.ENDC}")
    print(f"{Colors.BLUE}   Dataset: {Colors.BOLD}{run_info['dataset']}{Colors.ENDC}")
    
    # Test configuration
    print(f"\n{Colors.BOLD}🎯 Test Configuration:{Colors.ENDC}")
    print(f"{Colors.BLUE}   Test Method: {Colors.BOLD}{run_info['test_method']}{Colors.ENDC}")
    print(f"{Colors.BLUE}   Test FDR Level: {Colors.BOLD}{run_info['test_fdr']}%{Colors.ENDC}")
    
    # Training configuration
    print(f"\n{Colors.BOLD}🏋️  Training Configuration:{Colors.ENDC}")
    train_methods = run_info['train_methods']
    train_fdr_levels = run_info['train_fdr_levels']
    
    print(f"{Colors.BLUE}   Training FDR Levels: {Colors.BOLD}{', '.join(map(str, train_fdr_levels))}%{Colors.ENDC}")
    print(f"{Colors.BLUE}   Training Methods ({len(train_methods)} total):{Colors.ENDC}")
    
    # Display training methods in a nice format
    for i, method in enumerate(train_methods, 1):
        # Truncate long method names for better display
        if len(method) > 60:
            display_method = method[:57] + "..."
        else:
            display_method = method
        
        print(f"{Colors.BLUE}     {i:>2}. {display_method}{Colors.ENDC}")
        
        # If there are many methods, show first few and summarize
        if len(train_methods) > 8 and i == 5:
            remaining = len(train_methods) - 5
            print(f"{Colors.BLUE}        ... and {remaining} more training methods{Colors.ENDC}")
            break
    
    print(f"{Colors.CYAN}{'='*70}{Colors.ENDC}")

def get_user_choice(results_info):
    """Get user's choice of which run to analyze."""
    while True:
        try:
            choice = input(f"{Colors.BOLD}Select run to analyze (1-{len(results_info)}), 'q' to quit, or 'info <number>' for details: {Colors.ENDC}")
            
            if choice.lower() == 'q':
                return None
            
            # Check if user wants info about a specific run
            if choice.lower().startswith('info '):
                try:
                    info_num = int(choice.split()[1])
                    if 1 <= info_num <= len(results_info):
                        display_run_details(results_info[info_num - 1])
                        continue
                    else:
                        print(f"{Colors.WARNING}⚠️  Please enter a number between 1 and {len(results_info)}{Colors.ENDC}")
                        continue
                except (ValueError, IndexError):
                    print(f"{Colors.WARNING}⚠️  Invalid format. Use 'info <number>' (e.g., 'info 1'){Colors.ENDC}")
                    continue
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(results_info):
                selected_run = results_info[choice_num - 1]
                # Show details of selected run before analysis
                display_run_details(selected_run)
                
                # Confirm selection
                confirm = input(f"\n{Colors.BOLD}Proceed with analysis of this run? (y/n): {Colors.ENDC}").lower().strip()
                if confirm in ['y', 'yes']:
                    return selected_run
                elif confirm in ['n', 'no']:
                    continue
                else:
                    print(f"{Colors.WARNING}Please enter 'y' or 'n'{Colors.ENDC}")
                    continue
            else:
                print(f"{Colors.WARNING}⚠️  Please enter a number between 1 and {len(results_info)}{Colors.ENDC}")
        
        except ValueError:
            print(f"{Colors.WARNING}⚠️  Please enter a valid number, 'info <number>', or 'q' to quit{Colors.ENDC}")
        except KeyboardInterrupt:
            print(f"\n{Colors.FAIL}❌ Operation cancelled{Colors.ENDC}")
            return None

def load_data_files(test_method, test_fdr=50):
    """Load baseline, test, and ground truth data files."""
    dataset = test_method.split('_')[0]
    
    print(f"{Colors.BLUE}🔍 Loading data for {Colors.BOLD}{test_method}{Colors.ENDC}")
    print(f"{Colors.BLUE}   Dataset: {Colors.BOLD}{dataset}{Colors.ENDC}")
    
    # Define file patterns
    baseline_pattern = f"data/{dataset}/short_gradient/FDR_1/*FDR1.parquet"
    test_pattern = f"data/{dataset}/short_gradient/FDR_{test_fdr}/*FDR{test_fdr}.parquet" 
    gt_pattern = f"data/{dataset}/long_gradient/FDR_1/*FDR1.parquet"
    
    # Find files
    baseline_files = glob.glob(baseline_pattern)
    test_files = glob.glob(test_pattern)
    gt_files = glob.glob(gt_pattern)
    
    print(f"{Colors.BLUE}   Found: {Colors.BOLD}{len(baseline_files)} baseline, {len(test_files)} test, {len(gt_files)} GT{Colors.ENDC} files")
    
    if not baseline_files:
        raise FileNotFoundError(f"No baseline files found for {dataset} (pattern: {baseline_pattern})")
    if not test_files:
        raise FileNotFoundError(f"No test files found for {dataset} at {test_fdr}% FDR (pattern: {test_pattern})")
    if not gt_files:
        raise FileNotFoundError(f"No ground truth files found for {dataset} (pattern: {gt_pattern})")
    
    # Load and combine files
    print(f"{Colors.BLUE}   Loading data files...{Colors.ENDC}")
    
    baseline_data = pd.concat([pd.read_parquet(f) for f in baseline_files], ignore_index=True)
    test_data = pd.concat([pd.read_parquet(f) for f in test_files], ignore_index=True)
    gt_data = pd.concat([pd.read_parquet(f) for f in gt_files], ignore_index=True)
    
    print(f"{Colors.GREEN}   ✅ Loaded: {len(baseline_data):,} baseline, {len(test_data):,} test, {len(gt_data):,} GT peptides{Colors.ENDC}")
    
    return baseline_data, test_data, gt_data

def calculate_peptide_sets(baseline_data, test_data, gt_data):
    """Calculate the different peptide sets for analysis."""
    baseline_peptides = set(baseline_data['Modified.Sequence'])
    gt_peptides = set(gt_data['Modified.Sequence'])
    test_peptides = set(test_data['Modified.Sequence'])
    additional_candidates = test_peptides - baseline_peptides
    validated_additional = additional_candidates & gt_peptides
    
    return {
        'baseline': baseline_peptides,
        'ground_truth': gt_peptides,
        'test_pool': test_peptides,
        'additional_candidates': additional_candidates,
        'validated_additional': validated_additional
    }

def print_dataset_summary(peptide_sets):
    """Print a summary of the dataset."""
    print(f"\n{Colors.BOLD}📊 DATASET SUMMARY{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*50}")
    
    validation_rate = len(peptide_sets['validated_additional']) / len(peptide_sets['additional_candidates']) * 100 if peptide_sets['additional_candidates'] else 0
    
    summary_data = [
        ("Baseline peptides (short 1% FDR)", len(peptide_sets['baseline'])),
        ("Ground truth peptides (long 1% FDR)", len(peptide_sets['ground_truth'])),
        ("Test pool peptides", len(peptide_sets['test_pool'])),
        ("Additional candidates", len(peptide_sets['additional_candidates'])),
        ("Validated additional", len(peptide_sets['validated_additional'])),
        ("Validation rate", f"{validation_rate:.1f}%")
    ]
    
    for label, value in summary_data:
        if isinstance(value, int):
            print(f"{Colors.BLUE}   {label:.<40} {Colors.BOLD}{value:>8,}{Colors.ENDC}")
        else:
            print(f"{Colors.BLUE}   {label:.<40} {Colors.BOLD}{value:>8}{Colors.ENDC}")
    
    print(f"{Colors.CYAN}{'='*50}{Colors.ENDC}")

def prepare_diann_data(test_data, peptide_sets):
    """Prepare DIA-NN data for comparison by aggregating Q-values per peptide."""
    additional_df = test_data[test_data['Modified.Sequence'].isin(peptide_sets['additional_candidates'])].copy()
    additional_df['is_validated'] = additional_df['Modified.Sequence'].isin(peptide_sets['validated_additional'])
    
    # Aggregate by peptide (min Q-value per peptide - best confidence)
    diann_agg = additional_df.groupby('Modified.Sequence').agg({
        'Q.Value': 'min',
        'is_validated': 'first'
    }).reset_index().sort_values('Q.Value')
    
    return diann_agg

def perform_comparison(ml_results_df, diann_agg, test_method):
    """Perform the ML vs DIA-NN comparison."""
    print(f"\n{Colors.BOLD}🏆 ML MODEL vs DIA-NN PERFORMANCE COMPARISON{Colors.ENDC}")
    print(f"{Colors.BLUE}Method: {Colors.BOLD}{test_method}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*95}")
    
    # Create detailed comparison table header
    header = f"{'Target':>8} | {'ML':>6} | {'ML':>8} | {'DIA-NN':>8} | {'DIA-NN':>8} | {'ML':>10} | {'Q-value':>10}"
    subheader = f"{'FDR':>8} | {'Pept':>6} | {'FDR':>8} | {'Pept':>8} | {'FDR':>8} | {'Advantage':>10} | {'Threshold':>10}"
    
    print(f"{Colors.BOLD}{header}")
    print(f"{subheader}{Colors.ENDC}")
    print(f"{Colors.CYAN}{'-'*95}{Colors.ENDC}")
    
    comparison_results = []
    
    for _, row in ml_results_df.iterrows():
        target_fdr = row['Target_FDR']
        ml_peptides = int(row['Additional_Peptides'])
        ml_actual_fdr = row['Actual_FDR']
        
        if ml_peptides <= len(diann_agg) and ml_peptides > 0:
            # Get DIA-NN performance for same peptide count
            selected = diann_agg.head(ml_peptides)
            tp = selected['is_validated'].sum()
            fp = len(selected) - tp
            diann_fdr = fp / len(selected) * 100 if len(selected) > 0 else 0
            
            # Get the Q-value threshold used
            q_threshold = selected['Q.Value'].max()
            
            advantage = diann_fdr - ml_actual_fdr
            
            # Color coding for advantage
            if advantage > 15:
                advantage_color = Colors.GREEN
                advantage_symbol = "+++"
            elif advantage > 5:
                advantage_color = Colors.CYAN
                advantage_symbol = "++"
            elif advantage > 0:
                advantage_color = Colors.BLUE
                advantage_symbol = "+"
            else:
                advantage_color = Colors.WARNING
                advantage_symbol = "-"
            
            print(f"{target_fdr:>7.1f}% | {ml_peptides:>6,} | {Colors.BOLD}{ml_actual_fdr:>7.1f}%{Colors.ENDC} | "
                  f"{ml_peptides:>8,} | {diann_fdr:>7.1f}% | "
                  f"{advantage_color}{advantage:>+9.1f}pp{Colors.ENDC} | {q_threshold:>10.4f}")
            
            comparison_results.append({
                'target_fdr': target_fdr,
                'ml_peptides': ml_peptides,
                'ml_fdr': ml_actual_fdr,
                'diann_fdr': diann_fdr,
                'advantage': advantage,
                'q_threshold': q_threshold
            })
        else:
            print(f"{target_fdr:>7.1f}% | {ml_peptides:>6,} | {Colors.BOLD}{ml_actual_fdr:>7.1f}%{Colors.ENDC} | "
                  f"{'N/A':>8} | {'N/A':>8} | {'N/A':>10} | {'N/A':>10}")
    
    print(f"{Colors.CYAN}{'-'*95}{Colors.ENDC}")
    
    return comparison_results

def print_summary_statistics(comparison_results):
    """Print summary statistics of the comparison."""
    if not comparison_results:
        return
    
    valid_results = [r for r in comparison_results if r['advantage'] is not None]
    
    if not valid_results:
        return
    
    advantages = [r['advantage'] for r in valid_results]
    avg_advantage = np.mean(advantages)
    max_advantage = max(advantages)
    min_advantage = min(advantages)
    
    print(f"\n{Colors.BOLD}📈 PERFORMANCE SUMMARY{Colors.ENDC}")
    print(f"{Colors.CYAN}{'='*50}")
    
    print(f"{Colors.BLUE}   Average ML Advantage: {Colors.BOLD}{avg_advantage:+7.1f} percentage points{Colors.ENDC}")
    print(f"{Colors.BLUE}   Maximum ML Advantage: {Colors.BOLD}{max_advantage:+7.1f} percentage points{Colors.ENDC}")
    print(f"{Colors.BLUE}   Minimum ML Advantage: {Colors.BOLD}{min_advantage:+7.1f} percentage points{Colors.ENDC}")
    
    # Performance assessment
    if avg_advantage > 15:
        assessment = f"{Colors.GREEN}🎯 EXCELLENT! ML model significantly outperforms DIA-NN{Colors.ENDC}"
    elif avg_advantage > 8:
        assessment = f"{Colors.CYAN}✅ VERY GOOD! ML model provides substantial improvement{Colors.ENDC}"
    elif avg_advantage > 3:
        assessment = f"{Colors.BLUE}👍 GOOD! ML model shows solid improvement over DIA-NN{Colors.ENDC}"
    elif avg_advantage > 0:
        assessment = f"{Colors.WARNING}⚠️  MODEST improvement - consider parameter tuning{Colors.ENDC}"
    else:
        assessment = f"{Colors.FAIL}❌ Poor performance - model needs optimization{Colors.ENDC}"
    
    print(f"\n   {assessment}")
    print(f"{Colors.CYAN}{'='*50}{Colors.ENDC}")

def analyze_selected_run(run_info):
    """Analyze the selected run."""
    try:
        print(f"\n{Colors.BOLD}🔬 ANALYZING SELECTED RUN{Colors.ENDC}")
        print(f"{Colors.BLUE}Run: {Colors.BOLD}{run_info['dir_name']}{Colors.ENDC}")
        print(f"{Colors.BLUE}Date: {Colors.BOLD}{run_info['timestamp']}{Colors.ENDC}")
        print(f"{Colors.BLUE}Dataset: {Colors.BOLD}{run_info['dataset']}{Colors.ENDC}")
        
        # Load ML results
        ml_results_df = pd.read_csv(run_info['csv_path'])
        print(f"{Colors.GREEN}✅ Loaded ML results: {len(ml_results_df)} FDR targets{Colors.ENDC}")
        
        # Load config for test FDR
        with open(run_info['json_path'], 'r') as f:
            config = json.load(f)
        
        test_fdr = config['config'].get('test_fdr', 50)
        
        # Load raw data files
        baseline_data, test_data, gt_data = load_data_files(run_info['test_method'], test_fdr)
        
        # Calculate peptide sets
        peptide_sets = calculate_peptide_sets(baseline_data, test_data, gt_data)
        
        # Print dataset summary
        print_dataset_summary(peptide_sets)
        
        # Prepare DIA-NN comparison data
        print(f"\n{Colors.BLUE}🔄 Preparing DIA-NN comparison data...{Colors.ENDC}")
        diann_agg = prepare_diann_data(test_data, peptide_sets)
        print(f"{Colors.GREEN}✅ Processed {len(diann_agg):,} additional peptides for comparison{Colors.ENDC}")
        
        # Perform comparison
        comparison_results = perform_comparison(ml_results_df, diann_agg, run_info['test_method'])
        
        # Print summary statistics
        print_summary_statistics(comparison_results)
        
        return True
        
    except Exception as e:
        print(f"{Colors.FAIL}❌ Error analyzing run: {str(e)}{Colors.ENDC}")
        return False

def main():
    """Main function."""
    print_banner()
    
    try:
        # Find all available results
        results_info = find_all_results()
        
        # Display available runs
        if not display_available_runs(results_info):
            sys.exit(1)
        
        # Let user choose which run to analyze
        selected_run = get_user_choice(results_info)
        
        if selected_run is None:
            print(f"{Colors.BLUE}👋 Goodbye!{Colors.ENDC}")
            sys.exit(0)
        
        # Analyze the selected run
        success = analyze_selected_run(selected_run)
        
        if success:
            print(f"\n{Colors.GREEN}🎉 Analysis completed successfully!{Colors.ENDC}")
            
            # Ask if user wants to analyze another run
            while True:
                try:
                    another = input(f"\n{Colors.BOLD}Analyze another run? (y/n): {Colors.ENDC}").lower().strip()
                    if another in ['y', 'yes']:
                        print()  # Add spacing
                        display_available_runs(results_info)
                        selected_run = get_user_choice(results_info)
                        if selected_run:
                            analyze_selected_run(selected_run)
                        else:
                            break
                    elif another in ['n', 'no']:
                        break
                    else:
                        print(f"{Colors.WARNING}Please enter 'y' or 'n'{Colors.ENDC}")
                except KeyboardInterrupt:
                    break
        
        print(f"\n{Colors.BLUE}👋 Thanks for using PeptiDIA comparison tool!{Colors.ENDC}")
        
    except KeyboardInterrupt:
        print(f"\n{Colors.FAIL}❌ Operation cancelled by user{Colors.ENDC}")
    except Exception as e:
        print(f"{Colors.FAIL}❌ Unexpected error: {str(e)}{Colors.ENDC}")

if __name__ == "__main__":
    main()
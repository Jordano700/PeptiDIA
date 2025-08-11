#!/usr/bin/env python3
"""
DATASET UTILITIES MODULE
================================================================================
Shared utilities for dataset configuration and file discovery.
Separated from streamlit_app.py to avoid circular imports with peptide_validator_api.py
================================================================================
"""

import os
import json
import glob
from pathlib import Path


def discover_available_files_by_dataset():
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


def get_configured_methods(dataset_filter=None):
    """Get available methods respecting saved dataset configurations.
    
    Args:
        dataset_filter: If provided, only return methods from this dataset (e.g., 'ASTRAL', 'HEK')
    """
    files_info_by_dataset = discover_available_files_by_dataset()
    all_methods = []
    
    for dataset_name, dataset_info in files_info_by_dataset.items():
        # Apply dataset filter if specified
        if dataset_filter and dataset_filter != 'All' and dataset_name != dataset_filter:
            continue
            
        # Check if there's a saved configuration for this dataset
        config_path = f"data/{dataset_name}/dataset_info.json"
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                ground_truth_mapping = config.get("ground_truth_mapping", {})
                file_mode = ground_truth_mapping.get("_file_mode", "individual")
                
                if file_mode == "triplicates":
                    # Use configured grouping method
                    strategy = ground_truth_mapping.get("_strategy", "direct_mapping")
                    
                    if strategy == "direct_mapping":
                        # Use the direct mapping rules from setup
                        direct_rules = ground_truth_mapping.get("_direct_rules", {})
                        configured_methods = list(direct_rules.keys())
                        all_methods.extend(configured_methods)
                    elif strategy == "use_all_ground_truth" and "_group_definitions" in ground_truth_mapping:
                        # Use the group definitions from string search setup
                        group_definitions = ground_truth_mapping.get("_group_definitions", {})
                        configured_methods = list(group_definitions.keys())
                        all_methods.extend(configured_methods)
                    else:
                        # Fallback to individual files if no group definitions
                        training_methods = []
                        for fdr_level in ['20', '50']:
                            if fdr_level in dataset_info.get('training', {}):
                                methods = list(dataset_info['training'][fdr_level].keys())
                                training_methods.extend(methods)
                        all_methods.extend(sorted(list(set(training_methods))))
                else:
                    # Individual files mode
                    training_methods = []
                    for fdr_level in ['20', '50']:
                        if fdr_level in dataset_info.get('training', {}):
                            methods = list(dataset_info['training'][fdr_level].keys())
                            training_methods.extend(methods)
                    all_methods.extend(sorted(list(set(training_methods))))
                    
            except Exception as e:
                # Fallback to individual files if config can't be read
                training_methods = []
                for fdr_level in ['20', '50']:
                    if fdr_level in dataset_info.get('training', {}):
                        methods = list(dataset_info['training'][fdr_level].keys())
                        training_methods.extend(methods)
                all_methods.extend(sorted(list(set(training_methods))))
        else:
            # No configuration file - use individual files
            training_methods = []
            for fdr_level in ['20', '50']:
                if fdr_level in dataset_info.get('training', {}):
                    methods = list(dataset_info['training'][fdr_level].keys())
                    training_methods.extend(methods)
            all_methods.extend(sorted(list(set(training_methods))))
    
    # Filter out methods with empty ground truth files
    validated_methods = []
    for method in sorted(list(set(all_methods))):
        if validate_ground_truth_files(method):
            validated_methods.append(method)
    
    return validated_methods


def get_files_for_configured_method(method_name: str, fdr_level: int):
    """Get file paths for a configured method (handles both individual files and groups)."""
    files_info_by_dataset = discover_available_files_by_dataset()
    
    # Check if this is a grouped method (contains "_Group_")
    if "_Group_" in method_name:
        # Extract dataset and group info
        parts = method_name.split("_Group_")
        if len(parts) == 2:
            dataset_name = parts[0]
            group_term = parts[1]
            
            # Check if there's a saved configuration for this dataset
            config_path = f"data/{dataset_name}/dataset_info.json"
            
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    
                    ground_truth_mapping = config.get("ground_truth_mapping", {})
                    strategy = ground_truth_mapping.get("_strategy", "direct_mapping")
                    
                    if strategy == "direct_mapping":
                        # Get the specific files from the direct mapping
                        direct_rules = ground_truth_mapping.get("_direct_rules", {})
                        if method_name in direct_rules:
                            # This group was created in setup - need to find actual files that match
                            dataset_info = files_info_by_dataset.get(dataset_name, {})
                            
                            # Find all files that contain the group term
                            matching_files = []
                            
                            # For FDR_1, search in baseline files
                            if fdr_level == 1:
                                baseline_files = dataset_info.get('baseline', {})
                                for file_method, file_infos in baseline_files.items():
                                    if group_term in file_method:
                                        for file_info in file_infos:
                                            matching_files.append(file_info['path'])
                            else:
                                # For FDR_20, FDR_50, search in training files
                                training_files = dataset_info.get('training', {}).get(str(fdr_level), {})
                                for file_method, file_infos in training_files.items():
                                    if group_term in file_method:
                                        for file_info in file_infos:
                                            matching_files.append(file_info['path'])
                            
                            return matching_files
                    elif strategy == "use_all_ground_truth":
                        # Check if this group was defined in string search setup
                        group_definitions = ground_truth_mapping.get("_group_definitions", {})
                        if method_name in group_definitions:
                            # This group was created in setup - need to find actual files that match
                            dataset_info = files_info_by_dataset.get(dataset_name, {})
                            
                            # Find all files that contain the group term
                            matching_files = []
                            
                            # For FDR_1, search in baseline files
                            if fdr_level == 1:
                                baseline_files = dataset_info.get('baseline', {})
                                for file_method, file_infos in baseline_files.items():
                                    if group_term in file_method:
                                        for file_info in file_infos:
                                            matching_files.append(file_info['path'])
                            else:
                                # For FDR_20, FDR_50, search in training files
                                training_files = dataset_info.get('training', {}).get(str(fdr_level), {})
                                for file_method, file_infos in training_files.items():
                                    if group_term in file_method:
                                        for file_info in file_infos:
                                            matching_files.append(file_info['path'])
                            
                            return matching_files
                except Exception as e:
                    print(f"Error reading config for {dataset_name}: {e}")
    
    # Fallback: treat as individual file method
    for dataset_name, dataset_info in files_info_by_dataset.items():
        # For FDR_1, look in baseline files; for others, look in training files
        if fdr_level == 1:
            target_files = dataset_info.get('baseline', {})
        else:
            target_files = dataset_info.get('training', {}).get(str(fdr_level), {})
        
        # Direct match first
        if method_name in target_files:
            return [file_info['path'] for file_info in target_files[method_name]]
        
        # Check if method_name matches any file methods
        for file_method, file_infos in target_files.items():
            if method_name == file_method:
                return [file_info['path'] for file_info in file_infos]
    
    print(f"Warning: No files found for method {method_name} at FDR {fdr_level}")
    return []


def validate_ground_truth_files(method_name: str) -> bool:
    """
    Validate that a method has non-empty ground truth files.
    
    Args:
        method_name: The method name to check
        
    Returns:
        bool: True if method has valid ground truth files, False if empty
    """
    import pandas as pd
    import glob
    
    try:
        # For Foie, Gras, etc. - extract sample ID and check corresponding ground truth file
        if any(dataset in method_name for dataset in ['Foie', 'Gras', 'Colon', 'Ileon', 'Artere', 'Coeur']):
            # Extract 3-digit sample ID from method name
            parts = method_name.split('_')
            sample_id = None
            dataset_name = None
            
            # Get dataset name from method
            for dataset in ['Foie', 'Gras', 'Colon', 'Ileon', 'Artere', 'Coeur']:
                if dataset in method_name:
                    dataset_name = dataset
                    break
            
            if not dataset_name:
                return True  # Default to valid if we can't determine dataset
                
            # Extract sample ID
            for part in parts:
                if part.isdigit() and len(part) == 3:
                    sample_id = part
                    break
            
            if not sample_id:
                return True  # Default to valid if no sample ID found
            
            # Check ground truth file
            data_dir = os.path.join(os.path.dirname(__file__), "data")
            pattern = f'{data_dir}/{dataset_name}/long_gradient/FDR_1/*{sample_id}*FDR1.parquet'
            ground_truth_files = glob.glob(pattern)
            
            for file_path in ground_truth_files:
                try:
                    df = pd.read_parquet(file_path)
                    if len(df) == 0:
                        print(f"Warning: Empty ground truth file for method {method_name} (sample {sample_id})")
                        return False
                except Exception as e:
                    print(f"Warning: Error reading ground truth file for method {method_name}: {e}")
                    return False
                    
        # For other datasets (HEK, ASTRAL), assume valid
        return True
        
    except Exception as e:
        print(f"Warning: Error validating ground truth for method {method_name}: {e}")
        return True  # Default to valid on error
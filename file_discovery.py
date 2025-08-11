#!/usr/bin/env python3
"""
FILE DISCOVERY MODULE
================================================================================
Shared file discovery functions for both Streamlit frontend and API backend.
This module contains the generic file discovery logic that can be imported
by both the Streamlit app and the API without causing circular imports.
================================================================================
"""

import os
import glob
from pathlib import Path
import re


def discover_available_files():
    """Discover available data files for analysis using generic folder structure.
    
    Expected structure:
    data/
    ├── [DATASET_NAME]/
    │   ├── short_gradient/
    │   │   ├── FDR_1/          # BASELINE files (1% FDR short gradient)
    │   │   ├── FDR_20/         # TRAINING files (20% FDR)
    │   │   └── FDR_50/         # TRAINING files (50% FDR)
    │   └── long_gradient/
    │       └── FDR_1/          # GROUND TRUTH files
    """
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    
    files_info = {
        'baseline': [],     # Short gradient FDR_1 (baseline reference)
        'ground_truth': [], # Long gradient FDR_1 (ground truth)
        'training': [],     # Short gradient FDR_20, FDR_50 (training candidates)
        'testing': []       # Short gradient FDR_20, FDR_50 (testing candidates)
    }
    
    if not os.path.exists(data_dir):
        return files_info
    
    # Scan all datasets
    for dataset_name in os.listdir(data_dir):
        dataset_path = os.path.join(data_dir, dataset_name)
        if not os.path.isdir(dataset_path):
            continue
            
        # Skip README files
        if dataset_name.startswith('README'):
            continue
        
        # Process short gradient files (baseline + training)
        short_gradient_path = os.path.join(dataset_path, "short_gradient")
        if os.path.exists(short_gradient_path):
            # Baseline files (FDR_1)
            baseline_path = os.path.join(short_gradient_path, "FDR_1")
            if os.path.exists(baseline_path):
                for file_path in glob.glob(os.path.join(baseline_path, "*.parquet")):
                    file_info = extract_file_info(file_path, dataset_name, "short_gradient", 1)
                    files_info['baseline'].append(file_info)
            
            # Training files (FDR_20, FDR_50)
            for fdr_level in [20, 50]:
                training_path = os.path.join(short_gradient_path, f"FDR_{fdr_level}")
                if os.path.exists(training_path):
                    for file_path in glob.glob(os.path.join(training_path, "*.parquet")):
                        file_info = extract_file_info(file_path, dataset_name, "short_gradient", fdr_level)
                        files_info['training'].append(file_info)
                        files_info['testing'].append(file_info)  # Same files can be used for testing
        
        # Process long gradient files (ground truth)
        long_gradient_path = os.path.join(dataset_path, "long_gradient")
        if os.path.exists(long_gradient_path):
            ground_truth_path = os.path.join(long_gradient_path, "FDR_1")
            if os.path.exists(ground_truth_path):
                for file_path in glob.glob(os.path.join(ground_truth_path, "*.parquet")):
                    file_info = extract_file_info(file_path, dataset_name, "long_gradient", 1)
                    files_info['ground_truth'].append(file_info)
    
    return files_info


def extract_file_info(file_path, dataset_name, gradient_type, fdr_level):
    """Extract file information from a file path."""
    file_path = Path(file_path)
    
    # Get basic file info
    file_info = {
        'path': str(file_path),
        'filename': file_path.name,
        'dataset': dataset_name,
        'gradient': gradient_type,
        'fdr': fdr_level,
        'size_mb': file_path.stat().st_size / 1024 / 1024,
        'method': None,
        'triplicate_group': None
    }
    
    # Extract method and triplicate group info
    method, group = extract_method_and_group_from_filename(file_path.name)
    file_info['method'] = method
    file_info['triplicate_group'] = group
    
    return file_info


def extract_method_and_group_from_filename(filename):
    """Extract method name and triplicate group from filename.
    
    Args:
        filename: The filename to analyze
        
    Returns:
        tuple: (method_name, triplicate_group)
    """
    # Remove file extension
    base_name = Path(filename).stem
    
    # For ASTRAL files, extract group information
    if 'ASTRAL' in filename.upper() or any(x in filename for x in ['_001_', '_002_', '_003_', '_004_', '_005_', '_006_', '_007_', '_008_', '_009_', '_010_']):
        # Look for patterns like _001_, _002_, etc.
        group_match = re.search(r'_(\d{3})_', filename)
        if group_match:
            group_id = group_match.group(1)
            method = f"ASTRAL_Group_{group_id}"
            return method, group_id
    
    # For HEK files, extract method from filename
    if 'HEK' in filename.upper():
        # Extract method patterns like MS15-DIA11-25, MS30-DIA7-5, etc.
        method_match = re.search(r'MS(\d+)-DIA([0-9\-\.]+)', filename)
        if method_match:
            ms_time = method_match.group(1)
            dia_params = method_match.group(2)
            method = f"MS{ms_time}-DIA{dia_params}"
            return method, None
    
    # Default fallback
    method = base_name.split('_')[0] if '_' in base_name else base_name
    return method, None


def group_files_by_triplicates(file_list):
    """Group files by their triplicate groups.
    
    Args:
        file_list: List of file info dictionaries
        
    Returns:
        tuple: (grouped_dict, ungrouped_list)
    """
    grouped = {}
    ungrouped = []
    
    for file_info in file_list:
        triplicate_group = file_info.get('triplicate_group')
        if triplicate_group:
            if triplicate_group not in grouped:
                grouped[triplicate_group] = []
            grouped[triplicate_group].append(file_info)
        else:
            ungrouped.append(file_info)
    
    return grouped, ungrouped


def get_available_methods(files_info, use_triplicates=False):
    """Get list of available methods from discovered files.
    
    Args:
        files_info: Discovered files information
        use_triplicates: If True, group by triplicate groups instead of individual methods
    """
    if use_triplicates:
        # Group by triplicate groups
        methods = set()
        for category in ['training', 'testing']:
            grouped, ungrouped = group_files_by_triplicates(files_info[category])
            # Add triplicate groups
            for group_id in grouped.keys():
                methods.add(f"Group_{group_id}_Triplicates")
            # Add individual files without groups
            for file_info in ungrouped:
                methods.add(file_info['method'])
        return sorted(list(methods))
    else:
        # Original individual file methods
        methods = set()
        for category in ['training', 'testing']:
            for file_info in files_info[category]:
                methods.add(file_info['method'])
        return sorted(list(methods))


def get_files_for_method(files_info, method_name, fdr_level, use_triplicates=False):
    """Get file paths for a specific method and FDR level.
    
    Args:
        files_info: Discovered files information
        method_name: Method name or triplicate group name
        fdr_level: FDR level (1, 20, 50)
        use_triplicates: If True, return all files in triplicate group
        
    Returns:
        list: File paths for the method
    """
    if use_triplicates and "Group_" in method_name and "_Triplicates" in method_name:
        # Extract group ID from method name like "Group_001_Triplicates"
        group_id = method_name.replace("Group_", "").replace("_Triplicates", "")
        
        # Find files in this triplicate group
        files = []
        for category in ['training', 'testing', 'baseline']:
            for file_info in files_info[category]:
                if (file_info.get('fdr') == fdr_level and 
                    file_info.get('triplicate_group') == group_id):
                    files.append(file_info['path'])
        return files
    else:
        # Single file method
        files = []
        for category in ['training', 'testing', 'baseline']:
            for file_info in files_info[category]:
                if (file_info['method'] == method_name and 
                    file_info.get('fdr') == fdr_level):
                    files.append(file_info['path'])
        return files
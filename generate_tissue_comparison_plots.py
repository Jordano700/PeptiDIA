#!/usr/bin/env python3
"""
Generate comparison plots showing unique peptides across FDR levels for tissue datasets.
Similar to the Astro data comparison showing baseline, validated additional, and unvalidated additional peptides.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob
import argparse
from typing import Dict, List, Tuple


def load_parquet_file(pattern: str, file_index: int = 0) -> pd.DataFrame:
    """
    Load a single parquet file matching the pattern.

    Args:
        pattern: Glob pattern to match files
        file_index: Index of file to load (default: 0, first file sorted alphabetically)

    Returns:
        DataFrame from the selected file
    """
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")

    if file_index >= len(files):
        raise IndexError(f"File index {file_index} out of range. Found {len(files)} files.")

    selected_file = files[file_index]
    return pd.read_parquet(selected_file), selected_file


def calculate_peptide_sets(dataset: str,
                          gradient_short: str = "short_gradient",
                          gradient_long: str = "long_gradient",
                          fdr_levels: List[int] = [20, 50],
                          file_index: int = 0) -> Dict:
    """
    Calculate peptide sets for different FDR levels using a single matched file.

    Args:
        dataset: Dataset name (e.g., 'Artere', 'Coeur', etc.)
        gradient_short: Short gradient directory name
        gradient_long: Long gradient directory name (ground truth)
        fdr_levels: FDR levels to compare (e.g., [20, 50])
        file_index: Index of file to use (default: 0, first file sorted alphabetically)

    Returns:
        Dictionary containing peptide counts for plotting
    """
    base_path = f"data/{dataset}"

    # Load baseline (short gradient, FDR 1%)
    baseline_pattern = f"{base_path}/{gradient_short}/FDR_1/*FDR1.parquet"
    baseline_df, baseline_file = load_parquet_file(baseline_pattern, file_index)
    baseline_peptides = set(baseline_df['Modified.Sequence'].unique())

    # Extract sample identifier from baseline filename for matching
    baseline_name = Path(baseline_file).name
    sample_id = baseline_name.replace('_FDR1.parquet', '')

    # Load ground truth (long gradient, FDR 1%)
    gt_pattern = f"{base_path}/{gradient_long}/FDR_1/*FDR1.parquet"
    gt_df, gt_file = load_parquet_file(gt_pattern, file_index)
    gt_peptides = set(gt_df['Modified.Sequence'].unique())

    results = {
        'dataset': dataset,
        'sample_id': sample_id,
        'baseline_file': Path(baseline_file).name,
        'baseline_count': len(baseline_peptides),
        'ground_truth_count': len(gt_peptides),
        'fdr_levels': []
    }

    # Process each FDR level
    for fdr in fdr_levels:
        fdr_pattern = f"{base_path}/{gradient_short}/FDR_{fdr}/*FDR{fdr}.parquet"
        fdr_df, fdr_file = load_parquet_file(fdr_pattern, file_index)

        # Get unique peptides across all replicates for this FDR level
        fdr_peptides = set(fdr_df['Modified.Sequence'].unique())

        # Calculate sets
        additional_candidates = fdr_peptides - baseline_peptides
        validated_additional = additional_candidates & gt_peptides
        unvalidated_additional = additional_candidates - gt_peptides

        # Calculate percentages
        total_unique = len(fdr_peptides)
        baseline_pct = len(baseline_peptides) / total_unique * 100 if total_unique > 0 else 0
        validated_pct = len(validated_additional) / total_unique * 100 if total_unique > 0 else 0
        unvalidated_pct = len(unvalidated_additional) / total_unique * 100 if total_unique > 0 else 0

        results['fdr_levels'].append({
            'fdr': fdr,
            'total_unique': total_unique,
            'baseline': len(baseline_peptides),
            'validated_additional': len(validated_additional),
            'unvalidated_additional': len(unvalidated_additional),
            'baseline_pct': baseline_pct,
            'validated_pct': validated_pct,
            'unvalidated_pct': unvalidated_pct
        })

    return results


def create_comparison_plot(results: Dict, output_path: str = None):
    """
    Create a stacked bar chart comparing FDR levels.

    Args:
        results: Results dictionary from calculate_peptide_sets
        output_path: Path to save the plot (if None, displays the plot)
    """
    # Set up the plot style
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Set background color
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Prepare data
    fdr_levels = [f"FDR {item['fdr']}%" for item in results['fdr_levels']]
    baseline_counts = [item['baseline'] for item in results['fdr_levels']]
    validated_counts = [item['validated_additional'] for item in results['fdr_levels']]
    unvalidated_counts = [item['unvalidated_additional'] for item in results['fdr_levels']]

    # Get percentages for labels
    baseline_pcts = [item['baseline_pct'] for item in results['fdr_levels']]
    validated_pcts = [item['validated_pct'] for item in results['fdr_levels']]
    unvalidated_pcts = [item['unvalidated_pct'] for item in results['fdr_levels']]
    total_counts = [item['total_unique'] for item in results['fdr_levels']]

    # Create positions for bars
    x = np.arange(len(fdr_levels))
    width = 0.6

    # Create stacked bars
    bars1 = ax.bar(x, baseline_counts, width, label='Baseline (FDR 1%)',
                   color='#5DA5DA', edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x, validated_counts, width, bottom=baseline_counts,
                   label='Validated Additional', color='#60BD68',
                   edgecolor='white', linewidth=1.5)
    bars3 = ax.bar(x, unvalidated_counts, width,
                   bottom=np.array(baseline_counts) + np.array(validated_counts),
                   label='Unvalidated Additional', color='#F15854',
                   edgecolor='white', linewidth=1.5)

    # Add percentage and count labels on each segment
    for i, (b1, b2, b3) in enumerate(zip(bars1, bars2, bars3)):
        # Baseline label
        height1 = b1.get_height()
        if height1 > 0:
            ax.text(b1.get_x() + b1.get_width()/2., height1/2,
                   f'{baseline_pcts[i]:.1f}%\n({baseline_counts[i]:,})',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   color='white')

        # Validated additional label
        height2 = b2.get_height()
        if height2 > 0:
            ax.text(b2.get_x() + b2.get_width()/2., baseline_counts[i] + height2/2,
                   f'{validated_pcts[i]:.1f}%\n({validated_counts[i]:,})',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   color='white')

        # Unvalidated additional label
        height3 = b3.get_height()
        if height3 > 0:
            ax.text(b3.get_x() + b3.get_width()/2.,
                   baseline_counts[i] + validated_counts[i] + height3/2,
                   f'{unvalidated_pcts[i]:.1f}%\n({unvalidated_counts[i]:,})',
                   ha='center', va='center', fontsize=11, fontweight='bold',
                   color='white')

        # Total count label above bar
        total_height = baseline_counts[i] + validated_counts[i] + unvalidated_counts[i]
        ax.text(b1.get_x() + b1.get_width()/2., total_height + max(total_counts) * 0.02,
               f'Total: {total_counts[i]:,}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Customize plot
    ax.set_xlabel('DIA-NN FDR Level', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Unique Peptides', fontsize=14, fontweight='bold')

    # Create descriptive title
    title = f"{results['dataset']} Data: Additional Peptides at Higher DIA-NN FDRs\nin Paired Fast and Long Gradient Runs"
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    ax.set_xticks(x)
    ax.set_xticklabels(fdr_levels, fontsize=12)
    ax.legend(fontsize=12, loc='upper left', frameon=True, shadow=False, fancybox=False)

    # Format y-axis
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x):,}'))
    ax.tick_params(axis='y', labelsize=11)

    # Set y-axis to start at 0 and add some top margin
    ax.set_ylim(bottom=0, top=max(total_counts) * 1.1)

    # Remove grid
    ax.grid(False)

    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Adjust layout
    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.3)
        print(f"✅ Plot saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Main function to generate tissue comparison plots."""
    parser = argparse.ArgumentParser(
        description='Generate comparison plots for tissue datasets'
    )
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=['Artere', 'Coeur', 'Colon', 'Foie', 'Gras', 'Ileon'],
        help='Dataset names to process (default: all non-ASTRAL tissues)'
    )
    parser.add_argument(
        '--fdr-levels',
        nargs='+',
        type=int,
        default=[20, 50],
        help='FDR levels to compare (default: 20 50)'
    )
    parser.add_argument(
        '--output-dir',
        default='plots/tissue_comparisons',
        help='Output directory for plots (default: plots/tissue_comparisons)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display plots instead of saving'
    )
    parser.add_argument(
        '--file-index',
        type=int,
        default=0,
        help='Index of file to use for each dataset (default: 0, first file alphabetically)'
    )

    args = parser.parse_args()

    # Create output directory if saving
    if not args.show:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 Output directory: {output_dir}")

    # Process each dataset
    print(f"\n{'='*70}")
    print(f"🧬 Generating Tissue Comparison Plots")
    print(f"{'='*70}\n")

    for dataset in args.datasets:
        print(f"\n📊 Processing {dataset}...")

        try:
            # Calculate peptide sets
            results = calculate_peptide_sets(
                dataset=dataset,
                fdr_levels=args.fdr_levels,
                file_index=args.file_index
            )

            # Print summary
            print(f"   Using file: {results['baseline_file']}")
            print(f"   Baseline (7min, FDR 1%): {results['baseline_count']:,} peptides")
            print(f"   Ground Truth (28min, FDR 1%): {results['ground_truth_count']:,} peptides")

            for fdr_data in results['fdr_levels']:
                print(f"   FDR {fdr_data['fdr']}%: {fdr_data['total_unique']:,} unique peptides "
                      f"({fdr_data['validated_additional']:,} validated, "
                      f"{fdr_data['unvalidated_additional']:,} unvalidated)")

            # Create plot
            if args.show:
                output_path = None
            else:
                output_path = output_dir / f"{dataset}_fdr_comparison.png"

            create_comparison_plot(results, str(output_path) if output_path else None)

        except FileNotFoundError as e:
            print(f"   ⚠️  Skipping {dataset}: {e}")
        except Exception as e:
            print(f"   ❌ Error processing {dataset}: {e}")

    print(f"\n{'='*70}")
    print("✅ Done!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

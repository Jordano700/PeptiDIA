#!/usr/bin/env python3
"""
K=5 canonical-methods comparison for Trial037 vs Sept15 baseline (NOASTRAL).

- Discovers the most common 5-method train set per dataset from STREAMLIT_RESULTS_*
  (or uses a provided JSON with explicit canonical sets).
- Filters live runs to those exact method sets and Target FDR=1% rows.
- Compares per-dataset median/max Current_Additional to baseline k=5 Additional.
- Writes outputs into the latest Trial037 NOASTRAL folder by default:
  - k5_canonical_methods.json
  - k5_canonical_runs.tsv (detailed per run)
  - k5_canonical_compare.tsv (per-dataset aggregated, median and max)
"""

import argparse
import glob
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

BASELINE_ROOT_DEFAULT = "results/K_SWEEP_20250915_145805_CROSS0/splits"
TRIAL_GLOB_DEFAULT = "results/TRIAL037_KSWEEP_*_NOASTRAL"
STREAMLIT_GLOB_DEFAULT = "results/STREAMLIT_RESULTS_*"


def _safe_read_json(path: Path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _latest_trial_dir(pattern: str = TRIAL_GLOB_DEFAULT) -> Optional[Path]:
    paths = [Path(p) for p in glob.glob(pattern)]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def discover_k5_combos(streamlit_glob: str,
                       exclude_datasets: Tuple[str, ...] = ("ASTRAL",)) -> Dict[str, Counter]:
    combos: Dict[str, Counter] = defaultdict(Counter)
    for res_dir in glob.glob(streamlit_glob):
        meta_path = os.path.join(res_dir, "raw_data", "analysis_summary.json")
        if not os.path.exists(meta_path):
            continue
        meta = _safe_read_json(Path(meta_path)) or {}
        cfg = meta.get("config", {})
        test_method = str(cfg.get("test_method", ""))
        train_methods = list(cfg.get("train_methods", []) or [])
        if not test_method or len(train_methods) != 5:
            continue
        dataset = test_method.split("_")[0]
        if dataset in exclude_datasets:
            continue
        key = tuple(sorted(map(str, train_methods)))
        combos[dataset][key] += 1
    return combos


def choose_canonical_sets(combos: Dict[str, Counter]) -> Dict[str, List[str]]:
    canonical: Dict[str, List[str]] = {}
    for dataset, counter in combos.items():
        if not counter:
            continue
        key, _ = counter.most_common(1)[0]
        canonical[dataset] = list(key)
    return canonical


def load_baseline_k5(baseline_root: str,
                     include_variants: Tuple[str, ...] = ("xgb",),
                     exclude_datasets: Tuple[str, ...] = ("ASTRAL",)) -> pd.DataFrame:
    root = Path(baseline_root)
    rows: List[dict] = []
    if not root.exists():
        return pd.DataFrame(columns=["dataset", "variant", "Baseline_Additional"]) 
    for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        dataset = dataset_dir.name
        if dataset in exclude_datasets:
            continue
        k_dir = dataset_dir / "k5"
        if not k_dir.exists():
            continue
        # Prefer repeat_01; else any
        rep = k_dir / "repeat_01"
        if not rep.exists():
            reps = [p for p in k_dir.iterdir() if p.is_dir()]
            rep = reps[0] if reps else None
        if not rep or not rep.exists():
            continue
        for variant in include_variants:
            best_path = rep / f"best_{variant}.json"
            if not best_path.exists():
                continue
            best = _safe_read_json(best_path) or {}
            row = best.get("row", {})
            addl = row.get("Additional_Peptides")
            if addl is None:
                continue
            rows.append({
                "dataset": dataset,
                "variant": variant,
                "Baseline_Additional": int(addl),
            })
    return pd.DataFrame(rows)


def collect_k5_runs(streamlit_glob: str,
                    canonical: Dict[str, List[str]],
                    exclude_datasets: Tuple[str, ...] = ("ASTRAL",)) -> pd.DataFrame:
    rows: List[dict] = []
    for res_dir in glob.glob(streamlit_glob):
        csv_path = os.path.join(res_dir, "tables", "detailed_results.csv")
        meta_path = os.path.join(res_dir, "raw_data", "analysis_summary.json")
        if not (os.path.exists(csv_path) and os.path.exists(meta_path)):
            continue
        meta = _safe_read_json(Path(meta_path)) or {}
        cfg = meta.get("config", {})
        test_method = str(cfg.get("test_method", ""))
        train_methods = list(cfg.get("train_methods", []) or [])
        if not test_method or len(train_methods) != 5:
            continue
        dataset = test_method.split("_")[0]
        if dataset in exclude_datasets:
            continue
        canon = canonical.get(dataset)
        if not canon:
            continue
        key = sorted(map(str, train_methods))
        if key != sorted(canon):
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        one = df[df["Target_FDR"].astype(float) == 1.0].copy()
        if one.empty:
            continue
        r = one.iloc[0]
        rows.append({
            "dataset": dataset,
            "variant": "xgb",
            "Additional_Peptides": int(r.get("Additional_Peptides", 0)),
            "Actual_FDR": float(r.get("Actual_FDR", 0.0)),
            "run_id": Path(res_dir).name,
            "train_methods": json.dumps(key),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="K=5 canonical comparison (Trial037 vs Sept15 baseline)")
    ap.add_argument("--trial_dir", type=str, default=None,
                    help="Output Trial037 dir. Defaults to latest results/TRIAL037_KSWEEP_*_NOASTRAL")
    ap.add_argument("--baseline_root", type=str, default=BASELINE_ROOT_DEFAULT,
                    help="Baseline K_SWEEP splits root")
    ap.add_argument("--streamlit_glob", type=str, default=STREAMLIT_GLOB_DEFAULT,
                    help="Glob for live Streamlit result dirs")
    ap.add_argument("--canonical_json", type=str, default=None,
                    help="Optional JSON file mapping dataset -> list of 5 train methods")
    args = ap.parse_args()

    trial_dir = Path(args.trial_dir) if args.trial_dir else _latest_trial_dir()
    if trial_dir is None:
        raise SystemExit("No Trial037 folder found; create one like results/TRIAL037_KSWEEP_*_NOASTRAL")
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Canonical method sets per dataset
    if args.canonical_json and Path(args.canonical_json).exists():
        canonical = _safe_read_json(Path(args.canonical_json)) or {}
    else:
        combos = discover_k5_combos(args.streamlit_glob)
        canonical = choose_canonical_sets(combos)
    (trial_dir / "k5_canonical_methods.json").write_text(json.dumps(canonical, indent=2))

    # Collect runs that match canonical sets
    runs = collect_k5_runs(args.streamlit_glob, canonical)
    runs_out = trial_dir / "k5_canonical_runs.tsv"
    if not runs.empty:
        runs.sort_values(["dataset", "Additional_Peptides"], ascending=[True, False]).to_csv(runs_out, index=False, sep='\t')

    # Baseline at k=5
    baseline = load_baseline_k5(args.baseline_root)

    # Aggregate median and max per dataset
    if runs.empty:
        print("No k=5 runs matched canonical sets; check k5_canonical_methods.json or provide --canonical_json")
        return
    agg = runs.groupby(["dataset", "variant"], as_index=False).agg(
        Current_Median=("Additional_Peptides", "median"),
        Current_Max=("Additional_Peptides", "max"),
        N_Runs=("Additional_Peptides", "count")
    )
    # Merge baseline
    cmp_df = agg.merge(baseline, on=["dataset", "variant"], how="left")
    cmp_df["Delta_Median"] = cmp_df["Current_Median"] - cmp_df["Baseline_Additional"]
    cmp_df["Delta_Max"] = cmp_df["Current_Max"] - cmp_df["Baseline_Additional"]
    # Order and write
    cmp_out = trial_dir / "k5_canonical_compare.tsv"
    cmp_df.sort_values(["dataset", "variant"]).to_csv(cmp_out, index=False, sep='\t')

    print(f"Wrote: {trial_dir / 'k5_canonical_methods.json'}")
    if not runs.empty:
        print(f"Wrote: {runs_out}")
    print(f"Wrote: {cmp_out}")


if __name__ == "__main__":
    main()


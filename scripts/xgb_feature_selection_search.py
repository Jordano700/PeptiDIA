#!/usr/bin/env python3
"""
XGB + Feature Selection Search for PeptiDIA

Searches XGBoost hyperparameters together with different feature subsets to
find the best-performing configuration on a fixed train/test split derived
from a reference run (run_metadata.json).

Feature subsets explored:
  - All features
  - Single blocks: logs, ratios, sequence, stats, other
  - Pairwise combinations of blocks
  - Top-K features by XGB importance (K in {25, 50, 100, 150})

Selection objective (strict):
  - Use the row at the primary target FDR (default 1.0)
  - Require Actual FDR <= primary FDR
  - Maximize Additional_Peptides; tie-break by MCC

Outputs:
  - trials/xgb_fs_trial_XXX.json (includes selected feature list and params)
  - tables/xgb_fs_trial_XXX_results.csv (FDR sweep per trial)
  - best_xgb_fs.json (best overall strict)
  - summary_xgb_fs.csv (all trials, sorted by strict score)
  - best_overall_fdr_summary_fs.csv (FDR sweep of the best trial)

Usage (recommended in tmux for long runs):
  tmux new -s peptidia_xgbfs \
    "python scripts/xgb_feature_selection_search.py \
      --results-ref results/HPARAM_SEARCH_PATHS_20250909_144216_tmux \
      --trials-per-subset 40 \
      --out-dir results/XGB_FS_$(date +%Y%m%d_%H%M%S)"
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# Make project src importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from peptidia.core.peptide_validator_api import PeptideValidatorAPI  # type: ignore
from peptidia.core.dataset_utils import extract_method_from_filename  # type: ignore


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def json_safe(o):
    import numpy as _np
    if isinstance(o, list):
        return [json_safe(x) for x in o]
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (pd.Series, pd.Index)):
        return json_safe(o.tolist())
    if isinstance(o, _np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, (_np.integer,)):
        return int(o)
    if isinstance(o, (_np.floating,)):
        return float(o)
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            return str(o)
    return o


def path_to_dataset_and_method(p: Path) -> Tuple[str, str]:
    parts = list(p.parts)
    if "data" in parts:
        idx = parts.index("data")
        dataset = parts[idx + 1] if idx + 1 < len(parts) else "Unknown"
    else:
        dataset = "Unknown"
    method = extract_method_from_filename(p.name, dataset)
    return dataset, method


def sample_xgb_params(rng: random.Random) -> Dict:
    return {
        "n_estimators": rng.choice([300, 600, 800, 1000, 1200]),
        "max_depth": rng.choice([4, 5, 6, 7, 8]),
        "learning_rate": rng.choice([0.03, 0.05, 0.08, 0.1]),
        "subsample": rng.choice([0.6, 0.7, 0.8, 0.9]),
        "colsample_bytree": rng.choice([0.6, 0.7, 0.8, 0.9]),
        "min_child_weight": rng.choice([1, 3, 5]),
        "gamma": rng.choice([0, 0.5, 1.0]),
        "reg_alpha": rng.choice([0, 0.1, 0.5]),
        "reg_lambda": rng.choice([1.0, 1.5, 2.0]),
        "tree_method": "hist",
        "base_score": 0.5,
        "random_state": rng.randint(1, 10_000),
    }


def pick_primary_row_strict(results_table: List[Dict], primary_fdr: float) -> Tuple[float, Dict]:
    # Score only if Actual_FDR <= primary_fdr
    for row in results_table:
        if abs(float(row.get("Target_FDR", 0)) - primary_fdr) < 1e-6:
            actual = float(row.get("Actual_FDR", 1e9))
            if actual <= primary_fdr:
                score = float(row.get("Additional_Peptides", 0)) + 0.01 * float(row.get("MCC", 0))
                return score, row
            return -math.inf, {}
    return -math.inf, {}


def make_feature_blocks(columns: List[str]) -> Dict[str, List[str]]:
    blocks = {
        "logs": [c for c in columns if c.startswith("log_")],
        "ratios": [c for c in columns if c.startswith("ratio_")],
        "sequence": [c for c in columns if c == "sequence_length" or c.startswith("aa_count_") or c.startswith("aa_freq_")],
        "stats": [c for c in columns if c.startswith("zscore_")],
    }
    other = [c for c in columns if all(c not in v for v in blocks.values())]
    # Exclude any leakage-like columns defensively
    other = [c for c in other if c not in {"source_fdr"}]
    if other:
        blocks["other"] = other
    return blocks


def generate_subsets(blocks: Dict[str, List[str]], include_topk: bool = True) -> List[Tuple[str, List[str]]]:
    named_sets: List[Tuple[str, List[str]]] = []
    # Single blocks
    for name, cols in blocks.items():
        if cols:
            named_sets.append((f"block:{name}", cols))
    # Pairwise block combinations
    keys = list(blocks.keys())
    for a, b in itertools.combinations(keys, 2):
        cols = blocks[a] + blocks[b]
        if cols:
            named_sets.append((f"blocks:{a}+{b}", cols))
    # All blocks
    all_cols = [c for cols in blocks.values() for c in cols]
    if all_cols:
        named_sets.append(("all", all_cols))
    # TopK placeholders (actual TopK picked at runtime using importances)
    if include_topk:
        for k in [25, 50, 100, 150]:
            named_sets.append((f"topk:{k}", []))
    return named_sets


def topk_features_by_importance(xgb_params: Dict, X: pd.DataFrame, y: pd.Series, k: int) -> List[str]:
    model = XGBClassifier(**{**xgb_params, "n_estimators": max(200, int(xgb_params.get("n_estimators", 300) * 0.5))})
    try:
        model.fit(X, y)
        booster = model.get_booster()
        fmap = booster.get_fscore()  # dict feature_name -> importance
    except Exception:
        # Fallback: uniform if importance unavailable
        return list(X.columns)[:k]
    # Sort by importance
    scored = sorted(fmap.items(), key=lambda kv: kv[1], reverse=True)
    names = [n for n, _ in scored if n in X.columns]
    if len(names) < k:
        # Include remaining by arbitrary order
        remain = [c for c in X.columns if c not in names]
        names.extend(remain)
    return names[:k]


def main():
    ap = argparse.ArgumentParser(description="XGB + Feature Selection search")
    ap.add_argument("--results-ref", required=True, help="Path to existing run with run_metadata.json")
    ap.add_argument("--trials-per-subset", type=int, default=20, help="Hparam trials per feature subset")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    ref_dir = Path(args.results_ref)
    meta_path = ref_dir / "run_metadata.json"
    if not meta_path.exists():
        print(f"[ERROR] run_metadata.json not found at {meta_path}")
        raise SystemExit(2)
    meta = json.loads(meta_path.read_text())

    train_files = [Path(p) for p in meta["train_files"]]
    test_file = Path(meta["test_file"])
    target_fdrs = meta.get("target_fdrs", [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])
    primary_fdr = float(meta.get("primary_fdr", 1.0))

    out_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "results" / f"XGB_FS_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    (out_dir / "trials").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "live.log"

    def log(msg: str):
        line = f"[{ts()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    api = PeptideValidatorAPI()
    xgb_device = api._detect_gpu_device()

    # Build train
    train_frames = []
    y_train_list: List[int] = []
    for p in train_files:
        df = pd.read_parquet(p)
        df["source_fdr"] = 50
        _, method = path_to_dataset_and_method(p)
        gt_peps = api._load_ground_truth_peptides(method)
        labels = df["Modified.Sequence"].isin(gt_peps).astype(int)
        y_train_list.extend(labels.tolist())
        train_frames.append(df)
    train_df = pd.concat(train_frames, ignore_index=True)
    y_train = pd.Series(y_train_list)

    # Test
    test_df = pd.read_parquet(test_file)
    test_df["source_fdr"] = 50
    _, method_test = path_to_dataset_and_method(test_file)
    baseline = api._load_baseline_peptides(method_test)
    gt_test = api._load_ground_truth_peptides(method_test)
    add_set = set(test_df["Modified.Sequence"].unique()) - set(baseline)
    test_add_df = test_df[test_df["Modified.Sequence"].isin(add_set)].copy()
    y_test = test_add_df["Modified.Sequence"].isin(gt_test)

    # Features
    X_train_full = api._make_advanced_features(train_df)
    feat_names_full = X_train_full.columns.tolist()
    X_test_full = api._make_advanced_features(test_add_df, feat_names_full)

    # Make feature blocks and candidate subsets
    blocks = make_feature_blocks(feat_names_full)
    subsets = generate_subsets(blocks, include_topk=True)
    log(f"Feature blocks: { {k: len(v) for k,v in blocks.items()} }")
    log(f"Total subsets to explore: {len(subsets)}")

    best = {"score": -math.inf}
    rows_out = []
    trial_idx = 1

    for subset_name, cols in subsets:
        # Determine concrete cols (for topk, compute using a quick importance fit on full feature set)
        if subset_name.startswith("topk:"):
            k = int(subset_name.split(":")[1])
            # quick importance model to get top-k
            seed_params = sample_xgb_params(rng)
            seed_params["device"] = xgb_device
            topk_cols = topk_features_by_importance(seed_params, X_train_full, y_train, k)
            cols = topk_cols
        if not cols:
            continue

        X_train = X_train_full[cols]
        X_test = X_test_full[cols]

        for _ in range(args.trials_per_subset):
            xgb_params = sample_xgb_params(rng)
            xgb_params["device"] = xgb_device
            model = XGBClassifier(**xgb_params)
            model.fit(X_train, y_train)

            # Evaluate via API pipeline (peptide aggregation + FDR sweep)
            results_table, _ = api._validate_and_optimize(
                model, test_add_df, y_test, cols, target_fdrs, len(baseline), aggregation_method="max"
            )
            score, row = pick_primary_row_strict(results_table, primary_fdr)

            payload = {
                "variant": "xgb_fs",
                "trial": trial_idx,
                "subset": subset_name,
                "features": cols,
                "n_features": len(cols),
                "xgb_params": xgb_params,
                "score": score,
                "primary_fdr": primary_fdr,
                "selected_row": row,
            }
            (out_dir / "trials" / f"xgb_fs_trial_{trial_idx:04d}.json").write_text(json.dumps(json_safe(payload), indent=2))
            try:
                pd.DataFrame(results_table).to_csv(out_dir / "tables" / f"xgb_fs_trial_{trial_idx:04d}_results.csv", index=False)
            except Exception:
                pass

            addp = int(row.get("Additional_Peptides", 0) or 0) if row else 0
            actual = float(row.get("Actual_FDR", 1e9) or 1e9) if row else 1e9
            mcc = float(row.get("MCC", 0) or 0) if row else 0
            rows_out.append({
                "trial": trial_idx,
                "subset": subset_name,
                "n_features": len(cols),
                "add_peptides": addp,
                "actual_fdr": actual,
                "mcc": mcc,
                "score": score,
            })

            if score > best["score"]:
                best = payload
                (out_dir / "best_xgb_fs.json").write_text(json.dumps(json_safe(best), indent=2))
                # Save the best sweep
                try:
                    pd.DataFrame(results_table)[["Target_FDR","Additional_Peptides","Recovery_Pct","Actual_FDR","Threshold"]].to_csv(
                        out_dir / "best_overall_fdr_summary_fs.csv", index=False
                    )
                except Exception:
                    pass

            if trial_idx % 10 == 0:
                log(f"Progress: trial {trial_idx} | subset={subset_name} | best_add={int(best.get('selected_row',{}).get('Additional_Peptides',0) or 0)} @ <= {primary_fdr}%")
            trial_idx += 1

    if rows_out:
        pd.DataFrame(rows_out).sort_values(["add_peptides","mcc"], ascending=False).to_csv(out_dir / "summary_xgb_fs.csv", index=False)

    print(f"✅ XGB feature-selection search complete: {out_dir}")


if __name__ == "__main__":
    main()


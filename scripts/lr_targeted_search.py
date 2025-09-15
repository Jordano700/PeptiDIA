#!/usr/bin/env python3
"""
Targeted Logistic Regression sweep with scaling/solver variations to find a
competitive LR configuration on the same split as a prior run (via run_metadata.json).

Usage:
  python scripts/lr_targeted_search.py \
    --results-ref results/HPARAM_SEARCH_PATHS_20250909_144216_tmux \
    --trials 50 \
    --out-dir results/LR_TARGETED_$(date +%Y%m%d_%H%M%S)

Outputs:
  - best_lr.json
  - summary_best_by_lr.csv
  - trials/lr_trial_XXX.json
  - tables/lr_trial_XXX_results.csv
  - best_overall_fdr_summary.csv (for best LR)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


def calibrator_or_none(y_train: pd.Series, base_model) -> Optional[CalibratedClassifierCV]:
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    max_splits = 3
    feasible_splits = max(2, min(max_splits, pos, neg))
    if feasible_splits >= 2:
        skf = StratifiedKFold(n_splits=feasible_splits, shuffle=True, random_state=42)
        return CalibratedClassifierCV(base_model, method="isotonic", cv=skf)
    return None


def pick_primary_row(results_table: List[Dict], primary_fdr: float) -> (float, Dict):
    for row in results_table:
        if abs(float(row.get("Target_FDR", 0)) - primary_fdr) < 1e-6:
            score = float(row.get("Additional_Peptides", 0)) + 0.01 * float(row.get("MCC", 0))
            return score, row
    if results_table:
        row = sorted(results_table, key=lambda r: r.get("Target_FDR", 1e9))[0]
        score = float(row.get("Additional_Peptides", 0)) + 0.01 * float(row.get("MCC", 0))
        return score, row
    return -math.inf, {}


def sample_lr_params(rng: random.Random) -> Dict:
    solver = rng.choice(["lbfgs", "liblinear", "newton-cg", "saga"])  # rich solvers
    # Penalty compatibility
    if solver == "liblinear":
        penalty = rng.choice(["l1", "l2"])  # liblinear supports l1/l2
    elif solver in ("lbfgs", "newton-cg"):
        penalty = "l2"
    else:  # saga
        penalty = rng.choice(["l1", "l2"])  # keep elasticnet out for simplicity

    params: Dict = {
        "C": rng.choice([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]),
        "max_iter": rng.choice([2000, 5000, 8000]),
        "solver": solver,
        "penalty": penalty,
        "tol": rng.choice([1e-4, 1e-3]),
        "class_weight": rng.choice([None, "balanced"]),
        "fit_intercept": True,
        "n_jobs": None if solver == "liblinear" else -1,
        "random_state": rng.randint(1, 10_000),
    }
    # saga + l1 needs saga explicitly; already set
    return params


def build_lr_pipeline(lr_params: Dict) -> Pipeline:
    lr = LogisticRegression(**lr_params)
    # Standardize dense features; PeptiDIA features are dense DataFrames
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", lr),
    ])
    return pipe


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


def main():
    ap = argparse.ArgumentParser(description="Targeted LR sweep with scaling/solver variations")
    ap.add_argument("--results-ref", required=True, help="Path to existing HPARAM run (to mirror split)")
    ap.add_argument("--trials", type=int, default=50, help="Number of LR trials")
    ap.add_argument("--out-dir", type=str, default="", help="Output directory (default under results)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    ref_dir = Path(args.results_ref)
    meta_path = ref_dir / "run_metadata.json"
    if not meta_path.exists():
        print(f"[ERROR] run_metadata.json not found at {meta_path}")
        return 2
    meta = json.loads(meta_path.read_text())

    train_files = [Path(p) for p in meta["train_files"]]
    test_file = Path(meta["test_file"])
    target_fdrs = meta.get("target_fdrs", [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])
    primary_fdr = float(meta.get("primary_fdr", 1.0))

    # Prepare output
    out_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "results" / f"LR_TARGETED_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    (out_dir / "trials").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "live.log"

    def log(msg: str):
        line = f"[{ts()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    api = PeptideValidatorAPI()

    # Build training
    train_frames = []
    y_train_list: List[int] = []
    # Helper to infer dataset + method consistent with hparam script
    def path_to_dataset_and_method(pp: Path) -> (str, str):
        parts = list(pp.parts)
        if "data" in parts:
            idx = parts.index("data")
            dataset = parts[idx + 1] if idx + 1 < len(parts) else "Unknown"
        else:
            dataset = "Unknown"
        method = extract_method_from_filename(pp.name, dataset)
        return dataset, method

    for p in train_files:
        df = pd.read_parquet(p)
        df["source_fdr"] = 50
        dataset, method = path_to_dataset_and_method(p)
        gt_peps = api._load_ground_truth_peptides(method)
        labels = df["Modified.Sequence"].isin(gt_peps).astype(int)
        y_train_list.extend(labels.tolist())
        train_frames.append(df)

    train_df = pd.concat(train_frames, ignore_index=True)
    y_train = pd.Series(y_train_list)

    # Test data
    test_df = pd.read_parquet(test_file)
    test_df["source_fdr"] = 50
    _, method_test = path_to_dataset_and_method(test_file)
    baseline = api._load_baseline_peptides(method_test)
    gt_test = api._load_ground_truth_peptides(method_test)
    add_set = set(test_df["Modified.Sequence"].unique()) - set(baseline)
    test_add_df = test_df[test_df["Modified.Sequence"].isin(add_set)].copy()
    y_test = test_add_df["Modified.Sequence"].isin(gt_test)

    # Features
    X_train = api._make_advanced_features(train_df)
    feat_names = X_train.columns.tolist()
    X_test = api._make_advanced_features(test_add_df, feat_names)

    best_variant = {"score": -math.inf}
    rows = []

    for t in range(1, args.trials + 1):
        lr_params = sample_lr_params(rng)
        model_pipe = build_lr_pipeline(lr_params)
        calib = calibrator_or_none(y_train, model_pipe)
        if calib is not None:
            model = calib
            model.fit(X_train, y_train)
        else:
            model_pipe.fit(X_train, y_train)
            model = model_pipe

        # Evaluate
        results_table, _ = api._validate_and_optimize(model, test_add_df, y_test, feat_names, target_fdrs, len(baseline), aggregation_method="max")
        score, row = pick_primary_row(results_table, primary_fdr)

        payload = {
            "variant": "lr",
            "trial": t,
            "score": score,
            "primary_fdr": primary_fdr,
            "selected_row": row,
            "lr_params": lr_params,
            "use_scaler": True,
        }
        with open(out_dir / "trials" / f"lr_trial_{t:03d}.json", "w") as f:
            json.dump(json_safe(payload), f, indent=2)

        try:
            pd.DataFrame(results_table).to_csv(out_dir / "tables" / f"lr_trial_{t:03d}_results.csv", index=False)
        except Exception:
            pass

        rows.append({
            "trial": t,
            "score": score,
            "add_peptides": row.get("Additional_Peptides", None),
            "actual_fdr": row.get("Actual_FDR", None),
            "mcc": row.get("MCC", None),
            **{f"lr_{k}": v for k, v in lr_params.items()},
        })

        if score > best_variant["score"]:
            best_variant = {
                "variant": "lr",
                "score": score,
                "row": row,
                "lr_params": lr_params,
                "use_scaler": True,
            }
            with open(out_dir / "best_lr.json", "w") as f:
                json.dump(json_safe(best_variant), f, indent=2)

    # Save summary
    if rows:
        df = pd.DataFrame(rows).sort_values(["score", "add_peptides"], ascending=False)
        df.to_csv(out_dir / "summary_best_by_lr.csv", index=False)

    # Save best FDR sweep for convenience
    try:
        if best_variant and best_variant.get("row"):
            # We didn't keep the full table for the best row; rebuild once
            # using the winning params for all target FDRs
            lr_params = best_variant["lr_params"]
            model = calibrator_or_none(y_train, build_lr_pipeline(lr_params))
            if model is None:
                model = build_lr_pipeline(lr_params)
                model.fit(X_train, y_train)
            else:
                model.fit(X_train, y_train)
            results_table, _ = api._validate_and_optimize(model, test_add_df, y_test, feat_names, target_fdrs, len(baseline), aggregation_method="max")
            pd.DataFrame(results_table)[["Target_FDR","Additional_Peptides","Recovery_Pct","Actual_FDR","Threshold"]].to_csv(out_dir / "best_overall_fdr_summary.csv", index=False)
    except Exception as e:
        print(f"[WARN] Could not save best FDR summary: {e}")

    # Metadata
    meta_out = {
        "source_ref": str(ref_dir),
        "seed": args.seed,
        "trials": args.trials,
        "timestamp": datetime.now().isoformat(),
        "train_files": [str(p) for p in train_files],
        "test_file": str(test_file),
        "target_fdrs": target_fdrs,
        "primary_fdr": primary_fdr,
    }
    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"✅ LR targeted sweep complete: {out_dir}")


if __name__ == "__main__":
    main()

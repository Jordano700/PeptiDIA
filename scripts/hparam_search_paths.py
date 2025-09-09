#!/usr/bin/env python3
"""
Hyperparameter search with explicit parquet paths (FDR 50 train/test).

- Train on one or more FDR_50 parquet files (same dataset).
- Test on a held-out FDR_50 parquet file (same dataset).
- Reproduces Streamlit/CLI logic: baseline = test-method FDR_1; labels via dataset ground-truth mapping.
- Compares variants: ensemble (XGB+RF+LR), xgb, rf, lr.
- Random search over hyperparameters; saves per-trial JSON, best-per-variant JSON, and a summary CSV.

Example (Artere):
  tmux new -s peptidia_hparam
  python scripts/hparam_search_paths.py \
    --train-files \
      data/Artere/short_gradient/FDR_50/20250728_RD201_EXB_EV1107_300SPD_Artere_CTL_C46_01_FDR50.parquet,\
      data/Artere/short_gradient/FDR_50/20250728_RD201_EXB_EV1107_300SPD_Artere_CTL_C55_01_FDR50.parquet,\
      data/Artere/short_gradient/FDR_50/20250728_RD201_EXB_EV1107_300SPD_Artere_CTL_C57_01_FDR50.parquet \
    --test-file \
      data/Artere/short_gradient/FDR_50/20250728_RD201_EXB_EV1107_300SPD_Artere_CTL_C139_01_FDR50.parquet \
    --trials-per-variant 125 --variants ensemble,xgb,lr,rf --primary-fdr 1.0

Tip: 4 variants × 125 trials = 500 configurations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from contextlib import redirect_stdout

# Ensure project src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from peptidia.core.peptide_validator_api import PeptideValidatorAPI  # type: ignore
from peptidia.core.dataset_utils import extract_method_from_filename  # type: ignore

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def parse_percent_like(s: str) -> float:
    """Parse numbers like '1', '1.0', '1,0', '1%', '1.0%'.

    Accepts comma decimal separators and optional trailing percent sign.
    Returns float percentage value (e.g., '1.0' -> 1.0).
    """
    s = (s or "").strip()
    if s.endswith("%"):
        s = s[:-1]
    # Normalize decimal comma to dot
    s = s.replace(",", ".")
    return float(s)


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def path_to_dataset_and_method(p: Path) -> (str, str):
    """Infer dataset name and method name from a data path.

    dataset = the path segment after 'data/'.
    method = f"{dataset}_<filename-without-ext-and-without-_FDR##>"
    """
    parts = list(p.parts)
    if "data" in parts:
        idx = parts.index("data")
        if idx + 1 < len(parts):
            dataset = parts[idx + 1]
        else:
            dataset = "Unknown"
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


def sample_rf_params(rng: random.Random) -> Dict:
    return {
        "n_estimators": rng.choice([300, 500, 800, 1000]),
        "max_depth": rng.choice([6, 8, 10, None]),
        "min_samples_split": rng.choice([2, 5, 10]),
        "min_samples_leaf": rng.choice([1, 2, 4]),
        "n_jobs": -1,
        "random_state": rng.randint(1, 10_000),
    }


def sample_lr_params(rng: random.Random) -> Dict:
    return {
        "C": rng.choice([0.25, 0.5, 1.0, 2.0, 4.0]),
        "max_iter": 1000,
        "random_state": rng.randint(1, 10_000),
    }


def build_model(variant: str, xgb_params: Dict, rf_params: Dict, lr_params: Dict):
    variant = variant.lower()
    if variant == "xgb":
        return XGBClassifier(**xgb_params)
    if variant == "rf":
        return RandomForestClassifier(**rf_params)
    if variant == "lr":
        return LogisticRegression(**lr_params)
    if variant == "ensemble":
        xgb = XGBClassifier(**xgb_params)
        rf = RandomForestClassifier(**rf_params)
        lr = LogisticRegression(**lr_params)
        return VotingClassifier(
            estimators=[("xgb", xgb), ("rf", rf), ("lr", lr)], voting="soft", weights=[3, 2, 1]
        )
    raise ValueError(f"Unknown variant: {variant}")


def calibrator_or_none(base_model, y_train: pd.Series) -> Optional[CalibratedClassifierCV]:
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


def main():
    ap = argparse.ArgumentParser(description="PeptiDIA hparam search with explicit FDR_50 paths")
    ap.add_argument("--train-files", type=str, required=True,
                   help="Comma-separated FDR_50 parquet paths for training")
    ap.add_argument("--test-file", type=str, required=True,
                   help="FDR_50 parquet path for testing")
    ap.add_argument("--variants", type=str, default="ensemble,xgb,lr,rf",
                   help="Comma-separated variants: ensemble,xgb,rf,lr")
    ap.add_argument("--trials-per-variant", type=int, default=125,
                   help="Trials per variant (e.g., 125 × 4 variants = 500)")
    ap.add_argument("--target-fdrs", type=str, default="1,2,3,4,5,6,7,8,9",
                   help="Comma-separated target FDR levels")
    ap.add_argument("--primary-fdr", type=str, default="1.0",
                   help="Primary FDR used for ranking (e.g., 1, 1.0, 1%)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--results-dir", type=str, default="",
                   help="Output dir (default: results/HPARAM_SEARCH_PATHS_<ts>)")

    args = ap.parse_args()

    rng = random.Random(args.seed)

    train_paths = [Path(p.strip()) for p in args.train_files.split(",") if p.strip()]
    test_path = Path(args.test_file)
    variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]
    try:
        target_fdrs = [parse_percent_like(x) for x in args.target_fdrs.split(",") if x.strip()]
        primary_fdr_val = parse_percent_like(args.primary_fdr)
    except ValueError as e:
        print(f"[ERROR] Could not parse FDR values: {e}")
        print("Hint: use formats like '1', '1.0', '1%', '1,0'.")
        sys.exit(2)

    results_dir = Path(args.results_dir) if args.results_dir else (
        PROJECT_ROOT / "results" / f"HPARAM_SEARCH_PATHS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    (results_dir / "trials").mkdir(parents=True, exist_ok=True)
    (results_dir / "tables").mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "live.log"

    def log(msg: str):
        line = f"[{ts()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log("=== PeptiDIA HPARAM SEARCH (paths mode) ===")
    log(f"Results dir: {results_dir}")

    # API
    api = PeptideValidatorAPI()
    xgb_device = api._detect_gpu_device()

    # Build training dataframe (FDR=50 only) with per-method labeling
    train_frames = []
    y_train_list: List[int] = []
    for p in train_paths:
        df = pd.read_parquet(p)
        dataset, method = path_to_dataset_and_method(p)
        df["source_fdr"] = 50
        df["source_method"] = method
        train_frames.append(df)

        # Method-specific ground truth
        gt_peps = api._load_ground_truth_peptides(method)
        labels = df["Modified.Sequence"].isin(gt_peps).astype(int)
        y_train_list.extend(labels.tolist())

    train_df = pd.concat(train_frames, ignore_index=True)
    y_train = pd.Series(y_train_list)

    log(f"Training samples: {len(train_df):,}; positives: {int(y_train.sum()):,} ({y_train.mean()*100:.1f}%)")

    # Test set
    test_df = pd.read_parquet(test_path)
    test_df["source_fdr"] = 50
    dataset_test, method_test = path_to_dataset_and_method(test_path)

    # Baseline peptides from FDR_1 for the test method
    baseline_peptides = api._load_baseline_peptides(method_test)
    if not baseline_peptides:
        log(f"[WARN] No baseline peptides found for {method_test} (FDR 1). Proceeding, counts may differ.")

    # Ground truth for the test method
    gt_test = api._load_ground_truth_peptides(method_test)

    # Filter to additional peptides beyond baseline
    test_peps = set(test_df["Modified.Sequence"].unique())
    additional_peps = test_peps - set(baseline_peptides)
    test_mask = test_df["Modified.Sequence"].isin(additional_peps)
    test_add_df = test_df.loc[test_mask].copy()
    y_test = test_add_df["Modified.Sequence"].isin(gt_test)

    log(f"Test rows: {len(test_df):,}; additional rows: {len(test_add_df):,}; unique additional peptides: {test_add_df['Modified.Sequence'].nunique():,}")

    # Features once
    X_train = api._make_advanced_features(train_df)
    feat_names = X_train.columns.tolist()
    X_test = api._make_advanced_features(test_add_df, feat_names)

    # Search
    best_overall = {"score": -math.inf}
    summary_rows = []
    all_trial_rows = []

    total_trials = len(variants) * args.trials_per_variant
    done_trials = 0
    best_overall_table = None
    for variant in variants:
        log(f"--- Variant: {variant} ---")
        best_variant = {"score": -math.inf}
        for t in range(1, args.trials_per_variant + 1):
            xgb_p = sample_xgb_params(rng)
            xgb_p["device"] = xgb_device
            rf_p = sample_rf_params(rng)
            lr_p = sample_lr_params(rng)

            model_base = build_model(variant, xgb_p, rf_p, lr_p)
            calibrator = calibrator_or_none(model_base, y_train)
            if calibrator is not None:
                model = calibrator
                model.fit(X_train, y_train)
            else:
                model_base.fit(X_train, y_train)
                model = model_base

            # Evaluate using API's peptide-level aggregation + thresholding
            # Redirect verbose API prints to live.log to keep console clean
            with open(log_path, "a") as _lf, redirect_stdout(_lf):
                results_table, _ = api._validate_and_optimize(
                    model, test_add_df, y_test, feat_names, target_fdrs, len(baseline_peptides), aggregation_method="max"
                )

            score, row = pick_primary_row(results_table, primary_fdr_val)

            # JSON-safe conversion helper
            def _json_safe(o):
                import numpy as _np
                if isinstance(o, list):
                    return [_json_safe(x) for x in o]
                if isinstance(o, dict):
                    return {str(k): _json_safe(v) for k, v in o.items()}
                if isinstance(o, (pd.Series, pd.Index)):
                    return _json_safe(o.tolist())
                if isinstance(o, _np.ndarray):
                    return _json_safe(o.tolist())
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

            trial_payload = {
                "variant": variant,
                "trial": t,
                "score": score,
                "primary_fdr": primary_fdr_val,
                "selected_row": row,
                "xgb_params": xgb_p,
                "rf_params": rf_p,
                "lr_params": lr_p,
            }
            with open(results_dir / "trials" / f"{variant}_trial_{t:03d}.json", "w") as f:
                json.dump(_json_safe(trial_payload), f, indent=2)

            # Also save a CLI/Streamlit-like results table CSV for this trial
            try:
                pd.DataFrame(results_table).to_csv(
                    results_dir / "tables" / f"{variant}_trial_{t:03d}_results.csv", index=False
                )
            except Exception as e:
                log(f"[WARN] Could not save trial CSV for {variant} {t}: {e}")

            if score > best_variant["score"]:
                best_variant = {
                    "variant": variant,
                    "score": score,
                    "row": row,
                    "xgb_params": xgb_p,
                    "rf_params": rf_p,
                    "lr_params": lr_p,
                }
                with open(results_dir / f"best_{variant}.json", "w") as f:
                    json.dump(_json_safe(best_variant), f, indent=2)

                # Save best-of-variant results table CSV as well
                try:
                    pd.DataFrame(results_table).to_csv(
                        results_dir / "tables" / f"best_{variant}_results.csv", index=False
                    )
                except Exception as e:
                    log(f"[WARN] Could not save best-of-variant CSV for {variant}: {e}")

            if score > best_overall["score"]:
                best_overall = {"variant": variant, "score": score, "row": row}
                best_overall_table = results_table

            done_trials += 1
            prog = int(100 * done_trials / max(1, total_trials))
            # Minimal progress line to console
            print(
                f"Progress: {done_trials}/{total_trials} ({prog}%) - {variant} t={t} | Best={best_overall['variant']} {best_overall['score']:.2f}",
                flush=True,
            )

            # Accumulate compact per-trial row for an all-trials leaderboard CSV
            all_trial_rows.append({
                "variant": variant,
                "trial": t,
                "score": score,
                "primary_fdr": primary_fdr_val,
                "add_peptides": row.get("Additional_Peptides", None),
                "actual_fdr": row.get("Actual_FDR", None),
                "mcc": row.get("MCC", None),
            })

        # append best-of-variant
        summary_rows.append({
            "variant": best_variant.get("variant"),
            "score": best_variant.get("score"),
            "add_peptides": best_variant.get("row", {}).get("Additional_Peptides", None),
            "actual_fdr": best_variant.get("row", {}).get("Actual_FDR", None),
            "mcc": best_variant.get("row", {}).get("MCC", None),
            "params_xgb": best_variant.get("xgb_params", {}),
            "params_rf": best_variant.get("rf_params", {}),
            "params_lr": best_variant.get("lr_params", {}),
        })

    # Save summary
    if summary_rows:
        df = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
        df.to_csv(results_dir / "summary_best_by_variant.csv", index=False)

    # Save all trials leaderboard
    if all_trial_rows:
        df_all = pd.DataFrame(all_trial_rows).sort_values(["score", "add_peptides"], ascending=False)
        df_all.to_csv(results_dir / "all_trials_leaderboard.csv", index=False)

    # Run metadata
    metadata = {
        "train_files": [str(p) for p in train_paths],
        "test_file": str(test_path),
        "variants": variants,
        "trials_per_variant": args.trials_per_variant,
        "target_fdrs": target_fdrs,
        "primary_fdr": primary_fdr_val,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "results_dir": str(results_dir),
    }
    with open(results_dir / "run_metadata.json", "w") as f:
        json.dump({k: (float(v) if isinstance(v, np.floating) else v) for k, v in metadata.items()}, f, indent=2)

    # Minimal final FDR summary for the best overall configuration
    if best_overall_table:
        df_best = pd.DataFrame(best_overall_table)
        cols = ["Target_FDR", "Additional_Peptides", "Recovery_Pct", "Actual_FDR", "Threshold"]
        df_best_min = df_best[cols]
        df_best_min.to_csv(results_dir / "best_overall_fdr_summary.csv", index=False)
        # Also write a compact text summary for quick glance
        with open(results_dir / "best_overall_fdr_summary.txt", "w") as f:
            f.write("Target_FDR,Additional_Peptides,Recovery_Pct,Actual_FDR\n")
            for _, r in df_best_min.iterrows():
                f.write(f"{float(r['Target_FDR']):.1f}%,{int(r['Additional_Peptides'])},{float(r['Recovery_Pct']):.1f}%,{float(r['Actual_FDR']):.1f}%\n")

        # Print a concise final summary to console
        print("\nBest model FDR summary (Additional peptides and % recovery):")
        for _, r in df_best_min.iterrows():
            print(
                f"  {float(r['Target_FDR']):.1f}%: {int(r['Additional_Peptides'])} peptides, {float(r['Recovery_Pct']):.1f}% recovery"
            )

    log("=== Done ===")


if __name__ == "__main__":
    main()

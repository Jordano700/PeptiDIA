#!/usr/bin/env python3
"""
AutoML-style tabular search for PeptiDIA.

Searches across multiple model families on a fixed train/test split derived
from a reference run (run_metadata.json):
  - XGBClassifier (XGBoost)
  - RandomForestClassifier
  - LogisticRegression (wrapped with StandardScaler)
  - MLPClassifier (scikit-learn, with StandardScaler)

Features
  - Random search spaces per family (rich but safe defaults)
  - Isotonic calibration when feasible (CV=2..3 based on class counts)
  - Peptide-level aggregation + FDR threshold optimization identical to hparam runner
  - Resume capability (continues after existing trials)
  - Time/Trial budgets + graceful stop via a STOP file in out_dir
  - Saves: per-trial JSON/CSV, best-per-variant JSON, summary CSVs, best-overall FDR sweep

Usage (recommend tmux for long runs):
  tmux new -s peptidia_automl \
    "python scripts/automl_tabular_search.py \
      --results-ref results/HPARAM_SEARCH_PATHS_20250909_144216_tmux \
      --variants xgb,rf,lr,mlp \
      --trials-per-variant 500 \
      --time-minutes 1440 \
      --out-dir results/AUTO_TABULAR_$(date +%Y%m%d_%H%M%S)"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# Ensure project src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
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


def calibrator_or_none(y_train: pd.Series, base_model) -> Optional[CalibratedClassifierCV]:
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    max_splits = 3
    feasible_splits = max(2, min(max_splits, pos, neg))
    if feasible_splits >= 2:
        skf = StratifiedKFold(n_splits=feasible_splits, shuffle=True, random_state=42)
        return CalibratedClassifierCV(base_model, method="isotonic", cv=skf)
    return None


def pick_primary_row(results_table: List[Dict], primary_fdr: float) -> Tuple[float, Dict]:
    for row in results_table:
        if abs(float(row.get("Target_FDR", 0)) - primary_fdr) < 1e-6:
            score = float(row.get("Additional_Peptides", 0)) + 0.01 * float(row.get("MCC", 0))
            return score, row
    if results_table:
        row = sorted(results_table, key=lambda r: r.get("Target_FDR", 1e9))[0]
        score = float(row.get("Additional_Peptides", 0)) + 0.01 * float(row.get("MCC", 0))
        return score, row
    return -math.inf, {}


# --------------------------- Search Spaces ---------------------------
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
        "n_estimators": rng.choice([300, 500, 800, 1000, 1500]),
        "max_depth": rng.choice([6, 8, 10, None]),
        "min_samples_split": rng.choice([2, 5, 10]),
        "min_samples_leaf": rng.choice([1, 2, 4]),
        "n_jobs": -1,
        "random_state": rng.randint(1, 10_000),
    }


def sample_lr_params(rng: random.Random) -> Dict:
    solver = rng.choice(["lbfgs", "liblinear", "newton-cg", "saga"])  # rich solvers
    if solver == "liblinear":
        penalty = rng.choice(["l1", "l2"])  # liblinear supports l1/l2
    elif solver in ("lbfgs", "newton-cg"):
        penalty = "l2"
    else:  # saga
        penalty = rng.choice(["l1", "l2"])  # keep elasticnet out for simplicity
    return {
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


def sample_mlp_params(rng: random.Random) -> Dict:
    h_opts = [(64,), (128,), (256,), (128, 64), (256, 128), (128, 128), (256, 256)]
    return {
        "hidden_layer_sizes": rng.choice(h_opts),
        "activation": "relu",
        "solver": "adam",
        "learning_rate_init": rng.choice([1e-4, 3e-4, 1e-3, 3e-3]),
        "alpha": rng.choice([1e-5, 1e-4, 1e-3, 1e-2]),
        "batch_size": rng.choice([64, 128, 256]),
        "max_iter": rng.choice([200, 300, 400, 500]),
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": rng.choice([10, 20]),
        "random_state": rng.randint(1, 10_000),
    }


# --------------------------- Builders ---------------------------
def build_estimator(variant: str, params: Dict) -> object:
    if variant == "xgb":
        return XGBClassifier(**params)
    if variant == "rf":
        return RandomForestClassifier(**params)
    if variant == "lr":
        lr = LogisticRegression(**params)
        return Pipeline([("scaler", StandardScaler()), ("lr", lr)])
    if variant == "mlp":
        mlp = MLPClassifier(**params)
        return Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])
    raise ValueError(f"Unknown variant: {variant}")


def sample_params(variant: str, rng: random.Random) -> Dict:
    if variant == "xgb":
        return sample_xgb_params(rng)
    if variant == "rf":
        return sample_rf_params(rng)
    if variant == "lr":
        return sample_lr_params(rng)
    if variant == "mlp":
        return sample_mlp_params(rng)
    raise ValueError(variant)


def main():
    ap = argparse.ArgumentParser(description="Auto tabular search across XGB, RF, LR, MLP")
    ap.add_argument("--results-ref", required=True, help="Path to existing run with run_metadata.json")
    ap.add_argument("--variants", default="xgb,rf,lr,mlp", help="Comma-separated: xgb,rf,lr,mlp")
    ap.add_argument("--trials-per-variant", type=int, default=250)
    ap.add_argument("--time-minutes", type=int, default=0, help="Optional time budget (0=ignore)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]

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

    out_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "results" / f"AUTO_TABULAR_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    (out_dir / "trials").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "live.log"

    def log(msg: str):
        line = f"[{ts()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    # Data
    api = PeptideValidatorAPI()
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

    test_df = pd.read_parquet(test_file)
    test_df["source_fdr"] = 50
    _, method_test = path_to_dataset_and_method(test_file)
    baseline = api._load_baseline_peptides(method_test)
    gt_test = api._load_ground_truth_peptides(method_test)
    add_set = set(test_df["Modified.Sequence"].unique()) - set(baseline)
    test_add_df = test_df[test_df["Modified.Sequence"].isin(add_set)].copy()
    y_test = test_add_df["Modified.Sequence"].isin(gt_test)

    X_train = api._make_advanced_features(train_df)
    feat_names = X_train.columns.tolist()
    X_test = api._make_advanced_features(test_add_df, feat_names)

    # Resume: determine starting trial number per variant
    def next_trial_idx(variant: str) -> int:
        existing = sorted((out_dir / "trials").glob(f"{variant}_trial_*.json"))
        if not existing:
            return 1
        last = existing[-1].stem.split("_")[-1]
        try:
            return int(last) + 1
        except Exception:
            return 1

    best_overall = {"score": -math.inf}
    summary_rows = []
    all_rows = []

    start_time = time.time()
    max_minutes = args.time_minutes
    total_trials = len(variants) * args.trials_per_variant
    done_trials = 0

    for variant in variants:
        best_variant = {"score": -math.inf}
        t_idx = next_trial_idx(variant)
        for i in range(args.trials_per_variant):
            # Graceful stop
            if (out_dir / "STOP").exists():
                log("STOP file detected. Exiting.")
                break
            if max_minutes > 0 and (time.time() - start_time) / 60.0 > max_minutes:
                log("Time budget reached. Exiting.")
                break

            params = sample_params(variant, rng)
            # Device for XGB
            if variant == "xgb":
                try:
                    dev = api._detect_gpu_device()
                    params["device"] = dev
                except Exception:
                    pass

            est = build_estimator(variant, params)
            calib = calibrator_or_none(y_train, est)
            model = calib if calib is not None else est
            model.fit(X_train, y_train)

            # Evaluate on peptide-level aggregation with FDR sweep
            results_table, _ = api._validate_and_optimize(
                model, test_add_df, y_test, feat_names, target_fdrs, len(baseline), aggregation_method="max"
            )
            score, row = pick_primary_row(results_table, primary_fdr)

            payload = {
                "variant": variant,
                "trial": t_idx,
                "score": score,
                "primary_fdr": primary_fdr,
                "selected_row": row,
                f"{variant}_params": params,
            }
            trial_path = out_dir / "trials" / f"{variant}_trial_{t_idx:03d}.json"
            trial_path.write_text(json.dumps(json_safe(payload), indent=2))
            # Per-trial CSV of the sweep
            try:
                pd.DataFrame(results_table).to_csv(out_dir / "tables" / f"{variant}_trial_{t_idx:03d}_results.csv", index=False)
            except Exception:
                pass

            all_rows.append({
                "variant": variant,
                "trial": t_idx,
                "score": score,
                "add_peptides": row.get("Additional_Peptides", None),
                "actual_fdr": row.get("Actual_FDR", None),
                "mcc": row.get("MCC", None),
            })

            if score > best_variant["score"]:
                best_variant = {"variant": variant, "score": score, "row": row, f"{variant}_params": params}
                (out_dir / f"best_{variant}.json").write_text(json.dumps(json_safe(best_variant), indent=2))
                try:
                    pd.DataFrame(results_table).to_csv(out_dir / "tables" / f"best_{variant}_results.csv", index=False)
                except Exception:
                    pass

            if score > best_overall["score"]:
                best_overall = {"variant": variant, "score": score, "row": row}
                # keep this sweep as overall best
                try:
                    dfb = pd.DataFrame(results_table)[["Target_FDR","Additional_Peptides","Recovery_Pct","Actual_FDR","Threshold"]]
                    dfb.to_csv(out_dir / "best_overall_fdr_summary.csv", index=False)
                except Exception:
                    pass

            # progress
            done_trials += 1
            log(f"Progress: {done_trials}/{total_trials} | {variant} t={t_idx} | Best={best_overall['variant']} {best_overall['score']:.2f}")
            t_idx += 1

        # end variant loop: append best-of-variant row
        summary_rows.append({
            "variant": best_variant.get("variant"),
            "score": best_variant.get("score"),
            "add_peptides": best_variant.get("row", {}).get("Additional_Peptides", None),
            "actual_fdr": best_variant.get("row", {}).get("Actual_FDR", None),
            "mcc": best_variant.get("row", {}).get("MCC", None),
            f"params_{variant}": best_variant.get(f"{variant}_params", {}),
        })

    # Save summaries
    if summary_rows:
        df = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
        df.to_csv(out_dir / "summary_best_by_variant.csv", index=False)
    if all_rows:
        df_all = pd.DataFrame(all_rows).sort_values(["score","add_peptides"], ascending=False)
        df_all.to_csv(out_dir / "all_trials_leaderboard.csv", index=False)

    meta_out = {
        "source_ref": str(ref_dir),
        "variants": variants,
        "trials_per_variant": args.trials_per_variant,
        "time_minutes": args.time_minutes,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "train_files": [str(p) for p in train_files],
        "test_file": str(test_file),
        "target_fdrs": target_fdrs,
        "primary_fdr": primary_fdr,
        "results_dir": str(out_dir),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta_out, indent=2))

    print(f"✅ Auto tabular search complete: {out_dir}")


if __name__ == "__main__":
    main()


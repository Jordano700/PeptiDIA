#!/usr/bin/env python3
"""
Broad search for Logistic Regression (with scaling/solver variations) and a simple
feed‑forward MLP (via scikit‑learn) on the same split as an existing run.

- Mirrors train/test split from a reference results folder's run_metadata.json
- Evaluates 1..N random configurations for LR and MLP
- Saves per‑trial JSON, per‑trial CSV results, best‑of summaries, and a minimal FDR sweep

Usage examples:
  # Quick sanity run
  python scripts/lr_mlp_search.py \
    --results-ref results/HPARAM_SEARCH_PATHS_20250909_144216_tmux \
    --trials-lr 10 --trials-mlp 5 \
    --out-dir results/LR_MLP_SEARCH_quick

  # Broader background run (suggested in tmux)
  tmux new -s peptidia_lr_mlp \
    "python scripts/lr_mlp_search.py --results-ref results/HPARAM_SEARCH_PATHS_20250909_144216_tmux --trials-lr 200 --trials-mlp 120 --out-dir results/LR_MLP_SEARCH_$(date +%Y%m%d_%H%M%S)"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
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


def path_to_dataset_and_method(pp: Path) -> Tuple[str, str]:
    parts = list(pp.parts)
    if "data" in parts:
        idx = parts.index("data")
        dataset = parts[idx + 1] if idx + 1 < len(parts) else "Unknown"
    else:
        dataset = "Unknown"
    method = extract_method_from_filename(pp.name, dataset)
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


# ------------------ LR ------------------
def sample_lr_params(rng: random.Random) -> Dict:
    solver = rng.choice(["lbfgs", "liblinear", "newton-cg", "saga"])  # rich solvers
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
    return params


def build_lr_pipeline(lr_params: Dict) -> Pipeline:
    lr = LogisticRegression(**lr_params)
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", lr)])
    return pipe


# ------------------ MLP ------------------
def sample_mlp_params(rng: random.Random) -> Dict:
    h_opts = [(64,), (128,), (256,), (128, 64), (256, 128), (128, 128)]
    params = {
        "hidden_layer_sizes": rng.choice(h_opts),
        "activation": "relu",
        "solver": "adam",
        "learning_rate_init": rng.choice([1e-4, 3e-4, 1e-3, 3e-3]),
        "alpha": rng.choice([1e-5, 1e-4, 1e-3, 1e-2]),  # L2 reg
        "batch_size": rng.choice([64, 128, 256]),
        "max_iter": rng.choice([200, 300, 400]),
        "early_stopping": True,
        "validation_fraction": 0.1,
        "n_iter_no_change": rng.choice([10, 20]),
        "random_state": rng.randint(1, 10_000),
    }
    return params


def build_mlp_pipeline(mlp_params: Dict) -> Pipeline:
    mlp = MLPClassifier(**mlp_params)
    pipe = Pipeline([("scaler", StandardScaler()), ("mlp", mlp)])
    return pipe


def main():
    ap = argparse.ArgumentParser(description="LR + MLP search (mirrors a prior run's split)")
    ap.add_argument("--results-ref", required=True, help="Path to existing run with run_metadata.json")
    ap.add_argument("--trials-lr", type=int, default=50)
    ap.add_argument("--trials-mlp", type=int, default=30)
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

    # Prepare output
    out_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "results" / f"LR_MLP_SEARCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
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
    X_train = api._make_advanced_features(train_df)
    feat_names = X_train.columns.tolist()
    X_test = api._make_advanced_features(test_add_df, feat_names)

    # ------------------ LR loop ------------------
    best_lr = {"score": -math.inf}
    for t in range(1, args.trials_lr + 1):
        lr_params = sample_lr_params(rng)
        pipe = build_lr_pipeline(lr_params)
        calib = calibrator_or_none(y_train, pipe)
        if calib is not None:
            model = calib
            model.fit(X_train, y_train)
        else:
            pipe.fit(X_train, y_train)
            model = pipe
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
        pd.DataFrame(results_table).to_csv(out_dir / "tables" / f"lr_trial_{t:03d}_results.csv", index=False)
        if score > best_lr["score"]:
            best_lr = {"variant": "lr", "score": score, "row": row, "lr_params": lr_params, "use_scaler": True}
            with open(out_dir / "best_lr.json", "w") as f:
                json.dump(json_safe(best_lr), f, indent=2)

    # ------------------ MLP loop ------------------
    best_mlp = {"score": -math.inf}
    for t in range(1, args.trials_mlp + 1):
        mlp_params = sample_mlp_params(rng)
        pipe = build_mlp_pipeline(mlp_params)
        calib = calibrator_or_none(y_train, pipe)
        if calib is not None:
            model = calib
            model.fit(X_train, y_train)
        else:
            pipe.fit(X_train, y_train)
            model = pipe
        results_table, _ = api._validate_and_optimize(model, test_add_df, y_test, feat_names, target_fdrs, len(baseline), aggregation_method="max")
        score, row = pick_primary_row(results_table, primary_fdr)
        payload = {
            "variant": "mlp",
            "trial": t,
            "score": score,
            "primary_fdr": primary_fdr,
            "selected_row": row,
            "mlp_params": mlp_params,
            "use_scaler": True,
        }
        with open(out_dir / "trials" / f"mlp_trial_{t:03d}.json", "w") as f:
            json.dump(json_safe(payload), f, indent=2)
        pd.DataFrame(results_table).to_csv(out_dir / "tables" / f"mlp_trial_{t:03d}_results.csv", index=False)
        if score > best_mlp["score"]:
            best_mlp = {"variant": "mlp", "score": score, "row": row, "mlp_params": mlp_params, "use_scaler": True}
            with open(out_dir / "best_mlp.json", "w") as f:
                json.dump(json_safe(best_mlp), f, indent=2)

    # Minimal FDR sweeps for bests
    def write_best_sweep(tag: str, best_obj: Dict):
        try:
            if not best_obj or not best_obj.get("row"):
                return
            if tag == "lr":
                pipe = build_lr_pipeline(best_obj["lr_params"])    
            else:
                pipe = build_mlp_pipeline(best_obj["mlp_params"])  
            calib = calibrator_or_none(y_train, pipe)
            model = calib if calib is not None else pipe
            model.fit(X_train, y_train)
            results_table, _ = api._validate_and_optimize(model, test_add_df, y_test, feat_names, target_fdrs, len(baseline), aggregation_method="max")
            pd.DataFrame(results_table)[["Target_FDR","Additional_Peptides","Recovery_Pct","Actual_FDR","Threshold"]].to_csv(out_dir / f"best_overall_fdr_summary_{tag}.csv", index=False)
        except Exception as e:
            print(f"[WARN] Could not save best sweep for {tag}: {e}")

    write_best_sweep("lr", best_lr)
    write_best_sweep("mlp", best_mlp)

    # Summary CSVs
    def extract_trial_rows(pattern: str, add_key: str) -> List[Dict]:
        rows = []
        for p in sorted((out_dir / "trials").glob(pattern)):
            d = json.loads(p.read_text())
            r = d.get("selected_row", {})
            rows.append({
                "trial": d.get("trial"),
                "score": d.get("score"),
                "add_peptides": r.get("Additional_Peptides"),
                "actual_fdr": r.get("Actual_FDR"),
                "mcc": r.get("MCC"),
            })
        return rows

    lr_rows = extract_trial_rows("lr_trial_*.json", "lr_params")
    mlp_rows = extract_trial_rows("mlp_trial_*.json", "mlp_params")
    if lr_rows:
        pd.DataFrame(lr_rows).sort_values(["score","add_peptides"], ascending=False).to_csv(out_dir / "summary_best_by_lr.csv", index=False)
    if mlp_rows:
        pd.DataFrame(mlp_rows).sort_values(["score","add_peptides"], ascending=False).to_csv(out_dir / "summary_best_by_mlp.csv", index=False)

    meta_out = {
        "source_ref": str(ref_dir),
        "seed": args.seed,
        "trials_lr": args.trials_lr,
        "trials_mlp": args.trials_mlp,
        "timestamp": datetime.now().isoformat(),
        "train_files": [str(p) for p in train_files],
        "test_file": str(test_file),
        "target_fdrs": target_fdrs,
        "primary_fdr": primary_fdr,
        "out_dir": str(out_dir),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta_out, indent=2))

    print(f"✅ LR+MLP search complete: {out_dir}")


if __name__ == "__main__":
    main()


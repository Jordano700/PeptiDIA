#!/usr/bin/env python3
"""
Run a fixed XGBoost model (from a params JSON) against the Streamlit-style
ensemble on an explicit Colon train/test split (FDR_50 files), and compare.

This mirrors the data preparation, feature engineering, calibration, and
FDR sweep used in the existing hparam runner, but uses fixed params for XGB.

Example:
  python scripts/run_fixed_xgb_vs_ensemble_paths.py \
    --train-files \
      data/Colon/short_gradient/FDR_50/20250728_RD201_EXB_EV1107_300SPD_Colon_676_01_FDR50.parquet,\
      data/Colon/short_gradient/FDR_50/20250728_RD201_EXB_EV1107_300SPD_Colon_677_01_FDR50.parquet,\
      data/Colon/short_gradient/FDR_50/20250728_RD201_EXB_EV1107_300SPD_Colon_678_01_FDR50.parquet \
    --test-file \
      data/Colon/short_gradient/FDR_50/20250728_RD201_EXB_EV1107_300SPD_Colon_679_01_FDR50.parquet \
    --xgb-params-json results/AUTO_TABULAR_20250910_113944/best_xgb.json \
    --primary-fdr 1.0

Outputs under results/FIXED_COMPARE_YYYYMMDD_HHMMSS/
 - best_xgb.json: metrics and params for fixed XGB
 - best_ensemble.json: metrics and params used for ensemble
 - tables/*_fdr_results.csv: per-FDR sweep tables per model
 - run_metadata.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project src/ is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
import sys
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from peptidia.core.peptide_validator_api import PeptideValidatorAPI  # type: ignore
from peptidia.core.dataset_utils import extract_method_from_filename  # type: ignore

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def path_to_dataset_and_method(p: Path) -> Tuple[str, str]:
    parts = list(p.parts)
    if "data" in parts:
        idx = parts.index("data")
        dataset = parts[idx + 1] if idx + 1 < len(parts) else "Unknown"
    else:
        dataset = "Unknown"
    method = extract_method_from_filename(p.name, dataset)
    return dataset, method


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


def parse_percent_like(s: str) -> float:
    s = (s or "").strip()
    if s.endswith("%"):
        s = s[:-1]
    s = s.replace(",", ".")
    return float(s)


def main():
    ap = argparse.ArgumentParser(description="Compare fixed XGB vs ensemble on explicit FDR_50 paths")
    ap.add_argument("--train-files", type=str, required=True, help="Comma-separated FDR_50 parquet paths for training")
    ap.add_argument("--test-file", type=str, required=True, help="FDR_50 parquet path for testing")
    ap.add_argument("--xgb-params-json", type=str, required=True, help="Path to JSON with XGB params under 'xgb_params'")
    ap.add_argument("--target-fdrs", type=str, default="1,2,3,4,5,6,7,8,9", help="Comma-separated target FDR levels")
    ap.add_argument("--primary-fdr", type=str, default="1.0", help="Primary FDR for ranking (e.g., 1 or 1%)")
    ap.add_argument("--results-dir", type=str, default="", help="Optional output directory")
    args = ap.parse_args()

    train_paths = [Path(p.strip()) for p in args.train_files.split(",") if p.strip()]
    test_path = Path(args.test_file)
    target_fdrs = [parse_percent_like(x) for x in args.target_fdrs.split(",") if x.strip()]
    primary_fdr = parse_percent_like(args.primary_fdr)

    # Output dir
    out_dir = Path(args.results_dir) if args.results_dir else (PROJECT_ROOT / "results" / f"FIXED_COMPARE_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "live.log"
    def log(msg: str):
        line = f"[{ts()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    # Load XGB params
    cfg = json.loads(Path(args.xgb_params_json).read_text())
    xgb_params = cfg.get("xgb_params", cfg)

    # API for device, features, evaluation
    api = PeptideValidatorAPI()
    # Ensure device compatibility
    try:
        xgb_params = dict(xgb_params)
        xgb_params["device"] = api._detect_gpu_device()
    except Exception:
        pass

    # Build training set (FDR=50) with per-method labels
    train_frames = []
    y_train_list: List[int] = []
    for p in train_paths:
        df = pd.read_parquet(p)
        df["source_fdr"] = 50
        dataset, method = path_to_dataset_and_method(p)
        gt_peps = api._load_ground_truth_peptides(method)
        labels = df["Modified.Sequence"].isin(gt_peps).astype(int)
        y_train_list.extend(labels.tolist())
        train_frames.append(df)
    train_df = pd.concat(train_frames, ignore_index=True)
    y_train = pd.Series(y_train_list)
    log(f"Training rows: {len(train_df):,}; positives: {int(y_train.sum()):,} ({y_train.mean()*100:.1f}%)")

    # Test set (FDR=50 additional peptides beyond baseline)
    test_df = pd.read_parquet(test_path)
    test_df["source_fdr"] = 50
    _, method_test = path_to_dataset_and_method(test_path)
    baseline = api._load_baseline_peptides(method_test)
    gt_test = api._load_ground_truth_peptides(method_test)
    add_set = set(test_df["Modified.Sequence"].unique()) - set(baseline)
    test_add_df = test_df[test_df["Modified.Sequence"].isin(add_set)].copy()
    y_test = test_add_df["Modified.Sequence"].isin(gt_test)
    log(f"Test rows: {len(test_df):,}; additional rows: {len(test_add_df):,}; unique additional peptides: {test_add_df['Modified.Sequence'].nunique():,}")

    # Features
    X_train = api._make_advanced_features(train_df)
    feat_names = X_train.columns.tolist()
    X_test = api._make_advanced_features(test_add_df, feat_names)

    # Helper to optionally calibrate (consistent with other runners)
    def maybe_calibrate(estimator, y: pd.Series):
        pos = int(y.sum())
        neg = int(len(y) - pos)
        feasible_splits = max(2, min(3, pos, neg))
        if feasible_splits >= 2:
            skf = StratifiedKFold(n_splits=feasible_splits, shuffle=True, random_state=42)
            return CalibratedClassifierCV(estimator, method='isotonic', cv=skf)
        return None

    # 1) Fixed XGB
    xgb_model = XGBClassifier(**xgb_params)
    xgb_cal = maybe_calibrate(xgb_model, y_train)
    model_xgb = xgb_cal if xgb_cal is not None else xgb_model
    model_xgb.fit(X_train, y_train)
    xgb_results, _ = api._validate_and_optimize(model_xgb, test_add_df, y_test, feat_names, target_fdrs, len(baseline), aggregation_method='max')
    xgb_score, xgb_row = pick_primary_row(xgb_results, primary_fdr)
    # Save
    best_xgb_payload = {"variant": "xgb", "score": xgb_score, "row": xgb_row, "xgb_params": xgb_params}
    (out_dir / "best_xgb.json").write_text(json.dumps(json_safe(best_xgb_payload), indent=2))
    try:
        pd.DataFrame(xgb_results).to_csv(out_dir / "tables" / "xgb_fdr_results.csv", index=False)
    except Exception:
        pass

    # 2) Ensemble (XGB with same params + RF + LR, soft voting + isotonic calibration)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    xgb_base = XGBClassifier(**xgb_params)
    ensemble = VotingClassifier(
        estimators=[("xgb", xgb_base), ("rf", rf), ("lr", lr)], voting="soft", weights=[3, 2, 1]
    )
    ens_cal = CalibratedClassifierCV(ensemble, method='isotonic', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42))
    ens_cal.fit(X_train, y_train)
    ens_results, _ = api._validate_and_optimize(ens_cal, test_add_df, y_test, feat_names, target_fdrs, len(baseline), aggregation_method='max')
    ens_score, ens_row = pick_primary_row(ens_results, primary_fdr)
    best_ensemble_payload = {
        "variant": "ensemble",
        "score": ens_score,
        "row": ens_row,
        "xgb_params": xgb_params,
        "rf_params": rf.get_params(),
        "lr_params": lr.get_params(),
    }
    (out_dir / "best_ensemble.json").write_text(json.dumps(json_safe(best_ensemble_payload), indent=2))
    try:
        pd.DataFrame(ens_results).to_csv(out_dir / "tables" / "ensemble_fdr_results.csv", index=False)
    except Exception:
        pass

    # Metadata and quick text summary
    meta = {
        "train_files": [str(p) for p in train_paths],
        "test_file": str(test_path),
        "target_fdrs": target_fdrs,
        "primary_fdr": primary_fdr,
        "results_dir": str(out_dir),
        "xgb_params_json": str(Path(args.xgb_params_json).resolve()),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    with open(out_dir / "best_overall_fdr_summary.txt", "w") as f:
        f.write("Model,Target_FDR,Additional_Peptides,Recovery_Pct,Actual_FDR\n")
        def write_rows(name: str, table: List[Dict]):
            for r in table:
                f.write(f"{name},{float(r['Target_FDR']):.1f}%,{int(r['Additional_Peptides'])},{float(r['Recovery_Pct']):.1f}%,{float(r['Actual_FDR']):.1f}%\n")
        write_rows("xgb", xgb_results)
        write_rows("ensemble", ens_results)

    # Print concise comparison line
    print("\nComparison at primary FDR {:.1f}%:".format(primary_fdr))
    def fmt_row(tag: str, row: Dict):
        return (
            f"{tag}: Additional={int(row.get('Additional_Peptides', 0))}, "
            f"Recovery={float(row.get('Recovery_Pct', 0)):.1f}%, "
            f"Actual_FDR={float(row.get('Actual_FDR', 0)):.1f}%, "
            f"MCC={float(row.get('MCC', 0)):.3f}"
        )
    print("  " + fmt_row("XGB", xgb_row))
    print("  " + fmt_row("Ensemble", ens_row))
    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()


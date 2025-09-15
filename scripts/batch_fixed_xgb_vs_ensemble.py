#!/usr/bin/env python3
"""
Batch runner: compare fixed XGB vs ensemble across datasets and random splits.

For each dataset, repeatedly sample a holdout test file (FDR_50) and
train on K other files. Runs both models using the same feature pipeline
and FDR sweep as Streamlit/CLI utilities, and aggregates per‑split results.

Outputs under results/FIXED_BATCH_YYYYMMDD_HHMMSS/:
 - Per‑split folders with FDR sweep CSVs and JSON summaries
 - summary_primary_fdr.csv: one row per (dataset, split) x model
 - compare_deltas.csv: XGB vs Ensemble deltas (Additional, MCC, Actual_FDR)
 - summary_text.txt: quick win/loss and averages

Example (Colon, Ileon, Artere; K in {2,4}; 5 repeats):
  tmux new -s peptidia_batch \
    "python3 scripts/batch_fixed_xgb_vs_ensemble.py \
      --datasets Colon,Ileon,Artere \
      --train-k 2,4 \
      --repeats 5 \
      --xgb-params-json results/AUTO_TABULAR_20250910_113944/best_xgb.json"
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
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


def path_to_dataset_and_method(p: Path) -> Tuple[str, str]:
    parts = list(p.parts)
    if "data" in parts:
        idx = parts.index("data")
        dataset = parts[idx + 1] if idx + 1 < len(parts) else "Unknown"
    else:
        dataset = "Unknown"
    method = extract_method_from_filename(p.name, dataset)
    return dataset, method


def maybe_calibrate(y: pd.Series, estimator):
    pos = int(y.sum())
    neg = int(len(y) - pos)
    feasible_splits = max(2, min(3, pos, neg))
    if feasible_splits >= 2:
        skf = StratifiedKFold(n_splits=feasible_splits, shuffle=True, random_state=42)
        return CalibratedClassifierCV(estimator, method='isotonic', cv=skf)
    return None


def run_split(api: PeptideValidatorAPI,
              train_paths: List[Path],
              test_path: Path,
              xgb_params: Dict,
              target_fdrs: List[float],
              primary_fdr: float,
              out_dir: Path) -> Dict:
    # Build training
    train_frames = []
    y_train_list: List[int] = []
    for p in train_paths:
        try:
            df = pd.read_parquet(p)
        except Exception:
            # Skip unreadable/malformed train file
            continue
        try:
            df["source_fdr"] = 50
            _, method = path_to_dataset_and_method(p)
            gt_peps = api._load_ground_truth_peptides(method)
            labels = df.get("Modified.Sequence")
            if labels is None:
                continue
            labels = labels.isin(gt_peps).astype(int)
            y_train_list.extend(labels.tolist())
            train_frames.append(df)
        except Exception:
            # Skip files missing required columns or labels
            continue
    train_df = pd.concat(train_frames, ignore_index=True)
    y_train = pd.Series(y_train_list)

    # Test additional beyond baseline
    try:
        test_df = pd.read_parquet(test_path)
    except Exception as e:
        # Propagate to caller to skip this split
        raise RuntimeError(f"Failed to read test file {test_path.name}: {e}")
    test_df["source_fdr"] = 50
    _, method_test = path_to_dataset_and_method(test_path)
    baseline = api._load_baseline_peptides(method_test)
    gt_test = api._load_ground_truth_peptides(method_test)
    add_set = set(test_df["Modified.Sequence"].unique()) - set(baseline)
    test_add_df = test_df[test_df["Modified.Sequence"].isin(add_set)].copy()
    y_test = test_add_df["Modified.Sequence"].isin(gt_test)

    # Features
    # Ensure we have non-empty training data after filtering
    if train_df.empty or y_train.empty:
        raise RuntimeError("Empty training data after filtering problematic files")
    X_train = api._make_advanced_features(train_df)
    feat_names = X_train.columns.tolist()
    X_test = api._make_advanced_features(test_add_df, feat_names)

    # 1) XGB
    xgb = XGBClassifier(**xgb_params)
    cal_xgb = maybe_calibrate(y_train, xgb)
    model_xgb = cal_xgb if cal_xgb is not None else xgb
    model_xgb.fit(X_train, y_train)
    xgb_results, _ = api._validate_and_optimize(model_xgb, test_add_df, y_test, feat_names, target_fdrs, len(baseline), aggregation_method='max')

    # 2) Ensemble
    rf = RandomForestClassifier(n_estimators=500, max_depth=8, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    xgb_base = XGBClassifier(**xgb_params)
    ensemble = VotingClassifier(estimators=[("xgb", xgb_base), ("rf", rf), ("lr", lr)], voting="soft", weights=[3, 2, 1])
    cal_ens = CalibratedClassifierCV(ensemble, method='isotonic', cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42))
    cal_ens.fit(X_train, y_train)
    ens_results, _ = api._validate_and_optimize(cal_ens, test_add_df, y_test, feat_names, target_fdrs, len(baseline), aggregation_method='max')

    # Save per‑split outputs
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    try:
        pd.DataFrame(xgb_results).to_csv(out_dir / "tables" / "xgb_fdr_results.csv", index=False)
        pd.DataFrame(ens_results).to_csv(out_dir / "tables" / "ensemble_fdr_results.csv", index=False)
    except Exception:
        pass

    def pick_primary(results_table: List[Dict]) -> Tuple[float, Dict]:
        for row in results_table:
            if abs(float(row.get("Target_FDR", 0)) - primary_fdr) < 1e-6:
                score = float(row.get("Additional_Peptides", 0)) + 0.01 * float(row.get("MCC", 0))
                return score, row
        if results_table:
            row = sorted(results_table, key=lambda r: r.get("Target_FDR", 1e9))[0]
            score = float(row.get("Additional_Peptides", 0)) + 0.01 * float(row.get("MCC", 0))
            return score, row
        return -math.inf, {}

    xgb_score, xgb_row = pick_primary(xgb_results)
    ens_score, ens_row = pick_primary(ens_results)

    (out_dir / "best_xgb.json").write_text(json.dumps(json_safe({
        "variant": "xgb",
        "score": xgb_score,
        "row": xgb_row,
        "xgb_params": xgb_params,
    }), indent=2))
    (out_dir / "best_ensemble.json").write_text(json.dumps(json_safe({
        "variant": "ensemble",
        "score": ens_score,
        "row": ens_row,
        "xgb_params": xgb_params,
        "rf_params": rf.get_params(),
        "lr_params": lr.get_params(),
    }), indent=2))

    return {
        "xgb": {"score": xgb_score, **xgb_row},
        "ensemble": {"score": ens_score, **ens_row},
    }


def main():
    ap = argparse.ArgumentParser(description="Batch fixed XGB vs ensemble across datasets and random splits")
    ap.add_argument("--datasets", type=str, default="Colon,Ileon,Artere", help="Comma-separated dataset folders under data/")
    ap.add_argument("--gradient", type=str, default="short_gradient", help="Gradient folder under each dataset")
    ap.add_argument("--train-k", type=str, default="2,4", help="Comma-separated train sizes (e.g., 2,4)")
    ap.add_argument("--repeats", type=int, default=5, help="Repeats per K per dataset")
    ap.add_argument("--xgb-params-json", type=str, required=True, help="Path to best_xgb.json (or a JSON with xgb_params)")
    ap.add_argument("--target-fdrs", type=str, default="1,2,3,4,5,6,7,8,9", help="Target FDRs for sweeps")
    ap.add_argument("--primary-fdr", type=str, default="1.0", help="Primary FDR used for ranking (e.g., 1 or 1%)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for split sampling")
    ap.add_argument("--out-dir", type=str, default="", help="Optional output directory root")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    train_k_list = [int(k.strip()) for k in args.train_k.split(",") if k.strip()]
    target_fdrs = [parse_percent_like(x) for x in args.target_fdrs.split(",") if x.strip()]
    primary_fdr = parse_percent_like(args.primary_fdr)

    # Load XGB params
    cfg = json.loads(Path(args.xgb_params_json).read_text())
    xgb_params = cfg.get("xgb_params", cfg)

    # API and device
    api = PeptideValidatorAPI()
    try:
        xgb_params = dict(xgb_params)
        xgb_params["device"] = api._detect_gpu_device()
    except Exception:
        pass

    out_root = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "results" / f"FIXED_BATCH_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "splits").mkdir(exist_ok=True)
    log_path = out_root / "live.log"

    def log(msg: str):
        line = f"[{ts()}] {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    # Aggregation containers
    rows_primary: List[Dict] = []
    rows_delta: List[Dict] = []

    for dataset in datasets:
        data_dir = PROJECT_ROOT / "data" / dataset / args.gradient / "FDR_50"
        files = sorted(p for p in data_dir.glob("*.parquet"))
        if len(files) < 3:
            log(f"[WARN] Skipping {dataset}: found {len(files)} FDR_50 files (<3)")
            continue
        log(f"Dataset {dataset}: {len(files)} FDR_50 candidates")

        for k in train_k_list:
            if len(files) <= k:
                log(f"[WARN] {dataset} k={k}: not enough files (need >{k})")
                continue
            for r in range(1, args.repeats + 1):
                # sample split
                test_path = rng.choice(files)
                train_pool = [p for p in files if p != test_path]
                train_paths = rng.sample(train_pool, k)
                split_dir = out_root / "splits" / dataset / f"k{k}" / f"repeat_{r:02d}"
                split_dir.mkdir(parents=True, exist_ok=True)

                log(f"{dataset} k={k} r={r}: test={test_path.name}; train={[p.name for p in train_paths]}")
                try:
                    res = run_split(api, train_paths, test_path, xgb_params, target_fdrs, primary_fdr, split_dir)
                except Exception as e:
                    log(f"[ERROR] Split failed: {e}")
                    continue

                # Extract primary FDR rows
                def extract_primary(d: Dict) -> Dict:
                    return {
                        "Additional_Peptides": int(d.get("Additional_Peptides", 0)),
                        "Recovery_Pct": float(d.get("Recovery_Pct", 0.0)),
                        "Actual_FDR": float(d.get("Actual_FDR", 0.0)),
                        "MCC": float(d.get("MCC", 0.0)),
                        "Threshold": float(d.get("Threshold", 0.0)),
                    }

                xgb_row = extract_primary(res["xgb"]) if res.get("xgb") else {}
                ens_row = extract_primary(res["ensemble"]) if res.get("ensemble") else {}

                rows_primary.append({
                    "dataset": dataset,
                    "train_k": k,
                    "repeat": r,
                    "test_file": str(test_path),
                    "variant": "xgb",
                    **xgb_row,
                })
                rows_primary.append({
                    "dataset": dataset,
                    "train_k": k,
                    "repeat": r,
                    "test_file": str(test_path),
                    "variant": "ensemble",
                    **ens_row,
                })

                rows_delta.append({
                    "dataset": dataset,
                    "train_k": k,
                    "repeat": r,
                    "test_file": str(test_path),
                    "delta_additional": xgb_row.get("Additional_Peptides", 0) - ens_row.get("Additional_Peptides", 0),
                    "delta_mcc": xgb_row.get("MCC", 0.0) - ens_row.get("MCC", 0.0),
                    "delta_actual_fdr": xgb_row.get("Actual_FDR", 0.0) - ens_row.get("Actual_FDR", 0.0),
                    "delta_recovery_pct": xgb_row.get("Recovery_Pct", 0.0) - ens_row.get("Recovery_Pct", 0.0),
                })

    # Save aggregates
    if rows_primary:
        df_p = pd.DataFrame(rows_primary)
        df_p.to_csv(out_root / "summary_primary_fdr.csv", index=False)

    if rows_delta:
        df_d = pd.DataFrame(rows_delta)
        df_d.to_csv(out_root / "compare_deltas.csv", index=False)

        # Quick text summary
        with open(out_root / "summary_text.txt", "w") as f:
            f.write("XGB vs Ensemble @ primary FDR\n")
            for dataset in sorted(set(d["dataset"] for d in rows_delta)):
                subset = [d for d in rows_delta if d["dataset"] == dataset]
                wins = sum(1 for d in subset if d["delta_additional"] > 0)
                ties = sum(1 for d in subset if d["delta_additional"] == 0)
                losses = sum(1 for d in subset if d["delta_additional"] < 0)
                mean_da = float(pd.Series([d["delta_additional"] for d in subset]).mean())
                mean_dm = float(pd.Series([d["delta_mcc"] for d in subset]).mean())
                mean_df = float(pd.Series([d["delta_actual_fdr"] for d in subset]).mean())
                f.write(f"{dataset}: W/T/L = {wins}/{ties}/{losses}; mean ΔAdditional={mean_da:+.1f}, ΔMCC={mean_dm:+.003f}, ΔActual_FDR={mean_df:+.2f}%\n")

    # Metadata
    meta = {
        "datasets": datasets,
        "gradient": args.gradient,
        "train_k": train_k_list,
        "repeats": args.repeats,
        "target_fdrs": target_fdrs,
        "primary_fdr": primary_fdr,
        "xgb_params_json": str(Path(args.xgb_params_json).resolve()),
        "timestamp": datetime.now().isoformat(),
        "results_dir": str(out_root),
    }
    (out_root / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"✅ Batch complete: {out_root}")


if __name__ == "__main__":
    main()

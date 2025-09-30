#!/usr/bin/env python3
"""
Build a live apples-to-apples K-sweep comparison (excluding ASTRAL) vs. Sept16 baseline.

Outputs (into the latest Trial037 NOASTRAL folder by default):
 - live_primary_fdr.csv: live snapshot at Target FDR 1% from STREAMLIT_RESULTS_*
 - compare_to_20250916.tsv: joined live vs baseline per run
 - compare_to_20250916_agg.tsv: aggregated to max Current_Additional per dataset/k/variant
 - summary_text.txt: concise per-dataset summary lines

Notes
 - k is inferred as len(train_methods) per run
 - variant is 'xgb' for current Streamlit runs; ensemble is supported when present
 - ASTRAL is excluded from live and baseline
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd


BASELINE_ROOT_DEFAULT = "results/K_SWEEP_20250915_145805_CROSS0/splits"
TRIAL_GLOB_DEFAULT = "results/TRIAL037_KSWEEP_*_NOASTRAL"
STREAMLIT_GLOB_DEFAULT = "results/STREAMLIT_RESULTS_*"


def _safe_read_json(path: Path) -> Optional[dict]:
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


def collect_live_snapshot(streamlit_glob: str = STREAMLIT_GLOB_DEFAULT,
                          exclude_datasets: Tuple[str, ...] = ("ASTRAL",),
                          include_datasets: Optional[Tuple[str, ...]] = None) -> pd.DataFrame:
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
        if not test_method:
            continue
        dataset = test_method.split("_")[0]
        if dataset in exclude_datasets:
            continue
        if include_datasets is not None and dataset not in include_datasets:
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            continue
        # Select the 1% target FDR row (apples-to-apples)
        one = df[df["Target_FDR"].astype(float) == 1.0].copy()
        if one.empty:
            continue
        # Allow multiple rows if a file has duplicates; keep them all as separate runs
        for _, r in one.iterrows():
            rows.append({
                "dataset": dataset,
                "train_k": int(len(train_methods)),
                "variant": "xgb",  # Streamlit runs are XGB-only for now
                "Additional_Peptides": int(r.get("Additional_Peptides", 0)),
                "Actual_FDR": float(r.get("Actual_FDR", 0.0)),
                "_run": Path(res_dir).name,
            })
    if not rows:
        return pd.DataFrame(columns=[
            "dataset", "train_k", "variant", "Additional_Peptides", "Actual_FDR", "_run"
        ])
    live = pd.DataFrame(rows)
    # Keep only plausible datasets (letters) and k>=1
    live = live[(live["dataset"].str.len() > 0) & (live["train_k"] >= 1)]
    return live


def collect_baseline(baseline_root: str = BASELINE_ROOT_DEFAULT,
                     include_variants: Tuple[str, ...] = ("xgb", "ensemble"),
                     exclude_datasets: Tuple[str, ...] = ("ASTRAL",)) -> pd.DataFrame:
    root = Path(baseline_root)
    rows: List[dict] = []
    if not root.exists():
        return pd.DataFrame(columns=["dataset", "train_k", "variant", "Baseline_Additional"])
    for dataset_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        dataset = dataset_dir.name
        if dataset in exclude_datasets:
            continue
        for k_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("k")]):
            try:
                k_val = int(k_dir.name[1:])
            except Exception:
                continue
            rep = k_dir / "repeat_01"
            if not rep.exists():
                # try first subdir
                reps = [p for p in k_dir.iterdir() if p.is_dir()]
                rep = reps[0] if reps else None
            if not rep or not rep.exists():
                continue
            for variant in include_variants:
                best_path = rep / f"best_{variant}.json"
                if not best_path.exists():
                    continue
                best = _safe_read_json(best_path)
                if not best:
                    continue
                row = best.get("row", {})
                # Expect Target_FDR == 1.0 in these best_* files
                addl = row.get("Additional_Peptides")
                if addl is None:
                    continue
                rows.append({
                    "dataset": dataset,
                    "train_k": k_val,
                    "variant": variant,
                    "Baseline_Additional": int(addl),
                })
    base = pd.DataFrame(rows)
    return base


def build_comparisons(live: pd.DataFrame, baseline: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if live.empty or baseline.empty:
        return pd.DataFrame(), pd.DataFrame()
    # Detailed: join per run
    detailed = (
        live.merge(baseline, on=["dataset", "train_k", "variant"], how="left")
            .assign(Current_Additional=lambda d: d["Additional_Peptides"].astype("Int64"))
    )
    if "Baseline_Additional" not in detailed.columns:
        detailed["Baseline_Additional"] = pd.NA
    detailed["Delta"] = detailed["Current_Additional"] - detailed["Baseline_Additional"]
    # Aggregated: max Current per dataset/k/variant
    agg_live = (
        live.groupby(["dataset", "train_k", "variant"], as_index=False)
            .agg(Current_Additional=("Additional_Peptides", "max"))
    )
    agg = agg_live.merge(baseline, on=["dataset", "train_k", "variant"], how="left")
    agg["Delta"] = agg["Current_Additional"] - agg["Baseline_Additional"]
    return detailed[[
        "dataset", "train_k", "variant", "Current_Additional", "Baseline_Additional", "Delta"
    ]], agg[[
        "dataset", "train_k", "variant", "Current_Additional", "Baseline_Additional", "Delta"
    ]]


def write_summary_text_v2(agg: pd.DataFrame, out_dir: Path) -> None:
    lines: List[str] = []
    if not agg.empty:
        # Per-dataset breakdown: show by variant, sorted by name
        # If multiple k per dataset, pick the one with the most recent max Current (already max per k),
        # but include all ks to be explicit.
        for dataset in sorted(agg["dataset"].unique()):
            for variant in sorted(agg["variant"].unique()):
                sub = agg[(agg["dataset"] == dataset) & (agg["variant"] == variant)].copy()
                if sub.empty:
                    continue
                # Sort by k asc for readability
                sub = sub.sort_values(["train_k"]) 
                # Summary per k
                for _, r in sub.iterrows():
                    k = int(r["train_k"])
                    cur = int(r["Current_Additional"]) if pd.notna(r["Current_Additional"]) else 0
                    base = int(r["Baseline_Additional"]) if pd.notna(r["Baseline_Additional"]) else 0
                    delta = cur - base
                    lines.append(f"{dataset} k={k} {variant}: Current {cur} vs Baseline {base}  Delta {delta:+d}")
        # Also add a compact mean delta by dataset/variant
        lines.append("")
        means = (
            agg.groupby(["dataset", "variant"], as_index=False)["Delta"].mean()
               .sort_values(["dataset", "variant"]) 
        )
        for _, r in means.iterrows():
            lines.append(f"{r['dataset']} {r['variant']}: mean Additional={r['Delta']:.1f}")
    else:
        lines.append("No aggregated comparison available yet.")
    out = out_dir / "summary_text.txt"
    out.write_text("\n".join(lines))

def write_summary_ascii(agg: pd.DataFrame, out_dir: Path) -> None:
    lines: List[str] = []
    if not agg.empty:
        for dataset in sorted(agg["dataset"].unique()):
            for variant in sorted(agg["variant"].unique()):
                sub = agg[(agg["dataset"] == dataset) & (agg["variant"] == variant)].copy()
                if sub.empty:
                    continue
                sub = sub.sort_values(["train_k"]) 
                for _, r in sub.iterrows():
                    k = int(r["train_k"])
                    cur_val = r["Current_Additional"]
                    base_val = r["Baseline_Additional"]
                    if pd.isna(base_val):
                        lines.append(f"{dataset} k={k} {variant}: Current {int(cur_val)} vs Baseline N/A -> Delta N/A")
                    else:
                        cur = int(cur_val) if pd.notna(cur_val) else 0
                        base = int(base_val)
                        delta = cur - base
                        lines.append(f"{dataset} k={k} {variant}: Current {cur} vs Baseline {base} -> Delta {delta:+d}")
        lines.append("")
        means = (
            agg.groupby(["dataset", "variant"], as_index=False)["Delta"].mean()
               .sort_values(["dataset", "variant"]) 
        )
        for _, r in means.iterrows():
            val = r["Delta"]
            if pd.isna(val):
                lines.append(f"{r['dataset']} {r['variant']}: mean DeltaAdditional=N/A")
            else:
                lines.append(f"{r['dataset']} {r['variant']}: mean DeltaAdditional={val:.1f}")
    else:
        lines.append("No aggregated comparison available yet.")
    out = out_dir / "summary_text.txt"
    out.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="Build Trial037 k-sweep apples-to-apples comparison.")
    ap.add_argument("--trial_dir", type=str, default=None,
                    help="Output Trial037 dir. Defaults to latest results/TRIAL037_KSWEEP_*_NOASTRAL")
    ap.add_argument("--baseline_root", type=str, default=BASELINE_ROOT_DEFAULT,
                    help="Baseline K_SWEEP splits root")
    ap.add_argument("--streamlit_glob", type=str, default=STREAMLIT_GLOB_DEFAULT,
                    help="Glob for live Streamlit result dirs")
    args = ap.parse_args()

    trial_dir = Path(args.trial_dir) if args.trial_dir else _latest_trial_dir()
    if trial_dir is None:
        raise SystemExit("No Trial037 folder found; create one like results/TRIAL037_KSWEEP_*_NOASTRAL")
    trial_dir.mkdir(parents=True, exist_ok=True)

    # Discover allowed NOASTRAL datasets from baseline splits
    baseline = collect_baseline(args.baseline_root)
    allowed = tuple(sorted(d for d in baseline["dataset"].unique() if d != "ASTRAL")) if not baseline.empty else None

    # 1) Live snapshot (filtered to allowed baseline datasets when available)
    live = collect_live_snapshot(args.streamlit_glob, include_datasets=allowed)
    live_out = trial_dir / "live_primary_fdr.csv"
    live.sort_values(["dataset", "train_k", "_run"]).drop(columns=["_run"]).to_csv(live_out, index=False)

    # 2) Baseline (already loaded)

    # 3) Comparisons
    detailed, agg = build_comparisons(live, baseline)
    if not detailed.empty:
        detailed.sort_values(["dataset", "train_k", "variant"]).to_csv(trial_dir / "compare_to_20250916.tsv", index=False)
    if not agg.empty:
        agg.sort_values(["dataset", "train_k", "variant"]).to_csv(trial_dir / "compare_to_20250916_agg.tsv", index=False)

    # 4) Summary text
    write_summary_ascii(agg, trial_dir)

    print(f"Wrote: {live_out}")
    if not detailed.empty:
        print(f"Wrote: {trial_dir / 'compare_to_20250916.tsv'}")
    if not agg.empty:
        print(f"Wrote: {trial_dir / 'compare_to_20250916_agg.tsv'}")
        print(f"Wrote: {trial_dir / 'summary_text.txt'}")


if __name__ == "__main__":
    main()

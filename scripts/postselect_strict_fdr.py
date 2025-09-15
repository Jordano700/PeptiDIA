#!/usr/bin/env python3
"""
Select "best" trials under a strict Actual FDR constraint (<= primary_fdr).

Reads a results folder with `trials/` JSON files and produces:
  - summary_best_by_variant_strict.csv
  - best_overall_fdr_summary_strict.csv
  - best_<variant>_strict.json (one per variant found)

Usage:
  python scripts/postselect_strict_fdr.py --results-dir results/AUTO_TABULAR_YYYYMMDD_HHMMSS [--primary-fdr 1.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_primary_fdr(results_dir: Path) -> float:
    meta = results_dir / "run_metadata.json"
    if meta.exists():
        try:
            d = json.loads(meta.read_text())
            return float(d.get("primary_fdr", 1.0))
        except Exception:
            pass
    return 1.0


def main():
    ap = argparse.ArgumentParser(description="Post-select best trials under Actual FDR <= primary FDR")
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--primary-fdr", type=float, default=None)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    primary_fdr = args.primary_fdr if args.primary_fdr is not None else load_primary_fdr(results_dir)

    trials_dir = results_dir / "trials"
    tables_dir = results_dir / "tables"
    if not trials_dir.exists():
        raise SystemExit(f"No trials dir found: {trials_dir}")

    best_by_variant: Dict[str, Dict] = {}
    for p in sorted(trials_dir.glob("*_trial_*.json")):
        d = json.loads(p.read_text())
        var = d.get("variant", "unknown").lower()
        row = d.get("selected_row", {}) or {}
        if float(row.get("Target_FDR", -1)) <= 0:
            continue
        if abs(float(row.get("Target_FDR", 0)) - primary_fdr) > 1e-6:
            continue
        actual = float(row.get("Actual_FDR", 1e9))
        if actual > primary_fdr:
            # violates strict constraint
            continue
        addp = int(row.get("Additional_Peptides", 0) or 0)
        mcc = float(row.get("MCC", 0) or 0)
        key = (addp, mcc)
        prev = best_by_variant.get(var)
        if prev is None:
            best_by_variant[var] = d
        else:
            prev_row = prev.get('selected_row', {}) or {}
            prev_key = (int(prev_row.get('Additional_Peptides', 0) or 0), float(prev_row.get('MCC', 0) or 0))
            if key > prev_key:
                best_by_variant[var] = d

    # Write per-variant JSON and build summary
    rows = []
    for var, payload in best_by_variant.items():
        out_json = results_dir / f"best_{var}_strict.json"
        out_json.write_text(json.dumps(payload, indent=2))
        r = payload.get("selected_row", {})
        rows.append({
            "variant": var,
            "score": payload.get("score"),
            "add_peptides": r.get("Additional_Peptides"),
            "actual_fdr": r.get("Actual_FDR"),
            "mcc": r.get("MCC"),
        })
    if rows:
        sdf = pd.DataFrame(rows).sort_values(["add_peptides", "mcc"], ascending=False)
        sdf.to_csv(results_dir / "summary_best_by_variant_strict.csv", index=False)

        # Overall best under strict rule
        best = sdf.iloc[0]
        # try to locate the matching trial table to write a mini FDR summary
        # The trial id is embedded in the JSON filename, so we need to find it.
        # We re-scan for the exact matching payload by variant and add_peptides.
        chosen_payload: Optional[Dict] = None
        for p in sorted(trials_dir.glob(f"{best['variant']}_trial_*.json")):
            d = json.loads(p.read_text())
            r = d.get("selected_row", {})
            if int(r.get("Additional_Peptides", -1)) == int(best["add_peptides"]) and abs(float(r.get("Actual_FDR", 1e9)) - float(best["actual_fdr"])) < 1e-9:
                chosen_payload = d
                # write the best strict json as well
                (results_dir / f"best_{best['variant']}_strict.json").write_text(json.dumps(d, indent=2))
                # also try copying the CSV FDR sweep for this trial
                stem = p.stem.replace(".json", "")
                csv_path = tables_dir / f"{stem}_results.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        cols = ["Target_FDR", "Additional_Peptides", "Recovery_Pct", "Actual_FDR", "Threshold"]
                        df[cols].to_csv(results_dir / "best_overall_fdr_summary_strict.csv", index=False)
                    except Exception:
                        pass
                break
        if chosen_payload is None:
            # fallback: write a minimal one-row CSV from the summary table
            pd.DataFrame([{k: best[k] for k in ["variant", "add_peptides", "actual_fdr", "mcc"]}]).to_csv(
                results_dir / "best_overall_fdr_summary_strict.csv", index=False
            )

    print(f"✅ Wrote strict best selections to: {results_dir}")


if __name__ == "__main__":
    main()

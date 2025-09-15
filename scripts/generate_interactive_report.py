#!/usr/bin/env python3
"""
Generate a multi-page interactive Plotly report for a PeptiDIA hparam search run.

Outputs a folder with multiple HTML pages:
  - index.html: Landing page with links and key metrics
  - overview.html: Best-by-variant, best-overall FDR sweep, and XGB n_estimators scatter + heatmap
  - xgb.html: Rich XGB exploration (scatter, heatmaps, parallel coords, distributions)
  - rf.html: RF exploration (grids and distributions)
  - lr.html: LR summary (simple due to limited variation)
  - fdr.html: Best-overall FDR sweep details
  - trials.html: Leaderboard table and distributions

Usage:
  python scripts/generate_interactive_report.py --results-dir results/HPARAM_SEARCH_PATHS_YYYYMMDD_HHMMSS
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_core_tables(results_dir: Path):
    summary_path = results_dir / "summary_best_by_variant.csv"
    best_fdr_path = results_dir / "best_overall_fdr_summary.csv"
    leaderboard_path = results_dir / "all_trials_leaderboard.csv"
    summary_df = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    best_fdr_df = pd.read_csv(best_fdr_path) if best_fdr_path.exists() else pd.DataFrame()
    leaderboard_df = pd.read_csv(leaderboard_path) if leaderboard_path.exists() else pd.DataFrame()
    return summary_df, best_fdr_df, leaderboard_df


def load_trials(results_dir: Path) -> Dict[str, pd.DataFrame]:
    trials_dir = results_dir / "trials"
    data: Dict[str, List[Dict]] = {"xgb": [], "rf": [], "lr": [], "ensemble": []}
    if not trials_dir.exists():
        return {k: pd.DataFrame() for k in data}

    for p in sorted(trials_dir.glob("*_trial_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        variant = d.get("variant", "").lower()
        row = d.get("selected_row", {}) or {}
        base = {
            "file": p.name,
            "variant": variant,
            "trial": d.get("trial"),
            "score": d.get("score"),
            "target_fdr": d.get("primary_fdr"),
            "additional_peptides": row.get("Additional_Peptides"),
            "actual_fdr": row.get("Actual_FDR"),
            "mcc": row.get("MCC"),
            "threshold": row.get("Threshold"),
        }
        if variant == "xgb":
            base.update({f"xgb_{k}": v for k, v in (d.get("xgb_params") or {}).items()})
        if variant == "rf":
            base.update({f"rf_{k}": v for k, v in (d.get("rf_params") or {}).items()})
        if variant == "lr":
            base.update({f"lr_{k}": v for k, v in (d.get("lr_params") or {}).items()})
        if variant == "ensemble":
            # Keep sub-model params in case of diagnostics
            base.update({f"xgb_{k}": v for k, v in (d.get("xgb_params") or {}).items()})
            base.update({f"rf_{k}": v for k, v in (d.get("rf_params") or {}).items()})
            base.update({f"lr_{k}": v for k, v in (d.get("lr_params") or {}).items()})
        if variant in data:
            data[variant].append(base)

    return {k: pd.DataFrame(v) for k, v in data.items()}


def html_wrap(fig: go.Figure, title: str, nav: str) -> str:
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin: 0; padding: 0; }}
    .nav {{ background:#111827; color:#fff; padding: 10px 16px; position: sticky; top:0; z-index:10; }}
    .nav a {{ color:#93c5fd; margin-right: 14px; text-decoration:none; font-weight:600; }}
    .nav a:hover {{ text-decoration: underline; }}
    .container {{ padding: 16px; }}
    h1 {{ margin: 8px 0 16px; }}
    .meta {{ color:#374151; font-size: 14px; margin-bottom: 12px; }}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
  {nav}
  <script>
    // no-op; plotly loaded via CDN in body content produced by fig.to_html
  </script>
  <body>
    <div class="nav">
      <a href="index.html">Overview</a>
      <a href="overview.html">Summary</a>
      <a href="xgb.html">XGB</a>
      <a href="rf.html">RF</a>
      <a href="lr.html">LR</a>
      <a href="fdr.html">FDR Sweep</a>
      <a href="trials.html">Leaderboard</a>
    </div>
    <div class="container">
      <h1>{title}</h1>
      {fig.to_html(include_plotlyjs=False, full_html=False)}
    </div>
  </body>
</html>
"""


def write_html(path: Path, html: str):
    path.write_text(html, encoding="utf-8")


def make_overview_pages(results_dir: Path, summary_df: pd.DataFrame, best_fdr_df: pd.DataFrame, trials: Dict[str, pd.DataFrame], out_dir: Path):
    nav = ""  # no extra header scripts
    # Overview landing: key metrics and links
    cards = []
    if not summary_df.empty:
        best_row = summary_df.sort_values('score', ascending=False).iloc[0]
        cards.append(f"<div><b>Best Variant:</b> {best_row['variant']} &mdash; {int(best_row['add_peptides'])} peptides @ {best_row['actual_fdr']:.2f}% FDR</div>")
    if not best_fdr_df.empty:
        cards.append(f"<div><b>FDR sweep points:</b> {len(best_fdr_df)}</div>")
    total_trials = sum(len(df) for df in trials.values())
    if total_trials:
        cards.append(f"<div><b>Total trials:</b> {total_trials}</div>")
    html = f"""
<!DOCTYPE html><html><head>
  <meta charset='utf-8'/>
  <title>PeptiDIA Report — Overview</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif; margin:0; }}
    .nav {{ background:#111827; color:#fff; padding: 10px 16px; position: sticky; top:0; }}
    .nav a {{ color:#93c5fd; margin-right: 14px; text-decoration:none; font-weight:600; }}
    .container {{ padding: 18px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap:12px; margin-top:12px; }}
    .card {{ border:1px solid #e5e7eb; border-radius:8px; padding:12px; background:#fff; }}
    h1 {{ margin: 8px 0 4px; }}
    ul {{ line-height:1.7; }}
  </style>
</head>
<body>
  <div class="nav">
    <a href="index.html">Overview</a>
    <a href="overview.html">Summary</a>
    <a href="xgb.html">XGB</a>
    <a href="rf.html">RF</a>
    <a href="lr.html">LR</a>
    <a href="fdr.html">FDR Sweep</a>
    <a href="trials.html">Leaderboard</a>
  </div>
  <div class="container">
    <h1>PeptiDIA Hyperparameter Search — Report</h1>
    <div class="grid">
      {''.join(f'<div class="card">{c}</div>' for c in cards)}
    </div>
    <div class="card" style="margin-top:14px;">
      <b>Pages</b>
      <ul>
        <li><a href="overview.html">Summary charts</a></li>
        <li><a href="xgb.html">XGB deep dive</a></li>
        <li><a href="rf.html">RF exploration</a></li>
        <li><a href="lr.html">LR summary</a></li>
        <li><a href="fdr.html">FDR sweep</a></li>
        <li><a href="trials.html">All-trials leaderboard & distributions</a></li>
      </ul>
    </div>
  </div>
</body></html>
"""
    write_html(out_dir / "index.html", html)

    # Summary page: combined subplots
    fig = make_subplots(
        rows=2, cols=2, specs=[[{"type":"xy"}, {"secondary_y": True}], [{"type":"xy"}, {"type":"heatmap"}]],
        subplot_titles=(
            'Best Additional Peptides by Variant @ 1% target FDR',
            'Best-Overall FDR Sweep',
            'XGB: Additional Peptides vs n_estimators',
            'XGB: Mean Additional Peptides (n_estimators × max_depth)'
        )
    )
    # Bars
    if not summary_df.empty:
        order = summary_df.sort_values('score', ascending=False)
        fig.add_trace(go.Bar(x=order['variant'], y=order['add_peptides'], marker_color=px.colors.qualitative.Set2,
                             customdata=np.c_[order['actual_fdr'], order['mcc']],
                             hovertemplate='Variant: %{x}<br>Additional: %{y}<br>Actual FDR: %{customdata[0]:.2f}%<br>MCC: %{customdata[1]:.3f}<extra></extra>'),
                      row=1, col=1)
        fig.update_yaxes(title_text='Additional Peptides', row=1, col=1)
    # FDR sweep
    if not best_fdr_df.empty:
        fig.add_trace(go.Scatter(x=best_fdr_df['Target_FDR'], y=best_fdr_df['Additional_Peptides'], name='Additional', mode='lines+markers', line=dict(color='royalblue', width=3)), row=1, col=2, secondary_y=False)
        fig.add_trace(go.Scatter(x=best_fdr_df['Target_FDR'], y=best_fdr_df['Actual_FDR'], name='Actual FDR', mode='lines+markers', line=dict(color='firebrick', width=3)), row=1, col=2, secondary_y=True)
        fig.update_xaxes(title_text='Target FDR (%)', row=1, col=2)
        fig.update_yaxes(title_text='Additional Peptides', row=1, col=2, secondary_y=False)
        fig.update_yaxes(title_text='Actual FDR (%)', row=1, col=2, secondary_y=True)
    # XGB scatter + heatmap
    xgb_df = trials.get('xgb', pd.DataFrame())
    if not xgb_df.empty:
        fig.add_trace(go.Scatter(x=xgb_df['xgb_n_estimators'], y=xgb_df['additional_peptides'], mode='markers',
                                 marker=dict(color=xgb_df['actual_fdr'], colorscale='Viridis', showscale=True,
                                             colorbar=dict(title='Actual FDR (%)', x=0.47, len=0.45), size=8),
                                 hovertemplate='n_estimators: %{x}<br>Additional: %{y}<br>Actual FDR: %{marker.color:.2f}%<extra></extra>'),
                      row=2, col=1)
        piv = xgb_df.pivot_table(index='xgb_max_depth', columns='xgb_n_estimators', values='additional_peptides', aggfunc='mean')
        if isinstance(piv, pd.DataFrame) and len(piv) and len(piv.columns):
            piv = piv.sort_index().reindex(sorted(piv.columns), axis=1)
            fig.add_trace(go.Heatmap(z=piv.values, x=piv.columns.astype(str), y=piv.index.astype(str), colorscale='Blues', colorbar=dict(title='Mean Add. Peptides', len=0.45, x=1.02), hovertemplate='max_depth: %{y}<br>n_estimators: %{x}<br>Mean add.: %{z:.1f}<extra></extra>'), row=2, col=2)
    fig.update_layout(title='Summary', template='plotly_white', height=950, margin=dict(l=60, r=60, t=60, b=60))
    write_html(out_dir / "overview.html", html_wrap(fig, "Summary", nav))


def make_xgb_page(xgb: pd.DataFrame, out_dir: Path):
    nav = ""
    if xgb.empty:
        write_html(out_dir / "xgb.html", html_wrap(go.Figure(), "XGB (no trials found)", nav))
        return

    # Scatter with rich hover and facet by learning_rate
    fig1 = px.scatter(
        xgb, x='xgb_n_estimators', y='additional_peptides', color='actual_fdr', color_continuous_scale='Viridis',
        symbol='xgb_max_depth', size='xgb_learning_rate', facet_col='xgb_min_child_weight', facet_col_wrap=3,
        hover_data=['trial', 'score', 'xgb_max_depth', 'xgb_learning_rate', 'xgb_subsample', 'xgb_colsample_bytree', 'xgb_gamma', 'xgb_reg_alpha', 'xgb_reg_lambda'],
        title='XGB Trials — Additional Peptides vs n_estimators (facet by min_child_weight)'
    )
    fig1.update_layout(height=900, coloraxis_colorbar=dict(title='Actual FDR (%)'))

    # Heatmaps: mean Additional and mean Actual FDR by (n_estimators, max_depth)
    piv_add = xgb.pivot_table(index='xgb_max_depth', columns='xgb_n_estimators', values='additional_peptides', aggfunc='mean')
    piv_fdr = xgb.pivot_table(index='xgb_max_depth', columns='xgb_n_estimators', values='actual_fdr', aggfunc='mean')
    piv_add = piv_add.sort_index().reindex(sorted(piv_add.columns), axis=1)
    piv_fdr = piv_fdr.loc[piv_add.index, piv_add.columns]
    fig2 = go.Figure(go.Heatmap(z=piv_add.values, x=piv_add.columns.astype(str), y=piv_add.index.astype(str), colorscale='Blues', colorbar=dict(title='Mean Add. Peptides')))
    fig2.update_layout(title='Mean Additional Peptides — n_estimators × max_depth', xaxis_title='n_estimators', yaxis_title='max_depth', height=500, template='plotly_white')
    fig3 = go.Figure(go.Heatmap(z=piv_fdr.values, x=piv_fdr.columns.astype(str), y=piv_fdr.index.astype(str), colorscale='Viridis', colorbar=dict(title='Mean Actual FDR (%)')))
    fig3.update_layout(title='Mean Actual FDR — n_estimators × max_depth', xaxis_title='n_estimators', yaxis_title='max_depth', height=500, template='plotly_white')

    # Parallel coordinates over key hyperparams colored by Additional Peptides
    # Normalize columns for parallel coords
    pc_cols = ['xgb_n_estimators','xgb_max_depth','xgb_learning_rate','xgb_subsample','xgb_colsample_bytree','xgb_min_child_weight','xgb_gamma','xgb_reg_alpha','xgb_reg_lambda']
    df_pc = xgb.dropna(subset=['additional_peptides']).copy()
    dims = []
    for c in pc_cols:
        if c not in df_pc.columns:
            continue
        s = df_pc[c]
        dims.append(dict(label=c.replace('xgb_',''), values=s))
    fig4 = go.Figure(data=go.Parcoords(
        line=dict(color=df_pc['additional_peptides'], colorscale='Blues', showscale=True, colorbar=dict(title='Additional')), dimensions=dims
    ))
    fig4.update_layout(title='Parallel Coordinates — XGB Hyperparams vs Additional Peptides', height=500, template='plotly_white')

    html = "".join([
        html_wrap(fig1, "XGB — Scatter (faceted)", nav),
        html_wrap(fig2, "XGB — Heatmap (Additional)", nav),
        html_wrap(fig3, "XGB — Heatmap (Actual FDR)", nav),
        html_wrap(fig4, "XGB — Parallel Coordinates", nav),
    ])
    write_html(out_dir / "xgb.html", html)


def make_rf_page(rf: pd.DataFrame, out_dir: Path):
    nav = ""
    if rf.empty:
        write_html(out_dir / "rf.html", html_wrap(go.Figure(), "RF (no trials found)", nav))
        return
    # Heatmap by (n_estimators, max_depth)
    piv_add = rf.pivot_table(index='rf_max_depth', columns='rf_n_estimators', values='additional_peptides', aggfunc='mean')
    piv_add = piv_add.sort_index().reindex(sorted(piv_add.columns), axis=1)
    fig1 = go.Figure(go.Heatmap(z=piv_add.values, x=piv_add.columns.astype(str), y=piv_add.index.astype(str), colorscale='Blues', colorbar=dict(title='Mean Add. Peptides')))
    fig1.update_layout(title='RF — Mean Additional Peptides (n_estimators × max_depth)', xaxis_title='n_estimators', yaxis_title='max_depth', height=500, template='plotly_white')
    # Scatter by n_estimators vs additional
    fig2 = px.scatter(rf, x='rf_n_estimators', y='additional_peptides', color='actual_fdr', color_continuous_scale='Viridis', symbol='rf_max_depth', hover_data=['trial','score','rf_min_samples_split','rf_min_samples_leaf'])
    fig2.update_layout(title='RF Trials — Additional Peptides vs n_estimators', height=500, template='plotly_white', coloraxis_colorbar=dict(title='Actual FDR (%)'))
    html = html_wrap(fig1, "RF — Heatmap", nav) + html_wrap(fig2, "RF — Scatter", nav)
    write_html(out_dir / "rf.html", html)


def make_lr_page(lr: pd.DataFrame, out_dir: Path):
    nav = ""
    if lr.empty:
        write_html(out_dir / "lr.html", html_wrap(go.Figure(), "LR (no trials found)", nav))
        return
    fig1 = px.box(lr, x='lr_C', y='additional_peptides', points='all', hover_data=['trial','score'])
    fig1.update_layout(title='LR — Additional Peptides vs C', xaxis_title='C', yaxis_title='Additional Peptides', template='plotly_white', height=500)
    html = html_wrap(fig1, "LR — Summary", nav)
    write_html(out_dir / "lr.html", html)


def make_fdr_page(best_fdr_df: pd.DataFrame, out_dir: Path):
    nav = ""
    if best_fdr_df.empty:
        write_html(out_dir / "fdr.html", html_wrap(go.Figure(), "FDR Sweep (no data)", nav))
        return
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=best_fdr_df['Target_FDR'], y=best_fdr_df['Additional_Peptides'], mode='lines+markers', name='Additional', line=dict(color='royalblue', width=3)), secondary_y=False)
    fig.add_trace(go.Scatter(x=best_fdr_df['Target_FDR'], y=best_fdr_df['Actual_FDR'], mode='lines+markers', name='Actual FDR', line=dict(color='firebrick', width=3)), secondary_y=True)
    fig.update_xaxes(title_text='Target FDR (%)')
    fig.update_yaxes(title_text='Additional Peptides', secondary_y=False)
    fig.update_yaxes(title_text='Actual FDR (%)', secondary_y=True)
    fig.update_layout(title='Best-Overall FDR Sweep', height=500, template='plotly_white')
    write_html(out_dir / "fdr.html", html_wrap(fig, "FDR Sweep", nav))


def make_leaderboard_page(leaderboard_df: pd.DataFrame, out_dir: Path):
    nav = ""
    if leaderboard_df.empty:
        write_html(out_dir / "trials.html", html_wrap(go.Figure(), "Leaderboard (no data)", nav))
        return
    # Table
    df = leaderboard_df.copy()
    # truncate for display but keep all rows
    table = go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color='#111827', font=dict(color='white', size=12), align='left'),
        cells=dict(values=[df[c] for c in df.columns], fill_color='white', align='left'))
    ])
    table.update_layout(title='All Trials Leaderboard', height=700, template='plotly_white')
    # Distribution by variant
    dist = px.box(df, x='variant', y='add_peptides', points='all', color='variant', title='Distribution of Additional Peptides by Variant')
    dist.update_layout(template='plotly_white', height=600)
    html = html_wrap(table, "Trials Leaderboard", nav) + html_wrap(dist, "Trials Distribution", nav)
    write_html(out_dir / "trials.html", html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True, help='Path to a HPARAM_SEARCH_PATHS_* results folder')
    args = ap.parse_args()

    results_dir = Path(args.results_dir).resolve()
    out_dir = results_dir / 'interactive_report'
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_df, best_fdr_df, leaderboard_df = load_core_tables(results_dir)
    trials = load_trials(results_dir)

    make_overview_pages(results_dir, summary_df, best_fdr_df, trials, out_dir)
    make_xgb_page(trials.get('xgb', pd.DataFrame()), out_dir)
    make_rf_page(trials.get('rf', pd.DataFrame()), out_dir)
    make_lr_page(trials.get('lr', pd.DataFrame()), out_dir)
    make_fdr_page(best_fdr_df, out_dir)
    make_leaderboard_page(leaderboard_df, out_dir)

    print(f"✅ Interactive report written to: {out_dir}")


if __name__ == '__main__':
    main()


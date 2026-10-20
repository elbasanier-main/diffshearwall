#!/usr/bin/env python3
"""
step1_analyze_distribution.py
==============================
Phase 1, Step 1: Read all raw drift values from training set, compute
statistics and detect features that determine which single-step
normalization is feasible.

Output:
  - Console summary: percentiles, min/max, zeros, negative count
  - distribution_stats.json: machine-readable summary
  - distribution_hist.png: histograms in raw / log / sqrt space

Run:
  python step1_analyze_distribution.py \
      --raw-folder /home/elbasanier/CGVAE_RL_Tran/disc_diffusion/data/all_before/ \
      --output-dir /home/elbasanier/CGVAE_RL_Tran/disc_diffusion/analysis_singlestep/

DOES NOT modify any project file. Read-only.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def load_all_drifts(folder: Path):
    x_all, y_all = [], []
    n_files, n_skipped = 0, 0
    n_zeros_x, n_zeros_y, n_neg_x, n_neg_y = 0, 0, 0, 0

    for fp in sorted(folder.glob('*.json')):
        try:
            with open(fp, 'r') as f:
                data = json.load(f)
        except Exception:
            n_skipped += 1
            continue

        n_files += 1
        for fe in data.get('File_details', []):
            for s in fe.get('Story_detail', []):
                x = s.get('X-Dir')
                y = s.get('Y-Dir')
                if x is None or y is None:
                    continue
                x = float(x)
                y = float(y)
                if x == 0:
                    n_zeros_x += 1
                if y == 0:
                    n_zeros_y += 1
                if x < 0:
                    n_neg_x += 1
                if y < 0:
                    n_neg_y += 1
                x_all.append(x)
                y_all.append(y)

    return {
        'x': np.asarray(x_all),
        'y': np.asarray(y_all),
        'n_files': n_files,
        'n_skipped': n_skipped,
        'n_zeros_x': n_zeros_x,
        'n_zeros_y': n_zeros_y,
        'n_neg_x': n_neg_x,
        'n_neg_y': n_neg_y,
    }


def summarize(values: np.ndarray, label: str):
    if len(values) == 0:
        print(f'  {label}: NO VALUES')
        return {}

    pos = values[values > 0]
    pctls = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]
    stats = {
        'count': int(len(values)),
        'count_positive': int(len(pos)),
        'min': float(values.min()),
        'max': float(values.max()),
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'std': float(values.std()),
        'percentiles': {f'P{p}': float(np.percentile(values, p)) for p in pctls},
    }
    if len(pos) > 0:
        stats['log10_range_orders'] = float(np.log10(pos.max() / pos.min()))
        stats['ratio_max_min'] = float(pos.max() / pos.min())

    print(f'\n  {label}:  n={stats["count"]}')
    print(f'    min  = {stats["min"]:.4e}    max = {stats["max"]:.4e}')
    print(f'    mean = {stats["mean"]:.4e}    std = {stats["std"]:.4e}')
    for p in pctls:
        print(f'    P{p:>5} = {stats["percentiles"][f"P{p}"]:.4e}')
    if 'log10_range_orders' in stats:
        print(f'    Dynamic range: {stats["log10_range_orders"]:.2f} orders of magnitude')
        print(f'    max/min ratio: {stats["ratio_max_min"]:.1f}')
    return stats


def assess_distribution(values: np.ndarray, label: str):
    """Quick assessment for single-step feasibility."""
    if len(values) == 0:
        return {}

    pos = values[values > 0]
    log_v = np.log10(pos)
    log_skew = float(((log_v - log_v.mean()) ** 3).mean() / log_v.std() ** 3)
    log_kurt = float(((log_v - log_v.mean()) ** 4).mean() / log_v.std() ** 4 - 3.0)

    z = (values - values.mean()) / values.std()
    raw_skew = float((z ** 3).mean())
    raw_kurt = float((z ** 4).mean() - 3.0)

    sqrt_v = np.sqrt(pos)
    z_sqrt = (sqrt_v - sqrt_v.mean()) / sqrt_v.std()
    sqrt_skew = float((z_sqrt ** 3).mean())
    sqrt_kurt = float((z_sqrt ** 4).mean() - 3.0)

    print(f'\n  {label} — DISTRIBUTION SHAPE (skewness, kurtosis):')
    print(f'    raw      : skew={raw_skew:+.2f}  excess kurt={raw_kurt:+.2f}'
          f'  {"(highly skewed)" if abs(raw_skew) > 1.5 else "(moderate)"}')
    print(f'    sqrt(raw): skew={sqrt_skew:+.2f}  excess kurt={sqrt_kurt:+.2f}')
    print(f'    log10    : skew={log_skew:+.2f}  excess kurt={log_kurt:+.2f}'
          f'  {"(close to lognormal)" if abs(log_skew) < 0.7 else "(not lognormal)"}')

    return {
        'raw_skew': raw_skew, 'raw_kurt': raw_kurt,
        'sqrt_skew': sqrt_skew, 'sqrt_kurt': sqrt_kurt,
        'log10_skew': log_skew, 'log10_kurt': log_kurt,
    }


def maybe_plot_histograms(values_x, values_y, out_path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('\n[INFO] matplotlib not available, skipping histogram plot.')
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for row, (label, vals) in enumerate([('X-Dir', values_x), ('Y-Dir', values_y)]):
        pos = vals[vals > 0]
        axes[row, 0].hist(vals, bins=80, color='steelblue')
        axes[row, 0].set_title(f'{label}: raw drift')
        axes[row, 0].set_xlabel('drift value')
        axes[row, 0].set_ylabel('count')

        axes[row, 1].hist(np.sqrt(pos), bins=80, color='seagreen')
        axes[row, 1].set_title(f'{label}: sqrt(drift)')
        axes[row, 1].set_xlabel('sqrt(drift)')

        axes[row, 2].hist(np.log10(pos), bins=80, color='indianred')
        axes[row, 2].set_title(f'{label}: log10(drift)')
        axes[row, 2].set_xlabel('log10(drift)')

    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f'\n[OK] Saved histogram: {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-folder', required=True,
                        help='Folder with raw GT JSONs (all_before/)')
    parser.add_argument('--output-dir', required=True,
                        help='Where to write distribution_stats.json and histogram PNG')
    args = parser.parse_args()

    raw_folder = Path(args.raw_folder)
    out_dir = Path(args.output_dir)
    if not raw_folder.is_dir():
        raise SystemExit(f'Raw folder not found: {raw_folder}')
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[INFO] Scanning {raw_folder}')
    data = load_all_drifts(raw_folder)

    print(f'[INFO] Loaded {data["n_files"]} JSON files, skipped {data["n_skipped"]}')
    print(f'[INFO] Total drift points: X={len(data["x"])}, Y={len(data["y"])}')
    print(f'[INFO] Zeros — X: {data["n_zeros_x"]}    Y: {data["n_zeros_y"]}')
    print(f'[INFO] Negatives — X: {data["n_neg_x"]}    Y: {data["n_neg_y"]}')

    stats_x = summarize(data['x'], 'X-Dir')
    stats_y = summarize(data['y'], 'Y-Dir')

    assess_x = assess_distribution(data['x'], 'X-Dir')
    assess_y = assess_distribution(data['y'], 'Y-Dir')

    print('\n' + '=' * 70)
    print('FEASIBILITY HINTS')
    print('=' * 70)
    if data['n_zeros_x'] + data['n_zeros_y'] > 0:
        print('  ! Zeros present — log-transform NOT viable without offset/clip.')
    if data['n_neg_x'] + data['n_neg_y'] > 0:
        print('  ! Negatives present — sqrt/log undefined for these values.')

    for label, s, a in [('X-Dir', stats_x, assess_x), ('Y-Dir', stats_y, assess_y)]:
        if not s:
            continue
        orders = s.get('log10_range_orders', 0)
        skew_log = a.get('log10_skew', 999)
        print(f'\n  {label}:')
        print(f'    dynamic range = {orders:.2f} orders of magnitude')
        if orders < 1.5:
            print(f'      → narrow enough that z-score may work directly')
        elif orders < 3.0:
            print(f'      → sqrt or log both feasible')
        else:
            print(f'      → wide range — log is preferred over sqrt')
        if abs(skew_log) < 0.7:
            print(f'    log10 skew = {skew_log:+.2f} → log-normal distribution, '
                  f'log-transform is best choice')
        elif abs(skew_log) < 1.5:
            print(f'    log10 skew = {skew_log:+.2f} → moderate, log still helps')
        else:
            print(f'    log10 skew = {skew_log:+.2f} → log transform less effective')

    out_stats = {
        'n_files': data['n_files'],
        'n_skipped': data['n_skipped'],
        'n_zeros_x': data['n_zeros_x'],
        'n_zeros_y': data['n_zeros_y'],
        'n_neg_x': data['n_neg_x'],
        'n_neg_y': data['n_neg_y'],
        'x': {**stats_x, **assess_x} if stats_x else {},
        'y': {**stats_y, **assess_y} if stats_y else {},
    }
    out_json = out_dir / 'distribution_stats.json'
    with open(out_json, 'w') as f:
        json.dump(out_stats, f, indent=2)
    print(f'\n[OK] Statistics saved: {out_json}')

    maybe_plot_histograms(data['x'], data['y'], out_dir / 'distribution_hist.png')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
step2_simulate_candidates.py
=============================
Phase 1, Step 2: simulate expected raw-MAPE for each candidate
normalization on the true raw drift distribution.

Methodology:
  1. Load distribution_stats.json from step1.
  2. Sample synthetic GT raw values from a fitted log-normal that
     matches the real percentiles.
  3. For each candidate:
       forward(GT)   -> y_norm_true
       y_norm_pred = y_norm_true + Gaussian noise (std = ratio * range)
       inverse(y_norm_pred) -> y_raw_pred
       measure MAPE(y_raw_pred, GT)
  4. Report mean and std across trials.

DOES NOT train any model. Pure offline simulation.

Usage:
  python step2_simulate_candidates.py \
      --stats-json analysis_singlestep/distribution_stats.json \
      --output-dir analysis_singlestep/

The simulator runs for several noise levels (1%, 2%, 3.6%, 5%) so the
output table shows how each candidate behaves under stronger or weaker
model error.
"""

import argparse
import json
from pathlib import Path

import numpy as np

ORIG_SCALE = 1e7


# ---------------------------------------------------------------------------
# Forward / inverse for each candidate
# ---------------------------------------------------------------------------
def fwd_sqrt_2step(y):
    s1 = np.sqrt(y * ORIG_SCALE) / 100
    return np.sqrt(s1 * ORIG_SCALE) / 100


def inv_sqrt_2step(yn):
    s1 = (yn * 100) ** 2 / ORIG_SCALE
    return (s1 * 100) ** 2 / ORIG_SCALE


def fwd_sqrt_4step(y, norm=(100.0, 100.0, 1000.0, 250.0)):
    v = y
    for c in norm:
        v = np.sqrt(v * ORIG_SCALE) / c
    return v


def inv_sqrt_4step(yn, norm=(100.0, 100.0, 1000.0, 250.0)):
    v = yn
    for c in reversed(norm):
        v = (v * c) ** 2 / ORIG_SCALE
    return v


def fwd_log10_scaled(y, scale, shift=7):
    return (np.log10(y) + shift) * scale


def inv_log10_scaled(yn, scale, shift=7):
    return 10 ** (yn / scale - shift)


def fwd_ln_zscore(y, mu, sigma):
    return (np.log(y) - mu) / sigma


def inv_ln_zscore(yn, mu, sigma):
    return np.exp(yn * sigma + mu)


# ---------------------------------------------------------------------------
# Distribution sampling
# ---------------------------------------------------------------------------
def sample_from_stats(axis_stats, n_samples, rng):
    """
    Sample y_raw from a log-normal that matches the real percentiles.

    The training data has log10_skew close to 0 (X: +0.36, Y: +0.32),
    so log-normal is a reasonable surrogate. We fit log_mean and log_std
    from P25/P50/P75 using a robust quantile match (IQR-based std).
    """
    p25 = axis_stats['percentiles']['P25']
    p50 = axis_stats['percentiles']['P50']
    p75 = axis_stats['percentiles']['P75']

    log_mean = np.log10(p50)
    log_std = (np.log10(p75) - np.log10(p25)) / 1.349  # IQR / 1.349 ≈ std for normal

    log_samples = rng.normal(log_mean, log_std, n_samples)
    y = 10 ** log_samples
    # Clip into real observed range to avoid extreme tails
    y = np.clip(y, axis_stats['min'], axis_stats['max'])
    return y, log_mean, log_std


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------
def simulate_candidate(name, fwd_fn, inv_fn, y_true, noise_ratio, n_trials, rng):
    yn_true = fwd_fn(y_true)
    yn_range = yn_true.max() - yn_true.min()
    sigma_abs = noise_ratio * yn_range

    raw_mapes = []
    norm_mapes = []
    for _ in range(n_trials):
        noise = rng.normal(0.0, sigma_abs, size=y_true.shape)
        yn_pred = yn_true + noise
        # Inverse — guard against pathological exp() blowup
        try:
            y_pred = inv_fn(yn_pred)
        except (OverflowError, FloatingPointError):
            y_pred = np.full_like(y_true, np.nan)
        y_pred = np.where(np.isfinite(y_pred), y_pred, y_true.max() * 10)
        y_pred = np.clip(y_pred, 1e-15, 1.0)

        raw_mape = np.mean(np.abs(y_pred - y_true) / y_true) * 100
        norm_mape = np.mean(np.abs(yn_pred - yn_true) /
                            np.where(np.abs(yn_true) > 1e-9, np.abs(yn_true), 1)) * 100
        raw_mapes.append(raw_mape)
        norm_mapes.append(norm_mape)

    return {
        'name': name,
        'yn_range': float(yn_range),
        'sigma_abs': float(sigma_abs),
        'raw_mape_mean': float(np.mean(raw_mapes)),
        'raw_mape_std': float(np.std(raw_mapes)),
        'norm_mape_mean': float(np.mean(norm_mapes)),
    }


def run_simulation(axis_name, axis_stats, n_samples, n_trials, noise_ratios, rng):
    y_true, log_mean, log_std = sample_from_stats(axis_stats, n_samples, rng)
    ln_mean = log_mean * np.log(10)
    ln_std = log_std * np.log(10)

    print(f'\n{"="*86}')
    print(f'  AXIS: {axis_name}   (n_samples={n_samples}, n_trials={n_trials})')
    print(f'  Data range: [{axis_stats["min"]:.3e}, {axis_stats["max"]:.3e}]')
    print(f'  log10: mean={log_mean:+.3f}  std={log_std:.3f}')
    print(f'{"="*86}')

    candidates = [
        ('sqrt-2step (current)',
         fwd_sqrt_2step, inv_sqrt_2step),
        ('sqrt-4step (paper)',
         fwd_sqrt_4step, inv_sqrt_4step),
        ('log10 x5',
         lambda y: fwd_log10_scaled(y, scale=5),
         lambda yn: inv_log10_scaled(yn, scale=5)),
        ('log10 x8',
         lambda y: fwd_log10_scaled(y, scale=8),
         lambda yn: inv_log10_scaled(yn, scale=8)),
        ('log10 x10',
         lambda y: fwd_log10_scaled(y, scale=10),
         lambda yn: inv_log10_scaled(yn, scale=10)),
        ('ln + zscore',
         lambda y: fwd_ln_zscore(y, ln_mean, ln_std),
         lambda yn: inv_ln_zscore(yn, ln_mean, ln_std)),
    ]

    results_by_noise = {}
    for nr in noise_ratios:
        print(f'\n  --- noise ratio {nr*100:.1f}% of yn_range ---')
        print(f'  {"Candidate":<24} | {"yn range":>9} | {"sigma_abs":>10} | '
              f'{"raw MAPE":>15} | {"norm MAPE":>10}')
        print(f'  {"-"*84}')
        results = []
        for name, fwd, inv in candidates:
            r = simulate_candidate(name, fwd, inv, y_true, nr, n_trials, rng)
            results.append(r)
            print(f'  {name:<24} | {r["yn_range"]:>9.2f} | {r["sigma_abs"]:>10.4f} | '
                  f'{r["raw_mape_mean"]:>10.2f}% ±{r["raw_mape_std"]:>4.2f} | '
                  f'{r["norm_mape_mean"]:>9.2f}%')
        results_by_noise[f'{nr*100:.1f}pct'] = results

    return results_by_noise, log_mean, log_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stats-json', required=True,
                        help='distribution_stats.json from step1')
    parser.add_argument('--output-dir', required=True,
                        help='Folder to save simulation results')
    parser.add_argument('--n-samples', type=int, default=5000,
                        help='Number of synthetic GT samples per axis')
    parser.add_argument('--n-trials', type=int, default=20,
                        help='Number of noise realizations per candidate')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    with open(args.stats_json, 'r') as f:
        stats = json.load(f)

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Noise ratios reflect realistic surrogate convergence (calibrated to your
    # current 4-step run: RMSE 1.80 on range ~50 → 3.6%).
    noise_ratios = [0.01, 0.02, 0.036, 0.05]

    all_results = {}
    for axis_name, key in [('X-Dir', 'x'), ('Y-Dir', 'y')]:
        axis_stats = stats[key]
        if not axis_stats:
            print(f'[WARN] no stats for {axis_name}, skipping.')
            continue
        rby, lm, ls = run_simulation(axis_name, axis_stats,
                                     args.n_samples, args.n_trials,
                                     noise_ratios, rng)
        all_results[axis_name] = {
            'log_mean': lm,
            'log_std': ls,
            'by_noise': rby,
        }

    out_path = out_dir / 'simulation_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n[OK] Results saved: {out_path}')

    # Summary recommendation
    print('\n' + '=' * 86)
    print('SUMMARY & RECOMMENDATION')
    print('=' * 86)
    for axis_name, info in all_results.items():
        target_noise = '3.6pct'   # realistic noise level
        if target_noise not in info['by_noise']:
            continue
        results = info['by_noise'][target_noise]
        best = min(results, key=lambda r: r['raw_mape_mean'])
        baseline = next((r for r in results if 'sqrt-2step' in r['name']), None)
        print(f'\n  {axis_name} (at realistic 3.6% norm-noise):')
        for r in results:
            tag = ' <-- best' if r is best else ''
            improvement = ''
            if baseline and r is not baseline:
                imp = (baseline['raw_mape_mean'] - r['raw_mape_mean']) \
                      / baseline['raw_mape_mean'] * 100
                improvement = f'  ({imp:+.1f}% vs sqrt-2step)'
            print(f'    {r["name"]:<24}: {r["raw_mape_mean"]:.2f}%{tag}{improvement}')


if __name__ == '__main__':
    main()

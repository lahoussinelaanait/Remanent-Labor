"""
Temporality Analysis of the R&D -> EBIT Relation
=================================================

This script tests the temporal signature of augmented remanent labor by
computing cross-correlations between R&D of year t and EBIT of year t+k,
for k = 0, 1, 2, 3, separately for each production regime (1-cycle and
2-cycle).

Under the framework of Laanait (2026), the 1-cycle regime consumes its
R&D within the year: R&D_t predicts EBIT_t, and the correlation decays
with k. The 2-cycle regime invests R&D in a long cycle whose fruits are
realized years later: R&D_t predicts EBIT_{t+k}, and the correlation
grows with k.

Metrics:
  - Pearson correlation (linear association)
  - Spearman correlation (monotone association)
  - Delta (variation of correlation relative to k=0)
  - Slope (rate of change per year of lag)

Reference:
  Laanait, L. (2026). Remanent Surplus Value: An Essay in the Political
  Economy of Cognitive Capitalism with a Proposed Empirical Measurement
  Instrument.

Repository: https://github.com/lahoussinelaanait/Remanent-Labor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. BUILD LAGGED EBIT COLUMNS
# ============================================================
def build_lagged_ebit(df: pd.DataFrame, max_lag: int = 3) -> pd.DataFrame:
    """Create EBIT_{t+k} columns for k = 1 .. max_lag, by firm."""
    df = df.sort_values(['firm_id', 'year']).reset_index(drop=True)
    for k in range(1, max_lag + 1):
        df[f'EBIT_fut{k}'] = df.groupby('firm_id')['EBIT'].shift(-k)
    return df


# ============================================================
# 2. CROSS-CORRELATION FOR ONE REGIME
# ============================================================
def cross_correlation(sub: pd.DataFrame,
                      regime_label: str,
                      max_lag: int = 3) -> dict:
    """Compute Pearson and Spearman correlations for k = 0 .. max_lag."""
    print(f"\n{'='*70}")
    print(f"REGIME {regime_label}  —  "
          f"{sub['firm_id'].nunique()} firms, {len(sub)} obs")
    print(f"{'='*70}")
    print(f"{'Lag k':<8} {'N pairs':<12} {'Pearson':<12} {'Spearman':<12}")
    print("-" * 50)

    results = {}
    for k in range(0, max_lag + 1):
        ebit_col = 'EBIT' if k == 0 else f'EBIT_fut{k}'
        work = sub.dropna(subset=['RD', ebit_col])
        if len(work) < 20:
            print(f"  k={k:<5} {len(work):<12} --")
            results[k] = {'n': len(work), 'pearson': np.nan,
                          'spearman': np.nan}
            continue

        rd = work['RD'].values.astype(float)
        ebit = work[ebit_col].values.astype(float)
        try:
            r_p, _ = pearsonr(rd, ebit)
            r_s, _ = spearmanr(rd, ebit)
        except Exception:
            r_p, r_s = np.nan, np.nan

        print(f"  k={k:<5} {len(work):<12} {r_p:<12.4f} {r_s:<12.4f}")
        results[k] = {'n': len(work), 'pearson': r_p, 'spearman': r_s}

    return results


# ============================================================
# 3. DELTAS AND SLOPES
# ============================================================
def compute_deltas(results: dict, max_lag: int = 3) -> dict:
    """Compute Delta(k) = r(k) - r(0) and slopes."""
    base_p = results[0]['pearson']
    base_s = results[0]['spearman']

    deltas = {}
    for k in range(1, max_lag + 1):
        deltas[k] = {
            'delta_pearson': results[k]['pearson'] - base_p,
            'delta_spearman': results[k]['spearman'] - base_s,
        }

    # Slopes: (r(k) - r(0)) / k for k = max_lag
    slope_p = (results[max_lag]['pearson'] - base_p) / max_lag
    slope_s = (results[max_lag]['spearman'] - base_s) / max_lag

    return {
        'deltas': deltas,
        'slope_pearson': slope_p,
        'slope_spearman': slope_s,
    }


# ============================================================
# 4. MAIN PIPELINE
# ============================================================
def run_pipeline(panel_path: Path,
                 output_dir: Path,
                 max_lag: int = 3) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("TEMPORALITY ANALYSIS — CROSS-CORRELATION RD_t ↔ EBIT_{t+k}")
    print(f"{'='*70}")
    print(f"Panel : {panel_path}")

    df = pd.read_csv(panel_path, sep=';', encoding='utf-8-sig')

    for c in ['Revenue', 'EBIT', 'RD', 'Personnel', 'PPE', 'year']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df[(df['Revenue'] > 0) & (df['RD'] > 0)]
    df = build_lagged_ebit(df, max_lag)

    print(f"\nTotal : {len(df)} obs, {df['firm_id'].nunique()} firms")

    # Cross-correlations per regime
    raw_results = {}
    for regime in ['SHORT', 'LONG']:
        sub = df[df['regime'] == regime]
        if len(sub) < 30:
            print(f"\n{regime} : insufficient data")
            continue
        raw_results[regime] = cross_correlation(sub, regime, max_lag)

    # Deltas and slopes
    derived = {}
    for regime, res in raw_results.items():
        derived[regime] = compute_deltas(res, max_lag)

    # ============================================================
    # SUMMARY TABLE 1 — Cross-correlations
    # ============================================================
    print(f"\n\n{'='*80}")
    print("TABLE 1 — CROSS-CORRELATION RD_t vs EBIT_{t+k}")
    print(f"{'='*80}")
    header = f"{'Lag k':<8}"
    for regime in raw_results:
        header += f"{regime + ' (P)':<14}{regime + ' (S)':<14}"
    print(header)
    print("-" * len(header))
    for k in range(max_lag + 1):
        row = f"k={k:<6}"
        for regime in raw_results:
            p = raw_results[regime][k]['pearson']
            s = raw_results[regime][k]['spearman']
            row += f"{p:<14.4f}{s:<14.4f}"
        print(row)

    # ============================================================
    # SUMMARY TABLE 2 — Deltas
    # ============================================================
    print(f"\n\n{'='*80}")
    print("TABLE 2 — DELTA RELATIVE TO k = 0")
    print(f"{'='*80}")
    header = f"{'Lag k':<8}"
    for regime in derived:
        header += f"{regime + ' (Pearson)':<18}{regime + ' (Spearman)':<18}"
    print(header)
    print("-" * len(header))
    for k in range(1, max_lag + 1):
        row = f"k={k:<6}"
        for regime in derived:
            dp = derived[regime]['deltas'][k]['delta_pearson']
            ds = derived[regime]['deltas'][k]['delta_spearman']
            row += f"{dp:<18.4f}{ds:<18.4f}"
        print(row)

    # ============================================================
    # SUMMARY TABLE 3 — Slopes
    # ============================================================
    print(f"\n\n{'='*80}")
    print("TABLE 3 — SLOPES (r(k) - r(0)) / k")
    print(f"{'='*80}")
    print(f"{'Regime':<10} {'Pearson slope':<18} {'Spearman slope':<18} "
          f"{'Sign':<10}")
    print("-" * 60)
    for regime, d in derived.items():
        sp = d['slope_pearson']
        ss = d['slope_spearman']
        sign = 'Positive' if sp > 0 else 'Negative'
        print(f"{regime:<10} {sp:<18.4f} {ss:<18.4f} {sign:<10}")

    # ============================================================
    # VERDICT
    # ============================================================
    print(f"\n\n{'='*80}")
    print("VERDICT — Temporal Signature of the Two Regimes")
    print(f"{'='*80}")

    max_k = {}
    for regime, res in raw_results.items():
        valid = {k: v['pearson'] for k, v in res.items()
                 if not np.isnan(v['pearson'])}
        if valid:
            max_k[regime] = max(valid, key=valid.get)

    for regime, k in max_k.items():
        print(f"  {regime:<8} : peak correlation at k = {k}")

    if len(max_k) == 2 and max_k.get('LONG', 0) > max_k.get('SHORT', 0):
        print(f"\n  ✅ Framework confirmed:")
        print(f"     The LONG regime peaks LATER than the SHORT regime "
              f"(k={max_k['LONG']} vs k={max_k['SHORT']})")
        print(f"     → R&D in the 2-cycle regime has a delayed effect;")
        print(f"       R&D in the 1-cycle regime has an immediate effect.")
    else:
        print(f"\n  ⚠ The framework is not confirmed by this test.")

    # ============================================================
    # SAVE
    # ============================================================
    summary = []
    for regime in raw_results:
        for k in range(max_lag + 1):
            summary.append({
                'regime': regime,
                'lag_k': k,
                'n_pairs': raw_results[regime][k]['n'],
                'pearson': raw_results[regime][k]['pearson'],
                'spearman': raw_results[regime][k]['spearman'],
            })
    out_csv = output_dir / 'temporality_results.csv'
    pd.DataFrame(summary).to_csv(out_csv, index=False)
    print(f"\n✅ Results saved: {out_csv}")

    # Save slopes
    slopes = []
    for regime, d in derived.items():
        slopes.append({
            'regime': regime,
            'slope_pearson': d['slope_pearson'],
            'slope_spearman': d['slope_spearman'],
        })
    out_slopes = output_dir / 'temporality_slopes.csv'
    pd.DataFrame(slopes).to_csv(out_slopes, index=False)
    print(f"✅ Slopes saved: {out_slopes}")


# ============================================================
# 5. MAIN
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Temporality analysis of R&D → EBIT by regime.")
    parser.add_argument(
        '--panel', type=Path,
        default=Path('/content/drive/My Drive/Github_Remanent/new_panel/'
                     'panel_LONG_vs_SHORT.csv'),
        help='Path to the panel CSV file.')
    parser.add_argument(
        '--out', type=Path,
        default=Path('/content/drive/My Drive/Github_Remanent/new_panel/'),
        help='Output directory.')
    parser.add_argument(
        '--max-lag', type=int, default=3,
        help='Maximum lag (default: 3).')
    args = parser.parse_args()
    run_pipeline(args.panel, args.out, args.max_lag)

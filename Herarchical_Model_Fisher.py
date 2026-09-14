# ============================================================
# FISHER Z TESTS
# Statistical comparison of MLE correlations across regimes
#
# Remanent Surplus Value — Lahoussine Laanait (2026)
# https://github.com/lahoussinelaanait/Remanent-Labor
#
# This script performs Fisher z-tests on the MLE correlations
# estimated by 01_hierarchical_mle.py. The test compares two
# correlations at a time and determines whether their difference
# is statistically significant.
#
# The Fisher z-transformation converts a correlation coefficient
# r into a variable z that is approximately normally distributed:
#
#     z = arctanh(r) = 0.5 * ln((1+r) / (1-r))
#
# The difference between two transformed correlations is then
# compared to its standard error:
#
#     z_diff = (z1 - z2) / sqrt(1/(n1-3) + 1/(n2-3))
#
# where n1 and n2 are the number of FIRMS (not observations) in
# each regime. Using firm-level n is essential: observations of
# the same firm over ten years are not independent, so using the
# number of observations would inflate the p-values.
#
# Input:  results/correlations/correlations_*.csv
# Output: results/fisher_z_<run_id>.csv
# ============================================================

import os
import glob
import itertools
from datetime import datetime

import pandas as pd
import numpy as np
from scipy.stats import norm

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = '../data/panel_FINAL_REMANENT.csv'
RESULTS_DIR = '../results'
CORR_DIR = f'{RESULTS_DIR}/correlations'
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

# Production regimes (SERVICES excluded by design)
REGIMES = ['TECH', 'PHARMA', 'INDUS']


# ============================================================
# DATA LOADING
# ============================================================

def load_panel(path):
    """Load the panel and convert numeric columns."""
    df = pd.read_csv(path, sep=';', encoding='utf-8')
    df.columns = df.columns.str.strip().str.lower()

    for col in ['year', 'ebit', 'personnel']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    return df


def load_mle_correlations():
    """
    Read the MLE correlations produced by 01_hierarchical_mle.py.
    Keep the latest run for each (group, period) and retain the
    global period (2015-2024).
    """
    pattern = f'{CORR_DIR}/correlations_*.csv'
    files = sorted(glob.glob(pattern))
    print(f"\n📂 Reading {len(files)} correlation files")

    rows = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
            needed = {'Var1', 'Var2', 'Correlation', 'group', 'period', 'run_id'}
            if not needed.issubset(tmp.columns):
                continue
            sel = tmp[
                ((tmp['Var1'] == 'EBIT') & (tmp['Var2'] == 'Personnel')) |
                ((tmp['Var1'] == 'Personnel') & (tmp['Var2'] == 'EBIT'))
            ]
            for _, r in sel.iterrows():
                rows.append((r['group'], str(r['period']), r['Correlation'], r['run_id']))
        except Exception as e:
            print(f"  ⚠ {f}: {e}")

    if not rows:
        return pd.DataFrame(columns=['group', 'period', 'corr_mle'])

    df = pd.DataFrame(rows, columns=['group', 'period', 'corr_mle', 'run_id'])
    df = df.sort_values('run_id').drop_duplicates(subset=['group', 'period'], keep='last')

    # Keep only the global period (2015-2024)
    global_df = df[df['period'] == '2015-2024'].copy()
    if len(global_df) == 0:
        global_df = df[~df['period'].str.isdigit()].copy()

    return global_df[['group', 'period', 'corr_mle']].reset_index(drop=True)


# ============================================================
# FISHER Z TEST
# ============================================================

def fisher_z(r1, n1, r2, n2):
    """
    Fisher z-test for the difference between two correlations.

    Parameters
    ----------
    r1, r2 : float
        The two correlations to compare.
    n1, n2 : int
        The number of FIRMS (not observations) in each regime.

    Returns
    -------
    z : float
        The z-statistic.
    p : float
        The two-sided p-value.
    """
    z1 = np.arctanh(r1)
    z2 = np.arctanh(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("FISHER Z TESTS — REMANENT SURPLUS VALUE")
    print("=" * 70)
    print(f"Run ID: {RUN_ID}")

    # --- Load panel ---
    df_panel = load_panel(DATA_PATH)
    print(f"\nPanel: {df_panel.shape[0]} rows, {df_panel['company'].nunique()} firms")

    # --- Firm counts per regime ---
    n_firms = {g: df_panel[df_panel['groupe'] == g]['company'].nunique()
               for g in REGIMES}

    print("\nFirms per regime:")
    for g in REGIMES:
        print(f"  {g:>6}: {n_firms[g]} firms")

    # --- Load MLE correlations ---
    df_mle = load_mle_correlations()
    if len(df_mle) == 0:
        print("\n⚠ No MLE correlations found. Run 01_hierarchical_mle.py first.")
        return

    mle_corr = dict(zip(df_mle['group'], df_mle['corr_mle']))
    print("\nMLE correlations (global period):")
    for g in REGIMES:
        v = mle_corr.get(g, np.nan)
        print(f"  {g:>6}: {v:.3f}")

    # --- Fisher z tests for all pairs ---
    print("\n" + "=" * 70)
    print("FISHER Z TESTS (n = firms)")
    print("=" * 70)

    rows = []
    for g1, g2 in itertools.combinations(REGIMES, 2):
        r1 = mle_corr.get(g1)
        r2 = mle_corr.get(g2)
        n1 = n_firms.get(g1)
        n2 = n_firms.get(g2)

        if r1 is None or r2 is None or n1 is None or n2 is None:
            print(f"  ⚠ Missing data for {g1} vs {g2}, skipping")
            continue

        z, p = fisher_z(r1, n1, r2, n2)
        sig = 'Yes' if p < 0.05 else 'No'

        print(f"  {g1:>6} vs {g2:<6}: z = {z:>6.2f}, p = {p:.4e}, significant = {sig}")

        rows.append({
            'group1': g1, 'r1': r1, 'n1': n1,
            'group2': g2, 'r2': r2, 'n2': n2,
            'z': z, 'p_value': p, 'significant': sig,
            'run_id': RUN_ID,
        })

    # --- Save ---
    if rows:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out = f'{RESULTS_DIR}/fisher_z_{RUN_ID}.csv'
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\n  ✓ Fisher z saved: {out}")

    print(f"\n{'=' * 70}")
    print(f"✅ Run {RUN_ID} complete.")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
# ============================================================
# FISHER Z TESTS AND BOOTSTRAP BY FIRM
# Remanent Surplus Value — Lahoussine Laanait (2026)
# ============================================================

import os
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import norm

DATA_PATH = '../data/panel_FINAL_REMANENT.csv'
OUTPUT_DIR = '../results'
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

# --- MLE correlations (from 01_hierarchical_mle.py) ---
MLE_CORR = {
    'TECH': 0.338,
    'INDUS': 0.616,
    'PHARMA': 0.643,
}


def load_panel(path):
    df = pd.read_csv(path, sep=';', encoding='utf-8')
    df.columns = df.columns.str.strip().str.lower()
    for col in ['year', 'ebit', 'personnel']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def n_firms(df, group):
    return df[df['groupe'] == group]['company'].nunique()


def fisher_z(r1, n1, r2, n2):
    z1, z2 = np.arctanh(r1), np.arctanh(r2)
    se = np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    z = (z1 - z2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


def bootstrap_correlation(df, group, n_boot=2000):
    """Resample firms (not observations) with replacement."""
    sub = df[(df['groupe'] == group) & (df['ebit'].notna()) & (df['personnel'].notna())]
    firms = sub['company'].unique()
    corrs = []
    for _ in range(n_boot):
        sample = np.random.choice(firms, size=len(firms), replace=True)
        boot = pd.concat([sub[sub['company'] == f] for f in sample])
        if len(boot) > 5:
            corrs.append(boot['personnel'].corr(boot['ebit']))
    return np.percentile(corrs, [2.5, 50, 97.5])


def main():
    print("=" * 70)
    print("FISHER Z TESTS AND BOOTSTRAP")
    print("=" * 70)

    df = load_panel(DATA_PATH)
    groups = ['TECH', 'INDUS', 'PHARMA']

    # --- Firm counts ---
    n = {g: n_firms(df, g) for g in groups}
    print("\nFirms per regime:")
    for g in groups:
        print(f"  {g}: {n[g]} firms")

    # --- Fisher z on firm-level n ---
    print("\n--- Fisher z tests (n = firms) ---")
    rows = []
    for g1, g2 in itertools.combinations(groups, 2):
        z, p = fisher_z(MLE_CORR[g1], n[g1], MLE_CORR[g2], n[g2])
        sig = 'Yes' if p < 0.05 else 'No'
        print(f"  {g1} vs {g2}: z = {z:.2f}, p = {p:.2e}, significant = {sig}")
        rows.append({'group1': g1, 'r1': MLE_CORR[g1], 'n1': n[g1],
                     'group2': g2, 'r2': MLE_CORR[g2], 'n2': n[g2],
                     'z': z, 'p_value': p, 'significant': sig})

    pd.DataFrame(rows).to_csv(f'{OUTPUT_DIR}/fisher_z_{RUN_ID}.csv', index=False)
    print(f"\n  ✓ Fisher z saved: {OUTPUT_DIR}/fisher_z_{RUN_ID}.csv")

    # --- Bootstrap by firm ---
    print("\n--- Bootstrap by firm (2000 iterations) ---")
    boot_rows = []
    for g in groups:
        lo, med, hi = bootstrap_correlation(df, g)
        print(f"  {g}: median = {med:.3f}, 95% CI = [{lo:.3f}, {hi:.3f}]")
        boot_rows.append({'group': g, 'median': med, 'ci_lower': lo, 'ci_upper': hi})

    pd.DataFrame(boot_rows).to_csv(f'{OUTPUT_DIR}/bootstrap_{RUN_ID}.csv', index=False)
    print(f"\n  ✓ Bootstrap saved: {OUTPUT_DIR}/bootstrap_{RUN_ID}.csv")


if __name__ == '__main__':
    main()
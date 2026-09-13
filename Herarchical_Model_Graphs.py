# ============================================================
# RECAPITULATIVE GRAPHS
# Reads the CSVs produced by 01_hierarchical_mle.py
# Produces PDF and PNG figures for correlations and margins
#
# Remanent Surplus Value — Lahoussine Laanait (2026)
# https://github.com/lahoussinelaanait/Remanent
# ============================================================

import os
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = '../results'
FIG_DIR = f'{RESULTS_DIR}/figures'
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

os.makedirs(FIG_DIR, exist_ok=True)

# Regimes analyzed (SERVICES excluded by design)
REGIMES = ['TECH', 'PHARMA', 'INDUS']

# Color palette
COLORS = {
    'TECH': 'red',
    'INDUS': 'orange',
    'PHARMA': 'purple',
}
FALLBACK = ['red', 'orange', 'purple']


# ============================================================
# HELPERS
# ============================================================

def period_to_year(period):
    """Convert a period string to a year. '2015' -> 2015; '2015-2024' -> None."""
    p = str(period).strip()
    return int(p) if p.isdigit() else None


def clean_label(g):
    """Remove the V3_ prefix and replace underscores with spaces."""
    return str(g).replace('V3_', '').replace('_', ' ')


def get_color(group, idx):
    """Return a color for a group."""
    return COLORS.get(group, FALLBACK[idx % len(FALLBACK)])


# ============================================================
# DATA LOADING
# ============================================================

def read_correlations():
    """Read all correlation CSVs, keep the latest run per (group, period)."""
    pattern = f'{RESULTS_DIR}/correlations/correlations_*.csv'
    files = sorted(glob.glob(pattern))
    print(f"\n📂 Correlations: {len(files)} files")

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
        return pd.DataFrame(columns=['group', 'period', 'value', 'run_id', 'year'])

    df = pd.DataFrame(rows, columns=['group', 'period', 'value', 'run_id'])
    df = df[df['group'].isin(REGIMES)]
    df = df.sort_values('run_id').drop_duplicates(subset=['group', 'period'], keep='last')
    df['year'] = df['period'].apply(period_to_year)
    print(f"  → {len(df)} rows after dedup")
    return df


def read_margins():
    """Read all margin CSVs, keep the latest run per (group, period)."""
    pattern = f'{RESULTS_DIR}/margins/marge_mle_*.csv'
    files = sorted(glob.glob(pattern))
    print(f"\n📂 Margins: {len(files)} files")

    rows = []
    for f in files:
        try:
            tmp = pd.read_csv(f)
            needed = {'group', 'period', 'marge_mle', 'run_id'}
            if not needed.issubset(tmp.columns):
                continue
            for _, r in tmp.iterrows():
                rows.append((r['group'], str(r['period']), r['marge_mle'], r['run_id']))
        except Exception as e:
            print(f"  ⚠ {f}: {e}")

    if not rows:
        return pd.DataFrame(columns=['group', 'period', 'value', 'run_id', 'year'])

    df = pd.DataFrame(rows, columns=['group', 'period', 'value', 'run_id'])
    df = df[df['group'].isin(REGIMES)]
    df = df.sort_values('run_id').drop_duplicates(subset=['group', 'period'], keep='last')
    df['year'] = df['period'].apply(period_to_year)
    print(f"  → {len(df)} rows after dedup")
    return df


# ============================================================
# PLOTTING
# ============================================================

def plot_recap(df, title, ylabel, file_base):
    """
    Plot annual curves and global reference lines for each group.

    Parameters
    ----------
    df : DataFrame with columns ['group', 'year', 'value']
    title : str
    ylabel : str
    file_base : str — output path WITHOUT extension. Produces .pdf and .png.
    """
    if len(df) == 0:
        print(f"  ⚠ Nothing to plot: {title}")
        return

    plt.close('all')
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, g in enumerate(sorted(df['group'].unique())):
        sub = df[df['group'] == g]
        color = get_color(g, i)

        # Annual curve
        ann = sub[sub['year'].notna()].sort_values('year')
        if len(ann) > 0:
            ax.plot(ann['year'], ann['value'],
                    marker='o', color=color, lw=2.5, markersize=6,
                    label=f"{clean_label(g)} (annual)")

        # Global reference line (prefer 2015-2024)
        glo = sub[sub['year'].isna()]
        if len(glo) > 0:
            preferred = glo[glo['period'] == '2015-2024']
            v = (preferred if len(preferred) > 0 else glo.tail(1))['value'].iloc[0]
            ax.axhline(v, color=color, ls='--', lw=2, alpha=0.8,
                       label=f"{clean_label(g)} global = {v:.3f}")
            ax.annotate(f"{v:.3f}",
                        xy=(1.002, v), xycoords=('axes fraction', 'data'),
                        fontsize=10, color=color, va='center', fontweight='bold')

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Year')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.93, 1])

    # Save PDF (vector, for LaTeX)
    pdf_path = f"{file_base}.pdf"
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  ✅ PDF: {pdf_path}")

    # Save PNG (raster, for quick viewing)
    png_path = f"{file_base}.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ PNG: {png_path}")

    plt.show()
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("GENERATING RECAPITULATIVE GRAPHS")
    print("=" * 70)

    df_corr = read_correlations()
    df_margin = read_margins()

    plot_recap(
        df_corr,
        'MLE Correlation EBIT–Personnel by regime\n'
        '(annual curves + global references in dashed lines)',
        'MLE Correlation (EBIT, Personnel)',
        f'{FIG_DIR}/graph_correlations_{RUN_ID}'
    )

    plot_recap(
        df_margin,
        'EBIT/Revenue Margin (MLE) by regime\n'
        '(annual curves + global references in dashed lines)',
        'EBIT/Revenue Margin (MLE)',
        f'{FIG_DIR}/graph_margins_{RUN_ID}'
    )

    print(f"\n{'='*70}")
    print(f"✅ Run {RUN_ID} complete.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
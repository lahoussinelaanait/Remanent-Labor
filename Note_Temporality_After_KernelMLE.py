# ============================================================
# TEMPORAL ANALYSIS — Cross-correlation RD_t ↔ EBIT_{t+k}
# To run AFTER the main kernel has been executed.
# Requires in memory: TwoPartModel, df, variables_analyse
# Estimator: hierarchical MLE (identical to the one used in mediation).
# ============================================================

# ------------------------------------------------------------
# 1. LAG CONSTRUCTION
# ------------------------------------------------------------

def build_lagged_ebit(df_regime, max_k=6):
    """
    For each firm, build EBIT_fut{k} = EBIT of year t+k.
    Requires columns: company, year, ebit.
    """
    df_regime = df_regime.sort_values(['company', 'year']).copy()
    for k in range(1, max_k + 1):
        df_regime[f'EBIT_fut{k}'] = df_regime.groupby('company')['ebit'].shift(-k)
    return df_regime


# ------------------------------------------------------------
# 2. MLE CORRELATION (same estimator as mediation)
# ------------------------------------------------------------

def compute_mle_correlation(x, y):
    """
    Hierarchical MLE correlation:
        Corr(X,Y) = (E[XY] - E[X]E[Y]) / (σ_X σ_Y)
    Same estimator used in mediation (TwoPartModel on product variable).
    """
    z = x * y
    model_x = TwoPartModel().fit(x)
    model_y = TwoPartModel().fit(y)
    model_z = TwoPartModel().fit(z)
    ex = model_x.expectation()
    ey = model_y.expectation()
    ez = model_z.expectation()
    vx = model_x.variance()
    vy = model_y.variance()
    cov = ez - ex * ey
    denom = np.sqrt(vx * vy)
    return cov / denom if denom > 0 else np.nan


# ------------------------------------------------------------
# 3. TEMPORAL ANALYSIS FOR ONE REGIME
# ------------------------------------------------------------

def temporal_analysis(df_regime, regime_name, max_k=6):
    """
    Cross-correlation RD_t ↔ EBIT_{t+k}, k = 0..max_k.
    Estimator: hierarchical MLE — identical to the one used in mediation.
    """
    print(f"\n{'='*60}")
    print(f"TEMPORAL ANALYSIS: RD_t → EBIT_(t+k) - {regime_name}")
    print(f"Estimator: hierarchical MLE (same as mediation)")
    print("="*60)

    df_lag = build_lagged_ebit(df_regime, max_k=max_k)

    rows = []
    for k in range(0, max_k + 1):
        if k == 0:
            x = df_lag['rd'].values
            y = df_lag['ebit'].values
        else:
            col = f'EBIT_fut{k}'
            if col not in df_lag.columns:
                continue
            x = df_lag['rd'].values
            y = df_lag[col].values

        mask = ~(np.isnan(x) | np.isnan(y))
        x_, y_ = x[mask], y[mask]
        n = len(x_)

        if n < 20:
            rows.append({'k': k, 'r': np.nan, 'n': n})
            print(f"  k={k}: too few obs (n={n})")
            continue

        r = compute_mle_correlation(x_, y_)
        rows.append({'k': k, 'r': r, 'n': n})
        print(f"  k={k}: r={r:.4f}  (n={n})")

    df_temp = pd.DataFrame(rows)

    df_temp.to_csv(
        f'/content/drive/MyDrive/temporal_{regime_name}.csv',
        index=False
    )
    print(f"\n✓ Temporal results saved: temporal_{regime_name}.csv")

    return df_temp


# ------------------------------------------------------------
# 4. MAIN EXECUTION — SHORT vs LONG
# ------------------------------------------------------------

print("\n" + "="*80)
print("STARTING TEMPORAL ANALYSIS — SHORT vs LONG")
print("="*80)

REGIMES_TEMPORAL = ['SHORT', 'LONG']
temporal_results = {}

for regime in REGIMES_TEMPORAL:
    df_regime = df[df['REGIME'] == regime].copy()
    n_firms = df_regime['company'].nunique() if 'company' in df_regime.columns else '?'

    print(f"\n{'='*80}")
    print(f"🔍 REGIME: {regime}  —  {len(df_regime)} obs, {n_firms} firms")
    print(f"{'='*80}")

    if len(df_regime) < 30:
        print(f"  ⚠ Insufficient data ({len(df_regime)} < 30), regime ignored.")
        continue

    df_temp = temporal_analysis(df_regime, regime, max_k=6)
    temporal_results[regime] = df_temp


# ------------------------------------------------------------
# 5. SUMMARY TABLE — SHORT vs LONG
# ------------------------------------------------------------

print(f"\n{'='*80}")
print("TEMPORAL SUMMARY — SHORT vs LONG")
print(f"{'='*80}")

if len(temporal_results) == 2:
    short = temporal_results['SHORT'].set_index('k')
    long_ = temporal_results['LONG'].set_index('k')

    summary = pd.DataFrame({
        'k': short.index,
        'SHORT_r': short['r'].values,
        'SHORT_n': short['n'].values,
        'LONG_r': long_['r'].values,
        'LONG_n': long_['n'].values,
    })
    summary['diff_LONG_minus_SHORT'] = summary['LONG_r'] - summary['SHORT_r']

    print(summary.to_string(index=False))

    summary.to_csv(
        '/content/drive/MyDrive/TEMPORAL_SUMMARY_SHORT_LONG.csv',
        index=False
    )
    print(f"\n✓ Saved: /content/drive/MyDrive/TEMPORAL_SUMMARY_SHORT_LONG.csv")

    # Locate the peak
    k_short_star = int(short['r'].idxmax())
    k_long_star = int(long_['r'].idxmax())
    print(f"\nPeak SHORT: k* = {k_short_star} (r = {short['r'].max():.4f})")
    print(f"Peak LONG : k* = {k_long_star} (r = {long_['r'].max():.4f})")

    # Figure
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(short.index, short['r'], 'o-', color='#1f4e79',
            linewidth=2, markersize=8, label='SHORT (1-cycle)')
    ax.plot(long_.index, long_['r'], 's-', color='#c00000',
            linewidth=2, markersize=8, label='LONG (2-cycle)')
    ax.axvline(k_long_star, color='grey', linestyle='--', alpha=0.5)
    ax.annotate(f'k* = {k_long_star}',
                xy=(k_long_star, long_['r'].max()),
                xytext=(k_long_star + 0.3, long_['r'].max() + 0.005),
                color='#c00000')
    ax.annotate(f'k* = {k_short_star}',
                xy=(k_short_star, short['r'].max()),
                xytext=(k_short_star + 0.3, short['r'].max() + 0.005),
                color='#1f4e79')
    ax.set_xlabel('Lag k (years)')
    ax.set_ylabel('Corr(RD_t, EBIT_{t+k})')
    ax.set_title('Temporal signature RD → EBIT by regime')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        '/content/drive/MyDrive/temporal_signature_SHORT_LONG.png',
        dpi=300
    )
    plt.show()

else:
    print("  ⚠ Not enough regimes analysed to build the summary.")
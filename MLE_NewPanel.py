"""
MLE Two-Part / Two-Block Analysis of Production Regimes
========================================================

This script estimates, for each production regime (1-cycle and 2-cycle),
the following quantities under a hierarchical Maximum Likelihood framework:

  1. Correlation Personnel - EBIT   (via product variable Z = Personnel * EBIT)
  2. Correlation RD       - EBIT    (via product variable Z = RD * EBIT)
  3. Correlation Personnel - RD     (via product variable Z = Personnel * RD)
  4. MLE expectation of the margin EBIT/Revenue
  5. MLE expectation of the ratio RD/CA

Model specification (per variable):
  - Two-part:  Student-t for negative values, positive part for positives
  - Two-block: segmentation of the positive part at an optimal threshold
               (search over {60, 65, 70, 75, 80, 85, 90}), selected by
               maximizing the minimum Kolmogorov-Smirnov p-value across blocks
  - Within each block: mixture of K Gamma distributions (K in {1, 2, 3, 4}),
                       selected by BIC with a KS correction

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
from scipy.special import gammaln, logsumexp
from scipy.stats import gamma, kstest, t
from scipy.optimize import minimize
from sklearn.cluster import KMeans

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# 1. STUDENT-T (for the negative part)
# ============================================================
class StudentT:
    """Student-t distribution fitted by MLE on negative values."""

    def __init__(self):
        self.nu = None
        self.mu = None
        self.sigma = None

    def fit(self, X: np.ndarray) -> "StudentT":
        X = X[(~np.isnan(X)) & (X < 0)]
        if len(X) < 5:
            return self

        def nll(p):
            nu, mu, sg = p
            if nu <= 2 or sg <= 0:
                return 1e10
            const = (gammaln((nu + 1) / 2) - gammaln(nu / 2)
                     - 0.5 * np.log(np.pi * nu) - np.log(sg))
            return -np.sum(const - ((nu + 1) / 2)
                           * np.log1p((X - mu) ** 2 / (nu * sg ** 2)))

        r = minimize(nll, [3.0, np.mean(X), np.std(X)],
                     method='L-BFGS-B',
                     bounds=[(2.1, 50), (-1e5, 0), (0.001, 1e5)])
        if r.success:
            self.nu, self.mu, self.sigma = r.x
        return self

    def expectation(self) -> float:
        return self.mu if self.mu is not None else 0.0

    def variance(self) -> float:
        if self.nu is None or self.sigma is None:
            return 0.0
        if self.nu > 2:
            return self.sigma ** 2 * self.nu / (self.nu - 2)
        return np.inf


# ============================================================
# 2. GAMMA MIXTURE (within each block)
# ============================================================
class GammaMixture:
    """Mixture of K Gamma distributions fitted by EM."""

    def __init__(self, Kmin: int = 1, Kmax: int = 4):
        self.Kmin, self.Kmax = Kmin, Kmax
        self.K = None
        self.alphas = self.betas = self.weights = None
        self.ks_p = None

    @staticmethod
    def _logpdf(x, a, b):
        x = np.maximum(x, 1e-10)
        return (a - 1) * np.log(x) - x / b - a * np.log(b) - gammaln(a)

    def _fit_single(self, X, w=None):
        if w is None:
            w = np.ones(len(X))
        w = w / w.sum()
        mu = np.sum(w * X)
        var = np.sum(w * (X - mu) ** 2)
        a0 = min(max(mu ** 2 / var, 0.1), 500) if var > 0 else 2.0
        b0 = min(max(var / mu, 0.01), 1000) if var > 0 else mu / 2

        def nll(p):
            a, b = p
            if a <= 0 or b <= 0:
                return 1e10
            return -np.sum(w * self._logpdf(X, a, b))

        r = minimize(nll, [a0, b0], method='L-BFGS-B',
                     bounds=[(0.05, 1000), (1e-4, 5000)])
        return r.x if r.success else np.array([a0, b0])

    def _cdf(self, x):
        v = 0.0
        for k in range(self.K):
            v += self.weights[k] * gamma.cdf(x, self.alphas[k],
                                             scale=self.betas[k])
        return v

    def fit(self, X: np.ndarray) -> "GammaMixture":
        X = X[(~np.isnan(X)) & (X > 0)]
        if len(X) < 10:
            a, b = self._fit_single(X)
            self.K = 1
            self.alphas = np.array([a])
            self.betas = np.array([b])
            self.weights = np.array([1.0])
            return self

        models = {}
        for K in range(self.Kmin, min(self.Kmax, len(X) // 5) + 1):
            try:
                Xl = np.log(np.maximum(X, 1e-10))
                lab = KMeans(K, random_state=42, n_init=10).fit_predict(
                    Xl.reshape(-1, 1))
                al, be, we = [], [], []
                for k in range(K):
                    Xk = X[lab == k]
                    a, b = (self._fit_single(Xk) if len(Xk) >= 3
                            else self._fit_single(X))
                    al.append(a)
                    be.append(b)
                    we.append(len(Xk) if len(Xk) >= 3 else 1)
                we = np.array(we) / np.sum(we)
                al = np.array(al)
                be = np.array(be)

                ll_old = -np.inf
                for _ in range(100):
                    lr = np.zeros((len(X), K))
                    for k in range(K):
                        lr[:, k] = (self._logpdf(X, al[k], be[k])
                                    + np.log(max(we[k], 1e-10)))
                    ls = logsumexp(lr, axis=1, keepdims=True)
                    resp = np.exp(lr - ls)
                    Nk = resp.sum(0)
                    we_new = Nk / len(X)

                    al_new, be_new = [], []
                    for k in range(K):
                        if Nk[k] > 2:
                            a, b = self._fit_single(X, w=resp[:, k])
                        else:
                            a, b = al[k], be[k]
                        al_new.append(a)
                        be_new.append(b)
                    al = np.array(al_new)
                    be = np.array(be_new)
                    we = we_new

                    lp = np.zeros((len(X), K))
                    for k in range(K):
                        lp[:, k] = (self._logpdf(X, al[k], be[k])
                                    + np.log(max(we[k], 1e-10)))
                    ll = logsumexp(lp, axis=1).sum()
                    if abs(ll - ll_old) < 1e-6:
                        break
                    ll_old = ll

                bic = np.log(len(X)) * (3 * K - 1) - 2 * ll
                m = GammaMixture(K, K)
                m.K = K
                m.alphas = al
                m.betas = be
                m.weights = we
                _, m.ks_p = kstest(X, m._cdf)
                models[K] = {'K': K, 'bic': bic, 'ks': m.ks_p, 'model': m}
            except Exception:
                continue

        if not models:
            a, b = self._fit_single(X)
            self.K = 1
            self.alphas = np.array([a])
            self.betas = np.array([b])
            self.weights = np.array([1.0])
            _, self.ks_p = kstest(X, self._cdf)
            return self

        best_K = min(models, key=lambda k: models[k]['bic'])
        sel = models[best_K]['model']
        self.K = sel.K
        self.alphas = sel.alphas
        self.betas = sel.betas
        self.weights = sel.weights
        self.ks_p = sel.ks_p
        return self

    def expectation(self) -> float:
        return float(np.sum(self.weights * self.alphas * self.betas))

    def variance(self) -> float:
        exp = self.expectation()
        exp2 = sum(self.weights[k] * self.alphas[k] * (self.alphas[k] + 1)
                   * self.betas[k] ** 2 for k in range(self.K))
        return float(exp2 - exp ** 2)


# ============================================================
# 3. TWO-BLOCK MODEL
# ============================================================
class TwoBlock:
    """Two-block segmentation of the positive part with optimal threshold."""

    def __init__(self, thresholds=(60, 65, 70, 75, 80, 85, 90)):
        self.thresholds = thresholds
        self.threshold = None
        self.m1 = self.m2 = None
        self.prop1 = self.prop2 = 0.5
        self.ks1 = self.ks2 = None

    def fit(self, X: np.ndarray) -> "TwoBlock":
        X = X[(~np.isnan(X)) & (X > 0)]
        if len(X) < 20:
            return self

        best_min_ks = 0.0
        best = None
        best_th = 60
        for th in self.thresholds:
            cut = np.percentile(X, th)
            b1, b2 = X[X <= cut], X[X > cut]
            if len(b1) < 10 or len(b2) < 5:
                continue
            m1 = GammaMixture().fit(b1)
            m2 = GammaMixture().fit(b2)
            mks = min(m1.ks_p, m2.ks_p)
            if mks > best_min_ks:
                best_min_ks = mks
                best = (m1, m2, m1.ks_p, m2.ks_p)
                best_th = th
                self.prop1 = len(b1) / len(X)
                self.prop2 = len(b2) / len(X)

        if best is None:
            return self
        self.threshold = best_th
        self.m1, self.m2 = best[0], best[1]
        self.ks1, self.ks2 = best[2], best[3]
        return self

    def expectation(self) -> float:
        if self.m1 is None:
            return 0.0
        return (self.prop1 * self.m1.expectation()
                + self.prop2 * self.m2.expectation())

    def variance(self) -> float:
        if self.m1 is None:
            return 0.0
        e_tot = self.expectation()
        e1, e2 = self.m1.expectation(), self.m2.expectation()
        v1, v2 = self.m1.variance(), self.m2.variance()
        cond_var = self.prop1 * v1 + self.prop2 * v2
        cond_exp = (self.prop1 * (e1 - e_tot) ** 2
                    + self.prop2 * (e2 - e_tot) ** 2)
        return cond_var + cond_exp


# ============================================================
# 4. TWO-PART MODEL
# ============================================================
class TwoPart:
    """Two-part model: Student-t for negatives, TwoBlock for positives."""

    def __init__(self):
        self.student = None
        self.twoblock = None
        self.p_neg = 0.0
        self.p_pos = 1.0
        self.n_neg = 0
        self.n_pos = 0

    def fit(self, X: np.ndarray) -> "TwoPart":
        X = np.asarray(X)
        X = X[~np.isnan(X)]
        neg, pos = X[X < 0], X[X > 0]
        self.n_neg, self.n_pos = len(neg), len(pos)
        self.p_neg = len(neg) / len(X) if len(X) > 0 else 0.0
        self.p_pos = 1.0 - self.p_neg
        if len(neg) >= 5:
            self.student = StudentT().fit(neg)
        if len(pos) >= 10:
            self.twoblock = TwoBlock().fit(pos)
        return self

    def expectation(self) -> float:
        e_neg = self.student.expectation() if self.student else 0.0
        e_pos = self.twoblock.expectation() if self.twoblock else 0.0
        return self.p_neg * e_neg + self.p_pos * e_pos

    def variance(self) -> float:
        e_tot = self.expectation()
        e_neg = self.student.expectation() if self.student else 0.0
        e_pos = self.twoblock.expectation() if self.twoblock else 0.0
        v_neg = self.student.variance() if self.student else 0.0
        v_pos = self.twoblock.variance() if self.twoblock else 0.0
        cond_var = self.p_neg * v_neg + self.p_pos * v_pos
        cond_exp = (self.p_neg * (e_neg - e_tot) ** 2
                    + self.p_pos * (e_pos - e_tot) ** 2)
        return cond_var + cond_exp


# ============================================================
# 5. MLE CORRELATION VIA PRODUCT VARIABLE
# ============================================================
def mle_correlation(x: np.ndarray, y: np.ndarray):
    """
    Compute the MLE correlation between x and y using the product variable
    Z = X * Y, as described in the framework.

    Returns
    -------
    corr : float
    E_x, E_y : float
    V_x, V_y : float
    """
    z = x * y
    mx = TwoPart().fit(x)
    my = TwoPart().fit(y)
    mz = TwoPart().fit(z)

    ex, ey, ez = mx.expectation(), my.expectation(), mz.expectation()
    vx, vy = mx.variance(), my.variance()
    cov = ez - ex * ey
    denom = np.sqrt(vx * vy)
    corr = cov / denom if denom > 0 else np.nan
    return corr, ex, ey, vx, vy


# ============================================================
# 6. ANALYSIS PIPELINE
# ============================================================
def analyze_regime(sub: pd.DataFrame, regime_name: str) -> dict:
    """Run all MLE analyses for one regime."""
    print(f"\n{'='*70}")
    print(f"REGIME {regime_name}  —  {len(sub)} obs, "
          f"{sub['firm_id'].nunique()} firms")
    print(f"{'='*70}")

    # Personnel - EBIT
    x = sub['Personnel'].values.astype(float)
    y = sub['EBIT'].values.astype(float)
    r_pe, _, _, _, _ = mle_correlation(x, y)
    print(f"\n  Personnel - EBIT correlation (MLE) : {r_pe:.4f}")

    # RD - EBIT
    x = sub['RD'].values.astype(float)
    r_re, _, _, _, _ = mle_correlation(x, y)
    print(f"  RD        - EBIT correlation (MLE) : {r_re:.4f}")

    # Personnel - RD
    x = sub['Personnel'].values.astype(float)
    y = sub['RD'].values.astype(float)
    r_pr, _, _, _, _ = mle_correlation(x, y)
    print(f"  Personnel - RD   correlation (MLE) : {r_pr:.4f}")

    # Margin EBIT / Revenue
    margin = (sub['EBIT'] / sub['Revenue']).values
    model_m = TwoPart().fit(margin)
    e_margin = model_m.expectation()
    print(f"\n  MLE margin EBIT/Revenue            : {e_margin:.4f}")
    if model_m.twoblock:
        tb = model_m.twoblock
        print(f"    Threshold = {tb.threshold}%  "
              f"K1={tb.m1.K} (KS={tb.ks1:.3f})  "
              f"K2={tb.m2.K} (KS={tb.ks2:.3f})")

    # RD / CA
    rd_ca = (sub['RD'] / sub['Revenue']).dropna().values
    model_rd = TwoPart().fit(rd_ca)
    e_rd = model_rd.expectation()
    print(f"\n  MLE expectation E(RD/CA)           : {e_rd:.4f}")

    return {
        'regime': regime_name,
        'n_firms': sub['firm_id'].nunique(),
        'n_obs': len(sub),
        'corr_pers_ebit': r_pe,
        'corr_rd_ebit': r_re,
        'corr_pers_rd': r_pr,
        'mle_margin': e_margin,
        'mle_rd_ca': e_rd,
    }


def run_pipeline(panel_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("MLE TWO-PART / TWO-BLOCK ANALYSIS")
    print(f"{'='*70}")
    print(f"Panel : {panel_path}")

    df = pd.read_csv(panel_path, sep=';', encoding='utf-8-sig')

    for c in ['Revenue', 'EBIT', 'RD', 'Personnel', 'PPE', 'year']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df = df[(df['Revenue'] > 0) & (df['RD'] > 0)
            & (df['Personnel'] > 0)]

    print(f"\nTotal : {len(df)} obs, {df['firm_id'].nunique()} firms")
    print(df.groupby('regime').agg(
        firms=('firm_id', 'nunique'),
        obs=('year', 'count')
    ).to_string())

    # Run analyses
    results = []
    for regime in ['SHORT', 'LONG']:
        sub = df[df['regime'] == regime]
        if len(sub) < 30:
            print(f"\n{regime} : insufficient data")
            continue
        results.append(analyze_regime(sub, regime))

    # Summary table
    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE — MLE CORRELATIONS AND EXPECTATIONS")
    print(f"{'='*80}")
    print(f"{'Regime':<8} {'Firms':<7} {'Obs':<7} "
          f"{'Pers-EBIT':<11} {'RD-EBIT':<10} {'Pers-RD':<10} "
          f"{'Margin':<10} {'E(RD/CA)':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['regime']:<8} {r['n_firms']:<7} {r['n_obs']:<7} "
              f"{r['corr_pers_ebit']:<11.4f} "
              f"{r['corr_rd_ebit']:<10.4f} "
              f"{r['corr_pers_rd']:<10.4f} "
              f"{r['mle_margin']:<10.4f} "
              f"{r['mle_rd_ca']:<10.4f}")

    # Save
    out_csv = output_dir / 'mle_results_summary.csv'
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n✅ Results saved: {out_csv}")


# ============================================================
# 7. MAIN
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="MLE two-part / two-block analysis by production regime.")
    parser.add_argument(
        '--panel', type=Path,
        default=Path('/content/drive/My Drive/Github_Remanent/new_panel/'
                     'panel_LONG_vs_SHORT.csv'),
        help='Path to the panel CSV file.')
    parser.add_argument(
        '--out', type=Path,
        default=Path('/content/drive/My Drive/Github_Remanent/new_panel/'),
        help='Output directory.')
    args = parser.parse_args()
    run_pipeline(args.panel, args.out)
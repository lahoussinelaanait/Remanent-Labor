# ============================================================
# HIERARCHICAL MAXIMUM LIKELIHOOD MODEL
# Two-block segmentation | Gamma mixtures | Two-part specification
#
# Remanent Surplus Value — Lahoussine Laanait (2026)
# https://github.com/lahoussinelaanait/Remanent
#
# This script estimates correlations between economic variables
# for three production regimes (TECH, PHARMA, INDUS) using a
# hierarchical maximum likelihood framework.
# ============================================================

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import gamma, kstest, t
from scipy.optimize import minimize
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = '../data/panel_FINAL_REMANENT.csv'
OUTPUT_DIR = '../results'
RUN_ID = datetime.now().strftime('%Y%m%d_%H%M%S')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/correlations', exist_ok=True)
os.makedirs(f'{OUTPUT_DIR}/margins', exist_ok=True)

# Variables to analyze
VARIABLES = [
    ('ebit', 'EBIT'),
    ('personnel', 'Personnel'),
    ('z1', 'EBIT x Personnel'),
    ('ebit_revenue', 'EBIT/Revenue'),
    ('rd', 'R&D'),
    ('ppe', 'PP&E'),
    ('z2', 'EBIT x PP&E'),
    ('z3', 'EBIT x R&D'),
    ('z4', 'R&D x Personnel'),
]

# Variables that use the two-part model when negatives are present
TWO_PART_VARS = ['ebit', 'ebit_revenue', 'z1', 'z2', 'z3']

# Production regimes (SERVICES excluded by design)
REGIMES = ['TECH', 'PHARMA', 'INDUS']


# ============================================================
# DATA LOADING
# ============================================================

def load_panel(path):
    """
    Load the panel and compute derived variables.

    Derived variables:
        ebit_revenue = ebit / ca
        z1 = ebit * personnel   (for Cov(EBIT, Personnel))
        z2 = ebit * ppe         (for Cov(EBIT, PP&E))
        z3 = ebit * rd          (for Cov(EBIT, R&D))
        z4 = rd * personnel     (for Cov(R&D, Personnel))
    """
    df = pd.read_csv(path, sep=';', encoding='utf-8')
    df.columns = df.columns.str.strip().str.lower()

    for col in ['year', 'ca', 'ebit', 'ppe', 'rd', 'personnel']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if all(c in df.columns for c in ['ebit', 'ca']):
        df['ebit_revenue'] = df['ebit'] / df['ca']
    if all(c in df.columns for c in ['ebit', 'personnel']):
        df['z1'] = df['ebit'] * df['personnel']
    if all(c in df.columns for c in ['ebit', 'ppe']):
        df['z2'] = df['ebit'] * df['ppe']
    if all(c in df.columns for c in ['ebit', 'rd']):
        df['z3'] = df['ebit'] * df['rd']
    if all(c in df.columns for c in ['rd', 'personnel']):
        df['z4'] = df['rd'] * df['personnel']

    return df


# ============================================================
# STUDENT-T MLE (for the negative part of a two-part model)
# ============================================================

class StudentTMLE:
    """
    Student-t distribution fitted by maximum likelihood on the
    negative subset of a variable.

    The degrees-of-freedom parameter nu controls tail thickness:
    small nu = heavy tails, appropriate for extreme losses.

    Minimum sample size: 3 negative observations.
    """

    def __init__(self):
        self.nu = None
        self.mu = None
        self.sigma = None
        self.logL = None

    def _logpdf(self, x, nu, mu, sigma):
        nu = max(nu, 2.1)
        sigma = max(sigma, 1e-8)
        const = (gammaln((nu + 1) / 2) - gammaln(nu / 2)
                 - 0.5 * np.log(np.pi * nu) - np.log(sigma))
        return const - ((nu + 1) / 2) * np.log1p((x - mu) ** 2 / (nu * sigma ** 2))

    def fit(self, X):
        """Fit on negative values only. Requires at least 3 observations."""
        Xc = X[(~np.isnan(X)) & (X < 0)]
        if len(Xc) < 3:
            return self

        def neg_loglike(params):
            nu, mu, sigma = params
            if nu <= 2 or sigma <= 0:
                return 1e10
            return -np.sum(self._logpdf(Xc, nu, mu, sigma))

        mu0 = np.mean(Xc)
        sigma0 = max(np.std(Xc), 1e-3)
        result = minimize(neg_loglike, [3.0, mu0, sigma0],
                          method='L-BFGS-B',
                          bounds=[(2.1, 50), (-1e3, 0), (1e-3, 1e3)])

        if result.success:
            self.nu, self.mu, self.sigma = result.x
            self.logL = -result.fun

        return self

    def expectation(self):
        return self.mu if self.mu is not None else 0

    def variance(self):
        if self.nu is None or self.mu is None or self.sigma is None:
            return 0
        if self.nu > 2:
            return (self.sigma ** 2) * self.nu / (self.nu - 2)
        return np.inf

    def pdf(self, x):
        if self.nu is None:
            return 0
        return t.pdf(x, self.nu, self.mu, self.sigma)


# ============================================================
# GAMMA MIXTURE (intra-block model)
# ============================================================

class GammaMixture:
    """
    Mixture of K Gamma distributions, fitted by EM.

    Number of components K selected by a combined BIC + KS rule:
      - Start from the K that minimizes BIC.
      - If KS p-value < 0.05 (under-fitting), add a component.
      - If KS p-value > 0.10, keep the BIC choice.
    """

    def __init__(self, K_min=1, K_max=4):
        self.K_min = K_min
        self.K_max = K_max
        self.n_components = None
        self.alphas = None
        self.betas = None
        self.weights = None
        self.logL = None
        self.ks_pvalue = None
        self.wasserstein = None

    def _logpdf(self, x, alpha, beta):
        x = np.maximum(x, 1e-8)
        alpha = max(alpha, 1e-8)
        beta = max(beta, 1e-8)
        return (alpha - 1) * np.log(x) - x / beta - alpha * np.log(beta) - gammaln(alpha)

    def _fit_single(self, X, weights=None):
        """Fit a single Gamma component by weighted MLE."""
        if weights is None:
            weights = np.ones(len(X))
        weights = weights / weights.sum()

        mean_w = np.sum(weights * X)
        var_w = np.sum(weights * (X - mean_w) ** 2)

        if var_w > 0 and mean_w > 0:
            alpha0 = min(max(mean_w ** 2 / var_w, 0.1), 500)
            beta0 = min(max(var_w / mean_w, 0.01), 1000)
        else:
            alpha0, beta0 = 2.0, mean_w / 2

        def neg_loglike(params):
            a, b = params
            if a <= 0 or b <= 0:
                return 1e10
            return -np.sum(weights * self._logpdf(X, a, b))

        result = minimize(neg_loglike, [alpha0, beta0],
                          method='L-BFGS-B',
                          bounds=[(0.05, 1000), (1e-3, 5000)])

        return result.x if result.success else np.array([alpha0, beta0])

    def _cdf_mixture(self, x):
        return sum(self.weights[k] * gamma.cdf(x, self.alphas[k], scale=self.betas[k])
                   for k in range(self.n_components))

    def _compute_tests(self, X):
        _, self.ks_pvalue = kstest(X, self._cdf_mixture)
        X_sorted = np.sort(X)
        n = len(X)
        cdf_emp = np.arange(1, n + 1) / n
        cdf_theo = np.array([self._cdf_mixture(x) for x in X_sorted])
        self.wasserstein = np.mean(np.abs(cdf_emp - cdf_theo))

    def fit(self, X):
        Xc = X[(~np.isnan(X)) & (X > 0)]

        if len(Xc) < 10:
            self.n_components = 1
            a, b = self._fit_single(Xc)
            self.alphas = np.array([a])
            self.betas = np.array([b])
            self.weights = np.array([1.0])
            return self

        models = {}
        best_bic = np.inf
        best_K_by_bic = None

        for K in range(self.K_min, min(self.K_max, len(Xc) // 5) + 1):
            try:
                X_log = np.log(np.maximum(Xc, 1e-8))
                labels = KMeans(n_clusters=K, random_state=42, n_init=10).fit_predict(
                    X_log.reshape(-1, 1))

                alphas, betas, weights = [], [], []
                for k in range(K):
                    X_k = Xc[labels == k]
                    a, b = self._fit_single(X_k) if len(X_k) >= 3 else self._fit_single(Xc)
                    alphas.append(a)
                    betas.append(b)
                    weights.append(max(len(X_k), 1))

                weights = np.array(weights) / np.sum(weights)
                alphas = np.array(alphas)
                betas = np.array(betas)

                # EM refinement
                logL_old = -np.inf
                for _ in range(100):
                    log_resp = np.zeros((len(Xc), K))
                    for k in range(K):
                        log_resp[:, k] = (self._logpdf(Xc, alphas[k], betas[k])
                                          + np.log(max(weights[k], 1e-10)))
                    log_sum = logsumexp(log_resp, axis=1, keepdims=True)
                    resp = np.exp(log_resp - log_sum)

                    Nk = resp.sum(axis=0)
                    weights = Nk / len(Xc)

                    for k in range(K):
                        if Nk[k] > 2:
                            alphas[k], betas[k] = self._fit_single(Xc, weights=resp[:, k])

                    log_probs = np.zeros((len(Xc), K))
                    for k in range(K):
                        log_probs[:, k] = (self._logpdf(Xc, alphas[k], betas[k])
                                           + np.log(max(weights[k], 1e-10)))
                    logL = logsumexp(log_probs, axis=1).sum()

                    if abs(logL - logL_old) < 1e-6:
                        break
                    logL_old = logL

                n_params = 3 * K - 1
                bic = np.log(len(Xc)) * n_params - 2 * logL

                temp = GammaMixture(K_min=K, K_max=K)
                temp.n_components = K
                temp.alphas = alphas.copy()
                temp.betas = betas.copy()
                temp.weights = weights.copy()
                temp.logL = logL
                temp._compute_tests(Xc)

                models[K] = {'K': K, 'bic': bic, 'ks_pvalue': temp.ks_pvalue, 'model': temp}

                if bic < best_bic:
                    best_bic = bic
                    best_K_by_bic = K

            except Exception:
                continue

        if not models:
            self.n_components = 1
            a, b = self._fit_single(Xc)
            self.alphas = np.array([a])
            self.betas = np.array([b])
            self.weights = np.array([1.0])
            self._compute_tests(Xc)
            return self

        selected_K = best_K_by_bic

        # BIC + KS decision rule
        if selected_K == 1 and 1 in models:
            ks1 = models[1]['ks_pvalue']
            if ks1 < 0.05 and 2 in models:
                selected_K = 2
                print(f"    Intra-block: BIC chose K=1 but KS={ks1:.3f} < 0.05 -> K=2")
            elif ks1 > 0.10:
                selected_K = 1
                print(f"    Intra-block: BIC chose K=1 with KS={ks1:.3f} > 0.10 -> Keep K=1")
            else:
                if 2 in models and models[2]['ks_pvalue'] > 0.05:
                    selected_K = 2
                else:
                    selected_K = 1
        elif selected_K in [2, 3, 4]:
            print(f"    Intra-block: BIC chose K={selected_K} -> Keep K={selected_K}")

        sel = models[selected_K]['model']
        self.n_components = sel.n_components
        self.alphas = sel.alphas
        self.betas = sel.betas
        self.weights = sel.weights
        self.logL = sel.logL
        self.ks_pvalue = sel.ks_pvalue
        self.wasserstein = sel.wasserstein

        return self

    def expectation(self):
        return np.sum(self.weights * self.alphas * self.betas)

    def variance(self):
        e = self.expectation()
        e2 = sum(self.weights[k] * self.alphas[k] * (self.alphas[k] + 1) * self.betas[k] ** 2
                 for k in range(self.n_components))
        return e2 - e ** 2

    def pdf(self, x):
        if x <= 0:
            return 0
        return sum(self.weights[k] * gamma.pdf(x, self.alphas[k], scale=self.betas[k])
                   for k in range(self.n_components))


# ============================================================
# TWO-BLOCK MODEL
# ============================================================

class TwoBlockModel:
    """
    Two-block hierarchical model with adaptive threshold.

    Searches a threshold from 60% to 90% (percentile) to maximize
    min(KS1, KS2). Each block is fitted with a Gamma mixture.
    """

    def __init__(self, thresholds=(60, 65, 70, 75, 80, 85, 90)):
        self.thresholds = thresholds
        self.threshold = None
        self.prop1 = None
        self.prop2 = None
        self.model1 = None
        self.model2 = None
        self.ks_pvalue = None
        self.wasserstein = None

    def fit(self, X):
        Xc = X[(~np.isnan(X)) & (X > 0)]
        if len(Xc) < 25:
            return self

        best_threshold = 60
        best_min_ks = 0
        best_models = None

        print(f"\n  Searching optimal threshold...")

        for thresh in self.thresholds:
            cutoff = np.percentile(Xc, thresh)
            block1 = Xc[Xc <= cutoff]
            block2 = Xc[Xc > cutoff]

            if len(block1) < 20 or len(block2) < 10:
                continue

            m1 = GammaMixture(K_min=1, K_max=4).fit(block1)
            m2 = GammaMixture(K_min=1, K_max=4).fit(block2)

            min_ks = min(m1.ks_pvalue, m2.ks_pvalue)
            print(f"    thresh={thresh}%: KS1={m1.ks_pvalue:.3f}(K={m1.n_components}), "
                  f"KS2={m2.ks_pvalue:.3f}(K={m2.n_components}) -> min={min_ks:.3f}")

            if min_ks > best_min_ks:
                best_min_ks = min_ks
                best_threshold = thresh
                best_models = (m1, m2)

        if best_models is None:
            return self

        self.threshold = best_threshold
        self.model1, self.model2 = best_models

        cutoff = np.percentile(Xc, self.threshold)
        self.prop1 = np.mean(Xc <= cutoff)
        self.prop2 = 1 - self.prop1

        print(f"\n  -> Selected threshold: {self.threshold}%")
        print(f"    Block1: K={self.model1.n_components}, KS={self.model1.ks_pvalue:.3f}")
        print(f"    Block2: K={self.model2.n_components}, KS={self.model2.ks_pvalue:.3f}")

        self._compute_tests(Xc)
        return self

    def _compute_tests(self, X):
        def cdf(x):
            v = 0
            for k in range(self.model1.n_components):
                v += (self.model1.weights[k] * self.prop1
                      * gamma.cdf(x, self.model1.alphas[k], scale=self.model1.betas[k]))
            for k in range(self.model2.n_components):
                v += (self.model2.weights[k] * self.prop2
                      * gamma.cdf(x, self.model2.alphas[k], scale=self.model2.betas[k]))
            return v

        _, self.ks_pvalue = kstest(X, cdf)
        X_sorted = np.sort(X)
        n = len(X)
        cdf_emp = np.arange(1, n + 1) / n
        cdf_theo = np.array([cdf(x) for x in X_sorted])
        self.wasserstein = np.mean(np.abs(cdf_emp - cdf_theo))

    def expectation(self):
        return self.prop1 * self.model1.expectation() + self.prop2 * self.model2.expectation()

    def variance(self):
        e = self.expectation()
        e1 = self.model1.expectation()
        e2 = self.model2.expectation()
        v1 = self.model1.variance()
        v2 = self.model2.variance()
        return (self.prop1 * v1 + self.prop2 * v2
                + self.prop1 * (e1 - e) ** 2 + self.prop2 * (e2 - e) ** 2)

    def pdf(self, x):
        if x <= 0:
            return 0
        return (sum(self.model1.weights[k] * self.prop1
                    * gamma.pdf(x, self.model1.alphas[k], scale=self.model1.betas[k])
                    for k in range(self.model1.n_components))
                + sum(self.model2.weights[k] * self.prop2
                      * gamma.pdf(x, self.model2.alphas[k], scale=self.model2.betas[k])
                      for k in range(self.model2.n_components)))

    def get_structure(self):
        return (self.model1.n_components, self.model2.n_components)


# ============================================================
# TWO-PART MODEL (Student for losses + 2-Block Gamma for profits)
# ============================================================

class TwoPartModel:
    """
    Two-part model for variables with both negative and positive values.

    - Negative part: Student-t (captures extreme losses)
    - Positive part: two-block hierarchical Gamma mixture (captures profits)

    This model uses the FULL SAMPLE. It does not exclude negative values.
    A Student-t is fitted whenever there are at least 3 negative observations,
    so that no information is discarded.
    """

    def __init__(self):
        self.prob_neg = None
        self.prob_pos = None
        self.student = None
        self.twoblock = None
        self.ks_pvalue = None
        self.wasserstein = None

    def fit(self, X):
        pos_mask = X > 0
        neg_mask = X < 0

        n_total = len(X)
        n_pos = np.sum(pos_mask)
        n_neg = np.sum(neg_mask)

        self.prob_pos = n_pos / n_total
        self.prob_neg = n_neg / n_total

        print(f"\n  Data distribution:")
        print(f"    Negative: {n_neg} ({self.prob_neg*100:.1f}%)")
        print(f"    Positive: {n_pos} ({self.prob_pos*100:.1f}%)")

        # Student-t fitted whenever there are at least 3 negatives
        if n_neg >= 3:
            self.student = StudentTMLE().fit(X[neg_mask])
            print(f"  Student-t fitted on {n_neg} negatives")

        if n_pos >= 10:
            self.twoblock = TwoBlockModel().fit(X[pos_mask])
            print(f"  2-Block fitted on {n_pos} positives")
            self.ks_pvalue = self.twoblock.ks_pvalue
            self.wasserstein = self.twoblock.wasserstein

        return self

    def pdf(self, x):
        if x < 0 and self.student:
            return self.prob_neg * self.student.pdf(x)
        if x > 0 and self.twoblock:
            return self.prob_pos * self.twoblock.pdf(x)
        return 0

    def expectation(self):
        e_neg = self.student.expectation() if self.student else 0
        e_pos = self.twoblock.expectation() if self.twoblock else 0
        return self.prob_neg * e_neg + self.prob_pos * e_pos

    def variance(self):
        e = self.expectation()
        e_neg = self.student.expectation() if self.student else 0
        e_pos = self.twoblock.expectation() if self.twoblock else 0
        v_neg = self.student.variance() if self.student else 0
        v_pos = self.twoblock.variance() if self.twoblock else 0
        return (self.prob_neg * v_neg + self.prob_pos * v_pos
                + self.prob_neg * (e_neg - e) ** 2
                + self.prob_pos * (e_pos - e) ** 2)

    def get_structure(self):
        return self.twoblock.get_structure() if self.twoblock else (0, 0)


# ============================================================
# ANALYSIS
# ============================================================

def analyze_variable(df, var_name, display_name):
    """
    Analyze a single variable with the appropriate hierarchical model.

    Uses the two-part model whenever the variable has at least one
    negative value and belongs to TWO_PART_VARS. This ensures that
    the full sample is used, without discarding losses.
    """
    if var_name not in df.columns:
        return None

    data = df[var_name].dropna().values
    if len(data) < 10:
        return None

    n_pos = np.sum(data > 0)
    n_neg = np.sum(data < 0)

    print(f"\n{'='*60}")
    print(f"ANALYZING: {display_name}")
    print(f"{'='*60}")
    print(f"  Data: {len(data)} obs | {n_pos} positive | {n_neg} negative")

    # Two-part model whenever there is at least one negative
    if n_neg > 0 and var_name in TWO_PART_VARS:
        print("  Using Two-Part model (Student-t + 2-Block)")
        model = TwoPartModel().fit(data)
        model_type = 'Two-Part'
    else:
        data_pos = data[data > 0]
        if len(data_pos) < 10:
            return None
        print("  Using 2-Block Hierarchical model")
        model = TwoBlockModel().fit(data_pos)
        model_type = '2-Block'

    result = {
        'variable': var_name,
        'display_name': display_name,
        'model_type': model_type,
        'model': model,
        'n_obs': len(data),
        'n_pos': n_pos,
        'n_neg': n_neg,
        'mean_empirical': np.mean(data),
        'mle_expectation': model.expectation(),
        'mle_variance': model.variance(),
        'mle_std': np.sqrt(model.variance()),
        'ks_pvalue': getattr(model, 'ks_pvalue', np.nan),
        'wasserstein': getattr(model, 'wasserstein', np.nan),
        'structure': model.get_structure() if hasattr(model, 'get_structure') else (0, 0),
    }

    print(f"\n  Empirical mean: {result['mean_empirical']:.4f}")
    print(f"  MLE expectation: {result['mle_expectation']:.4f}")
    print(f"  KS p-value: {result['ks_pvalue']:.4f}")
    print(f"  Wasserstein: {result['wasserstein']:.4f}")

    return result


def compute_correlations(results, group, period):
    """
    Compute covariances and correlations from product variables.

    Cov(X, Y) = E[Z] - E[X] * E[Y], where Z = X * Y.
    Corr(X, Y) = Cov(X, Y) / (std(X) * std(Y)).
    """
    E = {r['variable']: r['mle_expectation'] for r in results}
    S = {r['variable']: r['mle_std'] for r in results}

    pairs = [
        ('ebit', 'personnel', 'z1', 'EBIT', 'Personnel'),
        ('ebit', 'ppe', 'z2', 'EBIT', 'PP&E'),
        ('ebit', 'rd', 'z3', 'EBIT', 'R&D'),
        ('rd', 'personnel', 'z4', 'R&D', 'Personnel'),
    ]

    rows = []
    print(f"\n{'='*70}")
    print(f"COVARIANCES AND CORRELATIONS — {group} — {period}")
    print(f"{'='*70}")

    for v1, v2, z, name1, name2 in pairs:
        if all(k in E for k in [v1, v2, z]) and S[v1] > 0 and S[v2] > 0:
            cov = E[z] - E[v1] * E[v2]
            corr = cov / (S[v1] * S[v2])
            print(f"  Cov({name1}, {name2}) = {cov:.4f}")
            print(f"  Corr({name1}, {name2}) = {corr:.4f}\n")
            rows.append({
                'Var1': name1, 'Var2': name2,
                'Covariance': cov, 'Correlation': corr,
                'group': group, 'period': period, 'run_id': RUN_ID,
            })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("HIERARCHICAL MLE ANALYSIS — REMANENT SURPLUS VALUE")
    print("=" * 70)
    print(f"Run ID: {RUN_ID}")

    df = load_panel(DATA_PATH)
    print(f"\nPanel loaded: {df.shape[0]} rows, {df['company'].nunique()} firms")

    for group in REGIMES:
        dfg = df[df['groupe'] == group].copy()
        if len(dfg) == 0:
            print(f"\n⚠ No data for regime {group}, skipping")
            continue

        print(f"\n\n{'#'*70}")
        print(f"# REGIME: {group}")
        print(f"# {dfg['company'].nunique()} firms, {len(dfg)} observations")
        print(f"{'#'*70}")

        # Global period (2015-2024)
        results = []
        for var, name in VARIABLES:
            r = analyze_variable(dfg, var, name)
            if r:
                results.append(r)

        # Correlations
        corr_rows = compute_correlations(results, group, '2015-2024')
        if corr_rows:
            out = f'{OUTPUT_DIR}/correlations/correlations_{group}_2015-2024_{RUN_ID}.csv'
            pd.DataFrame(corr_rows).to_csv(out, index=False)
            print(f"  ✓ Correlations saved: {out}")

        # Margin export
        if 'ebit_revenue' in [r['variable'] for r in results]:
            m = next(r for r in results if r['variable'] == 'ebit_revenue')
            margin_path = f'{OUTPUT_DIR}/margins/marge_mle_{group}_2015-2024_{RUN_ID}.csv'
            pd.DataFrame([{
                'group': group,
                'period': '2015-2024',
                'marge_mle': m['mle_expectation'],
                'run_id': RUN_ID,
            }]).to_csv(margin_path, index=False)
            print(f"  ✓ Margin saved: {margin_path}")

    print(f"\n{'='*70}")
    print(f"✅ Run {RUN_ID} complete.")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
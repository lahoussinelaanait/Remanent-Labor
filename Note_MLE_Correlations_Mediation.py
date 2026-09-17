# ============================================================
# HIERARCHICAL MLE — PANEL LONG vs SHORT
# TwoPartModel + TwoBlockModel + threshold 60-90%
output: expectations,correlations and mediation
# ============================================================

!pip install scipy numpy pandas scikit-learn matplotlib seaborn -q
from google.colab import drive
import pandas as pd
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import gamma, kstest, t, norm
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# ========== MOUNT DRIVE ==========
print("=" * 80)
print("MOUNTING GOOGLE DRIVE")
print("=" * 80)
drive.mount('/content/drive')
print("Google Drive mounted successfully!")

# ========== DATA LOADING ==========
panel_file = "/content/drive/My Drive/panel_LONG_vs_SHORT.csv"
df = pd.read_csv(panel_file, sep=';', encoding='utf-8-sig')

df = df.rename(columns={
    'Revenue': 'ca',
    'firm_id': 'company',
    'PPE': 'ppe',
    'EBIT': 'ebit',
    'RD': 'rd',
    'Personnel': 'personnel'
})

for c in ['year', 'ca', 'ebit', 'ppe', 'personnel', 'rd']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df = df[(df['ca'] > 0) & (df['rd'] > 0)]
df = df.dropna(subset=['ebit', 'rd', 'personnel', 'ppe'])

print(f"\nPanel loaded: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# ========== IDENTIFY REGIME COLUMN ==========
REGIME_COL = None
for c in df.columns:
    cl = c.lower()
    if 'regime' in cl or 'sector' in cl or 'group' in cl or 'groupe' in cl:
        REGIME_COL = c
        break

if REGIME_COL is None:
    print("⚠ Regime column not found. Available columns:")
    print(df.columns.tolist())
    raise SystemExit("Specify the regime column name.")

print(f"\nRegime column detected: {REGIME_COL}")
print(df[REGIME_COL].value_counts())

def normalize_regime(x):
    if pd.isna(x):
        return None
    x = str(x).upper().strip()
    # SHORT = industry only
    if 'INDUS' in x or x == 'IND':
        return 'SHORT'
    # LONG = TECH + PHARMA
    if 'TECH' in x or 'PHARMA' in x:
        return 'LONG'
    return x

df['REGIME'] = df[REGIME_COL].apply(normalize_regime)
print(f"\nNormalized regimes:")
print(df['REGIME'].value_counts())

# ========== DATA PREPARATION ==========
def prepare_dataframe(df_original):
    df = df_original.copy()
    print("Original columns:", df.columns.tolist())

    mapping = {}
    for col in df.columns:
        col_lower = col.lower().strip()
        if 'company' in col_lower or 'firm' in col_lower:
            mapping[col] = 'company'
        elif 'fiscal' in col_lower and 'year' in col_lower:
            mapping[col] = 'year'
        elif col_lower == 'year' or col_lower == 'annee':
            mapping[col] = 'year'
        elif 'revenue' in col_lower or col_lower == 'ca':
            mapping[col] = 'ca'
        elif 'ebit' in col_lower and 'revenue' not in col_lower:
            mapping[col] = 'ebit'
        elif 'pp&e' in col_lower or col_lower == 'ppe':
            mapping[col] = 'ppe'
        elif 'r&d' in col_lower or col_lower == 'rd':
            mapping[col] = 'rd'
        elif 'employee' in col_lower or 'personnel' in col_lower:
            mapping[col] = 'personnel'

    print("Mapping applied:", mapping)
    df = df.rename(columns=mapping)
    print("Columns after mapping:", df.columns.tolist())

    for col in ['year', 'ca', 'ebit', 'ppe', 'personnel', 'rd']:
        if col in df.columns:
            if col != 'year':
                df[col] = df[col].astype(str).str.replace(' ', '', regex=False)
                df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[col] = df[col].astype(str).str.replace('[^\d.-]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'year' in df.columns:
        available_years = sorted(df['year'].dropna().unique())
        print(f"Available years: {available_years}")

    if 'ca' in df.columns:
        df['revenue'] = df['ca']
    if 'ebit' in df.columns and 'ca' in df.columns:
        df['ebit_revenue'] = df['ebit'] / df['ca']

    if all(x in df.columns for x in ['ebit', 'personnel']):
        df['z1'] = df['ebit'] * df['personnel']
    if all(x in df.columns for x in ['ebit', 'ppe']):
        df['z2'] = df['ebit'] * df['ppe']
    if all(x in df.columns for x in ['ebit', 'rd']):
        df['z3'] = df['ebit'] * df['rd']
    if all(x in df.columns for x in ['rd', 'personnel']):
        df['z4'] = df['rd'] * df['personnel']
    if all(x in df.columns for x in ['ppe', 'personnel']):
        df['z9'] = df['ppe'] * df['personnel']

    return df

print("\nPreparing data...")
df = prepare_dataframe(df)
print(f"✓ Data prepared: {df.shape[0]} rows")

# ============================================================
# MLE CLASSES — IDENTICAL
# ============================================================

class StudentTMLE:
    """Student t distribution for negative values"""
    def __init__(self):
        self.nu = None
        self.mu = None
        self.sigma = None
        self.logL = None
        self.se_nu = None
        self.se_mu = None
        self.se_sigma = None

    def _logpdf(self, x, nu, mu, sigma):
        nu = max(nu, 2.1)
        sigma = max(sigma, 1e-8)
        const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * nu) - np.log(sigma)
        return const - ((nu + 1) / 2) * np.log1p((x - mu)**2 / (nu * sigma**2))

    def fit(self, X):
        X_clean = X[(~np.isnan(X)) & (X < 0)]
        if len(X_clean) < 10:
            return self

        def neg_loglike(params):
            nu, mu, sigma = params
            if nu <= 2 or sigma <= 0:
                return 1e10
            return -np.sum(self._logpdf(X_clean, nu, mu, sigma))

        mu0 = np.mean(X_clean)
        sigma0 = np.std(X_clean)
        result = minimize(neg_loglike, [3.0, mu0, sigma0],
                         method='L-BFGS-B',
                         bounds=[(2.1, 50), (-1000, 0), (0.001, 1000)])

        if result.success:
            self.nu, self.mu, self.sigma = result.x
            self.logL = -result.fun
            try:
                eps = 1e-6
                hess = np.zeros((3, 3))
                for i in range(3):
                    for j in range(3):
                        params = [self.nu, self.mu, self.sigma]
                        params[i] += eps
                        params[j] += eps
                        fpp = neg_loglike(params)
                        params[i] -= 2*eps
                        fmm = neg_loglike(params)
                        params[j] -= 2*eps
                        fpm = neg_loglike(params)
                        params[i] += 2*eps
                        fmp = neg_loglike(params)
                        hess[i, j] = (fpp + fmm - fpm - fmp) / (4 * eps**2)
                    self.se_nu, self.se_mu, self.se_sigma = np.sqrt(np.diag(np.linalg.inv(hess)))
            except:
                pass
        return self

    def expectation(self):
        return self.mu if self.mu is not None else 0

    def variance(self):
        if self.nu is None or self.mu is None or self.sigma is None:
            return 0
        if self.nu > 2:
            return (self.sigma**2) * self.nu / (self.nu - 2)
        return np.inf

    def pdf(self, x):
        if self.nu is None or self.mu is None or self.sigma is None:
            return 0
        return t.pdf(x, self.nu, self.mu, self.sigma)


class GammaMixture:
    """Gamma mixture with intra-block rule (BIC + KS)"""
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
        return (alpha - 1) * np.log(x) - x/beta - alpha * np.log(beta) - gammaln(alpha)

    def _fit_single(self, X, weights=None):
        if weights is None:
            weights = np.ones(len(X))
        weights = weights / weights.sum()

        mean_w = np.sum(weights * X)
        var_w = np.sum(weights * (X - mean_w)**2)

        if var_w > 0:
            alpha0 = min(max(mean_w**2 / var_w, 0.1), 500)
            beta0 = min(max(var_w / mean_w, 0.01), 1000)
        else:
            alpha0, beta0 = 2.0, mean_w/2

        def neg_loglike(params):
            a, b = params
            if a <= 0 or b <= 0:
                return 1e10
            return -np.sum(weights * self._logpdf(X, a, b))

        result = minimize(neg_loglike, [alpha0, beta0],
                         method='L-BFGS-B',
                         bounds=[(0.05, 1000), (0.001, 5000)])

        if result.success:
            return result.x
        return np.array([alpha0, beta0])

    def _cdf_mixture(self, x):
        val = 0
        for k in range(self.n_components):
            val += self.weights[k] * gamma.cdf(x, self.alphas[k], scale=self.betas[k])
        return val

    def _compute_tests(self, X):
        ks_stat, self.ks_pvalue = kstest(X, self._cdf_mixture)
        X_sorted = np.sort(X)
        n = len(X)
        cdf_emp = np.arange(1, n + 1) / n
        cdf_theo = np.array([self._cdf_mixture(x) for x in X_sorted])
        self.wasserstein = np.mean(np.abs(cdf_emp - cdf_theo))

    def fit(self, X):
        X_clean = X[(~np.isnan(X)) & (X > 0)]
        if len(X_clean) < 10:
            self.n_components = 1
            a, b = self._fit_single(X_clean)
            self.alphas = np.array([a])
            self.betas = np.array([b])
            self.weights = np.array([1.0])
            return self

        from sklearn.cluster import KMeans

        models = {}
        best_bic = np.inf
        best_K_by_bic = None

        for K in range(self.K_min, min(self.K_max, len(X_clean)//5) + 1):
            try:
                X_log = np.log(np.maximum(X_clean, 1e-8))
                kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
                labels = kmeans.fit_predict(X_log.reshape(-1, 1))

                alphas, betas, weights = [], [], []
                for k in range(K):
                    X_k = X_clean[labels == k]
                    if len(X_k) >= 3:
                        a, b = self._fit_single(X_k)
                    else:
                        a, b = self._fit_single(X_clean)
                    alphas.append(a)
                    betas.append(b)
                    weights.append(len(X_k) if len(X_k) >= 3 else 1)

                weights = np.array(weights) / np.sum(weights)
                alphas = np.array(alphas)
                betas = np.array(betas)

                logL_old = -np.inf
                for _ in range(100):
                    log_resp = np.zeros((len(X_clean), K))
                    for k in range(K):
                        log_resp[:, k] = self._logpdf(X_clean, alphas[k], betas[k]) + np.log(max(weights[k], 1e-10))

                    log_sum = logsumexp(log_resp, axis=1, keepdims=True)
                    resp = np.exp(log_resp - log_sum)

                    Nk = resp.sum(axis=0)
                    weights_new = Nk / len(X_clean)

                    alphas_new, betas_new = [], []
                    for k in range(K):
                        if Nk[k] > 2:
                            a, b = self._fit_single(X_clean, weights=resp[:, k])
                        else:
                            a, b = alphas[k], betas[k]
                        alphas_new.append(a)
                        betas_new.append(b)

                    alphas, betas, weights = np.array(alphas_new), np.array(betas_new), weights_new

                    log_probs = np.zeros((len(X_clean), K))
                    for k in range(K):
                        log_probs[:, k] = self._logpdf(X_clean, alphas[k], betas[k]) + np.log(max(weights[k], 1e-10))
                    logL = logsumexp(log_probs, axis=1).sum()

                    if abs(logL - logL_old) < 1e-6:
                        break
                    logL_old = logL

                n_params = 3*K - 1
                bic = np.log(len(X_clean)) * n_params - 2 * logL

                temp_model = GammaMixture(K_min=K, K_max=K)
                temp_model.n_components = K
                temp_model.alphas = alphas.copy()
                temp_model.betas = betas.copy()
                temp_model.weights = weights.copy()
                temp_model.logL = logL
                temp_model._compute_tests(X_clean)

                models[K] = {
                    'K': K,
                    'bic': bic,
                    'ks_pvalue': temp_model.ks_pvalue,
                    'model': temp_model
                }

                if bic < best_bic:
                    best_bic = bic
                    best_K_by_bic = K

            except Exception as e:
                continue

        if not models:
            self.n_components = 1
            a, b = self._fit_single(X_clean)
            self.alphas = np.array([a])
            self.betas = np.array([b])
            self.weights = np.array([1.0])
            self._compute_tests(X_clean)
            return self

        selected_K = best_K_by_bic

        if selected_K == 1 and 1 in models:
            ks1 = models[1]['ks_pvalue']
            if ks1 < 0.05:
                if 2 in models:
                    selected_K = 2
                    print(f"    Intra-block: BIC chose K=1 but KS={ks1:.3f} < 0.05 → K=2")
            elif ks1 > 0.10:
                selected_K = 1
                print(f"    Intra-block: BIC chose K=1 with KS={ks1:.3f} > 0.10 → Keep K=1")
            else:
                if 2 in models:
                    ks2 = models[2]['ks_pvalue']
                    if ks2 > 0.05:
                        selected_K = 2
                        print(f"    Intra-block: BIC K=1 (KS={ks1:.3f}), K=2 KS={ks2:.3f} > 0.05 → K=2")
                    else:
                        selected_K = 1
                        print(f"    Intra-block: BIC K=1 (KS={ks1:.3f}), K=2 also poor → Keep K=1")
                else:
                    selected_K = 1
        elif selected_K in [2, 3, 4]:
            print(f"    Intra-block: BIC chose K={selected_K} → Keep K={selected_K}")

        selected = models[selected_K]['model']
        self.n_components = selected.n_components
        self.alphas = selected.alphas
        self.betas = selected.betas
        self.weights = selected.weights
        self.logL = selected.logL
        self.ks_pvalue = selected.ks_pvalue
        self.wasserstein = selected.wasserstein

        return self

    def expectation(self):
        return np.sum(self.weights * self.alphas * self.betas)

    def variance(self):
        exp = self.expectation()
        exp2 = 0
        for k in range(self.n_components):
            a, b, w = self.alphas[k], self.betas[k], self.weights[k]
            exp2 += w * a * (a + 1) * (b**2)
        return exp2 - exp**2

    def pdf(self, x):
        if x <= 0:
            return 0
        val = 0
        for k in range(self.n_components):
            val += self.weights[k] * gamma.pdf(x, self.alphas[k], scale=self.betas[k])
        return val


class TwoBlockModel:
    """
    2-Block Hierarchical Model with OPTIMAL THRESHOLD
    - Searches threshold from 60% to 90% to maximize min(KS1, KS2)
    - Each block uses intra-block rule (BIC+KS) for K
    """
    def __init__(self, threshold_candidates=[60, 65, 70, 75, 80, 85, 90]):
        self.threshold_candidates = threshold_candidates
        self.threshold = None
        self.prop1 = None
        self.prop2 = None
        self.model1 = None
        self.model2 = None
        self.mean1 = None
        self.mean2 = None
        self.ks_pvalue = None
        self.wasserstein = None

    def fit(self, X):
        X_clean = X[(~np.isnan(X)) & (X > 0)]
        if len(X_clean) < 30:
            return self

        best_threshold = 60
        best_min_ks = 0
        best_models = None

        print(f"\n  Searching optimal threshold...")

        for thresh in self.threshold_candidates:
            cutoff = np.percentile(X_clean, thresh)
            block1 = X_clean[X_clean <= cutoff]
            block2 = X_clean[X_clean > cutoff]

            if len(block1) < 20 or len(block2) < 10:
                continue

            model1 = GammaMixture(K_min=1, K_max=4)
            model1.fit(block1)

            model2 = GammaMixture(K_min=1, K_max=4)
            model2.fit(block2)

            min_ks = min(model1.ks_pvalue, model2.ks_pvalue)

            ks1_str = f"{model1.ks_pvalue:.3f}(K={model1.n_components})"
            ks2_str = f"{model2.ks_pvalue:.3f}(K={model2.n_components})"
            print(f"    thresh={thresh}%: KS1={ks1_str}, KS2={ks2_str} → min={min_ks:.3f}")

            if min_ks > best_min_ks:
                best_min_ks = min_ks
                best_threshold = thresh
                best_models = (model1, model2)

        if best_models is None:
            return self

        self.threshold = best_threshold
        self.model1, self.model2 = best_models

        cutoff = np.percentile(X_clean, self.threshold)
        block1 = X_clean[X_clean <= cutoff]
        block2 = X_clean[X_clean > cutoff]

        self.mean1 = np.mean(block1)
        self.mean2 = np.mean(block2)
        self.prop1 = len(block1) / len(X_clean)
        self.prop2 = len(block2) / len(X_clean)

        print(f"\n  → Selected threshold: {self.threshold}%")
        print(f"    Block1: K={self.model1.n_components}, KS={self.model1.ks_pvalue:.3f}")
        print(f"    Block2: K={self.model2.n_components}, KS={self.model2.ks_pvalue:.3f}")

        self._compute_tests(X_clean)
        return self

    def _compute_tests(self, X):
        def cdf_mixture(x):
            val = 0
            for k in range(self.model1.n_components):
                val += self.model1.weights[k] * self.prop1 * gamma.cdf(x, self.model1.alphas[k], scale=self.model1.betas[k])
            for k in range(self.model2.n_components):
                val += self.model2.weights[k] * self.prop2 * gamma.cdf(x, self.model2.alphas[k], scale=self.model2.betas[k])
            return val

        ks_stat, self.ks_pvalue = kstest(X, cdf_mixture)

        X_sorted = np.sort(X)
        n = len(X)
        cdf_emp = np.arange(1, n + 1) / n
        cdf_theo = np.array([cdf_mixture(x) for x in X_sorted])
        self.wasserstein = np.mean(np.abs(cdf_emp - cdf_theo))

    def pdf(self, x):
        if x <= 0:
            return 0
        val = 0
        for k in range(self.model1.n_components):
            val += self.model1.weights[k] * self.prop1 * gamma.pdf(x, self.model1.alphas[k], scale=self.model1.betas[k])
        for k in range(self.model2.n_components):
            val += self.model2.weights[k] * self.prop2 * gamma.pdf(x, self.model2.alphas[k], scale=self.model2.betas[k])
        return val

    def expectation(self):
        e1 = self.model1.expectation()
        e2 = self.model2.expectation()
        return self.prop1 * e1 + self.prop2 * e2

    def variance(self):
        e_total = self.expectation()
        e1 = self.model1.expectation()
        e2 = self.model2.expectation()
        var1 = self.model1.variance()
        var2 = self.model2.variance()
        cond_var = self.prop1 * var1 + self.prop2 * var2
        cond_exp_var = self.prop1 * (e1 - e_total)**2 + self.prop2 * (e2 - e_total)**2
        return cond_var + cond_exp_var

    def get_structure(self):
        return (self.model1.n_components, self.model2.n_components)


class TwoPartModel:
    """
    Two-part model:
    - Student t for negatives
    - 2-Block Hierarchical for positives (optimal threshold)
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

        if n_neg >= 10:
            self.student = StudentTMLE()
            self.student.fit(X[neg_mask])
            print(f"  Student t fitted on {n_neg} negatives")

        if n_pos >= 10:
            self.twoblock = TwoBlockModel()
            self.twoblock.fit(X[pos_mask])
            print(f"  2-Block fitted on {n_pos} positives")
            self.ks_pvalue = self.twoblock.ks_pvalue
            self.wasserstein = self.twoblock.wasserstein

        return self

    def pdf(self, x):
        if x < 0:
            if self.student:
                return self.prob_neg * self.student.pdf(x)
            return 0
        elif x > 0:
            if self.twoblock:
                return self.prob_pos * self.twoblock.pdf(x)
            return 0
        else:
            return 0

    def expectation(self):
        e_neg = self.student.expectation() if self.student else 0
        e_pos = self.twoblock.expectation() if self.twoblock else 0
        return self.prob_neg * e_neg + self.prob_pos * e_pos

    def variance(self):
        e_total = self.expectation()
        e_neg = self.student.expectation() if self.student else 0
        e_pos = self.twoblock.expectation() if self.twoblock else 0
        var_neg = self.student.variance() if self.student else 0
        var_pos = self.twoblock.variance() if self.twoblock else 0
        cond_var = self.prob_neg * var_neg + self.prob_pos * var_pos
        cond_exp_var = self.prob_neg * (e_neg - e_total)**2 + self.prob_pos * (e_pos - e_total)**2
        return cond_var + cond_exp_var

    def get_structure(self):
        if self.twoblock:
            return self.twoblock.get_structure()
        return (0, 0)


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_ks(pvalue):
    if np.isnan(pvalue):
        return "NOT AVAILABLE"
    elif pvalue >= 0.10:
        return "VERY GOOD"
    elif pvalue >= 0.05:
        return "ACCEPTABLE"
    elif pvalue >= 0.01:
        return "POOR"
    else:
        return "REJECTED"

def evaluate_wasserstein(d):
    if np.isnan(d):
        return "NOT AVAILABLE"
    elif d < 0.01:
        return "EXCELLENT"
    elif d < 0.05:
        return "VERY GOOD"
    elif d < 0.10:
        return "ACCEPTABLE"
    elif d < 0.20:
        return "POOR"
    else:
        return "REJECTED"

def evaluate_quality(ks, wass):
    if ks in ["VERY GOOD"] and wass in ["EXCELLENT", "VERY GOOD"]:
        return "EXCELLENT"
    elif ks in ["VERY GOOD", "ACCEPTABLE"] and wass in ["VERY GOOD", "ACCEPTABLE"]:
        return "GOOD"
    elif ks == "ACCEPTABLE" or wass == "ACCEPTABLE":
        return "ACCEPTABLE"
    else:
        return "POOR"

def find_plot_limit(data):
    if len(data) < 100:
        return np.percentile(data, 95)
    if np.min(data) < 0:
        return np.percentile(data, 95)
    return np.percentile(data, 90)


# ============================================================
# ANALYZE VARIABLE
# ============================================================

def analyze_variable(df, var_name, display_name, period_name=""):
    if period_name:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {display_name} - {period_name}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print(f"ANALYZING: {display_name}")
        print(f"{'='*60}")

    if var_name in df.columns:
        data = df[var_name].dropna().values
    else:
        print(f"  Variable {var_name} not found")
        return None

    if len(data) < 10:
        print(f"  Too few data: {len(data)}")
        return None

    mean_emp = np.mean(data)
    std_emp = np.std(data)
    n_pos = np.sum(data > 0)
    n_neg = np.sum(data < 0)

    print(f"\n  Data: {len(data)} obs")
    print(f"    Positive: {n_pos} ({n_pos/len(data)*100:.1f}%)")
    print(f"    Negative: {n_neg} ({n_neg/len(data)*100:.1f}%)")

    result = {
        'variable': var_name,
        'display_name': display_name,
        'n_obs': len(data),
        'mean_emp': mean_emp,
        'std_emp': std_emp,
        'data': data
    }

    if n_neg >= 10 and var_name in ['ebit', 'ebit_revenue', 'z1', 'z2', 'z3']:
        print("\n  Using Two-Part model (Student t + 2-Block)")
        model = TwoPartModel()
        model.fit(data)
        result['model_type'] = 'Two-Part'
    else:
        data_pos = data[data > 0]
        if len(data_pos) < 10:
            print(f"  Too few positives: {len(data_pos)}")
            return None
        print("\n  Using 2-Block Hierarchical model")
        model = TwoBlockModel()
        model.fit(data_pos)
        result['model_type'] = '2-Block'

    result['model'] = model
    result['mle_expectation'] = model.expectation()
    result['mle_variance'] = model.variance()
    result['mle_std'] = np.sqrt(model.variance())

    if hasattr(model, 'ks_pvalue'):
        result['ks_pvalue'] = model.ks_pvalue
    if hasattr(model, 'wasserstein'):
        result['wasserstein'] = model.wasserstein

    if hasattr(model, 'get_structure'):
        result['structure'] = model.get_structure()
    else:
        result['structure'] = (0, 0)

    if hasattr(model, 'threshold') and model.threshold:
        result['threshold'] = model.threshold

    result['ks_eval'] = evaluate_ks(result.get('ks_pvalue', np.nan))
    result['wass_eval'] = evaluate_wasserstein(result.get('wasserstein', np.nan))
    result['quality'] = evaluate_quality(result['ks_eval'], result['wass_eval'])

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"  Empirical mean: {mean_emp:.4f}")
    print(f"  MLE expectation: {result['mle_expectation']:.4f}")
    print(f"  Difference: {result['mle_expectation'] - mean_emp:.4f}")
    print(f"\n  Structure: ({result['structure'][0]},{result['structure'][1]})")
    if 'threshold' in result:
        print(f"  Threshold: {result['threshold']}%")
    print(f"  KS test: {result['ks_eval']} (p={result.get('ks_pvalue', np.nan):.4f})")
    print(f"  Wasserstein: {result['wass_eval']} ({result.get('wasserstein', np.nan):.4f})")
    print(f"  Quality: {result['quality']}")

    return result


# ============================================================
# CREATE PDF
# ============================================================

def create_pdf(results, dataset, period_name):
    print(f"\n{'='*80}")
    print(f"CREATING PDF WITH FITS - {period_name}")
    print("="*80)

    pdf_path = f'/content/drive/MyDrive/fits_{dataset}_{period_name}.pdf'
    print(f"  PDF: {pdf_path}")

    with PdfPages(pdf_path) as pdf:
        fig = plt.figure(figsize=(11, 8.5))
        plt.axis('off')
        plt.text(0.5, 0.7, '2-BLOCK HIERARCHICAL MODEL',
                fontsize=24, ha='center', va='center', fontweight='bold')
        plt.text(0.5, 0.6, 'Optimal threshold (60-90%) | Intra-block BIC+KS rule',
                fontsize=16, ha='center', va='center')
        plt.text(0.5, 0.5, f'Dataset: {dataset}\nPeriod: {period_name}',
                fontsize=14, ha='center', va='center')
        pdf.savefig(fig)
        plt.close()

        for res in results:
            if res['variable'] in ['z1', 'z2', 'z3', 'z4', 'z9']:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            data = res['data']
            x_min = min(np.min(data), 0)
            x_max = find_plot_limit(data)
            data_plot = data[(data >= x_min) & (data <= x_max)]

            bins = min(50, int(len(data_plot)**0.5) * 2)
            ax.hist(data_plot, bins=bins, density=True,
                   color='white', edgecolor='black', linewidth=1,
                   alpha=1, label='Empirical')

            model = res['model']
            x_fit = np.linspace(x_min, x_max, 500)
            y_fit = np.array([model.pdf(x) for x in x_fit])

            if np.max(y_fit) > 0:
                hist_max = np.max(ax.get_ylim())
                y_fit = y_fit / np.max(y_fit) * hist_max * 0.9
                ax.plot(x_fit, y_fit, 'k-', linewidth=2, label='MLE fit')

            stats = (f"n = {len(data)}\n"
                    f"E[X] MLE = {res['mle_expectation']:.2f}\n"
                    f"Structure = ({res['structure'][0]},{res['structure'][1]})\n")

            if 'threshold' in res:
                stats += f"Threshold = {res['threshold']}%\n"

            stats += (f"KS: {res['ks_eval']} (p={res.get('ks_pvalue', np.nan):.3f})\n"
                     f"Wasserstein: {res['wass_eval']} ({res.get('wasserstein', np.nan):.4f})")

            ax.text(0.98, 0.98, stats, transform=ax.transAxes,
                   ha='right', va='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_xlabel(res['display_name'])
            ax.set_ylabel('Density')
            ax.set_title(f"{res['display_name']} - {dataset} ({period_name})")
            ax.legend(loc='upper left')
            ax.grid(True, alpha=0.3)

            pdf.savefig(fig)
            plt.close()
            print(f"  ✓ {res['display_name']}")

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')

        table_data = []
        for res in results:
            if res['variable'] in ['z1', 'z2', 'z3', 'z4', 'z9']:
                continue
            threshold_str = f"{res.get('threshold', '—')}%" if 'threshold' in res else "—"
            table_data.append([
                res['display_name'],
                f"{res['mle_expectation']:.2f}",
                f"({res['structure'][0]},{res['structure'][1]})",
                threshold_str,
                res['ks_eval'],
                res['wass_eval'],
                res['quality']
            ])

        col_labels = ['Variable', 'E[X] MLE', 'Structure', 'Threshold', 'KS', 'Wasserstein', 'Quality']
        table = ax.table(cellText=table_data, colLabels=col_labels,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        ax.set_title(f'SUMMARY - {dataset} ({period_name})', fontsize=16, pad=20)
        pdf.savefig(fig)
        plt.close()

    print(f"\n✅ PDF saved: {pdf_path}")


# ============================================================
# COVARIANCES AND CORRELATIONS
# ============================================================

def compute_covariances_correlations(results, dataset, period_name):
    print(f"\n{'='*80}")
    print(f"COVARIANCES AND CORRELATIONS - {period_name}")
    print("="*80)

    expectations = {r['variable']: r['mle_expectation'] for r in results}
    stds = {r['variable']: r['mle_std'] for r in results}

    covariances = []
    correlations_list = []

    if all(v in expectations for v in ['ebit', 'personnel', 'z1']):
        cov = expectations['z1'] - expectations['ebit'] * expectations['personnel']
        corr = cov / (stds['ebit'] * stds['personnel']) if stds['ebit'] > 0 and stds['personnel'] > 0 else np.nan
        covariances.append(('EBIT', 'Personnel', cov))
        correlations_list.append(('EBIT', 'Personnel', corr))
        print(f"\nCov(EBIT, Personnel) = {cov:.4f}")
        print(f"Corr(EBIT, Personnel) = {corr:.4f}")

    if all(v in expectations for v in ['ebit', 'ppe', 'z2']):
        cov = expectations['z2'] - expectations['ebit'] * expectations['ppe']
        corr = cov / (stds['ebit'] * stds['ppe']) if stds['ebit'] > 0 and stds['ppe'] > 0 else np.nan
        covariances.append(('EBIT', 'PP&E', cov))
        correlations_list.append(('EBIT', 'PP&E', corr))
        print(f"\nCov(EBIT, PP&E) = {cov:.4f}")
        print(f"Corr(EBIT, PP&E) = {corr:.4f}")

    if all(v in expectations for v in ['ebit', 'rd', 'z3']):
        cov = expectations['z3'] - expectations['ebit'] * expectations['rd']
        corr = cov / (stds['ebit'] * stds['rd']) if stds['ebit'] > 0 and stds['rd'] > 0 else np.nan
        covariances.append(('EBIT', 'R&D', cov))
        correlations_list.append(('EBIT', 'R&D', corr))
        print(f"\nCov(EBIT, R&D) = {cov:.4f}")
        print(f"Corr(EBIT, R&D) = {corr:.4f}")

    if all(v in expectations for v in ['rd', 'personnel', 'z4']):
        cov = expectations['z4'] - expectations['rd'] * expectations['personnel']
        corr = cov / (stds['rd'] * stds['personnel']) if stds['rd'] > 0 and stds['personnel'] > 0 else np.nan
        covariances.append(('R&D', 'Personnel', cov))
        correlations_list.append(('R&D', 'Personnel', corr))
        print(f"\nCov(R&D, Personnel) = {cov:.4f}")
        print(f"Corr(R&D, Personnel) = {corr:.4f}")

    if all(v in expectations for v in ['personnel', 'ppe', 'z9']):
        cov = expectations['z9'] - expectations['personnel'] * expectations['ppe']
        corr = cov / (stds['personnel'] * stds['ppe']) if stds['personnel'] > 0 and stds['ppe'] > 0 else np.nan
        covariances.append(('Personnel', 'PP&E', cov))
        correlations_list.append(('Personnel', 'PP&E', corr))
        print(f"\nCov(Personnel, PP&E) = {cov:.4f}")
        print(f"Corr(Personnel, PP&E) = {corr:.4f}")

    if covariances:
        df_cov = pd.DataFrame(covariances, columns=['Var1', 'Var2', 'Covariance'])
        df_cov.to_csv(f'/content/drive/MyDrive/covariances_{dataset}_{period_name}.csv', index=False)
        print(f"\n✓ Covariances saved")

    if correlations_list:
        df_corr = pd.DataFrame(correlations_list, columns=['Var1', 'Var2', 'Correlation'])
        df_corr.to_csv(f'/content/drive/MyDrive/correlations_{dataset}_{period_name}.csv', index=False)
        print(f"✓ Correlations saved")

    return covariances, correlations_list


# ============================================================
# 4x4 CORRELATION MATRIX
# ============================================================

def create_correlation_matrix(correlations_list, dataset, period_name):
    print(f"\n{'='*80}")
    print(f"CREATING 4x4 CORRELATION MATRIX - {period_name}")
    print("="*80)

    variables = ['EBIT', 'R&D', 'PP&E', 'Personnel']
    n = len(variables)
    matrix = np.full((n, n), np.nan)

    for i in range(n):
        matrix[i, i] = 1.0

    var_to_idx = {'EBIT': 0, 'R&D': 1, 'PP&E': 2, 'Personnel': 3}

    for v1, v2, corr in correlations_list:
        if v1 in var_to_idx and v2 in var_to_idx:
            i, j = var_to_idx[v1], var_to_idx[v2]
            matrix[i, j] = corr
            matrix[j, i] = corr

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                color = 'white' if abs(matrix[i, j]) > 0.5 else 'black'
                ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color=color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(variables)
    ax.set_yticklabels(variables)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    ax.set_title(f'Correlation Matrix - {dataset} ({period_name})')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    plt.savefig(f'/content/drive/MyDrive/correlation_matrix_{dataset}_{period_name}.png', dpi=300)
    print(f"✓ Correlation matrix image saved")

    df_matrix = pd.DataFrame(matrix, index=variables, columns=variables)
    df_matrix.to_csv(f'/content/drive/MyDrive/correlation_matrix_{dataset}_{period_name}.csv')
    print(f"✓ Correlation matrix CSV saved")

    plt.show()
    plt.close()

    return matrix


# ============================================================
# MEDIATION
# ============================================================

def mediation_rd(results, dataset, period_name):
    print(f"\n{'='*60}")
    print(f"MEDIATION: Personnel → R&D → EBIT - {period_name}")
    print("="*60)

    try:
        e_x = next(r['mle_expectation'] for r in results if r['variable'] == 'personnel')
        e_m = next(r['mle_expectation'] for r in results if r['variable'] == 'rd')
        e_y = next(r['mle_expectation'] for r in results if r['variable'] == 'ebit')
        e_z1 = next(r['mle_expectation'] for r in results if r['variable'] == 'z1')
        e_z3 = next(r['mle_expectation'] for r in results if r['variable'] == 'z3')
        e_z4 = next(r['mle_expectation'] for r in results if r['variable'] == 'z4')
        var_x = next(r['mle_variance'] for r in results if r['variable'] == 'personnel')
        var_m = next(r['mle_variance'] for r in results if r['variable'] == 'rd')
        var_y = next(r['mle_variance'] for r in results if r['variable'] == 'ebit')
        n = min(next(r['n_obs'] for r in results if r['variable'] == 'personnel'),
                next(r['n_obs'] for r in results if r['variable'] == 'rd'),
                next(r['n_obs'] for r in results if r['variable'] == 'ebit'))
    except Exception as e:
        print(f"  Missing data: {e}")
        return

    cov_xm = e_z4 - e_x * e_m
    cov_xy = e_z1 - e_x * e_y
    cov_my = e_z3 - e_m * e_y

    a = cov_xm / var_x if var_x != 0 else 0
    c = cov_xy / var_x if var_x != 0 else 0

    if var_x != 0:
        num = cov_my - (cov_xm * cov_xy / var_x)
        den = var_m - (cov_xm**2 / var_x)
        b = num / den if den != 0 else 0
        c_prime = c - a * b
    else:
        b = 0
        c_prime = 0

    var_resid_m = var_m - a**2 * var_x
    se_a = np.sqrt(var_resid_m / (n * var_x)) if var_x > 0 else 0

    var_resid_y = var_y - c_prime**2 * var_x - b**2 * var_m - 2 * c_prime * b * cov_xm
    var_resid_y = max(var_resid_y, 1e-10)

    se_b = np.sqrt(var_resid_y / (n * var_m)) if var_m > 0 else 0
    se_c_prime = np.sqrt(var_resid_y / (n * var_x)) if var_x > 0 else 0

    indirect = a * b
    se_indirect = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)

    if se_indirect > 0:
        z = indirect / se_indirect
        p = 2 * (1 - norm.cdf(abs(z)))
    else:
        z, p = 0, 1.0

    prop = indirect / c if c != 0 else np.nan

    print(f"\nPath coefficients:")
    print(f"  a (Personnel→R&D)     = {a:.4f} (SE = {se_a:.4f})")
    print(f"  b (R&D→EBIT)          = {b:.4f} (SE = {se_b:.4f})")
    print(f"  c (total)             = {c:.4f}")
    print(f"  c' (direct)           = {c_prime:.4f} (SE = {se_c_prime:.4f})")
    print(f"\nIndirect effect (a×b)   = {indirect:.4f}")
    print(f"  SE                     = {se_indirect:.4f}")
    print(f"  Sobel z                = {z:.4f}")
    print(f"  p-value                = {p:.6f}")
    print(f"  Proportion mediated    = {prop:.2%}")
    print(f"\nSignificant at 5%?       = {'YES' if p < 0.05 else 'NO'}")

    res_dict = {
        'Dataset': dataset, 'Period': period_name,
        'Model': 'Personnel → R&D → EBIT', 'n': n,
        'a': a, 'se_a': se_a, 'b': b, 'se_b': se_b,
        'c': c, 'c_prime': c_prime, 'se_c_prime': se_c_prime,
        'indirect': indirect, 'se_indirect': se_indirect,
        'z_sobel': z, 'p_value': p, 'prop_mediated': prop,
        'significant': p < 0.05
    }

    pd.DataFrame([res_dict]).to_csv(
        f'/content/drive/MyDrive/mediation_RD_{dataset}_{period_name}.csv',
        index=False
    )
    print(f"\n✓ Results saved")


def mediation_ppe(results, dataset, period_name):
    print(f"\n{'='*60}")
    print(f"MEDIATION: Personnel → PP&E → EBIT - {period_name}")
    print("="*60)

    try:
        e_x = next(r['mle_expectation'] for r in results if r['variable'] == 'personnel')
        e_m = next(r['mle_expectation'] for r in results if r['variable'] == 'ppe')
        e_y = next(r['mle_expectation'] for r in results if r['variable'] == 'ebit')
        e_z1 = next(r['mle_expectation'] for r in results if r['variable'] == 'z1')
        e_z2 = next(r['mle_expectation'] for r in results if r['variable'] == 'z2')
        e_z9 = next(r['mle_expectation'] for r in results if r['variable'] == 'z9')
        var_x = next(r['mle_variance'] for r in results if r['variable'] == 'personnel')
        var_m = next(r['mle_variance'] for r in results if r['variable'] == 'ppe')
        var_y = next(r['mle_variance'] for r in results if r['variable'] == 'ebit')
        n = min(next(r['n_obs'] for r in results if r['variable'] == 'personnel'),
                next(r['n_obs'] for r in results if r['variable'] == 'ppe'),
                next(r['n_obs'] for r in results if r['variable'] == 'ebit'))
    except Exception as e:
        print(f"  Missing data: {e}")
        return

    cov_xm = e_z9 - e_x * e_m
    cov_xy = e_z1 - e_x * e_y
    cov_my = e_z2 - e_m * e_y

    a = cov_xm / var_x if var_x != 0 else 0
    c = cov_xy / var_x if var_x != 0 else 0

    if var_x != 0:
        num = cov_my - (cov_xm * cov_xy / var_x)
        den = var_m - (cov_xm**2 / var_x)
        b = num / den if den != 0 else 0
        c_prime = c - a * b
    else:
        b = 0
        c_prime = 0

    var_resid_m = var_m - a**2 * var_x
    se_a = np.sqrt(var_resid_m / (n * var_x)) if var_x > 0 else 0

    var_resid_y = var_y - c_prime**2 * var_x - b**2 * var_m - 2 * c_prime * b * cov_xm
    var_resid_y = max(var_resid_y, 1e-10)

    se_b = np.sqrt(var_resid_y / (n * var_m)) if var_m > 0 else 0
    se_c_prime = np.sqrt(var_resid_y / (n * var_x)) if var_x > 0 else 0

    indirect = a * b
    se_indirect = np.sqrt(a**2 * se_b**2 + b**2 * se_a**2)

    if se_indirect > 0:
        z = indirect / se_indirect
        p = 2 * (1 - norm.cdf(abs(z)))
    else:
        z, p = 0, 1.0

    prop = indirect / c if c != 0 else np.nan

    print(f"\nPath coefficients:")
    print(f"  a (Personnel→PP&E)    = {a:.4f} (SE = {se_a:.4f})")
    print(f"  b (PP&E→EBIT)         = {b:.4f} (SE = {se_b:.4f})")
    print(f"  c (total)             = {c:.4f}")
    print(f"  c' (direct)           = {c_prime:.4f} (SE = {se_c_prime:.4f})")
    print(f"\nIndirect effect (a×b)   = {indirect:.4f}")
    print(f"  SE                     = {se_indirect:.4f}")
    print(f"  Sobel z                = {z:.4f}")
    print(f"  p-value                = {p:.6f}")
    print(f"  Proportion mediated    = {prop:.2%}")
    print(f"\nSignificant at 5%?       = {'YES' if p < 0.05 else 'NO'}")

    res_dict = {
        'Dataset': dataset, 'Period': period_name,
        'Model': 'Personnel → PP&E → EBIT', 'n': n,
        'a': a, 'se_a': se_a, 'b': b, 'se_b': se_b,
        'c': c, 'c_prime': c_prime, 'se_c_prime': se_c_prime,
        'indirect': indirect, 'se_indirect': se_indirect,
        'z_sobel': z, 'p_value': p, 'prop_mediated': prop,
        'significant': p < 0.05
    }

    pd.DataFrame([res_dict]).to_csv(
        f'/content/drive/MyDrive/mediation_PPE_{dataset}_{period_name}.csv',
        index=False
    )
    print(f"\n✓ Results saved")


# ============================================================
# MAIN ANALYSIS FUNCTION FOR ONE REGIME
# ============================================================

def analyze_regime(df_regime, regime_name):
    print(f"\n{'='*80}")
    print(f"ANALYZING REGIME: {regime_name}")
    print(f"{'='*80}")
    print(f"Observations: {len(df_regime)}")

    if len(df_regime) < 30:
        print(f"  ⚠ Insufficient data for {regime_name}: {len(df_regime)} observations (<30)")
        return False

    results = []
    for var_name, display_name in variables_analysis:
        res = analyze_variable(df_regime, var_name, display_name, regime_name)
        if res:
            results.append(res)

    print(f"\n✓ {len(results)} variables analyzed for {regime_name}")

    if results:
        df_main = pd.DataFrame([{
            'Variable': r['display_name'],
            'E_X_MLE': r['mle_expectation'],
            'Std_MLE': r['mle_std'],
            'Std_Emp': r['std_emp'],
            'Structure': f"({r['structure'][0]},{r['structure'][1]})",
            'Threshold': r.get('threshold', np.nan),
            'KS_pvalue': r.get('ks_pvalue', np.nan),
            'KS_Eval': r['ks_eval'],
            'Wasserstein': r.get('wasserstein', np.nan),
            'Wasserstein_Eval': r['wass_eval'],
            'Quality': r['quality']
        } for r in results])

        df_main.to_csv(f'/content/drive/MyDrive/MLE_results_{regime_name}.csv', index=False)
        print(f"\n✓ Main results saved for {regime_name}")

        create_pdf(results, regime_name, regime_name)

        covariances, correlations = compute_covariances_correlations(results, regime_name, regime_name)

        if correlations:
            create_correlation_matrix(correlations, regime_name, regime_name)

        print(f"\n{'='*60}")
        print(f"MEDIATION ANALYSES - {regime_name}")
        print("="*60)
        mediation_rd(results, regime_name, regime_name)
        mediation_ppe(results, regime_name, regime_name)

        return True
    return False


# ============================================================
# ANALYSIS VARIABLES
# ============================================================

variables_analysis = [
    ('ebit', 'EBIT'),
    ('rd', 'R&D'),
    ('ppe', 'PP&E'),
    ('personnel', 'Personnel'),
    ('revenue', 'Revenue'),
    ('ebit_revenue', 'EBIT/Revenue'),
    ('z1', 'EBIT × Personnel'),
    ('z2', 'EBIT × PP&E'),
    ('z3', 'EBIT × R&D'),
    ('z4', 'R&D × Personnel'),
    ('z9', 'Personnel × PP&E')
]


# ============================================================
# MAIN EXECUTION — SHORT vs LONG
# ============================================================

print("\n" + "="*80)
print("STARTING COMPLETE ANALYSIS — SHORT vs LONG")
print("="*80)

REGIMES = ['SHORT', 'LONG']
global_results = []

for regime in REGIMES:
    df_regime = df[df['REGIME'] == regime].copy()
    n_firms = df_regime['company'].nunique() if 'company' in df_regime.columns else '?'

    print(f"\n{'='*80}")
    print(f"🔍 REGIME: {regime}  —  {len(df_regime)} obs, {n_firms} firms")
    print(f"{'='*80}")

    if len(df_regime) < 30:
        print(f"  ⚠ Insufficient data ({len(df_regime)} < 30), regime ignored.")
        continue

    if analyze_regime(df_regime, regime):
        global_results.append(regime)
        print(f"  ✓ Analysis completed for {regime}")
    else:
        print(f"  ✗ Analysis failed for {regime}")


# ============================================================
# SUMMARY TABLE — SHORT vs LONG
# ============================================================

print(f"\n{'='*80}")
print("SUMMARY — SHORT vs LONG")
print(f"{'='*80}")

recap_rows = []
for regime in global_results:
    path = f'/content/drive/MyDrive/correlations_{regime}_{regime}.csv'
    if os.path.exists(path):
        corr_df = pd.read_csv(path)
        row = {'Regime': regime}
        for _, r in corr_df.iterrows():
            key = f"{r['Var1']}–{r['Var2']}"
            row[key] = r['Correlation']
        recap_rows.append(row)

if recap_rows:
    recap = pd.DataFrame(recap_rows)
    print(recap.to_string(index=False))
    recap.to_csv('/content/drive/MyDrive/RECAP_SHORT_LONG.csv', index=False)
    print(f"\n✓ Saved: /content/drive/MyDrive/RECAP_SHORT_LONG.csv")

print(f"\n📊 SUMMARY:")
print(f"   Regimes analyzed: {global_results}")
print(f"\n📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")
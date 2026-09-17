Laanait, L. (2026). Remanent Surplus Value: An Essay in the Political Economy
of Cognitive Capitalism. Working paper.

## Data

The panel  "panel_FINAL_REMANENT.csv" covers publicly listed firms from 2015 to 2024, across three production regimes:

| Regime | Firms | Obs. |
|--------|-------|------|
| TECH   | 74    | 713  |
| PHARMA | 37    | 370  |
| INDUS  | 417   | 4128 |

The panel excludes clinical-stage firms and duplicates. All values are in harmonized currency.

**Columns:**
- `company` — firm name
- `year` — fiscal year
- `ca` — revenue
- `ppe` — property, plant and equipment
- `ebit` — earnings before interest and taxes
- `rd` — research and development expenditure
- `personnel` — number of employees
- `groupe` — production regime (TECH, PHARMA, INDUS)

See `data/README.md` for details on exclusions.

---

## Code

The `code/` folder contains three Python scripts. They are designed to be run in order.

### `01_hierarchical_Model.py` — Hierarchical MLE model

**What it does:** Estimates the hierarchical maximum likelihood model for each production regime (TECH, PHARMA, INDUS). For each variable (EBIT, Personnel, R&D, PP&E, EBIT/Revenue, and the product variables Z = X × Y), the script fits:

- a **two-block segmentation** (optimal threshold 60–90%) for positive variables;
- a **Gamma mixture** within each block, with the number of components selected by a combined BIC + KS criterion;
- a **two-part specification** (Student-t for losses + Gamma mixture for profits) for variables that can take negative values.

The script then computes covariances and correlations from the product variables.

**Input:** `data/panel_FINAL_REMANENT.csv`

**Output:**
- `results/correlations/correlations_<regime>_2015-2024_<run_id>.csv`
- `results/margins/marge_mle_<regime>_2015-2024_<run_id>.cs

python code/02_generate_graphs.py
python code/03_fisher_z_bootstrap.py


cd code

# Step 1: Estimate the hierarchical MLE model
python 01_hierarchical_mle.py

# Step 2: Generate the recapitulative graphs
python 02_generate_graphs.py

# Step 3: Run Fisher z tests and bootstrap
python 03_fisher_z_bootstrap.py


## Relation to the Additive Note
  ## Panel composition — `panel_LONG_vs_SHORT.csv`

### Summary

The cleaned panel    " Note_panel_LONG_vs_SHORT.csv" contains **4,660 observations** across **474 firms**,
covering the period **2015–2024** (10 fiscal years).

### Composition by regime

| Regime | Firms | Observations | Share of firms | Share of obs |
|---|---|---|---|---|
| 1-cycle (SHORT) | **363** | **3,582** | 76.6 % | 76.9 % |
| 2-cycle (LONG) | **111** | **1,078** | 23.4 % | 23.1 % |
| **Total** | **474** | **4,660** | 100 % | 100 % |

### Composition by sector (before normalization)

| Sector | Observations |
|---|---|
| INDUS | 3,582 |
| TECH | 708 |
| PHARMA | 370 |
| **Total** | **4,660** |

The kernel maps sectors to regimes:

- `INDUS → SHORT`
- `TECH → LONG`
- `PHARMA → LONG`

### Why SHORT is larger than LONG

The SHORT regime aggregates **363 industrial firms**. The LONG regime
aggregates **111 firms** across two sectors:

- TECH firms,
- PHARMA firms.

The number of firms is therefore not balanced. This is a structural
feature of the panel, not an artifact. The asymmetry is handled in the
note by comparing **within-regime** correlations and by using the MLE
estimator, which is robust to sample size differences.

### Notes on the sample size

- SHORT has approximately **3.3 times more firms** than LONG.
- SHORT has approximately **3.3 times more observations** than LONG.
- The minimum sample size for a regime to be analysed is **30 observations**
  (checked by the kernel before any fit).
- At high lags (k = 4, 5, 6), the effective sample size per regime
  decreases because firms must survive k years to enter the computation.
  The `n` column of `TEMPORAL_SUMMARY_SHORT_LONG.csv` reports this.
The two scripts ( Note_MLE_Correlations_Mediation.py , Note_Temporality_After_KernelMLE.py ) in this repository ) constitute the
empirical apparatus of the **Additive Note** (Laanait, 2026), which
accompanies the article *Remanent Surplus Value*. They construct the
two-regime panel, estimate the MLE correlations and expectations by
regime and mediation and test the temporal signature of augmented remanent labor.

## How to run the temporal analysis   Note_Temporality_After_KernelMLE.py 

### 1. Prerequisites

The main kernel Note_MLE_Correlations_Mediation.py must already have been executed in the same Colab session.
It defines:

- `df` — the prepared panel, with column `REGIME` in `{SHORT, LONG}`
- `TwoPartModel` — the hierarchical MLE class
- `np`, `pd`, `plt` — standard imports


### 2. File placement

Create a new cell below the main kernel. Paste the entire content of
`temporal_analysis.py` into that cell.

### 3. Path configuration

The script writes its outputs to:

    /content/drive/MyDrive/

If your Drive folder is different, edit the three paths inside the script:

```python
df_temp.to_csv(f'/content/drive/MyDrive/temporal_{regime_name}.csv', ...)
summary.to_csv('/content/drive/MyDrive/TEMPORAL_SUMMARY_SHORT_LONG.csv', ...)
plt.savefig('/content/drive/MyDrive/temporal_signature_SHORT_LONG.png', ...)

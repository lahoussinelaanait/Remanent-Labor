Laanait, L. (2026). Remanent Surplus Value: An Essay in the Political Economy
of Cognitive Capitalism. Working paper.

## Data

The panel covers publicly listed firms from 2015 to 2024, across three production regimes:

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

### `01_hierarchical_mle.py` — Hierarchical MLE model

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

Outputs are saved to results/correlations/, results/margins/, and results/figures/.

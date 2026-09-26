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

# RVI — Remanent Value Indicator

code: ## `Remanent_Value_Indicator_calculation.py`

```python
#!/usr/bin/env python3

Panel: Global_Panel_2005_2024_RECLASSIFIED.csv

An empirical instrument for detecting the realization of **remanent value** in listed firms.

This repository provides the replication code for the paper:

> Laanait, L. (2026). *The Remanent Value Indicator (RVI): An Empirical Instrument for Detecting the Realization of Remanent Value in Cognitive Capitalism.*

---

## Concept

Remanent value is the value transmitted to each copy of a commodity by accumulated scientific labor — socialized in its origin, unwearable in its nature, and unpaid in the present. The RVI aims to capture the **successful monetization of an augmented remanent labor**: it rewards commercially viable, stable, large-scale innovation.

## Formula

The **creative effort term** $A$:

$$A = \frac{\bar{r} \times \left(1 + \ln\left(1 + \frac{RD_{\text{median}}}{RD_0}\right)\right)}{1 + \sigma_r}$$

The **final RVI**:

$$\boxed{RVI = \frac{\bar{m} \times A}{1 + \sigma_m}}$$

where:

| Symbol | Definition | Unit |
|--------|-----------|------|
| $\bar{m}$ | Average EBIT / Revenue over the period | dimensionless |
| $\bar{r}$ | Average RD / Revenue over the period | dimensionless |
| $RD_{\text{median}}$ | Median R&D over the period | billion USD |
| $RD_0$ | Reference value = 1.0 | billion USD |
| $\sigma_m$ | Standard deviation of EBIT / Revenue | dimensionless |
| $\sigma_r$ | Standard deviation of RD / Revenue | dimensionless |

## Method

- **Overlapping 5-year windows**, step of 2 years: `2006-2010, 2008-2012, ..., 2020-2024` (8 periods).
- **Strict filter**: each firm must have **all 5 years** of complete data (Revenue, EBIT, RD) within a window.
- **No global statistics on the entire panel**: each firm is treated independently.
- **Currencies harmonized** to billion USD using annual average exchange rates.

## Panel

| Sector | Number of firms |
|--------|----------------|
| PHARMA | 43 |
| TECH   | 87 |
| INDUS  | 169 |
| AUTO   | 32 |
| **Total** | **331** |

## Expected output

The code reproduces the sectoral hierarchy:

$$\text{PHARMA} > \text{TECH} > \text{INDUS} > \text{AUTO}$$

preserved across all 8 periods.

### Average RVI (top 6) by sector and period

| Period | PHARMA | TECH | INDUS | AUTO |
|--------|--------|------|-------|------|
| 2006-2010 | 0.1180 | 0.1086 | 0.0277 | 0.0070 |
| 2008-2012 | 0.1428 | 0.1281 | 0.0324 | 0.0079 |
| 2010-2014 | 0.1727 | 0.1546 | 0.0402 | 0.0108 |
| 2012-2016 | 0.1666 | 0.1547 | 0.0421 | 0.0124 |
| 2014-2018 | 0.1688 | 0.1643 | 0.0501 | 0.0134 |
| 2016-2020 | 0.1920 | 0.1762 | 0.0551 | 0.0132 |
| 2018-2022 | 0.2168 | 0.1805 | 0.0651 | 0.0147 |
| 2020-2024 | 0.2413 | 0.2145 | 0.0779 | 0.0165 |

## Installation

```bash
pip install -r requirements.txt




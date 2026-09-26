# =============================================================
# RVI — Remanent Value Indicator
# Google Colab version (no argparse)
# =============================================================

import pandas as pd
import numpy as np

# ---------- CONFIGURATION ----------
INPUT_PATH  = "/content/drive/MyDrive/verification/Global_Panel_2005_2024_RECLASSIFIED.csv"
OUTPUT_PATH = "/content/drive/MyDrive/verification/RVI_article_5y.csv"

SEP         = ";"
ENCODING    = "utf-8-sig"
MIN_YEARS   = 5
RD0         = 1.0
TOP_N       = 6

PERIODS = [
    (2006, 2010),
    (2008, 2012),
    (2010, 2014),
    (2012, 2016),
    (2014, 2018),
    (2016, 2020),
    (2018, 2022),
    (2020, 2024),
]

SECTORS = ["PHARMA", "TECH", "INDUS", "AUTO"]


# ---------- LOADING ----------
def load_panel(path, sep=SEP, encoding=ENCODING):
    df = pd.read_csv(path, sep=sep, encoding=encoding)
    for col in ["Revenue (Billion )", "EBIT", "RD", "Year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Company", "Year", "Sector_Final"])
    df["Year"] = df["Year"].astype(int)
    return df.reset_index(drop=True)


# ---------- COMPUTATION ----------
def compute_rvi(df, periods, min_years=MIN_YEARS, rd0=RD0):
    df = df.rename(columns={"Revenue (Billion )": "Revenue"})
    df = df.dropna(subset=["Revenue", "EBIT", "RD"])
    df = df[df["Revenue"] > 0].copy()
    df["m"] = df["EBIT"] / df["Revenue"]
    df["r"] = df["RD"] / df["Revenue"]

    records = []
    for y0, y1 in periods:
        label = f"{y0}-{y1}"
        window = df[(df["Year"] >= y0) & (df["Year"] <= y1)]
        for company, g in window.groupby("Company", sort=False):
            if g["Year"].nunique() < min_years or len(g) < min_years:
                continue
            g = g.sort_values("Year")
            m_bar = g["m"].mean()
            r_bar = g["r"].mean()
            rd_med = g["RD"].median()
            sigma_m = g["m"].std(ddof=1)
            sigma_r = g["r"].std(ddof=1)
            A = r_bar * (1 + np.log(1 + rd_med / rd0)) / (1 + sigma_r)
            RVI = m_bar * A / (1 + sigma_m)
            records.append({
                "Company": company,
                "Sector_Final": g["Sector_Final"].iloc[0],
                "Period": label,
                "n_years": int(len(g)),
                "RVI": RVI,
                "m_bar": m_bar,
                "r_bar": r_bar,
                "RD_median": rd_med,
                "sigma_m": sigma_m,
                "sigma_r": sigma_r,
                "A": A,
            })
    return pd.DataFrame(records)


# ---------- RUN ----------
print(f"→ Loading panel from {INPUT_PATH} …")
df = load_panel(INPUT_PATH)
print(f"  {len(df):,} rows | {df['Company'].nunique()} firms | "
      f"{df['Year'].min()}–{df['Year'].max()}")

print(f"→ Computing RVI over {len(PERIODS)} period(s) "
      f"(strict {MIN_YEARS}/5 years) …")
rvi = compute_rvi(df, PERIODS)
rvi.to_csv(OUTPUT_PATH, sep=SEP, index=False, encoding=ENCODING)
print(f"✅ Wrote {len(rvi):,} rows to {OUTPUT_PATH}")


# ---------- SUMMARY ----------
top = (rvi.sort_values(["Period", "Sector_Final", "RVI"],
                       ascending=[True, True, False])
          .groupby(["Period", "Sector_Final"]).head(TOP_N))
pivot = top.groupby(["Period", "Sector_Final"])["RVI"].mean().unstack()
pivot = pivot[[s for s in SECTORS if s in pivot.columns]]

print("\n" + "=" * 72)
print(f"Average RVI (top {TOP_N}) by sector and period")
print("=" * 72)
print(pivot.round(4).to_string())

print("\n" + "=" * 72)
print("Global ranking (mean across periods)")
print("=" * 72)
print(pivot.mean().sort_values(ascending=False).round(4).to_string())

print("\n" + "=" * 72)
print("Coverage — firms per period")
print("=" * 72)
print(rvi.groupby("Period")["Company"].nunique().to_string())
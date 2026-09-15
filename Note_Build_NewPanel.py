# ============================================================
# BUILD THE PANEL: panel_LONG_vs_SHORT.csv
# From panel_FINAL_REMANENT.csv
# Regimes:
#   SHORT = INDUS (cleaned)
#   LONG  = TECH + PHARMA
# Excluded: SERVICES + beverages, food, textile, sport, FMCG
# ============================================================

!pip install pandas numpy -q

from google.colab import drive
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

drive.mount('/content/drive')

# ============================================================
# 1. LOAD ORIGINAL PANEL
# ============================================================
panel_file = "/content/drive/My Drive/panel_FINAL_REMANENT.csv"
df = pd.read_csv(panel_file, sep=';', encoding='utf-8-sig')

# Rename columns
df = df.rename(columns={
    'ca': 'Revenue',
    'groupe': 'sector',
    'company': 'firm_id',
    'ppe': 'PPE',
    'ebit': 'EBIT',
    'rd': 'RD',
    'personnel': 'Personnel'
})

# Numeric conversion
for c in ['year', 'Revenue', 'PPE', 'EBIT', 'RD', 'Personnel']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

print(f"Original panel : {len(df)} obs, {df['firm_id'].nunique()} firms")
print(df['sector'].value_counts())

# ============================================================
# 2. FIRMS TO REMOVE FROM INDUS
# ============================================================
FIRMS_TO_REMOVE = {
    # === BEVERAGES ===
    'China Resources Beer', 'Chongqing Brewery', 'Coca-Cola (US)',
    'Gujinggong Liquor', 'Jiugui Liquor', 'Kweichow Moutai',
    'Luzhou Laojiao', 'PepsiCo (US)', 'Wuliangye Yibin', 'Yanghe Brewery',
    # === FOOD ===
    'Angel Yeast', 'Anjoy Food', 'Cofco Sugar', 'Dali Foods Group',
    'Danone (EU)', 'Foshan Haitian', 'General Mills (US)',
    'JinJian Rice', 'Juewei Food', 'Kraft Heinz (US)',
    'Mengniu Dairy', 'Mondelez (US)', 'Muyuan Foods', 'Nestlé (EU)',
    'Tingyi', 'Uni-President China', 'Want Want China',
    'Wens Foodstuff Group', 'Yili Group',
    # === TEXTILE ===
    'Inditex (EU)', 'Kering (EU)', 'LVMH (EU)', 'Xingyuan Textile',
    # === SPORT ===
    'Adidas (EU)', 'Anta Sports', 'Huali Industrial',
    'Li-Ning', 'Puma (EU)', 'Shenzhou Intl',
    # === FMCG / HYGIENE / COSMETICS ===
    'Procter & Gamble (US)', 'Unilever (EU)', 'Henkel (EU)',
    'Beiersdorf (EU)', 'Colgate-Palmolive (US)', 'Estee Lauder (US)',
    'Shiseido (JP)', 'Kao Corp (JP)', 'AmorePacific (KR)',
    'Hengan International', 'Vinda International', 'Blue Moon Group',
    # === FERRARI (luxury) ===
    'Ferrari (EU)',
}

print(f"\nFirms to remove from INDUS: {len(FIRMS_TO_REMOVE)}")

# ============================================================
# 3. BUILD REGIME LABEL
# ============================================================
# Exclude SERVICES
df = df[df['sector'] != 'SERVICES'].copy()

# Assign regime
df['regime'] = 'OTHER'
df.loc[df['sector'].isin(['TECH', 'PHARMA']), 'regime'] = 'LONG'
df.loc[(df['sector'] == 'INDUS')
       & (~df['firm_id'].isin(FIRMS_TO_REMOVE)), 'regime'] = 'SHORT'

# Keep only SHORT and LONG
df_new = df[df['regime'].isin(['SHORT', 'LONG'])].copy()

print(f"\nNew panel (before cleaning):")
print(df_new.groupby(['regime', 'sector']).agg(
    firms=('firm_id', 'nunique'),
    obs=('year', 'count')
).reset_index().to_string(index=False))

# ============================================================
# 4. CLEANING
# ============================================================
df_new = df_new[(df_new['Revenue'] > 0)
                & (df_new['PPE'] > 0)
                & (df_new['Personnel'] > 0)
                & (df_new['RD'] > 0)]
df_new = df_new.dropna(subset=['Revenue', 'PPE', 'EBIT',
                                'RD', 'Personnel'])
df_new = df_new.sort_values(['firm_id', 'year']).reset_index(drop=True)

print(f"\nAfter cleaning:")
print(df_new.groupby('regime').agg(
    firms=('firm_id', 'nunique'),
    obs=('year', 'count')
).to_string())

# ============================================================
# 5. SAVE
# ============================================================
out_dir = Path("/content/drive/My Drive/new_panel/")
out_dir.mkdir(exist_ok=True)

out_file = out_dir / "panel_LONG_vs_SHORT.csv"
df_new.to_csv(out_file, sep=';', index=False, encoding='utf-8-sig')

print(f"\n{'='*70}")
print(f"✅ Panel saved : {out_file}")
print(f"   Total        : {len(df_new)} obs, "
      f"{df_new['firm_id'].nunique()} firms")
print(f"   SHORT firms  : "
      f"{df_new[df_new['regime']=='SHORT']['firm_id'].nunique()}")
print(f"   LONG firms   : "
      f"{df_new[df_new['regime']=='LONG']['firm_id'].nunique()}")
print(f"{'='*70}")

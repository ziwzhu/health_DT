# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:25:47 2026

@author: ziz4012

Run models for level 1 to predict clustering based on characteristics
"""
import pandas as pd
import numpy as np
from scipy import stats
from config import (
    DATA_FILE,
    PATIENT_CAT_COLUMNS,
    PATIENT_BIN_COLUMNS,
    PATIENT_NUM_COLUMNS,
    COMORBIDITY_COLUMNS,
    LABEL_COLUMN,
    ALT_LABEL_COLUMNS,
)

# 1. Load the cat_processed file (after grouping, before one-hot)
df = pd.read_csv("processed_data.csv")
target_col = "target"
print(f"Loaded{df.shape[0]} rows. Target: {target_col}\n")
print(f"Target distribution: \n {df[target_col].value_counts()}\n")

results = []

# 2. Numeric + Binary: mean by group + ANOVA
num_bin_cols = PATIENT_NUM_COLUMNS + COMORBIDITY_COLUMNS + PATIENT_BIN_COLUMNS + ["comorbid_count"]
print("=" * 60)
print("Numeric/Binary Variables")
print("=" * 60)

num_rows = []
for col in num_bin_cols:
    if col not in df.columns:
        continue
    groups = [g.dropna() for _, g in df.groupby(target_col)[col]]
    
    if len(groups) < 2:
        continue
    
    f_stat, p_val = stats.f_oneway(*groups)
    means = df.groupby(target_col)[col].mean()
    row = {"variable": col}
    for label, val in means.items():
        row[str(label)] = round(val, 3)
    row["p_value"] = p_val
    num_rows.append(row)
    
    results.append({"variable": col, "type": "numeric/binary", "p_value": p_val})
    # print(f"\n{col} (ANOVA p = {p_val:.2e})")
    # print(means.to_string())

num_df = pd.DataFrame(num_rows)
print(num_df.to_string(index = False))
    
# 3. Categorical: chisq
print("=" * 60)
print("Categorical Variables")
print("=" * 60)

df_cat = pd.read_csv(DATA_FILE.replace(".csv", "") + "_cat_processed.csv")

cat_rows = []

cat_target = "cluster_lcs_3clus"

for col in PATIENT_CAT_COLUMNS:
    if col not in df_cat.columns:
        print(f"SKIPPING: {col} not found")
        continue
    ct = pd.crosstab(df_cat[col].fillna("Missing"), df_cat[cat_target])
    chi2, p_val, _, _ = stats.chi2_contingency(ct)
    # results.append({"variable": col, "type": "categorical", "p_value": p_val})
    print(f"\n{col} (Chi-square p = {p_val:.2e})")
    # row percent
    pct = ct.div(ct.sum(axis = 0), axis = 1).round(3)*100
    #print(pct.to_string())
    for level, row in pct.iterrows():
        r = {"variable": col, "category": level}
        for label, val in row.items():
            r[str(label)] = val
        r["p_value"] = p_val
        cat_rows.append(r)

#print(cat_rows)
cat_df = pd.DataFrame(cat_rows)
print(cat_df.to_string(index = False))

#print(results)
"""
Data processing and categorical variable check for multiclassification model

It well do: one-hot coding for catagorical. Check if there are more than 10 categories for some. Process comorbidity varialbes (3 = yes, others = no)

"""

import pandas as pd
import numpy as np
from config import (
    DATA_FILE,
    PATIENT_CAT_COLUMNS,
    PATIENT_BIN_COLUMNS,
    PATIENT_NUM_COLUMNS,
    COMORBIDITY_COLUMNS,
    LABEL_COLUMN,
    ALT_LABEL_COLUMNS,
)

# 1. Load data
df = pd.read_csv(DATA_FILE.replace(".csv", "") + "_cat_processed.csv")

# 2. Encode target
all_targets = LABEL_COLUMN + ALT_LABEL_COLUMNS
#target_col = LABEL_COLUMN[0]

for target_col in all_targets:
    label_map = {label: idx for idx, label in enumerate(sorted(df[target_col].dropna().unique()))}
    col_name = "target" if target_col == LABEL_COLUMN[0] else "target_" + target_col
    df[col_name] = df[target_col].map(label_map)
    print(f"Target mapping {target_col} -> {col_name}:", label_map)
    print(f"Target distribution {col_name}:\n{df[col_name].value_counts().sort_index()}\n")

# 3. Recode comorbidities: 3 = yes, all others = 0.
for col in COMORBIDITY_COLUMNS:
    df[col] = (df[col] == 3).astype(int)

# create comorbid_count
df["comorbid_count"] = df[COMORBIDITY_COLUMNS].sum(axis = 1)
# check comorbid county by cluster
print("Comorbid summary by cluster\n", df.groupby(LABEL_COLUMN[0])["comorbid_count"].agg(["mean", "std"]))

# 4. Categorical audit
print("=" * 60)
print("Categorical audit")
print("=" * 60)
for col in PATIENT_CAT_COLUMNS:
    n_unique = df[col].nunique(dropna=False)
    flag = "*** > 10 categories ***" if n_unique > 10 else ""
    #print(f"\n{col}: {n_unique} unique values {flag}")
    if n_unique > 10:
        print(df[col].value_counts(dropna=False))
print("=" * 60 + "\n")

# 5. Handle categoricals: fill missing with "Missing", one-hot encode
for col in PATIENT_CAT_COLUMNS:
    df[col] = df[col].fillna("Missing").astype(str)
    
df = pd.get_dummies(df, columns = PATIENT_CAT_COLUMNS, drop_first = False)

# 6. binary - check whether really binary
for col in PATIENT_BIN_COLUMNS:
    # unique_vals = df[col].nunique(dropna = False)
    # if unique_vals > 2:
    #     print(f"WARNING: Binary {col} has {unique_vals} unique values, encoding as binary(mode = 1, rest = 0)")
    #     most_common = df[col].mode()[0]
    #     df[col] = (df[col] == most_common).astype(int)
    # else:
    #     df[col] = df[col].fillna(0).astype(int)
    n_missing = df[col].isna().sum()
    unique_vals = df[col].dropna().nunique()
    if n_missing == 0 and unique_vals <= 2:
        continue
    print(f"{col}: {unique_vals} unique vals, {n_missing} missing ({n_missing/len(df)*100:.1f}%)")
        
# 7. Build final feature set
# Identify all one-hot columns that were created
onehot_cols = [c for c in df.columns if any(c.startswith(cat + "_") for cat in PATIENT_CAT_COLUMNS)]
                                            
feature_cols = (
    PATIENT_BIN_COLUMNS
    + PATIENT_NUM_COLUMNS
    + COMORBIDITY_COLUMNS
    + ["comorbid_count"]
    + onehot_cols
)

# keep only features + target
target_cols = [c for c in df.columns if c.startswith("target")]
out = df[feature_cols + target_cols].copy()
print(f"Final dataset: {out.shape[0]} rows, {out.shape[1]} columns ({len(feature_cols)} features + target")

# 8. Save
out.to_csv("processed_data.csv", index = False)
print("Saved to processed_data.csv")


# 9. Process another data for population synthesis - keep categorical grouping (already processed), fillna with "Missing", and substitute comorbid and total comorbid count
df_synth = pd.read_csv(DATA_FILE.replace(".csv", "") + "_cat_processed.csv")
# Recode comorbidities: 3 = yes, all others = 0.
for col in COMORBIDITY_COLUMNS:
    df_synth[col] = (df_synth[col] == 3).astype(int)
# create comorbid_count
df_synth["comorbid_count"] = df_synth[COMORBIDITY_COLUMNS].sum(axis = 1)
# Handle categoricals: fill missing with "Missing", one-hot encode
for col in PATIENT_CAT_COLUMNS:
    df_synth[col] = df_synth[col].fillna("Missing").astype(str)
# Relevant variables only
synth_feature_cols = (
    PATIENT_CAT_COLUMNS
    + PATIENT_BIN_COLUMNS
    + PATIENT_NUM_COLUMNS
    + COMORBIDITY_COLUMNS
    + ["comorbid_count"]
)
df_synth = df_synth[synth_feature_cols].copy()
df_synth.to_csv("processed_synth_raw_data.csv", index = False)


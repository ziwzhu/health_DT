# -*- coding: utf-8 -*-
"""
process categorical variables - group some before modeling

@author: ziz4012
"""
import pandas as pd
from config import DATA_FILE

# 1. Load data
df = pd.read_csv(DATA_FILE)

# 2. PRIMARY_PAYER_AT_DX -> 5 groups
payer_map = {
    61: "Medicare_supplement",
    63: "Medicare_supplement",
    
    60: "Medicare_basic",
    62: "Medicare_basic",
    
    64: "Dual_Medicare_Medicaid",
    
    10: "Private",
    20: "Private",
    21: "Private",

    31: "Medicaid_uninsured_gov",
    65: "Medicaid_uninsured_gov",
    67: "Medicaid_uninsured_gov",
    2:  "Medicaid_uninsured_gov",
    35: "Medicaid_uninsured_gov",
    1:  "Medicaid_uninsured_gov",
    66: "Medicaid_uninsured_gov",
    68: "Medicaid_uninsured_gov",
    
    99: "Unknown_missing",
}
df["PRIMARY_PAYER_AT_DX"] = df["PRIMARY_PAYER_AT_DX"].map(payer_map).fillna("Unknown_missing")
print("PRIMARY_PAYER_AT_DX groups:")
print(df["PRIMARY_PAYER_AT_DX"].value_counts(), "\n")

# 3. MARITAL_STATUS_AT_DIAGNOSIS -> 5 groups
marital_map = {
    2: "Married_partnered",
    5: "Widowed",
    4: "Divorced_separated",
    1: "Single",
    9: "Unknown_missing",
    3: "Divorced_separated",
    6: "Married_partnered",
    }
df["MARITAL_STATUS_AT_DIAGNOSIS"] = df["MARITAL_STATUS_AT_DIAGNOSIS"].map(marital_map).fillna("Unknown_missing")
print("MARITAL_STATUS_AT_DIAGNOSIS groups:")
print(df["MARITAL_STATUS_AT_DIAGNOSIS"].value_counts(), "\n")

# 4. RADIATION_RECODE -> 3 groups
radiation_map = {
    0: "No",
    1: "Yes",
    8: "Refused_unknown_missing",
    7: "Refused_unknown_missing",
    2: "Yes",
    5: "Yes",
    3: "Yes",
    4: "Yes",
    }
df["RADIATION_RECODE"] = df["RADIATION_RECODE"].map(radiation_map).fillna("Refused_unknown_missing")
print("RADIATION_RECODE groups:")
print(df["RADIATION_RECODE"].value_counts(), "\n")

df.rename (columns = {
    "PRIMARY_PAYER_AT_DX": "GRP_PRIMARY_PAYER_AT_DX",
    "MARITAL_STATUS_AT_DIAGNOSIS": "GRP_MARITAL_STATUS_AT_DIAGNOSIS",
    "RADIATION_RECODE": "GRP_RADIATION_RECODE",
    }, inplace=True)

out_name = DATA_FILE.replace(".csv", "") + "_cat_processed.csv"
df.to_csv(out_name, index=False)
print(f"Saved to {out_name}")




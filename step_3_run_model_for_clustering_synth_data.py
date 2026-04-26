# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 02:57:58 2026

@author: ziz4012

# Cluster prediction after synthesis data

"""
import joblib
import pandas as pd

from config import PATIENT_CAT_COLUMNS


def predict_clusters(model_path, data_paths):
    model = joblib.load(model_path)
    for path in data_paths:
        df = pd.read_csv(path)
        df = pd.get_dummies(df, columns = PATIENT_CAT_COLUMNS, drop_first=False)
        df["cluster"] = model.predict(df)
        df.to_csv(path.replace(".csv", "_clustered.csv"), index=False)
        print(f"Saved {path.replace('.csv', '_clustered.csv')}")

files = ["synthpop_patients_ageshift_plus_5.csv", 
         "synthpop_patients_ageshift_minus_5.csv", 
         "synthpop_patients_1.5x_stage1.csv", 
         "synthpop_patients_2x_stage7.csv",
         "synthpop_patients_orig_age.csv"
         ]
predict_clusters("best_model.pkl", files)

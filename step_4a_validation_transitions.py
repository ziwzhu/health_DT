# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 08:14:32 2026

@author: ziz4012

# Step 4A: Validation, transition extraction, and fidelity checks
# (Split from step_4_run_auc_comparison.py — final pipeline portion)
"""
import pandas as pd
import numpy as np
from config import (EVENT_COLUMNS,
                    OUTCOME_COLUMNS, PATTERN_TO_STATE, RARE_STATE,
                    N_STATES, N_WEEKS, STATE_LABELS)

from module0_data_prep import (encode_sequences)

df_raw = pd.read_csv("lung_above65_notx_merged.csv")
sequences = encode_sequences(df_raw)

df_patients = df_raw[["PATIENT_ID", "n_ed_12fu", "los", "mcr_cost"]].copy()
# Add sequence-derived features to patient characteristics
hospital_states = [7, 8, 9]
is_hospital = np.isin(sequences, hospital_states)
df_patients["had_hospital_encounter"] = is_hospital.any(axis=1).astype(int)
df_patients["n_hospital_weeks"] = is_hospital.sum(axis=1)
df_patients["had_rare_pattern"] = (sequences == RARE_STATE).any(axis=1).astype(int)
df_patients["n_rare_weeks"] = (sequences == RARE_STATE).sum(axis=1)
    
clustered = pd.read_csv("above65-notx-3-clusters-merged-from-4.csv")
df_patients["cluster"] = clustered["cluster"].values

labels = df_patients["cluster"].values

# also get state distribution for level 2 simulation
from module5_transitions import (extract_cluster_transitions, extract_initial_state_distribution)
cluster_trans = extract_cluster_transitions(sequences, labels)
init_dist = extract_initial_state_distribution(sequences, labels)
# save init_dist
np.savez("initial_distributions.npz", **{f"cluster_{c}": dist for c, dist in init_dist.items()})

### Table 1 - Module 4 check: diff between clusters
# Module 4: Validation & Profiling
print(f"\n--- Module 4: Validation & Profiling ---")
from module4_validation import (validate_outcomes, profile_clusters)

PATIENT_CHAR_COLUMNS = [
    "sex_num_y", "race_num_y", "age_at_followup_y", "STATE_CODE_y",
    "age_grp_y", "race_grp_y", "acttx_y",
    "MARITAL_STATUS_AT_DIAGNOSIS", "PRIMARY_SITE", "YEAR_OF_DIAGNOSIS",
    "GRADE_THRU_2017","YOSTACS20132017QUINTILESTATEBAS",
    "CHEMOTHERAPY_RECODE_YES_NO_UNK", "RADIATION_RECODE",
    "INSURANCE_RECODE_2007", "PRIMARY_PAYER_AT_DX",
    "SEERCOMBINEDSUMMARYSTAGE2000200",
]
df_patients[PATIENT_CHAR_COLUMNS] = df_raw[PATIENT_CHAR_COLUMNS]

outcome_results = validate_outcomes(df_patients, labels)
profile_df = profile_clusters(df_patients, labels, sequences)

### Table 2 - level 2 check
# get Frobenius norms
real_files = ["transition_cluster_1. Low use.csv", 
"transition_cluster_2. Medium use.csv", 
"transition_cluster_3. High use.csv"]

seq_synth = np.load("sequences_orig_age.npy")
df_synth = pd.read_csv("synthpop_patients_orig_age_clustered.csv")

def _tmat(seqs):
    mat = np.zeros((N_STATES, N_STATES))
    for seq in seqs:
        for t in range(len(seq)-1):
            mat[seq[t], seq[t+1]] += 1
    rs = mat.sum(axis=1, keepdims=True); rs[rs==0] = 1
    return mat/rs

# NOTE (future todo): enumerate(real_files) assumes real_files[0] = synthetic
# cluster 0, which holds because cluster_1 (Low use) maps to classifier
# class 0. Fragile if cluster ordering changes. See DEV_NOTES.md §1.

# Transition matrix fidelity per cluster
print(f"\n Transition Matrix Fidelity (Frobenius norm, lower=better:")
for c, fname in enumerate(real_files):
    r_mat = pd.read_csv(fname, index_col = 0).values
    s_mat = _tmat(seq_synth[df_synth["cluster"].values==c])
    frob = np.linalg.norm(r_mat - s_mat, "fro")
    print(f" Cluster{c}: {frob:.4f}")

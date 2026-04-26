"""
module0_data_prep.py — Load and validate data from SEER-Medicare CSV exports.

Handles the combined CSV format where patient characteristics and
52-week event sequences are in a single file.

Inputs:  Combined CSV on disk (exported from SAS)
Outputs: df_patients (DataFrame), sequences (ndarray), patient_ids (array)
"""

import pandas as pd
import numpy as np
from config import (COMBINED_CSV, PATIENTS_CSV, EVENT_COLUMNS,
                    PATIENT_CHAR_COLUMNS, COMORBIDITY_COLUMNS,
                    OUTCOME_COLUMNS, PATTERN_TO_STATE, RARE_STATE,
                    N_STATES, N_WEEKS, STATE_LABELS)


def load_combined_csv(path=None):
    """
    Load the combined CSV with both patient characteristics and event sequences.

    Returns
    -------
    df_raw : pd.DataFrame — full dataframe as-is from CSV
    """
    path = path or COMBINED_CSV
    df = pd.read_csv(path)
    print(f"[Module 0] Loaded {len(df)} rows × {len(df.columns)} columns from {path}")
    return df


def encode_sequences(df_raw):
    """
    Convert raw event columns (binary pattern strings like "100010", "rare")
    into integer state codes (0-10) using PATTERN_TO_STATE mapping.

    Parameters
    ----------
    df_raw : pd.DataFrame with event1..event52 columns

    Returns
    -------
    sequences : np.ndarray (n_patients × 52), dtype int
    """
    event_cols = [c for c in EVENT_COLUMNS if c in df_raw.columns]
    if len(event_cols) < N_WEEKS:
        print(f"  WARNING: Only {len(event_cols)} event columns found "
              f"(expected {N_WEEKS})")

    sequences = np.zeros((len(df_raw), len(event_cols)), dtype=int)

    for j, col in enumerate(event_cols):
        for i, val in enumerate(df_raw[col].astype(str)):
            val_clean = val.strip()
            if val_clean in PATTERN_TO_STATE:
                sequences[i, j] = PATTERN_TO_STATE[val_clean]
            elif val_clean.lower() == "rare":
                sequences[i, j] = RARE_STATE
            else:
                # Unknown pattern → Rare
                sequences[i, j] = RARE_STATE

    # Print state distribution
    print(f"[Module 0] Encoded sequences: {sequences.shape[0]} patients × "
          f"{sequences.shape[1]} weeks")
    print(f"  State distribution overall:")
    for s in range(N_STATES):
        pct = (sequences == s).mean() * 100
        label = STATE_LABELS.get(s, f"State {s}")
        print(f"    {s:>2} ({label:<25s}): {pct:.1f}%")

    return sequences

# NOTE (future todo): extract_patient_characteristics is commented out.
# Needed by load_and_prepare() for the orchestrator pipeline, not by the
# step-by-step scripts. Uncomment if restoring full pipeline. See DEV_NOTES.md §3.

# def extract_patient_characteristics(df_raw):
#     """
#     Extract patient characteristics (demographics, cancer features,
#     comorbidities, outcomes) from the combined CSV.

#     Generates a synthetic patient_id if none exists.

#     Returns
#     -------
#     df_patients : pd.DataFrame
#     """
#     # Generate patient_id
#     df = df_raw.copy()
#     df.insert(0, "patient_id", np.arange(1, len(df) + 1))

#     # Identify which characteristic columns are present
#     char_cols = ["patient_id"]
#     for col in PATIENT_CHAR_COLUMNS + COMORBIDITY_COLUMNS + OUTCOME_COLUMNS:
#         if col in df.columns:
#             char_cols.append(col)

#     # Also include cost columns if present
#     cost_cols = ["medpar_total", "Otpt_total", "nch_total", "mcr_cost", "pat_cost"]
#     for col in cost_cols:
#         if col in df.columns and col not in char_cols:
#             char_cols.append(col)

#     df_patients = df[char_cols].copy()

#     # Compute derived comorbidity index (sum of binary comorbidity flags)
#     comorbid_present = [c for c in COMORBIDITY_COLUMNS if c in df.columns]
#     if comorbid_present:
#         # Many comorbidity flags may have values > 1 (count of claims);
#         # binarize first then sum for a simple index
#         df_patients["charlson_approx"] = (
#             df[comorbid_present].apply(lambda x: (x > 0).sum(), axis=1)
#         )
#         print(f"  Computed charlson_approx from {len(comorbid_present)} "
#               f"comorbidity flags (mean={df_patients['charlson_approx'].mean():.1f})")

#     # Derive hospital encounter flags from sequences (will be set later)
#     print(f"[Module 0] Extracted {len(df_patients)} patient records "
#           f"with {len(char_cols)} characteristic columns")

#     # Check for outcome columns
#     found = [c for c in OUTCOME_COLUMNS if c in df_patients.columns]
#     missing = [c for c in OUTCOME_COLUMNS if c not in df_patients.columns]
#     if found:
#         print(f"  Outcome columns found: {found}")
#     if missing:
#         print(f"  WARNING: Outcome columns not found: {missing}")

#     return df_patients


def load_and_prepare(path=None):
    """
    Full data loading pipeline: load CSV → encode sequences → extract
    patient characteristics.

    Returns
    -------
    df_patients : pd.DataFrame
    sequences   : np.ndarray (n_patients × 52)
    patient_ids : np.ndarray
    """
    df_raw = load_combined_csv(path)
    sequences = encode_sequences(df_raw)
    df_patients = extract_patient_characteristics(df_raw)
    patient_ids = df_patients["patient_id"].values

    # Add sequence-derived features to patient characteristics
    hospital_states = [7, 8, 9]
    is_hospital = np.isin(sequences, hospital_states)
    df_patients["had_hospital_encounter"] = is_hospital.any(axis=1).astype(int)
    df_patients["n_hospital_weeks"] = is_hospital.sum(axis=1)
    df_patients["had_rare_pattern"] = (sequences == RARE_STATE).any(axis=1).astype(int)
    df_patients["n_rare_weeks"] = (sequences == RARE_STATE).sum(axis=1)

    print(f"\n[Module 0] Data preparation complete:")
    print(f"  {len(df_patients)} patients, {sequences.shape[1]} weeks, "
          f"{N_STATES} states")

    return df_patients, sequences, patient_ids


# Legacy loaders (for separate-file format)
def load_patients(path=None):
    """Load patient characteristics CSV (separate file format)."""
    path = path or PATIENTS_CSV
    df = pd.read_csv(path)
    assert "patient_id" in df.columns, (
        f"'patient_id' column not found. Columns: {list(df.columns)}")
    return df


def validate_patient_sequence_alignment(df_patients, sequence_ids):
    """Ensure patient IDs match between characteristics and sequences."""
    pat_ids = set(df_patients["patient_id"])
    seq_ids = set(sequence_ids)
    common = pat_ids & seq_ids
    print(f"[Module 0] Patient-Sequence alignment: {len(common)} in both")
    return common

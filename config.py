"""
config.py — Central configuration for the Digital Twin pipeline.

This is the trimmed version containing only parameters used by the
current step-by-step pipeline (step_0 through step_5).
See config_old.py for the full original including legacy parameters.

Edit paths and parameters here. Nothing else in the pipeline
should contain hardcoded paths or magic numbers.
"""

import os

# ============================================================================
# PATHS
# ============================================================================

# DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Combined CSV: patient characteristics + 52-week event columns.
# These path variables are required by module0_data_prep.py imports,
# though the step scripts load CSVs by name directly.
# COMBINED_CSV = os.path.join(DATA_DIR, "lung_above65_notx_merged.csv")
# SEQUENCE_CSV = os.path.join(DATA_DIR, "sequences.csv")
# PATIENTS_CSV = os.path.join(DATA_DIR, "patients.csv")

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# COLUMN DEFINITIONS
# ============================================================================

# 52-week event columns in the combined CSV
EVENT_COLUMNS = [f"event{i}" for i in range(1, 53)]

# Input file for Level 1 classification (post-clustering)
DATA_FILE = "patients_clustered_3clus.csv"

# Patient characteristic columns (demographics + cancer + comorbidities)
# Column names carry _y suffixes from a prior pandas merge step.
PATIENT_CHAR_COLUMNS = [
    "SEER_REGISTRY", "site_y", "followup_start", "followup_end",
    "lookback_start", "birth_dt_num",
    "sex_num_y", "race_num_y", "age_at_followup_y", "STATE_CODE_y",
    "age_grp_y", "race_grp_y", "acttx_y",
    "MARITAL_STATUS_AT_DIAGNOSIS", "PRIMARY_SITE", "YEAR_OF_DIAGNOSIS",
    "GRADE_THRU_2017", "YOSTACS20132017QUINTILESTATEBAS",
    "CHEMOTHERAPY_RECODE_YES_NO_UNK", "RADIATION_RECODE",
    "INSURANCE_RECODE_2007", "PRIMARY_PAYER_AT_DX",
    "SEERCOMBINEDSUMMARYSTAGE2000200",
]

PATIENT_CAT_COLUMNS = [
    "site_y", "PRIMARY_SITE",
    "race_grp_y", "sex_num_y",
    "GRP_MARITAL_STATUS_AT_DIAGNOSIS",
    "GRADE_THRU_2017", "YOSTACS20132017QUINTILESTATEBAS",
    "CHEMOTHERAPY_RECODE_YES_NO_UNK", "GRP_RADIATION_RECODE",
    "INSURANCE_RECODE_2007", "GRP_PRIMARY_PAYER_AT_DX",
    "SEERCOMBINEDSUMMARYSTAGE2000200",
]

PATIENT_BIN_COLUMNS = []

PATIENT_NUM_COLUMNS = [
    "age_at_followup_y",
]

# Comorbidity flags from claims data.
# Convention: value == 3 means "claim present" in SEER-Medicare.
# Binarized as (col == 3) in step_1 and comparison scripts.
COMORBIDITY_COLUMNS = [
    "AMI", "ALZH", "ATRIAL_FIB", "CATARACT", "CHRONICKIDNEY",
    "COPD", "CHF", "DIABETES", "GLAUCOMA", "HIP_FRACTURE",
    "ISCHEMICHEART", "DEPRESSION", "OSTEOPOROSIS", "RA_OA",
    "STROKE_TIA", "HYPERT", "HYPERL", "ANEMIA", "ASTHMA",
]

LABEL_COLUMN = ["cluster_lcs_3clus"]
ALT_LABEL_COLUMNS = ["cluster_lcs_4clus", "cluster_pca_3clus", "cluster_pca_5clus"]

# ============================================================================
# PROVIDER TYPE DEFINITIONS (the 6 binary flags per week)
# ============================================================================

PROVIDER_TYPES = {
    0: "PCP",
    1: "Other Spec",
    2: "Onc/Hem",
    3: "Mus_Skeletal",
    4: "Mental Health",
    5: "Hospital",
}

N_PROVIDER_TYPES = len(PROVIDER_TYPES)

# ============================================================================
# WEEKLY STATE DEFINITIONS (11 states from the actual data patterns)
# ============================================================================
# The top 10 most common weekly encounter patterns + Rare cover 100% of data.

STATE_LABELS = {
    0:  "No Encounter",
    1:  "PCP Only",
    2:  "Other Spec Only",
    3:  "Other Spec + PCP",
    4:  "Onc/Hem Only",
    5:  "Mus_Skeletal Only",
    6:  "Mental Health Only",
    7:  "Hospital only",
    8:  "Hospital + Other Spec",
    9:  "Hospital + Other Spec + PCP",
    10: "Rare",
}

# Maps the raw event string (6-digit padded) to a state integer.
PATTERN_TO_STATE = {
    "000000": 0,
    "000001": 1,
    "000010": 2,
    "000011": 3,
    "000100": 4,
    "001000": 5,
    "010000": 6,
    "100000": 7,
    "100010": 8,
    "100011": 9,
}
RARE_STATE = 10

N_STATES = len(STATE_LABELS)
N_WEEKS = 52

# ============================================================================
# OUTCOME COLUMNS
# ============================================================================

OUTCOME_COLUMNS = [
    "n_ed_12fu",       # number of ED visits in 12-month follow-up
    "n_inpt_stay",     # number of inpatient stays
    "los",             # length of stay
    "mcr_cost",        # Medicare cost
]

# ============================================================================
# GENERAL
# ============================================================================

RANDOM_SEED = 42
FIGURE_DPI = 150

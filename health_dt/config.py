"""All configuration for the pipeline: paths, data schema, and model parameters.

Nothing else in the codebase contains a file path, a column name, or a magic
number. To run on a different dataset, edit this file and nothing else.

Sections
--------
1. Paths
2. Weekly encounter encoding
3. Patient characteristic columns
4. Categorical regrouping
5. Level 1 parameters
6. Level 2 scenarios
7. Level 3 resources
"""

from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# 1. PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Input files (not distributed — SEER-Medicare Data Use Agreement).
COMBINED_CSV = DATA_DIR / "lung_above65_notx_merged.csv"
CLUSTER_LABELS_CSV = DATA_DIR / "above65-notx-3-clusters-merged-from-4.csv"
CLUSTERED_PATIENTS_CSV = DATA_DIR / "patients_clustered_3clus.csv"

# R synthesis script.
SYNTHPOP_SCRIPT = PROJECT_ROOT / "R" / "level2_synthpop.R"
RSCRIPT = "Rscript"  # full path on Windows, e.g. "C:/Program Files/R/R-4.5.1/bin/Rscript.exe"

RANDOM_SEED = 42
FIGURE_DPI = 150


# =============================================================================
# 2. WEEKLY ENCOUNTER ENCODING
# =============================================================================

N_WEEKS = 52
EVENT_COLUMNS = [f"event{i}" for i in range(1, N_WEEKS + 1)]

# Bit position -> provider type in the 6-digit event string.
PROVIDER_TYPES = {
    0: "PCP",
    1: "Other Spec",
    2: "Onc/Hem",
    3: "Mus_Skeletal",
    4: "Mental Health",
    5: "Hospital",
}

# The ten most frequent patterns plus a catch-all cover 100% of observed weeks.
STATE_LABELS = {
    0: "No Encounter",
    1: "PCP Only",
    2: "Other Spec Only",
    3: "Other Spec + PCP",
    4: "Onc/Hem Only",
    5: "Mus_Skeletal Only",
    6: "Mental Health Only",
    7: "Hospital only",
    8: "Hospital + Other Spec",
    9: "Hospital + Other Spec + PCP",
    10: "Rare",
}

# Raw zero-padded 6-character event string -> state code.
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

# States involving a hospital-based encounter. Note that state 4 is oncology,
# not hospital — an easy and previously-made mistake.
HOSPITAL_STATES = [7, 8, 9]

# Per-state colours, matched to the R figures so panels are comparable.
STATE_COLORS = [
    "#B2E8D4", "#FFFFB3", "#B3CDE3", "#F4A0A0", "#6BAED6", "#FDAE61",
    "#B2DF8A", "#AAAAAA", "#F5D0E2", "#C994E0", "#E2F5D0",
]


# =============================================================================
# 3. PATIENT CHARACTERISTIC COLUMNS
# =============================================================================
# Column names carry _y suffixes inherited from an upstream pandas merge. They
# are kept verbatim so the code matches the delivered files.

PATIENT_ID_COLUMN = "PATIENT_ID"

# Categorical predictors, AFTER regrouping (note the GRP_ prefixes).
PATIENT_CAT_COLUMNS = [
    "site_y",
    "PRIMARY_SITE",
    "race_grp_y",
    "sex_num_y",
    "GRP_MARITAL_STATUS_AT_DIAGNOSIS",
    "GRADE_THRU_2017",
    "YOSTACS20132017QUINTILESTATEBAS",
    "CHEMOTHERAPY_RECODE_YES_NO_UNK",
    "GRP_RADIATION_RECODE",
    "INSURANCE_RECODE_2007",
    "GRP_PRIMARY_PAYER_AT_DX",
    "SEERCOMBINEDSUMMARYSTAGE2000200",
]

PATIENT_NUM_COLUMNS = ["age_at_followup_y"]
PATIENT_BIN_COLUMNS = []

# Chronic condition flags from the Medicare CCW segment.
#
# WARNING: these are not counts. Value 3 means "met claims criteria with
# sufficient FFS coverage", i.e. the condition is present. Any other value means
# absent or insufficient coverage. Binarize with == 3, never > 0.
COMORBIDITY_COLUMNS = [
    "AMI", "ALZH", "ATRIAL_FIB", "CATARACT", "CHRONICKIDNEY",
    "COPD", "CHF", "DIABETES", "GLAUCOMA", "HIP_FRACTURE",
    "ISCHEMICHEART", "DEPRESSION", "OSTEOPOROSIS", "RA_OA",
    "STROKE_TIA", "HYPERT", "HYPERL", "ANEMIA", "ASTHMA",
]
COMORBIDITY_PRESENT_CODE = 3

# Phenotype labels from the external R LCS clustering.
LABEL_COLUMN = "cluster_lcs_3clus"
ALT_LABEL_COLUMNS = [
    "cluster_lcs_4clus",
    "cluster_pca_3clus",
    "cluster_pca_5clus",
]

# Clinical outcomes tested for between-phenotype differences (Table 2).
OUTCOME_COLUMNS = ["n_ed_12fu", "n_inpt_stay", "los", "mcr_cost"]

# Characteristics included in the Table 1 cluster profile.
PROFILE_CHAR_COLUMNS = [
    "sex_num_y", "race_num_y", "age_at_followup_y", "STATE_CODE_y",
    "age_grp_y", "race_grp_y", "acttx_y",
    "MARITAL_STATUS_AT_DIAGNOSIS", "PRIMARY_SITE", "YEAR_OF_DIAGNOSIS",
    "GRADE_THRU_2017", "YOSTACS20132017QUINTILESTATEBAS",
    "CHEMOTHERAPY_RECODE_YES_NO_UNK", "RADIATION_RECODE",
    "INSURANCE_RECODE_2007", "PRIMARY_PAYER_AT_DX",
    "SEERCOMBINEDSUMMARYSTAGE2000200",
]

# Columns never used as model features.
NON_FEATURE_COLUMNS = {"patient_id", PATIENT_ID_COLUMN, "cluster", "latent_profile"}


# =============================================================================
# 4. CATEGORICAL REGROUPING
# =============================================================================
# High-cardinality SEER codes collapsed into analysis groups before modelling.
# Format: source column -> (code map, default for unmapped, new column name).

PAYER_MAP = {
    60: "Medicare_basic", 62: "Medicare_basic",
    61: "Medicare_supplement", 63: "Medicare_supplement",
    64: "Dual_Medicare_Medicaid",
    10: "Private", 20: "Private", 21: "Private",
    1: "Medicaid_uninsured_gov", 2: "Medicaid_uninsured_gov",
    31: "Medicaid_uninsured_gov", 35: "Medicaid_uninsured_gov",
    65: "Medicaid_uninsured_gov", 66: "Medicaid_uninsured_gov",
    67: "Medicaid_uninsured_gov", 68: "Medicaid_uninsured_gov",
    99: "Unknown_missing",
}

MARITAL_MAP = {
    1: "Single",
    2: "Married_partnered", 6: "Married_partnered",
    3: "Divorced_separated", 4: "Divorced_separated",
    5: "Widowed",
    9: "Unknown_missing",
}

RADIATION_MAP = {
    0: "No",
    1: "Yes", 2: "Yes", 3: "Yes", 4: "Yes", 5: "Yes",
    7: "Refused_unknown_missing", 8: "Refused_unknown_missing",
}

CATEGORICAL_REGROUPINGS = {
    "PRIMARY_PAYER_AT_DX": (PAYER_MAP, "Unknown_missing", "GRP_PRIMARY_PAYER_AT_DX"),
    "MARITAL_STATUS_AT_DIAGNOSIS": (MARITAL_MAP, "Unknown_missing", "GRP_MARITAL_STATUS_AT_DIAGNOSIS"),
    "RADIATION_RECODE": (RADIATION_MAP, "Refused_unknown_missing", "GRP_RADIATION_RECODE"),
}


# =============================================================================
# 5. LEVEL 1 PARAMETERS
# =============================================================================

# Which encoded label to model. Alternatives: target_cluster_lcs_4clus,
# target_cluster_pca_3clus, target_cluster_pca_5clus.
TARGET_COLUMN = "target"

TEST_SIZE = 0.2
N_ESTIMATORS = 200
TOP_FEATURES_TO_REPORT = 15

# Candidate classifiers, compared by macro F1. Names must match the keys in
# level1.classify.build_models().
CLASSIFIER_MODELS = ["logistic_regression", "random_forest", "gradient_boosting"]


# =============================================================================
# 6. LEVEL 2 SCENARIOS
# =============================================================================


@dataclass(frozen=True)
class Scenario:
    """One counterfactual population.

    Attributes
    ----------
    name:
        Short key used in output filenames.
    synth_csv:
        Filename of the synthpop output, written to OUTPUT_DIR.
    age_shift:
        Mean age offset in years passed to the R synthesiser. Age is synthesised
        first, so every later variable is conditioned on the shifted ages.
    source_csv:
        Optional scenario-specific input (used by the stage-shift scenarios,
        whose source cohort is modified externally). None means the shared
        synthesis frame is used.
    description:
        Free text carried into result tables.
    """

    name: str
    synth_csv: str
    age_shift: int = 0
    source_csv: str | None = None
    description: str = ""

    @property
    def clustered_csv(self) -> str:
        """Filename after phenotype assignment."""
        return self.synth_csv.replace(".csv", "_clustered.csv")

    def sequence_csv(self, seed: int) -> str:
        """Filename for simulated trajectories at a given replication seed."""
        return f"sequences_{self.name}_seed{seed:05d}.npy"

    @property
    def sequence_glob(self) -> str:
        """Glob matching every replication for this scenario."""
        return f"sequences_{self.name}_seed*.npy"


N_SYNTHETIC = 24462  # matches the observed cohort size
N_REPLICATIONS = 100

SCENARIOS = [
    Scenario(
        name="orig_age",
        synth_csv="synthpop_patients_orig_age.csv",
        age_shift=0,
        description="Observed age distribution (reference population)",
    ),
    Scenario(
        name="Age_5yr_Younger",
        synth_csv="synthpop_patients_ageshift_minus_5.csv",
        age_shift=-5,
        description="Mean age shifted down 5 years",
    ),
    Scenario(
        name="Age_5yr_Older",
        synth_csv="synthpop_patients_ageshift_plus_5.csv",
        age_shift=5,
        description="Mean age shifted up 5 years",
    ),
    Scenario(
        name="1.5x_Stage_1",
        synth_csv="synthpop_patients_1.5x_stage1.csv",
        source_csv="synth_source_stage1.csv",
        description="Localised-stage share increased 1.5x",
    ),
    Scenario(
        name="2x_Stage_7",
        synth_csv="synthpop_patients_2x_stage7.csv",
        source_csv="synth_source_stage7.csv",
        description="Distant-stage share doubled",
    ),
]

BASELINE_SCENARIO = "orig_age"


# =============================================================================
# 7. LEVEL 3 RESOURCES
# =============================================================================


@dataclass(frozen=True)
class Resource:
    """A finite care-delivery resource.

    Attributes
    ----------
    name:
        Display name, also the key in result tables.
    weekly_capacity:
        Patients served per simulated week.
    required_by_states:
        Weekly states (see STATE_LABELS) that generate demand for this resource.
    priority:
        Queue discipline when demand exceeds capacity: "acuity", "fifo", or
        "random".
    spillover_state:
        State an unmet agent is forced into. None keeps the planned state and
        records the encounter as unmet.
    """

    name: str
    weekly_capacity: int
    required_by_states: tuple[int, ...]
    priority: str = "acuity"
    spillover_state: int | None = None


RESOURCES = [
    Resource("PCP", 3000, (1, 3, 9), priority="acuity"),
    Resource("Other_Specialty", 3300, (2, 3, 8, 9), priority="fifo"),
    Resource("Oncology_Hematology", 400, (4,), priority="acuity"),
    Resource("Hospital_Beds", 800, (7, 8, 9), priority="acuity"),
    Resource("Musculoskeletal", 650, (5,), priority="fifo"),
    Resource("Mental_Health", 200, (6,), priority="fifo"),
    Resource("Navigator", 2000, (7, 8, 9, 10), priority="acuity"),
]

# Weeks an agent waits before abandoning a queue. The sweep runs all three.
DROP_THRESHOLDS = [4, 8, 12]

# Probability a waiting agent abandons for reasons other than waiting time.
# Kept at 0 for the reported results.
RANDOM_DROPOUT_PROB = 0.0

# Columns consulted when scoring triage acuity, in priority order. The first
# present column in each group is used; groups are additive.
#
# WARNING: on the published runs none of these existed in the synthpop output
# (which carries comorbid_count, not charlson_approx), so every agent scored 0
# and "acuity" queues behaved as FIFO. build_agents() warns when this happens.
# Adding "comorbid_count" to the burden group makes triage active — that is a
# new analysis, not a bug fix.
ACUITY_COLUMNS = {
    "burden": ["charlson_approx", "charlson_index"],
    "stage": ["SUMMARY_STAGE_2000_1998_2017", "stage"],
    "history": ["n_hospital_weeks"],
}
HOSPITAL_HISTORY_WEIGHT = 0.5


def scale_resources(factor: float, resources: list[Resource] | None = None) -> list[Resource]:
    """Return a copy of the resource list with every capacity multiplied.

    Two uses: capacity what-if scenarios ("what if we had 20% more navigators"),
    and demo runs on a small cohort, where the real capacities are so far above
    demand that nothing binds and the simulation shows nothing.

        config.scale_resources(500 / config.N_SYNTHETIC)   # demo-sized system
        config.scale_resources(1.2)                        # 20% more of everything
    """
    return [
        Resource(
            r.name,
            max(1, int(round(r.weekly_capacity * factor))),
            r.required_by_states,
            r.priority,
            r.spillover_state,
        )
        for r in (resources or RESOURCES)
    ]


def scenario(name: str) -> Scenario:
    """Look up a scenario by name."""
    for spec in SCENARIOS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown scenario {name!r}; known: {[s.name for s in SCENARIOS]}")


def out(filename: str) -> Path:
    """Path to a file in the output directory, creating the directory if needed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR / filename

# Note: Superseded by v2.0.0. Code for this version lives on the v1-legacy branch.

# Digital Twin Pipeline for Cancer Survivorship Care Utilization

A three-level computational framework for modeling healthcare utilization patterns among older (65+) lung cancer survivors using SEER-Medicare claims data. The pipeline discovers utilization phenotypes from 52-week encounter sequences, generates synthetic populations with scenario-shifted demographics, and simulates resource-constrained care delivery via agent-based modeling.

## Overview

**Level 1 — Phenotype Discovery & Classification**
Patients are clustered into utilization phenotypes (e.g., low / medium / high use) based on their 52-week encounter sequences. Clustering was performed externally using Longest Common Subsequence (LCS) distance in R, producing 3 clusters. A multiclass classifier (Gradient Boosting) is trained to predict cluster membership from patient characteristics (demographics, cancer stage, comorbidities), enabling phenotype assignment for synthetic patients.

**Level 2 — Digital Twin Initialization**
Synthetic patient populations are generated using R's `synthpop` package (CART-based synthesis), with optional distribution shifts (e.g., age ±5 years, altered cancer stage mix). Each synthetic patient is assigned a phenotype via the trained classifier. A 52-week utilization sequence is then simulated per patient using cluster-specific first-order Markov transition matrices extracted from the real data.

**Level 3 — Agent-Based Resource Simulation**
Each synthetic patient becomes an agent with their Markov sequence as a planned care trajectory. Finite resource pools (PCP, specialty, hospital, mental health, oncology, musculoskeletal, navigator) constrain actual access via priority-based weekly allocation. The ABM tracks unmet demand, queue carryover (with configurable drop thresholds), and compares outcomes across population scenarios and resource constraint levels over 100 stochastic replications.

## Requirements

**Python 3.9+** with the following packages:

- numpy, pandas, scipy
- scikit-learn
- matplotlib
- joblib

**R 4.x** with:

- `synthpop`
- `truncnorm`

## Data

The pipeline expects SEER-Medicare claims data in a combined CSV format. The primary input file contains both patient characteristics and 52-week event columns (`event1` through `event52`), where each event is a 6-digit binary string encoding encounters with 6 provider types:

| Position | Provider Type |
|----------|--------------|
| 0 | PCP (Primary Care) |
| 1 | Other Specialty |
| 2 | Oncology / Hematology |
| 3 | Musculoskeletal |
| 4 | Mental Health |
| 5 | Hospital-based |

These binary patterns are mapped to 11 discrete weekly states (the 10 most frequent patterns + a "Rare" catch-all). See `config.py` `STATE_LABELS` for the full mapping.

**Required input files** (not included in this repository due to data use agreements):

- `lung_above65_notx_merged.csv` — Combined patient + sequence data from SEER-Medicare
- `above65-notx-3-clusters-merged-from-4.csv` — Cluster labels from R-based LCS analysis
- `patients_clustered_3clus.csv` — Patient data with cluster assignments (input to Level 1 classification)

## Execution Order

The pipeline runs as a series of sequential scripts. Each step produces intermediate files consumed by later steps.

### Step 0: Categorical Variable Processing

```bash
python step_0_categorical_processing.py
```

Groups categorical variables (payer type → 5 groups, marital status → 5 groups, radiation → 3 groups) to reduce cardinality before modeling. Reads `patients_clustered_3clus.csv`, writes `patients_clustered_3clus_cat_processed.csv`.

### Step 0 (side): Variable Association Check

```bash
python step_0_side_variable_check.py
```

Descriptive analysis: ANOVA for numeric/binary variables and chi-square for categorical variables against the cluster target. Used for Table 1 in the paper. Reads `processed_data.csv` (produced by Step 1 — run this after Step 1).

### Step 1: Data Processing

```bash
python step_1_data_processing.py
```

One-hot encodes categoricals, binarizes comorbidities (`== 3` means present in claims data), computes comorbidity count, and builds the final feature matrix. Produces `processed_data.csv` (for classification) and `processed_synth_raw_data.csv` (for synthpop, keeps categorical groupings intact without one-hot encoding).

### Step 2: Multiclass Classification

```bash
python step_2_run_multiclassification.py
```

Trains Logistic Regression, Random Forest, and Gradient Boosting to predict cluster from patient characteristics. Reports macro F1, confusion matrices, and feature importance. Saves the best model as `best_model.pkl`.

### Level 2A: Synthetic Population Generation (R)

```bash
Rscript level2_synthpop_ageshift.R <input.csv> <output.csv> <n_synthetic> [age_shift] [seed]
```

Run once per scenario. Example calls:

```bash
Rscript level2_synthpop_ageshift.R processed_synth_raw_data.csv synthpop_patients_orig_age.csv 24462 0
Rscript level2_synthpop_ageshift.R processed_synth_raw_data.csv synthpop_patients_ageshift_plus_5.csv 24462 5
Rscript level2_synthpop_ageshift.R processed_synth_raw_data.csv synthpop_patients_ageshift_minus_5.csv 24462 -5
```

For stage-shift scenarios, the input CSV is modified externally before synthesis.

### Step 3: Cluster Assignment for Synthetic Data

```bash
python step_3_run_model_for_clustering_synth_data.py
```

Applies the trained classifier (`best_model.pkl`) to each synthpop CSV, one-hot encodes to match training features, and appends a `cluster` column. Outputs `*_clustered.csv` files.

### Step 4A: Validation and Transition Extraction

```bash
python step_4a_validation_transitions.py
```

Encodes the real sequences and builds outcome variables, extracts cluster-specific Markov transition matrices and initial state distributions (saved as `cluster_transitions.npz` and `initial_distributions.npz`), runs cluster profiling and outcome validation (Table 1, Table 2), and computes transition matrix fidelity (Frobenius norms) between real and synthetic data.

### Step 5: Level 2B Simulation + Level 3 ABM

```bash
python step_5_run_level_2b_level_3.py
```

The main simulation script:

1. Loads transition matrices and initial distributions from the npz files
2. For each of 5 population scenarios × 100 random seeds (generated in two batches of 50), simulates 52-week Markov sequences for the synthetic populations
3. Runs the ABM for each simulated population across 3 resource constraint levels (drop thresholds: 4, 8, 12 weeks)
4. Aggregates results into `abm_runs_all_results.csv` and `abm_runs_results_summary.csv`

## Output Files

Key outputs saved to the `outputs/` directory:

| File | Description |
|------|-------------|
| `best_model.pkl` | Trained classifier for phenotype assignment |
| `cluster_transitions.npz` | Per-cluster Markov transition matrices |
| `initial_distributions.npz` | Per-cluster initial state distributions |
| `transition_cluster_*.csv` | Transition matrices in readable CSV format |
| `cluster_profiles.csv` | Table 1-style cluster characterization |
| `sequences_*_seed*.npy` | Simulated sequences per scenario per seed |
| `abm_runs_all_results.csv` | All ABM run results (scenarios × seeds × thresholds) |
| `abm_runs_results_summary.csv` | Aggregated ABM results with mean and std |
| `utilization_heatmaps.png` | Weekly state distributions by cluster |
| `state_distribution_stacked.png` | Stacked area charts of state proportions |

## Pre-Final Comparison Analyses

> **Warning:** The scripts in `pre_final_historical_code/` were developed and run during a pre-final iteration of the pipeline. They are conceptually close to the final version but **were not re-run or verified against the final data and outputs**. Column names, comorbidity encoding assumptions, or input file formats may not match the current pipeline without adjustments. See DEV_NOTES.md §8 for a verification checklist before trusting results from these scripts.

The `pre_final_historical_code/` directory contains comparison model implementations that benchmark the digital twin pipeline against traditional approaches. These were used during development to demonstrate the pipeline's advantages over conventional methods.

**Comparison 1a: Cox PH + Charlson — Python approximation** (`comparison1_cox_charlson.py`)
Logistic regression with the standard claims-research covariate set (CCI, demographics, cancer stage). This is a proxy for Cox PH since Python's `lifelines` was not used. Predicts binary utilization outcomes via 5-fold cross-validated AUC.

**Comparison 1b: Cox PH + Charlson — R implementation** (`comparison1_cox_R.R`)
Proper Cox Proportional Hazards using R's `survival` package. Produces concordance index and cross-validated AUC for utilization outcomes. Outputs `cox_r_survival.csv` and `cox_r_utilization.csv`, which `run_comparisons.py` can auto-detect via `_load_r_cox_results()`. Requires R packages `survival` and optionally `pROC` (falls back to manual Wilcoxon AUC if unavailable).

**Comparison 2: Random Survival Forest** (`comparison2_random_survival_forest.py`)
Random Forest classification with two feature sets: static-only (same as Cox) and static + sequence-derived temporal features (state frequencies, trends, entropy). Shows the value of temporal information.

**DT Pipeline Cluster-Based Prediction** (`run_comparisons.py` → `dt_cluster_prediction`)
Uses cluster membership + sequence features as predictors for the same utilization outcomes, enabling head-to-head AUC comparison.

**Step 4B: AUC Comparison** (`step_4b_auc_comparison.py`)
The AUC comparison portion that was originally in step_4, split out to separate the final pipeline from pre-final comparison code. Calls `run_comparisons.dt_cluster_prediction` and `comparison2_random_survival_forest.build_sequence_summary_features`.

The original unsplit `step_4_run_auc_comparison.py` is also preserved in the directory for reference.

## File Inventory

### Active — Final Version

These files produced the results reported in the paper.

| File | Role |
|------|------|
| `config.py` | Central configuration (column definitions, state mappings, parameters) |
| `step_0_categorical_processing.py` | Groups categorical variables |
| `step_0_side_variable_check.py` | Descriptive association tests |
| `step_1_data_processing.py` | Feature engineering and encoding |
| `step_2_run_multiclassification.py` | Classifier training and selection |
| `step_3_run_model_for_clustering_synth_data.py` | Phenotype assignment for synthetic data |
| `step_4a_validation_transitions.py` | Validation, transition extraction, fidelity checks |
| `step_5_run_level_2b_level_3.py` | Markov simulation + ABM (main results) |
| `level2_synthpop_ageshift.R` | R-based synthetic population generation |
| `level2_initialization.py` | `simulate_sequences` and `validate_twin` called by steps 4a and 5 |
| `level3_abm.py` | ABM engine, resource pools, agent definitions |
| `module0_data_prep.py` | `encode_sequences` used by step_4a (note: `load_and_prepare` is orchestrator-era dead code, not called by any step) |
| `module4_validation.py` | Cluster profiling and outcome validation used by step_4a |
| `module5_transitions.py` | Transition matrix extraction used by step_4a |

### Pre-Final — Developed, Not Verified Against Final Data

Located in `pre_final_historical_code/`. These were run during a pre-final iteration. They may work against the final data but have not been re-validated. See DEV_NOTES.md §8 for a verification checklist.

| File | Role |
|------|------|
| `comparison1_cox_charlson.py` | Cox/Charlson benchmark via Python logistic regression (proxy for Cox PH) |
| `comparison1_cox_R.R` | Cox/Charlson benchmark via R `survival` package (proper Cox PH) |
| `comparison2_random_survival_forest.py` | RSF benchmark with static and static+sequence feature sets |
| `run_comparisons.py` | Comparison orchestrator and DT cluster-based prediction. Note: `run_all_comparisons()` imports missing modules (comparison3, comparison4); only `dt_cluster_prediction()` and `build_unified_summary()` are usable |
| `step_4b_auc_comparison.py` | AUC comparison portion split from original step_4 |
| `step_4_run_auc_comparison.py` | Original unsplit step_4 (preserved for reference) |

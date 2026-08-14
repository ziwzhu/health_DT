# Data dictionary

Definitions for the SEER-Medicare extract this pipeline consumes and the derived quantities it produces. Everything here corresponds to a constant in `health_dt/config.py` — that module is the machine-readable version of this document, and the two must be changed together.

---

## Input files

| File | Rows | Contents |
|---|---|---|
| `lung_above65_notx_merged.csv` | one per patient | Characteristics, outcomes, and `event1`…`event52` |
| `above65-notx-3-clusters-merged-from-4.csv` | one per patient | `cluster` column from the R LCS clustering |
| `patients_clustered_3clus.csv` | one per patient | Characteristics joined to cluster labels |

> **The label file is joined positionally.** Row *i* of the label file must be the same patient as row *i* of the combined export. There is no key column to verify against, so `level1/sequences.py` checks the row counts match and refuses to proceed if they do not. Preserve row order in any upstream change.

---

## Weekly encounter encoding

Each `event<week>` cell is a **zero-padded 6-character binary string**, one bit per provider type:

| Position | Provider type |
|---|---|
| 0 | PCP (primary care) |
| 1 | Other specialty |
| 2 | Oncology / haematology |
| 3 | Musculoskeletal |
| 4 | Mental health |
| 5 | Hospital-based |

> **Zero-padding is required.** `"000011"` and `"11"` are different strings, and only the first matches. If leading zeros are stripped on export, every week collapses to `Rare` and the transition matrices become meaningless. `encode_sequences()` warns when more than 25% of patient-weeks land in `Rare`, which is the symptom.

### Weekly states

The ten most frequent patterns plus a catch-all account for 100% of observed weeks.

| Code | Pattern | Label |
|---|---|---|
| 0 | `000000` | No Encounter |
| 1 | `000001` | PCP Only |
| 2 | `000010` | Other Spec Only |
| 3 | `000011` | Other Spec + PCP |
| 4 | `000100` | Onc/Hem Only |
| 5 | `001000` | Mus_Skeletal Only |
| 6 | `010000` | Mental Health Only |
| 7 | `100000` | Hospital only |
| 8 | `100010` | Hospital + Other Spec |
| 9 | `100011` | Hospital + Other Spec + PCP |
| 10 | anything else, or `"rare"` | Rare |

**Hospital states** are 7, 8, and 9 (`config.HOSPITAL_STATES`). State 4 is oncology, not hospital — a distinction the pre-refactor `planned_hospital_weeks` column got wrong.

These codes are referenced directly in `config.RESOURCES` entries. Changing the encoding means changing every config and regenerating every transition archive.

---

## Patient characteristics

Column names carry `_y` suffixes inherited from an upstream pandas merge. They are kept verbatim rather than renamed, so the code matches the delivered files without a translation step.

### Demographics

| Column | Type | Notes |
|---|---|---|
| `age_at_followup_y` | numeric | Age at follow-up start. Cohort floor 65; the synthesiser truncates shifted distributions to [65, 95]. |
| `sex_num_y` | categorical | SEER coding |
| `race_num_y` | categorical | Full SEER detail |
| `race_grp_y` | categorical | Collapsed grouping used in modelling |
| `age_grp_y` | categorical | Banded age |
| `STATE_CODE_y` | categorical | State of residence |
| `SEER_REGISTRY` | categorical | Registry identifier |
| `MARITAL_STATUS_AT_DIAGNOSIS` | categorical | Regrouped → `GRP_MARITAL_STATUS_AT_DIAGNOSIS` |
| `YOSTACS20132017QUINTILESTATEBAS` | categorical | Yost socioeconomic index, state-based quintile |
| `INSURANCE_RECODE_2007` | categorical | SEER insurance recode |
| `PRIMARY_PAYER_AT_DX` | categorical | Regrouped → `GRP_PRIMARY_PAYER_AT_DX` |

### Cancer characteristics

| Column | Type | Notes |
|---|---|---|
| `PRIMARY_SITE` | categorical | ICD-O-3 topography |
| `site_y` | categorical | Derived site grouping |
| `SEERCOMBINEDSUMMARYSTAGE2000200` | categorical | SEER combined summary stage |
| `GRADE_THRU_2017` | categorical | Tumour grade |
| `YEAR_OF_DIAGNOSIS` | numeric | |
| `CHEMOTHERAPY_RECODE_YES_NO_UNK` | categorical | |
| `RADIATION_RECODE` | categorical | Regrouped → `GRP_RADIATION_RECODE` |
| `acttx_y` | binary | Active treatment indicator |

### Dates

`followup_start`, `followup_end`, `lookback_start`, `birth_dt_num` — SAS numeric dates. Carried through but not used as predictors.

---

## Comorbidity flags

Nineteen chronic conditions from the Medicare CCW segment: `AMI`, `ALZH`, `ATRIAL_FIB`, `CATARACT`, `CHRONICKIDNEY`, `COPD`, `CHF`, `DIABETES`, `GLAUCOMA`, `HIP_FRACTURE`, `ISCHEMICHEART`, `DEPRESSION`, `OSTEOPOROSIS`, `RA_OA`, `STROKE_TIA`, `HYPERT`, `HYPERL`, `ANEMIA`, `ASTHMA`.

> ### These are not counts
>
> A value of **`3`** means *"beneficiary met claims criteria and had sufficient FFS coverage"* — the condition is present. Every other value means absent, or that the coverage requirement was not met.
>
> **Binarise with `== 3`, never `> 0`.** Using `> 0` counts patients whose status is unknown as having the condition, inflating prevalence. The correct criterion is enforced in `preprocess.binarise_comorbidities()`, encoded as `config.COMORBIDITY_PRESENT_CODE`, and covered by a test. Commented-out `> 0` lines in the historical comparison scripts are wrong for raw data and should not be uncommented.

**`comorbid_count`** — the sum of the binarised flags, 0 to 19. Computed after binarisation, and recomputed rather than synthesised on the R side so it stays consistent with its components.

---

## Outcomes

Tested for between-phenotype differences in `validate_outcomes` (chi-square for binary, Kruskal-Wallis otherwise).

| Column | Definition |
|---|---|
| `n_ed_12fu` | Emergency department visits during 12-month follow-up |
| `n_inpt_stay` | Inpatient stays |
| `los` | Total length of stay (days) |
| `mcr_cost` | Total Medicare cost |

---

## Derived quantities

### From sequences (`derive_sequence_features`)

| Column | Definition |
|---|---|
| `had_hospital_encounter` | Any week in states 7, 8, or 9 |
| `n_hospital_weeks` | Count of such weeks |
| `had_rare_pattern` | Any week in state 10 |
| `n_rare_weeks` | Count of such weeks |

### Phenotype labels

| Column | Definition |
|---|---|
| `cluster_lcs_3clus` | Primary label: LCS distance, k=3 merged from k=4 |
| `cluster_lcs_4clus` | LCS, k=4 |
| `cluster_pca_3clus`, `cluster_pca_5clus` | PCA-based alternatives (sensitivity) |
| `target` | `cluster_lcs_3clus` encoded 0..K-1 |
| `target_<name>` | Encoded alternatives |
| `cluster` | 0-indexed phenotype — **the canonical form everywhere in Python** |

The R clustering emits 1-indexed labels; `preprocess.encode_targets` and `sequences.normalise_labels` map them to 0-indexed while preserving sort order, so `1, 2, 3` (low/medium/high use) becomes `0, 1, 2` with the same meaning.

---

## Level 3 resources

Defined in `config.RESOURCES`. Each has:

| Field | Meaning |
|---|---|
| `weekly_capacity` | Patients served per simulated week |
| `required_by_states` | Weekly states that generate demand for this resource |
| `priority` | `acuity` (sickest first), `fifo`, or `random` |
| `spillover_state` | State an unmet agent is forced into; `null` means the planned state is kept and the encounter is recorded as unmet |

Published capacities, for a cohort of 24,462:

| Resource | Capacity | Required by states |
|---|---|---|
| PCP | 3000 | 1, 3, 9 |
| Other_Specialty | 3300 | 2, 3, 8, 9 |
| Oncology_Hematology | 400 | 4 |
| Hospital_Beds | 800 | 7, 8, 9 |
| Musculoskeletal | 650 | 5 |
| Mental_Health | 200 | 6 |
| Navigator | 2000 | 7, 8, 9, 10 |

### Acuity score

Additive: comorbidity burden + stage + 0.5 × prior hospital weeks. Used only to order queues, never to predict.

> **On the published runs this evaluated to 0 for every agent.** The default lookup columns (`charlson_approx`, `charlson_index`) do not exist in the synthpop output, which carries `comorbid_count` instead. Because Python's sort is stable, `acuity` queues therefore behaved as FIFO. `build_agents()` warns when this happens. Adding `"comorbid_count"` to `config.ACUITY_COLUMNS["burden"]` makes triage active — that is a new analysis, not a bug fix.

---

## Output artefacts

| File | Contents |
|---|---|
| `processed_data.csv` | Encoded classifier matrix |
| `processed_synth_raw_data.csv` | Synthesis frame, categoricals as labelled groups |
| `best_model.pkl` | `{model, model_name, feature_names}` |
| `cluster_transitions.npz` | Per-phenotype transition matrices, 0-indexed keys |
| `initial_distributions.npz` | Per-phenotype week-1 state distributions |
| `transition_cluster_<k>.csv` | Human-readable transition matrices |
| `cluster_profiles.csv` | Table 1 |
| `outcome_tests.csv` | Table 2 |
| `real_sequences.npy`, `real_labels.npy` | Encoded real cohort, for fidelity comparison |
| `synthpop_patients_*.csv` | Synthetic cohorts |
| `*_clustered.csv` | Synthetic cohorts with assigned phenotypes |
| `sequences_<scenario>_seed<nnnnn>.npy` | Simulated trajectories, one per replication |
| `fidelity_<scenario>.csv` | Real vs synthetic comparison |
| `abm_runs_all_results.csv` | Long: one row per resource per (scenario, seed, threshold) |
| `abm_runs_summary.csv` | Aggregated means and standard deviations |
| `utilization_heatmaps.png`, `state_distribution_stacked.png`, `abm_resource_summary.png` | Figures |

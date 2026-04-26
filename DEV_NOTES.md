# DEV_NOTES — Technical Debt and Future Revision Notes

Internal notes for the developer. Not a user guide — see `README.md` for replication instructions.

## Known Technical Debt

### 1. Cluster Index Remapping (step_5, step_4a)

**What:** The real data uses 1-indexed cluster labels (1, 2, 3) from R. The classifier in step_2 maps these to 0-indexed targets (0, 1, 2) via `enumerate(sorted(...))`. The transition matrices are saved to npz with keys like `"cluster_1"`, `"cluster_2"`, `"cluster_3"`. In step_5, the line `int(k.split("_")[1].split(".")[0]) - 1` converts these back to 0-indexed to match the synthetic data's classifier-assigned clusters.

**Why it's fine for now:** Frobenius norms between real and synthetic transition matrices were all < 0.04, confirming correct alignment. The `.split(".")` handles numpy's occasional float-suffix on integer keys.

**Risk:** If cluster labels are ever changed to 0-indexed upstream (e.g., in the R clustering step), the `- 1` would silently produce wrong mappings (-1, 0, 1 instead of 0, 1, 2). There is no assertion guarding against this.

**Future fix:** Add an explicit mapping dict or assertion at load time, e.g.:
```python
# Explicit: map npz keys to synthetic cluster IDs
CLUSTER_KEY_MAP = {"cluster_1": 0, "cluster_2": 1, "cluster_3": 2}
cluster_trans = {CLUSTER_KEY_MAP[k]: ct_raw[k] for k in ct_raw.files}
assert set(cluster_trans.keys()) == set(df_synth["cluster"].unique()), \
    "Cluster ID mismatch between transitions and synthetic data"
```

**Files affected:** `step_5_run_level_2b_level_3.py` (lines ~10-13), `step_4a_validation_transitions.py` (Frobenius check loop uses `enumerate` which implicitly assumes alignment).

---

### 2. Legacy `np.random.seed` in `_build_agents`

**What:** `level3_abm.py` `_build_agents()` uses `np.random.seed(seed)` + `np.random.choice()` (global legacy API), while the rest of the codebase uses `np.random.default_rng(seed)` (isolated generators).

**Why it's fine for now:** `_build_agents` is called before the ABM simulation starts, and the simulation itself uses `default_rng`. No intervening code touches the global state between the seed and the choice call.

**Risk:** Any future code inserted between the `seed()` and `choice()` calls that touches `np.random.*` would silently alter results.

**Future fix:** Replace with `rng = np.random.default_rng(seed)` and `rng.choice(...)`. Note that this will change the sampled indices (different algorithm), so results will not be identical — do this only when starting a new version.

**Files affected:** `level3_abm.py` `_build_agents()` (around line 488-489).

---

### 3. `extract_patient_characteristics` Commented Out in module0

**What:** `module0_data_prep.py` has `extract_patient_characteristics` commented out (lines ~65-106), but `load_and_prepare()` still calls it. Both are dead code in the step-by-step workflow — no step script calls `load_and_prepare()`. The step_4a script constructs the patient DataFrame directly.

**Future fix:** Remove the commented-out block and `load_and_prepare`, or mark them explicitly as legacy with a `# LEGACY` comment.

**Files affected:** `module0_data_prep.py`.

---

### 4. config.py Cleanup

**What:** `DATA_DIR`, `COMBINED_CSV`, `SEQUENCE_CSV`, and `PATIENTS_CSV` were inherited from the orchestrator pipeline and not used by any step_0–step_5 script. These have been commented out in the current config. `FIGURE_DPI` remains as it is used by `module4_validation.py`. Original config preserved as `config_old.py`.

**Status:** Done. No further action needed for this version.

**Files affected:** `config.py`.

---

### 5. Comorbidity Encoding Convention

**What:** In SEER-Medicare claims data, comorbidity flags use value `3` to indicate "Beneficiary met claims criteria and had sufficient FFS coverage" (not a count or severity). The active pipeline consistently uses `(df[col] == 3).astype(int)` for binarization (step_1). There are commented-out `(df[col] > 0)` lines in the pre-final comparison scripts from a previous coding error. The R Cox script (`comparison1_cox_R.R`, line 51) also uses `> 0`, which is correct only if its input has already been binarized by step_1 — see §8 checklist.

**Why it matters:** If someone unfamiliar with the data convention sees `> 0` and uncomments it thinking it's a fix, results would be wrong. The R script's `> 0` is context-dependent on which input file it receives.

**Future fix:** Remove the commented-out `> 0` lines from the Python comparison files. In the R Cox script, either change to `== 3` for raw data compatibility or add a comment specifying it expects pre-binarized input.

**Files affected:** `pre_final_historical_code/comparison1_cox_charlson.py` (line ~42), `pre_final_historical_code/comparison2_random_survival_forest.py` (line ~57), `pre_final_historical_code/comparison1_cox_R.R` (line 51).

---

### 6. Hardcoded File Paths in Step Scripts

**What:** Several step scripts reference specific filenames inline (e.g., `"lung_above65_notx_merged.csv"`, `"above65-notx-3-clusters-merged-from-4.csv"`, `"processed_data.csv"`) rather than pulling from config.py. The R script paths are passed as command-line arguments, which is fine, but the Python steps have a mix.

**Future fix:** Centralize all file paths in config.py or a separate paths config. This would make it easier to run the pipeline on different datasets.

**Files affected:** `step_4a_validation_transitions.py`, `step_0_side_variable_check.py`, `step_1_data_processing.py`, `step_2_run_multiclassification.py`.

---

### 7. `run_comparisons.py` References Missing Modules

**What:** `run_all_comparisons()` imports `comparison3_risk_stratified_pathway` and `comparison4_dynamic_advantage`, which are not included in the current release. This function was part of the orchestrator pipeline. `build_unified_summary()` is safe to call (handles `None` inputs gracefully). `dt_cluster_prediction()` is used by `step_4b_auc_comparison.py` in the pre-final code.

**Status:** All affected files are now in `pre_final_historical_code/` and are not called by the active pipeline.

**Files affected:** `pre_final_historical_code/run_comparisons.py`.

---

### 8. Pre-Final Comparison Scripts — Verification Checklist

The comparison scripts in `pre_final_historical_code/` were developed and run during a pre-final iteration but **not re-validated against the final data and pipeline**. Before trusting results from these scripts, verify:

- [ ] **Input file format:** The Python comparison scripts auto-detect columns and fall back gracefully, but confirm the input DataFrame has the columns they expect (e.g., `age_at_fo` vs `age_at_followup_y`, `sex_num` vs `sex_num_y`). The R Cox script (`comparison1_cox_R.R`) expects `age_at_fo`, `sex_num`, `race_num` — these may not match the current `_y`-suffixed column names.
- [ ] **Comorbidity encoding:** The Python comparison scripts use `(df[col] == 3)` which is correct for raw SEER-Medicare data. The R Cox script uses `as.integer(df[[col]] > 0)` (line 51), which is correct only if the input has already been binarized by step_1. Confirm which input is being used. If feeding raw data to the R script, change `> 0` to `== 3`.
- [ ] **Input data path:** The R Cox script defaults to `outputs/patients_clustered.csv` (orchestrator-era). Update to match the current pipeline's output (e.g., `patients_dt3cluster_auc_ready.csv` from step_4b).
- [ ] **Missing module imports:** `run_comparisons.py`'s `run_all_comparisons()` imports comparison3 and comparison4 which don't exist. Only call `dt_cluster_prediction()` or `build_unified_summary()` directly.
- [ ] **Cluster labels in comparison context:** The comparison scripts receive `labels` from the caller. Confirm these are in the same index space (0-indexed) as the rest of the pipeline at point of use.

---

## Previous Approaches (Not in Current Version)

These were explored during development but replaced by the current step-by-step pipeline. Code may exist in earlier commits or local archives. These are distinct from the pre-final comparison scripts (§8 above), which are preserved in `pre_final_historical_code/`.

### LSTM Autoencoder Features

Used an LSTM autoencoder to learn latent representations of 52-week sequences, then clustered in the latent space. Config parameters (`LSTM_HIDDEN=64`, `LSTM_LATENT=16`, `LSTM_EPOCHS=50`, etc.) remain in `config_old.py`. Abandoned in favor of the LCS-distance + R-based clustering approach, which was more interpretable and produced clinically meaningful phenotypes.

### Alternative Clustering Methods

Explored k-means, hierarchical (Ward linkage), and GMM clustering with varying k (2–8) on PCA-reduced transition-enriched features. The config parameter `CLUSTER_METHOD = "gmm"` reflects the last-tested method. Final version used LCS distance computed in R with k=3 (merged from 4), selected based on clinical interpretability.

### Orchestrator Pipeline

A fully automated pipeline (`run_level1.py` → `run_level2` → `run_level3`) that chained all modules together. Replaced by the step-by-step scripts because:

- The step-by-step approach allowed manual inspection between stages
- The R-based synthpop generation couldn't be cleanly integrated into the Python orchestrator
- The clustering was done externally in R, breaking the Level 1 automation assumption
- Different scenarios required different intermediate files, which the orchestrator wasn't flexible enough to handle

### Additional Comparison Models (Legacy — Not Included)

`comparison3_risk_stratified_pathway.py` implemented ACS-ASCO risk-stratified follow-up care pathway scoring. `comparison4_dynamic_advantage.py` measured week-by-week prediction accuracy and trajectory divergence. Both were run during earlier development and dropped from the paper. Not included in the current release — these are further from the final version than the pre-final scripts in `pre_final_historical_code/`.

## Suggested Improvements for Next Version

1. **Unified cluster indexing:** Adopt 0-indexed clusters throughout (including R output), eliminate all remapping logic.
2. **Isolated RNG everywhere:** Replace all `np.random.seed()` / `np.random.choice()` calls with `default_rng` instances.
3. **Config splitting:** Separate path config, model parameters, and data schema definitions. Consider YAML or TOML for non-code config.
4. **Restore orchestrator (redesigned):** Build a new pipeline runner that calls the step scripts in sequence, with checkpointing and resume capability. Use the R script via `subprocess` for synthpop.
5. **Parameterize scenario definitions:** Move the scenario file list and ABM threshold values in step_5 into config rather than hardcoding.
6. **Add `requirements.txt`:** Pin package versions used for the paper.
7. **Validation assertions:** Add shape/value checks at module boundaries (e.g., confirm sequence array is n_patients × 52, cluster IDs are in expected range after loading npz).

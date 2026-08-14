# Modular Update Guide

Each component of this pipeline can be replaced independently. You might want a more sophisticated clustering algorithm, a different synthetic population generator, a true ABM framework, or a different classifier. This guide describes what each component expects and produces, so you can swap one out without breaking the rest.

## Architecture

The pipeline is data-flow: each level writes artifacts that the next level reads.

```
 ┌──────────────────────────────────────────────────────────────┐
 │  LEVEL 1 — Phenotype discovery            run_level1.py      │
 │                                                              │
 │  ┌────────────┐   ┌────────────┐   ┌────────────┐            │
 │  │ preprocess │──▶│ clustering │──▶│  classify  │            │
 │  │            │   │(external R)│   │            │            │
 │  └────────────┘   └─────┬──────┘   └────────────┘            │
 │                         │                                    │
 │                   ┌─────▼──────┐                             │
 │                   │transitions │                             │
 │                   └────────────┘                             │
 └──────────────────────────┬───────────────────────────────────┘
      best_model.pkl · cluster_transitions.npz
      initial_distributions.npz · processed_synth_raw_data.csv
 ┌──────────────────────────▼───────────────────────────────────┐
 │  LEVEL 2 — Digital twin initialisation    run_level2.py      │
 │                                                              │
 │  ┌────────────┐   ┌────────────┐   ┌────────────┐            │
 │  │ synthesis  │──▶│   assign   │──▶│  simulate  │            │
 │  │    (R)     │   │            │   │  (Markov)  │            │
 │  └────────────┘   └────────────┘   └─────┬──────┘            │
 │                                    ┌─────▼──────┐            │
 │                                    │  fidelity  │            │
 │                                    └────────────┘            │
 └──────────────────────────┬───────────────────────────────────┘
      *_clustered.csv · sequences_<scenario>_seed<n>.npy
 ┌──────────────────────────▼───────────────────────────────────┐
 │  LEVEL 3 — Resource simulation            run_level3.py      │
 │                                                              │
 │  ┌────────────┐   ┌────────────┐   ┌────────────┐            │
 │  │   agents   │──▶│    abm     │──▶│ scenarios  │            │
 │  └────────────┘   └────────────┘   └────────────┘            │
 └──────────────────────────────────────────────────────────────┘
```

The contract between levels is file formats and function signatures. As long as your replacement produces the same outputs in the same format, everything downstream works unchanged.

**Before any swap**, note two conventions that hold throughout:

- **Cluster IDs are 0-indexed integers in Python.** The R clustering emits 1-indexed labels; `level1.sequences.normalise_labels()` is the single translation point.
- **Sequences are `(n_patients, 52)` integer arrays** with values in `[0, 10]`, row-aligned with their population frame.

---

## Component 1: Preprocessing

**Current implementation:** `health_dt/level1/preprocess.py`

**What it does:** Groups high-cardinality categoricals, one-hot encodes, binarizes comorbidities (SEER-Medicare convention: value 3 = present), and produces two frames — one for classification, one for synthpop.

### Interface contract

**Input:** A CSV of raw patient characteristics with cluster labels.

**Output 1 — `processed_data.csv`:** one-hot categoricals + numeric + binary comorbidities + `comorbid_count` + `target` (0-indexed integer cluster). All features numeric, no missing values.

**Output 2 — `processed_synth_raw_data.csv`:** grouped categoricals **as strings, not one-hot**, plus numeric, binary comorbidities, `comorbid_count`. No target column.

### How to swap

Edit `build_model_matrix()` or `build_synthesis_frame()`. Different encoding, different variable selection, imputation instead of fill-with-"Missing" — all local changes. The constraints are:

- `processed_data.csv` must have a `target` column and all-numeric features.
- `processed_synth_raw_data.csv` must have `config.PATIENT_CAT_COLUMNS` as strings and comorbidities as 0/1.

If you add or remove features, the classifier adapts automatically because it reads whatever columns the CSV contains. **But the R script hardcodes variable ordering** — update the `DEMOGRAPHIC_VARS`, `CANCER_VARS`, and `COMORBIDITY_VARS` vectors in `R/level2_synthpop.R` to match.

**Files to touch:** `level1/preprocess.py`, `config.py` (column lists), `R/level2_synthpop.R` (if variables changed).

**Validation after swap:** Re-run `run_level1.py`. Check the classifier's macro F1 in `model_comparison.csv` hasn't degraded.

---

## Component 2: Clustering

**Current implementation:** External R script (LCS distance + hierarchical clustering, not in this repository). Python reads the labels via `level1/sequences.py :: load_labels()`.

**What it does:** Groups patients into utilisation phenotypes from their 52-week encounter sequences.

### Interface contract

**Input:** Patient encounter sequences, 52 weekly states per patient.

**Output:** A CSV with a `cluster` column, one row per patient, **in the same row order as the combined export**. The join is positional; `load_labels()` checks the row counts match and refuses to guess if they don't.

### How to swap

The most natural place for an upgrade. Alternatives:

- **LSTM autoencoder + latent-space clustering** (explored earlier and dropped for interpretability).
- **Dynamic Time Warping** distance with k-medoids.
- **Hidden Markov Model** clustering — fit one HMM per cluster, assign by likelihood.
- **Deep embedding clustering** (DEC, VaDE).

To plug one in:

1. Write a script or module that reads the sequences, runs your clustering, and saves a CSV with a `cluster` column.
2. Point `config.CLUSTER_LABELS_CSV` at your output.
3. Labels may be 1-indexed or 0-indexed — `normalise_labels()` handles either, preserving sort order. You do **not** need to touch any index mapping.
4. If the number of clusters changes from 3, see the warning below.

**Files to touch:** your clustering script, `config.CLUSTER_LABELS_CSV`. That's it — everything downstream reads from config and derives cluster IDs from the data.

**If K changes:** transition matrices, initial distributions, and the classifier are all keyed by cluster ID and regenerate automatically. What does *not* regenerate is interpretation — the current three phenotypes map to low/medium/high use, and a five-cluster solution needs its own clinical reading before any Level 3 result means anything.

**Validation after swap:** Re-run `run_level1.py`, then `run_level2.py`. The Frobenius norms in `fidelity_<scenario>.csv` will tell you whether the transition matrices are being extracted and applied consistently. The published run had all clusters below 0.04.

---

## Component 3: Classifier

**Current implementation:** `health_dt/level1/classify.py` — Logistic Regression, Random Forest, and Gradient Boosting compared by macro F1.

**What it does:** Learns to predict phenotype from patient characteristics, so synthetic patients — who have characteristics but no sequences — can be assigned a phenotype and therefore a transition matrix.

### Interface contract

**Input:** `processed_data.csv` — numeric features plus a `target` column.

**Output:** `best_model.pkl`, a dict of `{model, model_name, feature_names}`. The model must support `.predict(X)`.

**The `feature_names` entry is not optional.** `level2/assign.py` reindexes each synthetic cohort to exactly that column order, filling absent one-hot columns with 0. Without it, a cohort missing a rare category — quite possible after an age shift — produces a differently shaped matrix and a silently wrong prediction.

### How to swap

To use a different estimator (XGBoost, LightGBM, a neural net):

1. Add it to `build_models()` in `level1/classify.py`. It needs `.fit(X, y)` and `.predict(X)`; wrap a non-sklearn model in a small class exposing those.
2. Add its name to `config.CLASSIFIER_MODELS`.
3. If it needs different preprocessing, put that inside a `Pipeline` so the saved object includes it.

**Files to touch:** `level1/classify.py`, `config.CLASSIFIER_MODELS`.

**Validation after swap:** Compare macro F1 in `model_comparison.csv`. Then check the phenotype mix of the synthetic cohorts in the `run_level2.py` log — if the classifier is poor, synthetic patients land in the wrong phenotypes and every simulation result downstream is unreliable.

---

## Component 4: Population Synthesis

**Current implementation:** `R/level2_synthpop.R`, invoked from `health_dt/level2/synthesis.py`. CART-based sequential synthesis with an optional truncated-normal age shift.

**What it does:** Generates synthetic populations preserving the joint distribution of the real characteristics, with optional distributional shifts for scenario analysis.

### Interface contract

**Input:** `processed_synth_raw_data.csv` — characteristics with grouped categoricals, not one-hot.

**Output:** One CSV per scenario with the same column names as the input. No cluster column — that comes from Component 5.

### How to swap

Alternatives: **CTGAN / TVAE** (SDV library), **DataSynthesizer** (Bayesian network), **copula methods**, **bootstrap resampling with perturbation** (simple baseline).

To plug one in:

1. Add a branch to `synthesize()` in `level2/synthesis.py` alongside `synthpop` and `marginal` — it dispatches on a `method` string, so this is a two-line change plus your function.
2. Your function signature is `(source_csv, output_csv, n, age_shift, seed) -> Path`.
3. Output columns must match the input. Categorical values must be strings matching the categories the classifier was trained on, or the one-hot alignment will drop them.

**Files to touch:** `level2/synthesis.py`. Add the method name to the `--method` choices in `run_level2.py`.

### Getting scenario shifts right

This is the subtle part. In the R implementation, age is synthesised **first** in the visit sequence, so every later variable — comorbidities, stage, treatment — is generated *conditional on the shifted ages*. An older synthetic cohort therefore carries an older cohort's comorbidity burden.

Adding 5 to an age column after synthesis does **not** do this. It produces 80-year-olds with 75-year-olds' comorbidity profiles, and any Level 3 conclusion drawn from it is an artefact. If your synthesiser cannot condition this way, say so in its docstring and restrict it to non-demographic scenarios.

**Validation after swap:** `fidelity_<scenario>.csv`. The pMSE ratio should be near 1.0 — real and synthetic indistinguishable to a discriminator. Check the `marginal` method's ratio for comparison: it is deliberately terrible, and the gap between it and yours is what your synthesiser is buying you.

---

## Component 5: Phenotype Assignment

**Current implementation:** `health_dt/level2/assign.py`

**What it does:** Applies the Level 1 classifier to each synthetic cohort, appending a 0-indexed `cluster` column.

### Interface contract

**Input:** a synthetic cohort frame, plus the model bundle from `best_model.pkl`.

**Output:** the same frame with a `cluster` column of 0-indexed integers.

### How to swap

The most useful variation is **probabilistic assignment** — sampling from `predict_proba` instead of taking the argmax, which stops the synthetic cohort understating phenotype uncertainty:

```python
probabilities = bundle["model"].predict_proba(X)
clusters = np.array([rng.choice(len(p), p=p) for p in probabilities])
```

**Files to touch:** `level2/assign.py :: assign_clusters()`.

**Validation after swap:** the phenotype mix logged during `run_level2.py`, compared against the real mix in `fidelity_<scenario>.csv`. Soft assignment shifts the mix; make sure it shifts it *toward* the real distribution, not away.

---

## Component 6: Trajectory Simulation

**Current implementation:** `health_dt/level2/simulate.py` — first-order Markov chain using per-phenotype transition matrices.

**What it does:** For each synthetic patient, generates a 52-week state sequence following their phenotype's transition probabilities.

### Interface contract

**Input:**
- `population` — DataFrame with a 0-indexed `cluster` column.
- `transitions` — `{cluster_id: (N_STATES, N_STATES) array}`, rows summing to 1.
- `initial` — `{cluster_id: (N_STATES,) array}` summing to 1.
- `seed` — int.

**Output:** `np.ndarray` of shape `(n_patients, 52)`, dtype int, values in `[0, N_STATES)`, row-aligned with `population`. Reproducible at a given seed.

### How to swap

The first-order memoryless assumption is the pipeline's main simplification: the probability of next week's care depends only on this week's state. Real care has memory — a patient three weeks into a chemotherapy cycle is not exchangeable with one just starting.

| Upgrade | Buys you | Cost |
|---|---|---|
| Higher-order Markov | Short-range memory | State space grows as `N_STATES^order`; sparse fast |
| Semi-Markov | Explicit dwell-time distributions | Must estimate and store sojourn times |
| Hidden Markov Model | Latent clinical states behind observed encounters | Fitting and storing per-phenotype models |
| Covariate-conditioned | Variation within a phenotype | Transition estimation per covariate stratum |
| Neural autoregressive | Arbitrary dependence | Loses interpretability; needs many sequences |

To plug one in, write a function with the same signature as `simulate_sequences` and call it from `run_level2.py`.

**If your model needs artifacts Level 1 doesn't produce** — dwell-time distributions, a fitted HMM — you have two options:

1. **Compute them inside your simulator** from `outputs/real_sequences.npy` and `outputs/real_labels.npy`, which Level 1 already saves. Simplest, and keeps the change to one file.
2. **Extend Level 1** to emit them: add an extraction function to `level1/transitions.py`, save it in `run_level1.py`, and load it in your simulator. Cleaner if you intend to keep the method. Follow the existing pattern — save with validation, load with an `expected_clusters` check.

**Files to touch:** `level2/simulate.py`, `run_level2.py` (the call), and `level1/transitions.py` + `run_level1.py` if you took option 2.

**A warning about the random stream:** any implementation that consumes randomness differently will not reproduce published seeds. That is fine — it is a new method — but give it a new function name rather than editing `simulate_sequences` in place, so the distinction stays visible. A vectorised version of the *same* model was written during development and dropped for exactly this reason: 50× faster, distributionally identical, but different trajectories at the same seed.

**Validation after swap:** Frobenius norms in `fidelity_<scenario>.csv`, plus a visual check of `state_distribution_stacked.png` — the temporal pattern should resemble the real data.

---

## Component 7: Agent-Based Model

**Current implementation:** `health_dt/level3/abm.py` — weekly ticks, priority-based allocation, queue carryover with drop thresholds.

**What it does:** Takes simulated sequences as planned care trajectories and constrains them against finite resource pools, tracking unmet demand, waits, and abandonment.

### Be clear about what you are replacing

The current engine is a **synchronous weekly queueing approximation**, not a behavioural ABM:

- Time is discrete and weekly, matching claims granularity.
- All agents declare needs simultaneously; each pool sorts and serves to capacity.
- Unmet agents carry over until a drop threshold, then abandon.
- Agents do not interact, choose between providers, reschedule, or respond to waiting.

The largest structural gap: **unmet need never worsens a patient's condition.** The planned trajectory is fixed by Level 2, and only `spillover_state` can perturb the actual one. Today's unmet need cannot generate tomorrow's hospitalisation — which is arguably the central policy question — so the model likely *understates* the cost of constrained capacity.

### Interface contract

**Input:**
- `agents` — list of `SurvivorAgent`, each with `planned` (52-week int array), `cluster`, `acuity`.
- `resources` — list of `config.Resource`.
- `drop_threshold`, `seed`.

**Output:** a dict with four keys:
- `resources` — one row per resource: `resource`, `capacity`, `total_demand`, `total_served`, `total_unmet`, `mean_utilization`, `peak_demand`, and so on.
- `agents` — one row per agent: `agent_id`, `cluster`, `total_unmet`, `planned_hospital_weeks`, `actual_hospital_weeks`, `wait_*`.
- `weekly` — one row per week: `week`, `total_demand`, `total_unmet`.
- `timeseries` — per-resource weekly arrays.

`level3/scenarios.py :: aggregate()` picks up **every numeric column** in `resources` automatically and reports its mean and standard deviation across replications. Emit extra columns freely — `mean_wait_days`, `p95_wait`, anything — and they flow through to the summary without touching the sweep code.

### How to swap

Candidates: **SimPy** (discrete-event, continuous time, natural for queueing), **Mesa** (Python ABM framework with behaviours and schedulers), **NetLogo** (visual, good for stakeholder demos), or custom extensions adding provider agents, referral networks, or geography.

Three levels of change:

1. **Same interface, different engine.** Write a new function in `level3/abm.py` (or a new module, e.g. `level3/des_abm.py`) with the signature of `run_abm` returning the same dict. Update the call in `level3/scenarios.py :: sweep()` — one line.

2. **Different agent model.** Extend the `SurvivorAgent` dataclass in `level3/agents.py` with new fields (location, insurance, provider preference) and populate them in `build_agents`. The allocation loop reads `agent.planned[week]`; if your agent changes that interface, update the loop.

3. **Different resource model.** Add resource types to `config.RESOURCES`, add priority schemes to `ResourcePool.allocate()`, or add capacity variation (seasonal, day-of-week) to `ResourcePool.reset_week()`.

**Files to touch:** `level3/abm.py` or a new engine module; `level3/agents.py` if the agent changes; `config.RESOURCES` for new resource types; `level3/scenarios.py` for the call site.

**Validation after swap:** run both engines on the same population and sequences. Differences in unmet demand totals are expected — that's the point of a better model — but the *ordering* of resources by strain should be directionally consistent. If your new engine says Mental_Health is fine and the old one said it was the tightest constraint, understand why before proceeding.

### A note on acuity

`compute_acuity` in `level3/agents.py` looks for `charlson_approx` / `charlson_index`, neither of which exists in the synthpop output (which carries `comorbid_count`). On the published runs every agent scored 0, so `acuity` queues behaved as FIFO. `build_agents` now warns when this happens.

Adding `"comorbid_count"` to `config.ACUITY_COLUMNS["burden"]` makes triage active. Doing so **changes results** — it is a new analysis, not a bug fix, and conclusions about priority rules cannot be carried over from the published runs.

---

## Component 8: Validation and Analysis

**Current implementation:** `level1/profile.py`, `level2/fidelity.py`, `plots.py`

These are consumers, not producers: they read pipeline outputs and generate tables and figures. Nothing depends on them, so they have the most flexible interface.

- **Different statistical tests:** edit `level1/profile.py`. Currently chi-square for binary outcomes and Kruskal-Wallis for continuous; you might want logistic regression, permutation tests, or Bayesian alternatives.
- **Different fidelity metrics:** add to `level2/fidelity.py`. pMSE is one approach; KL divergence, Wasserstein distance, or per-variable tests are all reasonable additions.
- **Different figures:** edit `plots.py`. Sankey diagrams, alluvial plots, interactive dashboards.

**Files to touch:** only these. No other module imports them for anything but display.

---

## General Rules for Replacement

1. **Understand the interface before changing internals.** Read the docstring, the parameter types, and what downstream code does with the return value. The contract matters more than the implementation.

2. **Write the new module, then swap.** Create a new file alongside the old one, get it working, then change the call site. You can always revert one line.

3. **Keep the old version until the new one validates.** Running both and comparing outputs is the only way to know what changed.

4. **Match the output format exactly, or update every consumer.** Grep for the assumption you're changing.

5. **Re-run validation after every swap.** The pipeline has built-in checks — macro F1 for the classifier, pMSE for synthesis quality, Frobenius norms for transition fidelity. If a number gets worse, understand why before proceeding.

6. **Document what you changed and why.** Future you will want to know why the Markov simulator became an HMM.

## Quick Reference

| I want to… | Files to modify | Config changes |
|---|---|---|
| Add a new classifier | `level1/classify.py` | `CLASSIFIER_MODELS` |
| Change clustering method | Your clustering script | `CLUSTER_LABELS_CSV` |
| Change number of clusters | Your clustering script | `CLUSTER_LABELS_CSV`, `LABEL_COLUMN` |
| Use a different synthesiser | `level2/synthesis.py` | — |
| Add a population scenario | — | `SCENARIOS` |
| Replace Markov with HMM | `level2/simulate.py`, `run_level2.py` | — |
| Use SimPy or Mesa for the ABM | New module + `level3/scenarios.py` | — |
| Add a resource type | — | `RESOURCES` |
| Change resource capacities | — | `RESOURCES` |
| Change constraint levels | — | `DROP_THRESHOLDS` |
| Make acuity triage active | — | `ACUITY_COLUMNS` |
| Add outcome variables | — | `OUTCOME_COLUMNS` |
| Add comorbidities | — | `COMORBIDITY_COLUMNS` |
| Change the state encoding | `plots.py` (`STATE_COLORS`) | `STATE_LABELS`, `PATTERN_TO_STATE`, `HOSPITAL_STATES`, `RESOURCES` |
| Run on a different dataset | — | Paths and column lists |

# Reproducibility and limitations

## Determinism

Every stochastic stage takes an explicit seed derived from `config.RANDOM_SEED`. Nothing relies on global NumPy state.

| Stage | Generator |
|---|---|
| Train/test split, model fitting | sklearn `random_state` |
| Population synthesis | R `set.seed(seed)`, passed on the command line |
| Trajectory simulation | `np.random.default_rng(seed + scenario_index)` — the offset stops two scenarios sharing a stream |
| Replication seeds | `np.random.default_rng(master).integers(...)`, drawn as one vector so 100 extends 50 |
| Agent ordering | `np.random.RandomState(seed)` — see below |
| ABM allocation | `np.random.default_rng(seed)` |

**Why one legacy generator remains.** `build_agents` uses `RandomState`, not `default_rng`. The original pipeline used `np.random.seed()` + `np.random.choice()`, and `RandomState(seed).choice(...)` reproduces that index sequence bit-for-bit while `default_rng` — a different algorithm — would not. Since agent order determines who is served under FIFO, switching would silently change every published result. The improvement over the original is that no global state is touched.

**Draws that look pointless and aren't.** `ResourcePool.reset_week` calls `rng.random()` for every carried-over agent even when `RANDOM_DROPOUT_PROB` is 0. Removing it would shift every subsequent draw, and it keeps runs with and without dropout on the same stream.

**One deliberate non-reproducibility.** A vectorised Markov sampler written during development was ~50× faster and distributionally identical, but consumed the random stream in a different order and so produced different trajectories at the same seed. It was dropped rather than kept as an option. If you write one, give it a new function name — see `docs/MODULAR_UPDATE_GUIDE.md`.

## Verifying a run

**Environment.**

```bash
python -c "import numpy, pandas, sklearn, scipy; print(numpy.__version__, pandas.__version__, sklearn.__version__, scipy.__version__)"
Rscript -e 'cat(R.version.string, "\n"); packageVersion("synthpop")'
```

Reported results used the pins in `requirements.txt`. scikit-learn occasionally changes tree-splitting tie-breaks between minor versions; if macro F1 moves by more than ~0.01, check the version before checking the data.

**Smoke test, no data required.**

```bash
python tests/make_fixture.py --n 500
python run_level1.py
python run_level2.py --method marginal --replications 3
python run_level3.py
pytest
```

This proves the code executes. It says nothing about the science — and note that with a 500-patient fixture against capacities sized for 24,462, no resource binds, so Level 3 will show zero unmet demand.

**Checks on a real run.**

- **Rare share** in the Level 1 log should be small. A warning above 25% means the event columns lost their zero-padding.
- **Macro F1** in `model_comparison.csv` bounds how faithfully any scenario can be simulated.
- **Frobenius norms** in `fidelity_<scenario>.csv` — all clusters were below 0.04 in the published run. Above ~0.1 means synthesis or phenotype assignment has drifted.
- **pMSE ratio** near 1.0 means real and synthetic are hard to tell apart.
- **Phenotype mix** should track the real one in the reference scenario. It is *expected* to shift in counterfactual scenarios — that is the point.
- **Alignment warnings** ("training feature absent, filled with 0") mean the old pipeline would have produced a different, wrong answer at that point.
- **Acuity warning** means queues are running FIFO. True of the published runs.

## Three findings from the reorganisation

Each is documented in code, warns at runtime where possible, and was **not** silently fixed.

**1. Acuity triage was inert.** `compute_acuity` looks for `charlson_approx` / `charlson_index`; the synthpop output carries `comorbid_count`. Every agent scored 0, so `acuity` queues behaved as FIFO throughout. Conclusions about priority rules cannot be drawn from the published runs. Adding `comorbid_count` to `config.ACUITY_COLUMNS["burden"]` activates triage — and makes it a new analysis.

**2. `planned_hospital_weeks` counted the wrong state.** It counted state 4, which is *Onc/Hem Only*; hospital states are 7, 8, 9. Now computed from `config.HOSPITAL_STATES`. This was a reported column, not an input to allocation, so no simulation result moves.

**3. The classifier was pickled bare.** Scoring synthetic cohorts relied on `get_dummies` producing identical columns to training — not guaranteed when an age shift eliminates a rare category. The model bundle now stores `feature_names` and `level2/assign.py` reindexes to it. Results change only where the old behaviour was already wrong.

## Enclave handoff

**Before transfer**

- [ ] Confirm no data files are staged — check `git status --ignored`
- [ ] Confirm `config.py` contains paths only, no patient identifiers or disclosive counts
- [ ] Run the fixture smoke test on the origin machine

**Inside the enclave**

- [ ] Set `config.DATA_DIR` to the read-only mount and `config.OUTPUT_DIR` to a writable location
- [ ] Set `config.RSCRIPT` to the full interpreter path if `Rscript` is not on `PATH`
- [ ] Verify `synthpop` and `truncnorm` are installed, or plan to use pre-generated cohorts
- [ ] Run levels one at a time the first time; each writes `outputs/run_levelN.log`

**Before exporting results**

- [ ] Check cell sizes against the disclosure threshold — `cluster_profiles.csv` and `table1_categorical.csv` carry per-cluster counts and percentages
- [ ] `abm_runs_summary.csv` is aggregate-only and generally safe
- [ ] Agent-level tables describe *synthetic* patients, not real ones — but confirm your DUA's view before export

## Known limitations

These belong in a manuscript's limitations section, not just in code comments.

**Level 3 is a queueing approximation, not a behavioural agent-based model.** Time is discrete and weekly; all agents act simultaneously; nobody interacts, chooses between providers, reschedules, or responds to waiting.

**No feedback from unmet need to health state.** Planned trajectories are fixed at Level 2. Unmet demand is recorded but never worsens a patient's condition or generates later demand, so the model likely understates the downstream cost of constrained capacity. This is the largest structural gap.

**Trajectories are memoryless.** The first-order Markov assumption means next week's care depends only on this week's state. Treatment cycles, post-discharge follow-up, and disease progression all have memory the chain cannot express.

**Phenotypes are assigned, not observed, for synthetic patients.** Classifier error propagates into every Level 3 result — report macro F1 alongside the simulation findings.

**Capacities are assumptions, not measurements.** The values in `config.RESOURCES` were chosen to place the system near realistic strain, not estimated from delivery data. Absolute unmet-demand figures are therefore not interpretable on their own; the design supports *relative* comparison across scenarios at fixed capacity.

**Scenario shifts are demographic, not behavioural.** An older synthetic cohort carries older patients' comorbidity profiles, because age is synthesised first and everything conditions on it. It does not carry older patients' care-seeking behaviour — transition matrices are held fixed by phenotype.

**Acuity triage was inert in the published runs.** See above.

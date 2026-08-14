# Health Digital Twin — Cancer Survivorship Care Utilisation

A three-level simulation framework for modelling healthcare utilisation among older (65+) lung cancer survivors from SEER-Medicare claims. It discovers utilisation phenotypes from 52-week encounter sequences, generates synthetic populations under counterfactual demographics, and simulates care delivery under finite resource capacity.

## The three levels

**Level 1 — Phenotype discovery and classification.**
Patients are clustered into utilisation phenotypes (low / medium / high use) from their 52-week encounter sequences. Clustering runs externally in R using Longest Common Subsequence distance (k=3, merged from k=4). A multiclass classifier then learns to predict phenotype from patient characteristics, and per-phenotype Markov transition matrices are extracted from the real sequences.

**Level 2 — Digital twin initialisation.**
Synthetic patient populations are generated with R's `synthpop` (CART-based synthesis), optionally shifted — age ±5 years, altered stage mix. Each synthetic patient is assigned a phenotype by the Level 1 classifier, then given a 52-week *planned* care trajectory by running that phenotype's Markov chain.

**Level 3 — Resource-constrained delivery.**
Each synthetic patient becomes an agent whose planned trajectory meets finite clinic, bed, and navigator capacity. Weekly priority-based allocation produces unmet demand, queue waits, and abandonment, compared across population scenarios and constraint levels over 100 replications.

The quantity of interest is the gap between planned and delivered care, and how it moves when the population or the capacity changes.

## Start here — run it with no data

```bash
pip install -e .
python run_demo.py
```

Generates a SEER-shaped demo cohort, runs all three levels, and prints samples of
every key output. About a minute. Nothing real is used, and nothing needs to be
downloaded.

[`quickstart.ipynb`](quickstart.ipynb) is the same walkthrough as a notebook,
with the figures and tables rendered inline — open it on GitHub to see what the
pipeline produces before running anything.

## Running it

Each level is a script. Each reads the previous level's outputs from `outputs/` — you never re-run an earlier level to run a later one.

```bash
python run_level1.py     # phenotypes, classifier, transition matrices
python run_level2.py     # synthetic cohorts, planned trajectories
python run_level3.py     # capacity constraints, scenario sweep
```

Useful flags:

```bash
python run_level2.py --method marginal      # no R installed (smoke tests only)
python run_level2.py --replications 10      # quick run instead of 100
python run_level3.py --scenario orig_age    # one scenario
```

Each script writes a log to `outputs/run_levelN.log` alongside its results.

## Setup

```bash
pip install -e .
```

Level 2 population synthesis needs **R 4.x**:

```r
install.packages(c("synthpop", "truncnorm"))
```

If R is unavailable, `--method marginal` substitutes a pure-Python sampler that draws each variable independently. It runs the pipeline end to end but destroys the joint structure the twin depends on — use it to check the code works, never for results.

### Demo data

`health_dt.demo_data` writes a SEER-shaped cohort under the filenames `config.py` expects, so the pipeline runs unmodified:

```bash
python -m health_dt.demo_data --n 500
pytest
```

The three latent phenotypes have genuinely different utilisation intensity, and outcomes correlated with them — a cohort of pure noise would pass shape checks while hiding logic errors that only appear when there is signal to find.

Two differences from a real run, both flagged in the demo output: synthesis uses the `marginal` method (no R, but no joint structure either), and capacities are scaled to the demo cohort — at full capacity 500 patients strain nothing.

## Configuration

Everything lives in `health_dt/config.py`, in seven sections: paths, weekly encounter encoding, patient columns, categorical regrouping, Level 1 parameters, Level 2 scenarios, Level 3 resources. No other file contains a path, a column name, or a magic number.

Adding a scenario is one entry:

```python
SCENARIOS = [
    ...
    Scenario(
        name="Age_10yr_Older",
        synth_csv="synthpop_patients_ageshift_plus_10.csv",
        age_shift=10,
        description="Mean age shifted up 10 years",
    ),
]
```

Changing a capacity assumption is one number:

```python
Resource("Navigator", 3000, (7, 8, 9, 10), priority="acuity"),   # was 2000
```

## Data

Input files are **not** distributed — SEER-Medicare is governed by a Data Use Agreement. `.gitignore` blocks `data/`, `outputs/`, and all `*.csv` / `*.npy` / `*.pkl` by default.

Three inputs are expected in `data/`:

| File | Contents |
|---|---|
| `lung_above65_notx_merged.csv` | Patient characteristics plus `event1`…`event52` |
| `above65-notx-3-clusters-merged-from-4.csv` | Phenotype labels from the R clustering |
| `patients_clustered_3clus.csv` | Characteristics joined to labels |

Each weekly `event` cell is a zero-padded 6-bit string, one bit per provider type, mapped to 11 weekly states. See [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Project structure

```
health_dt/
├── config.py              all paths, columns, encodings, scenarios, capacities
├── logging_setup.py
├── plots.py               all figure generation
├── level1/
│   ├── preprocess.py      regroup categoricals, build modelling matrices
│   ├── sequences.py       event strings → weekly state codes
│   ├── classify.py        train and select the phenotype classifier
│   ├── transitions.py     per-phenotype Markov matrices, save/load
│   └── profile.py         Tables 1 and 2
├── level2/
│   ├── synthesis.py       synthpop via R, or marginal fallback
│   ├── assign.py          phenotypes for synthetic patients
│   ├── simulate.py        Markov trajectories
│   └── fidelity.py        real vs synthetic comparison
├── level3/
│   ├── agents.py          survivor agents and acuity
│   ├── abm.py             weekly allocation engine
│   └── scenarios.py       replication sweep
└── demo_data.py           generated SEER-shaped cohort, for demos and tests

run_level1.py  run_level2.py  run_level3.py     entry points, one per level
run_demo.py                                     end-to-end demo on generated data
quickstart.ipynb                                the same walkthrough, annotated
R/level2_synthpop.R                             population synthesis
tests/                                          test suite
docs/                                           see below
```

## Key outputs

| File | Contents |
|---|---|
| `best_model.pkl` | Classifier plus its training column order |
| `cluster_transitions.npz` | Per-phenotype Markov matrices (Level 1 → 2 handoff) |
| `initial_distributions.npz` | Per-phenotype week-1 state distributions |
| `cluster_profiles.csv`, `outcome_tests.csv` | Tables 1 and 2 |
| `sequences_<scenario>_seed<n>.npy` | Simulated trajectories per replication |
| `fidelity_<scenario>.csv` | Frobenius norms, pMSE, phenotype mix |
| `abm_runs_all_results.csv` | Every ABM run |
| `abm_runs_summary.csv` | Means and standard deviations across replications |

## Documentation

| Document | Purpose |
|---|---|
| [`quickstart.ipynb`](quickstart.ipynb) | Annotated end-to-end run with rendered outputs |
| [`docs/MODULAR_UPDATE_GUIDE.md`](docs/MODULAR_UPDATE_GUIDE.md) | How to replace any component — contracts, files to touch, validation after swap |
| [`docs/data_dictionary.md`](docs/data_dictionary.md) | Column definitions, state encoding, coding conventions |
| [`docs/reproducibility.md`](docs/reproducibility.md) | Seeds, verification checks, known limitations |

## Key design decisions

- **Three scripts, one per level.** Levels communicate through files, so any level can be re-run alone and inspected between stages.
- **All configuration in one file.** Running on a different dataset means editing `config.py` and nothing else.
- **Cluster IDs are 0-indexed everywhere in Python.** R emits 1-indexed labels; `normalise_labels()` is the single place that translates, and loaders assert the IDs match the population using them.
- **Comorbidities binarise with `== 3`, never `> 0`.** The SEER-Medicare code 3 means "condition present with sufficient coverage"; other non-zero codes mean the coverage criterion was not met.
- **R for synthpop.** The package is R-native, called via `subprocess` — one boundary, five arguments.

## Version

**2.0** — the reorganised package. Same science as 1.x, restructured into three levels with a single configuration file. See `docs/reproducibility.md` for what changed and what was preserved.

## License

MIT (code only). SEER-Medicare data remains subject to its own Data Use Agreement.

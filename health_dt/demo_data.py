"""Generate a SEER-Medicare-*shaped* demo cohort.

No real data is used or required. The output has the same column names, dtypes,
and value ranges as the true extract, and — importantly — the same latent
structure: three phenotypes with genuinely different utilisation intensity, and
outcomes correlated with them. A cohort of pure noise would pass shape checks
while hiding logic errors that only appear when there is signal to find.

This is what lets anyone clone the repository and run the whole pipeline without
data access: the tests use it, and so does ``run_demo.py`` and
``quickstart.ipynb``.

    from health_dt.demo_data import write_demo_data
    write_demo_data(n=500)

or from the command line::

    python -m health_dt.demo_data --n 500
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from health_dt import config

logger = logging.getLogger(__name__)

STATE_PATTERNS = list(config.PATTERN_TO_STATE) + ["rare"]


def make_cohort(n: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(combined_export, cluster_labels)`` frames."""
    rng = np.random.default_rng(seed)

    # Three latent phenotypes with genuinely different utilisation intensity,
    # so clustering and transition extraction have real signal to find.
    cluster = rng.choice([1, 2, 3], size=n, p=[0.5, 0.33, 0.17])
    intensity = {1: 0.10, 2: 0.35, 3: 0.65}

    frame = pd.DataFrame(
        {
            config.PATIENT_ID_COLUMN: np.arange(1, n + 1),
            "SEER_REGISTRY": rng.integers(1, 20, n),
            "site_y": rng.choice(["lung_upper", "lung_lower", "lung_other"], n),
            "followup_start": rng.integers(19000, 21000, n),
            "followup_end": rng.integers(21000, 22000, n),
            "lookback_start": rng.integers(18000, 19000, n),
            "birth_dt_num": rng.integers(-6000, -3000, n),
            "sex_num_y": rng.choice([1, 2], n),
            "race_num_y": rng.choice([1, 2, 3, 4], n),
            "age_at_followup_y": np.clip(rng.normal(75, 6, n), 65, 95).round().astype(int),
            "STATE_CODE_y": rng.integers(1, 50, n),
            "age_grp_y": rng.choice(["65-74", "75-84", "85+"], n),
            "race_grp_y": rng.choice(["White", "Black", "Other"], n, p=[0.75, 0.12, 0.13]),
            "acttx_y": rng.choice([0, 1], n),
            "MARITAL_STATUS_AT_DIAGNOSIS": rng.choice([1, 2, 3, 4, 5, 6, 9], n),
            "PRIMARY_SITE": rng.choice(["C340", "C341", "C343", "C349"], n),
            "YEAR_OF_DIAGNOSIS": rng.integers(2013, 2018, n),
            "GRADE_THRU_2017": rng.choice([1, 2, 3, 4, 9], n),
            "YOSTACS20132017QUINTILESTATEBAS": rng.choice([1, 2, 3, 4, 5], n),
            "CHEMOTHERAPY_RECODE_YES_NO_UNK": rng.choice([0, 1], n),
            "RADIATION_RECODE": rng.choice([0, 1, 2, 3, 4, 5, 7, 8], n),
            "INSURANCE_RECODE_2007": rng.choice([1, 2, 3, 4], n),
            "PRIMARY_PAYER_AT_DX": rng.choice([10, 20, 60, 61, 62, 63, 64, 65, 99], n),
            "SEERCOMBINEDSUMMARYSTAGE2000200": rng.choice([1, 2, 3, 4, 7, 9], n),
        }
    )

    # Comorbidity flags use the SEER-Medicare convention: 3 = present.
    intensity_vec = np.vectorize(intensity.get)(cluster)
    for i, condition in enumerate(config.COMORBIDITY_COLUMNS):
        prevalence = 0.08 + 0.35 * intensity_vec
        present = rng.random(n) < np.clip(prevalence * (0.4 + 0.9 * ((i % 5) / 4)), 0, 0.9)
        frame[condition] = np.where(present, 3, rng.choice([0, 1, 2], n))

    # Weekly encounter patterns, biased by phenotype.
    for week in range(1, config.N_WEEKS + 1):
        rate = np.vectorize(intensity.get)(cluster)
        has_encounter = rng.random(n) < rate
        pattern_idx = rng.choice(len(STATE_PATTERNS) - 1, size=n)
        values = np.where(has_encounter, np.array(STATE_PATTERNS)[pattern_idx], "000000")
        # A small share of genuinely rare patterns.
        values = np.where(rng.random(n) < 0.01, "rare", values)
        frame[f"event{week}"] = values

    # Outcomes correlated with phenotype, so the validation tests have signal.
    rate = np.vectorize(intensity.get)(cluster)
    frame["n_ed_12fu"] = rng.poisson(0.5 + 4 * rate)
    frame["n_inpt_stay"] = rng.poisson(0.2 + 2 * rate)
    frame["los"] = rng.poisson(1 + 12 * rate)
    frame["mcr_cost"] = (rng.gamma(2, 6000) * (0.5 + 2 * rate)).round(2)

    labels = pd.DataFrame({"cluster": cluster})
    return frame, labels


def write_demo_data(n: int = 500, seed: int = 0, out: Path | None = None) -> Path:
    """Write a demo cohort using the filenames config.py expects.

    After calling this, ``run_level1.py`` and friends work unmodified.

    Returns the directory written to.
    """
    out = Path(out) if out else config.DATA_DIR
    out.mkdir(parents=True, exist_ok=True)
    combined, labels = make_cohort(n, seed)

    combined.to_csv(out / config.COMBINED_CSV.name, index=False)
    labels.to_csv(out / config.CLUSTER_LABELS_CSV.name, index=False)

    # The classification branch starts from characteristics joined to labels.
    clustered = combined.drop(columns=[c for c in combined.columns if c.startswith("event")])
    clustered[config.LABEL_COLUMN] = labels["cluster"]
    for alt in config.ALT_LABEL_COLUMNS:
        clustered[alt] = labels["cluster"]
    clustered.to_csv(out / config.CLUSTERED_PATIENTS_CSV.name, index=False)

    print(f"Wrote demo cohort of {n} patients to {out}/")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a demo cohort.")
    parser.add_argument("--n", type=int, default=500, help="Number of patients")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="Output directory (default: config.DATA_DIR)")
    args = parser.parse_args()
    write_demo_data(args.n, args.seed, args.out)


if __name__ == "__main__":
    main()

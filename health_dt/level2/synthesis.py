"""Generate synthetic patient populations.

Two synthesisers:

``synthpop`` (default)
    Shells out to R, which builds each variable by CART regression on the
    variables already synthesised. This preserves the joint structure — the
    correlation between age, stage, and comorbidity burden — which is what a
    digital twin needs and what independent marginal sampling destroys.
    Scenario shifts override the age synthesis with a truncated normal at a
    shifted mean; because age is synthesised first, everything downstream is
    conditioned on it.

``marginal``
    Pure-Python fallback sampling each variable from its own marginal,
    independently. No R needed, so the pipeline runs anywhere — but all joint
    structure is destroyed by construction. Use it to smoke-test the pipeline or
    as a null comparator, never for reported results.
"""

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from health_dt import config

logger = logging.getLogger(__name__)


def synthesize(
    source_csv: Path,
    output_csv: Path,
    n: int,
    age_shift: int = 0,
    seed: int = config.RANDOM_SEED,
    method: str = "synthpop",
) -> Path:
    """Generate one synthetic cohort. Returns the path written."""
    if method == "synthpop":
        return _synthpop_r(source_csv, output_csv, n, age_shift, seed)
    if method == "marginal":
        return _marginal_python(source_csv, output_csv, n, age_shift, seed)
    raise ValueError(f"unknown synthesis method {method!r}; use 'synthpop' or 'marginal'")


def r_available() -> bool:
    """Whether the configured Rscript interpreter can be found."""
    return shutil.which(config.RSCRIPT) is not None or Path(config.RSCRIPT).is_file()


def _synthpop_r(source_csv, output_csv, n, age_shift, seed) -> Path:
    """Run the R synthpop script."""
    if not r_available():
        raise FileNotFoundError(
            f"cannot find {config.RSCRIPT!r}. Install R with the synthpop and truncnorm "
            "packages, or set config.RSCRIPT to the full Rscript path. To run without R, "
            "pass --method marginal (null comparator only)."
        )
    if not config.SYNTHPOP_SCRIPT.is_file():
        raise FileNotFoundError(f"R script not found: {config.SYNTHPOP_SCRIPT}")

    command = [
        config.RSCRIPT,
        str(config.SYNTHPOP_SCRIPT),
        str(source_csv),
        str(output_csv),
        str(n),
        str(age_shift),
        str(seed),
    ]
    logger.info("  $ %s", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.stdout:
        logger.info("%s", result.stdout.rstrip())
    if result.returncode != 0:
        raise RuntimeError(f"synthpop failed (exit {result.returncode}):\n{result.stderr.strip()}")
    return output_csv


def _marginal_python(source_csv, output_csv, n, age_shift, seed) -> Path:
    """Sample each column independently from its own marginal distribution.

    Numeric columns are drawn from a normal matched on mean and sd, clipped to
    the observed range and rounded back to integers where the source was
    integral. Categoricals are drawn from observed frequencies.
    """
    source = pd.read_csv(source_csv)
    rng = np.random.default_rng(seed)
    data = {}

    for col in source.columns:
        if col in config.NON_FEATURE_COLUMNS:
            continue
        series = source[col]
        if pd.api.types.is_numeric_dtype(series):
            mean = series.mean() + (age_shift if col in config.PATIENT_NUM_COLUMNS else 0)
            values = np.clip(
                rng.normal(mean, max(series.std(), 0.01), n), series.min(), series.max()
            )
            if (series.dropna() == series.dropna().round()).all():
                values = np.round(values).astype(int)
            data[col] = values
        else:
            counts = series.value_counts(normalize=True)
            data[col] = rng.choice(counts.index.tolist(), size=n, p=counts.to_numpy())

    frame = pd.DataFrame(data)
    frame.to_csv(output_csv, index=False)
    logger.info("  synthesised %d patients from independent marginals", n)
    return output_csv

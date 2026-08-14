"""Simulate weekly care trajectories from phenotype Markov chains.

Each synthetic patient draws an opening state from their phenotype's week-1
distribution, then steps forward 51 times through that phenotype's transition
matrix. The result is the *planned* trajectory: the care this patient would
receive in a system with no capacity limits. Level 3 turns planned into actual.

The first-order memoryless assumption is the main simplification in the
pipeline — see docs/MODULAR_UPDATE_GUIDE.md for how to replace it.
"""

import logging

import numpy as np
import pandas as pd

from health_dt import config
from health_dt.level1.sequences import validate_sequences
from health_dt.level1.transitions import validate_matrices

logger = logging.getLogger(__name__)


def simulate_sequences(
    population: pd.DataFrame,
    transitions: dict[int, np.ndarray],
    initial: dict[int, np.ndarray],
    seed: int = config.RANDOM_SEED,
) -> np.ndarray:
    """Sample one weekly state sequence per patient.

    Patients are visited in row order and each draw is a single ``rng.choice``
    call. Do not reorder or vectorise these calls without accepting that
    published seeds will no longer reproduce — the random stream is part of the
    result.

    Parameters
    ----------
    population:
        Synthetic cohort with a 0-indexed ``cluster`` column.
    transitions, initial:
        Per-phenotype matrices and week-1 distributions from Level 1.
    seed:
        Seed for the isolated generator. Same seed, same output.

    Returns
    -------
    np.ndarray
        ``(len(population), N_WEEKS)`` state codes, row-aligned with population.
    """
    if "cluster" not in population.columns:
        raise KeyError("population has no 'cluster' column; run assign.py first")
    validate_matrices(transitions)

    clusters = population["cluster"].to_numpy().astype(int)
    unknown = set(np.unique(clusters).tolist()) - set(transitions)
    if unknown:
        raise KeyError(
            f"no transition matrix for cluster(s) {sorted(unknown)}; available "
            f"{sorted(transitions)}. The transition archive and the population "
            "disagree on cluster indexing."
        )

    rng = np.random.default_rng(seed)
    sequences = np.zeros((len(population), config.N_WEEKS), dtype=int)

    for i in range(len(population)):
        matrix = transitions[clusters[i]]
        state = rng.choice(config.N_STATES, p=initial[clusters[i]])
        sequences[i, 0] = state
        for week in range(1, config.N_WEEKS):
            state = rng.choice(config.N_STATES, p=matrix[state])
            sequences[i, week] = state

    return validate_sequences(sequences)


def replication_seeds(n: int, master: int = config.RANDOM_SEED) -> np.ndarray:
    """Derive n reproducible child seeds from one master seed.

    Drawn as a single vector, so n=100 extends n=50 rather than replacing it —
    adding replications later does not invalidate earlier ones.
    """
    return np.random.default_rng(master).integers(0, 100_000, size=n)

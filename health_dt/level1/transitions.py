"""Per-phenotype Markov transition matrices — the Level 1 to Level 2 handoff.

Everything the digital twin knows about how care unfolds week to week is
encoded here: one row-stochastic ``N_STATES x N_STATES`` matrix per phenotype,
plus the distribution of week-1 states needed to start a synthetic patient.

Cluster IDs are 0-indexed on disk. The pre-refactor pipeline wrote R's 1-indexed
labels and converted them on load with a bare ``- 1``, which was correct but
unchecked; :func:`load_transitions` now verifies the IDs against the population
they will be applied to.
"""

import logging
import re

import numpy as np
import pandas as pd

from health_dt import config
from health_dt.level1.sequences import validate_sequences

logger = logging.getLogger(__name__)

_CLUSTER_KEY = re.compile(r"^cluster[_-]?(\d+)", re.IGNORECASE)


def transition_matrix(sequences: np.ndarray) -> np.ndarray:
    """Count week-to-week transitions and normalise each row to a probability.

    Rows for states never visited are left all-zero rather than smoothed: an
    unseen state is unreachable in simulation too, and a uniform row would send
    synthetic patients into states the phenotype never occupies.
    """
    counts = np.zeros((config.N_STATES, config.N_STATES))
    if len(sequences) == 0:
        return counts

    np.add.at(counts, (sequences[:, :-1].ravel(), sequences[:, 1:].ravel()), 1)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def extract_transitions(sequences: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    """One transition matrix per phenotype, keyed by 0-indexed cluster ID."""
    validate_sequences(sequences, n_weeks=sequences.shape[1])
    labels = np.asarray(labels)
    if len(labels) != len(sequences):
        raise ValueError(f"labels ({len(labels)}) and sequences ({len(sequences)}) differ")

    matrices = {
        int(c): transition_matrix(sequences[labels == c]) for c in sorted(np.unique(labels))
    }
    for cluster, matrix in matrices.items():
        logger.info(
            "Cluster %d (n=%d):\n%s",
            cluster,
            int((labels == cluster).sum()),
            format_matrix(matrix),
        )
    return matrices


def extract_initial_distributions(
    sequences: np.ndarray, labels: np.ndarray
) -> dict[int, np.ndarray]:
    """Distribution of week-1 states per phenotype.

    Level 2 draws each synthetic patient's opening state from this rather than
    from the stationary distribution, because survivorship follow-up has a real
    starting point — the first week after the index date.
    """
    labels = np.asarray(labels)
    distributions = {}
    for cluster in sorted(np.unique(labels)):
        counts = np.bincount(sequences[labels == cluster, 0], minlength=config.N_STATES)
        distributions[int(cluster)] = counts / counts.sum()
    return distributions


def validate_matrices(matrices: dict[int, np.ndarray]) -> None:
    """Check every matrix is correctly sized and row-stochastic."""
    if not matrices:
        raise ValueError("no transition matrices supplied")
    for cluster, matrix in matrices.items():
        matrix = np.asarray(matrix)
        expected = (config.N_STATES, config.N_STATES)
        if matrix.shape != expected:
            raise ValueError(f"cluster {cluster}: shape {matrix.shape}, expected {expected}")
        row_sums = matrix.sum(axis=1)
        bad = ~(np.isclose(row_sums, 1.0) | np.isclose(row_sums, 0.0))
        if bad.any():
            raise ValueError(
                f"cluster {cluster}: rows {np.flatnonzero(bad).tolist()} sum to "
                f"{row_sums[bad].round(4).tolist()}, expected 1.0"
            )


def save_transitions(matrices: dict[int, np.ndarray], path, write_csv: bool = True) -> None:
    """Write transition matrices to .npz with 0-indexed keys (plus readable CSVs)."""
    validate_matrices(matrices)
    np.savez(path, **{f"cluster_{c}": m for c, m in sorted(matrices.items())})

    if write_csv:
        labels = [config.STATE_LABELS[s] for s in range(config.N_STATES)]
        for cluster, matrix in matrices.items():
            pd.DataFrame(matrix, index=labels, columns=labels).to_csv(
                path.parent / f"transition_cluster_{cluster}.csv"
            )
    logger.info("Saved %s", path)


def load_transitions(path, expected_clusters=None) -> dict[int, np.ndarray]:
    """Load transition matrices, checking they match the population using them.

    Parameters
    ----------
    expected_clusters:
        Cluster IDs present in the population these matrices will be applied to.
        A mismatch raises rather than silently simulating the wrong phenotype.
    """
    with np.load(path) as archive:
        matrices = {_parse_key(k): archive[k] for k in archive.files}
    validate_matrices(matrices)
    _check_clusters(matrices, expected_clusters, path)
    return matrices


def save_initial_distributions(distributions: dict[int, np.ndarray], path) -> None:
    """Write per-phenotype week-1 state distributions to .npz."""
    for cluster, dist in distributions.items():
        if not np.isclose(dist.sum(), 1.0):
            raise ValueError(f"cluster {cluster}: initial distribution sums to {dist.sum():.6f}")
    np.savez(path, **{f"cluster_{c}": d for c, d in sorted(distributions.items())})
    logger.info("Saved %s", path)


def load_initial_distributions(path, expected_clusters=None) -> dict[int, np.ndarray]:
    """Load week-1 state distributions with the same cluster-ID check."""
    with np.load(path) as archive:
        distributions = {_parse_key(k): archive[k] for k in archive.files}
    _check_clusters(distributions, expected_clusters, path)
    return distributions


def format_matrix(matrix: np.ndarray, width: int = 12) -> str:
    """Render a transition matrix as a labelled fixed-width table."""
    labels = [config.STATE_LABELS[s][: width - 2] for s in range(config.N_STATES)]
    lines = [" " * width + "".join(f"{lab:>{width}}" for lab in labels)]
    for i, label in enumerate(labels):
        lines.append(
            f"{label:<{width}}" + "".join(f"{matrix[i, j]:>{width}.3f}" for j in range(config.N_STATES))
        )
    return "\n".join(lines)


def _parse_key(key: str) -> int:
    """Extract the cluster ID from an .npz archive key."""
    match = _CLUSTER_KEY.match(key.strip())
    if match is None:
        raise ValueError(f"cannot parse cluster key {key!r}; expected e.g. 'cluster_0'")
    return int(match.group(1))


def _check_clusters(matrices: dict, expected_clusters, path) -> None:
    if expected_clusters is None:
        return
    expected = {int(c) for c in expected_clusters}
    if expected != set(matrices):
        raise ValueError(
            f"cluster ID mismatch in {path.name}: archive has {sorted(matrices)}, "
            f"population has {sorted(expected)}. If the archive came from the old "
            "pipeline its keys are 1-indexed — regenerate with run_level1.py."
        )

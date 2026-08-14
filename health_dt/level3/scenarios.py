"""Replication sweep: populations x replications x constraint levels.

This is where the study's headline numbers come from. For every scenario
population, every Level 2 replication, and every drop threshold, the ABM runs
once and its resource summary is recorded. Aggregating to means and standard
deviations lets a difference between scenarios be read against the Monte Carlo
noise of the simulation itself rather than asserted from a single run.
"""

import logging

import numpy as np
import pandas as pd

from health_dt import config
from health_dt.level3.abm import run_abm
from health_dt.level3.agents import build_agents

logger = logging.getLogger(__name__)


def sweep(
    scenarios: list[config.Scenario] | None = None,
    resources: list[config.Resource] | None = None,
) -> pd.DataFrame:
    """Run the ABM across every population, replication, and constraint level.

    Parameters
    ----------
    scenarios:
        Defaults to config.SCENARIOS.
    resources:
        Defaults to config.RESOURCES. Pass config.scale_resources(f) for a
        capacity what-if.

    Returns
    -------
    pd.DataFrame
        One row per resource per cell, identified by ``scenario``, ``seed``,
        and ``drop_threshold``.
    """
    scenarios = scenarios or config.SCENARIOS
    frames = []
    total = 0

    for spec in scenarios:
        sequence_files = sorted(config.OUTPUT_DIR.glob(spec.sequence_glob))
        if not sequence_files:
            logger.warning("no simulated sequences for %s; skipping", spec.name)
            continue

        population = pd.read_csv(config.out(spec.clustered_csv))
        logger.info("%s: %d replications x %d thresholds",
                    spec.name, len(sequence_files), len(config.DROP_THRESHOLDS))

        for path in sequence_files:
            sequences = np.load(path)
            seed_label = path.stem.split("seed")[-1]

            for threshold in config.DROP_THRESHOLDS:
                agents = build_agents(population, sequences)
                result = run_abm(agents, drop_threshold=threshold, resources=resources)

                frame = result["resources"].copy()
                frame["scenario"] = spec.name
                frame["seed"] = seed_label
                frame["drop_threshold"] = threshold
                frames.append(frame)
                total += 1

    if not frames:
        raise RuntimeError("sweep produced no results — run run_level2.py first")

    logger.info("Completed %d simulations", total)
    return pd.concat(frames, ignore_index=True)


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    """Collapse the sweep to means and standard deviations across replications.

    The standard deviation is what tells you whether a between-scenario
    difference is larger than simulation noise.
    """
    keys = ["scenario", "drop_threshold", "resource"]
    metrics = [
        c for c in runs.columns
        if c not in keys + ["seed"] and pd.api.types.is_numeric_dtype(runs[c])
    ]
    aggregations = {"n_runs": ("seed", "count")}
    aggregations.update({f"{c}_mean": (c, "mean") for c in metrics})
    aggregations.update({f"{c}_std": (c, "std") for c in metrics})
    return runs.groupby(keys).agg(**aggregations).reset_index()

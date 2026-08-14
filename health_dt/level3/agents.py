"""Survivor agents: the unit of simulation in Level 3.

An agent carries one synthetic patient's planned trajectory plus what the
allocator needs to triage them. Two sequences are tracked: ``planned``, which
never changes, and ``actual``, which diverges when a resource is unavailable and
a spillover rule applies. The gap between them is the model's estimate of care
disrupted by capacity.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from health_dt import config
from health_dt.logging_setup import warn_once

logger = logging.getLogger(__name__)

# Fallback ordinal mapping for text-coded stage.
STAGE_ORDINAL = {"I": 0.0, "II": 1.0, "III": 2.0, "IV": 4.0}


@dataclass
class SurvivorAgent:
    """One synthetic survivor moving through the constrained system.

    Attributes
    ----------
    agent_id:
        Identifier carried from the synthetic population.
    cluster:
        Utilisation phenotype, 0-indexed.
    acuity:
        Triage priority under the "acuity" queue discipline; higher is served
        first.
    planned:
        Weekly states from Level 2 — the care this patient needs.
    actual:
        Weekly states after capacity constraints. Starts as a copy of planned
        and is mutated only where a spillover rule fires.
    unmet:
        One record per unfulfilled resource request.
    waits:
        Consecutive weeks waited per resource, compared against the drop
        threshold.
    """

    agent_id: int
    cluster: int
    acuity: float
    planned: np.ndarray
    actual: np.ndarray = field(default=None)
    unmet: list = field(default_factory=list)
    waits: dict = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self):
        if self.actual is None:
            self.actual = self.planned.copy()


def compute_acuity(row) -> float:
    """Score a patient's triage priority from whatever fields are available.

    Deliberately simple and additive — comorbidity burden, plus stage, plus
    weighted prior hospital weeks — because it only orders a queue, it does not
    predict anything. What matters is that it is monotone in severity and
    computable for every cohort, including ones missing columns.

    Unparseable stage values contribute nothing rather than raising.
    """
    score = 0.0

    for col in config.ACUITY_COLUMNS["burden"]:
        if col in row:
            score += float(row[col])
            break

    for col in config.ACUITY_COLUMNS["stage"]:
        if col not in row:
            continue
        try:
            score += float(row[col])
        except (TypeError, ValueError):
            score += STAGE_ORDINAL.get(str(row[col]).strip().upper(), 0.0)
        break

    for col in config.ACUITY_COLUMNS["history"]:
        if col in row:
            score += float(row[col]) * config.HOSPITAL_HISTORY_WEIGHT
            break

    return score


def build_agents(
    population: pd.DataFrame, sequences: np.ndarray, seed: int = config.RANDOM_SEED
) -> list[SurvivorAgent]:
    """Turn a synthetic cohort into agents, in randomised order.

    The order is not cosmetic: under the "fifo" discipline, position in this
    list determines who is served when demand exceeds capacity. Shuffling stops
    the arbitrary row order of the synthesis output becoming a systematic
    advantage.

    Note the legacy ``RandomState`` generator. This is deliberate — it
    reproduces the exact index sequence the published runs used, whereas
    ``default_rng`` uses a different algorithm and would silently change every
    result. It is isolated to this call and does not touch global NumPy state.
    """
    if "cluster" not in population.columns:
        raise KeyError("population has no 'cluster' column")
    if len(sequences) != len(population):
        raise ValueError(f"sequences ({len(sequences)}) and population ({len(population)}) differ")

    order = np.random.RandomState(seed).choice(len(population), size=len(population), replace=False)
    agents = [
        SurvivorAgent(
            agent_id=int(population.iloc[i].get("patient_id", i)),
            cluster=int(population.iloc[i]["cluster"]),
            acuity=compute_acuity(population.iloc[i]),
            planned=sequences[i],
        )
        for i in order
    ]

    if agents and all(a.acuity == 0.0 for a in agents):
        warn_once(
            logger,
            "acuity-zero",
            "Every agent scored acuity 0 — none of config.ACUITY_COLUMNS exist in this "
            "population, so 'acuity' queues behave as FIFO. This was true of the published "
            "runs; see docs/reproducibility.md.",
        )
    return agents


def summarise_agents(agents: list[SurvivorAgent]) -> pd.DataFrame:
    """Agent-level result table."""
    records = []
    for agent in agents:
        record = {
            "agent_id": agent.agent_id,
            "cluster": agent.cluster,
            "acuity": agent.acuity,
            "total_unmet": len(agent.unmet),
            "disrupted_weeks": int((agent.planned != agent.actual).sum()),
            "planned_hospital_weeks": int(np.isin(agent.planned, config.HOSPITAL_STATES).sum()),
            "actual_hospital_weeks": int(np.isin(agent.actual, config.HOSPITAL_STATES).sum()),
        }
        record.update({f"wait_{name}": weeks for name, weeks in agent.waits.items()})
        records.append(record)
    return pd.DataFrame(records)

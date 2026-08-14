"""The weekly resource allocation engine.

Each simulated week runs four phases:

1. **Reset** — every pool restores capacity and inspects its carryover queue,
   dropping anyone who has waited past the threshold or abandons at random.
2. **Declare** — each agent looks up its planned state and joins the queue of
   every resource that state requires.
3. **Allocate** — each pool orders its queue by the configured discipline and
   serves up to capacity. The remainder is unmet and carried over.
4. **Spill** — where a resource defines a spillover state, unmet agents have
   that week of their *actual* trajectory rewritten.

What this is not: agents do not interact, choose between providers, or
reschedule, and time is discrete and synchronous. It is a queueing
approximation at weekly claims granularity.
"""

import logging

import numpy as np
import pandas as pd

from health_dt import config
from health_dt.level3.agents import SurvivorAgent, summarise_agents

logger = logging.getLogger(__name__)


class ResourcePool:
    """A single finite resource with a weekly capacity and a waiting queue."""

    def __init__(self, resource: config.Resource, drop_threshold: int):
        self.resource = resource
        self.name = resource.name
        self.capacity = resource.weekly_capacity
        self.required_by = frozenset(resource.required_by_states)
        self.priority = resource.priority
        self.spillover_state = resource.spillover_state
        self.drop_threshold = drop_threshold

        self.queue: list[SurvivorAgent] = []
        self.carryover: list[SurvivorAgent] = []
        self.demand: list[int] = []
        self.served: list[int] = []
        self.unmet: list[int] = []
        self.utilization: list[float] = []
        self.dropped: list[int] = []
        self.arrivals: list[int] = []
        self._arrivals_this_week = 0

    def reset_week(self, rng: np.random.Generator) -> None:
        """Restore capacity and filter the carryover queue.

        The random-abandonment draw is taken for every carried-over agent even
        when RANDOM_DROPOUT_PROB is 0. That looks wasteful but keeps the random
        stream independent of the dropout setting, so runs with and without
        dropout stay comparable — and it reproduces the pre-refactor stream.
        """
        kept = []
        for agent in self.carryover:
            if agent.waits.get(self.name, 0) >= self.drop_threshold:
                continue
            if rng.random() < config.RANDOM_DROPOUT_PROB:
                continue
            kept.append(agent)

        self.dropped.append(len(self.carryover) - len(kept))
        self.queue = kept
        self.carryover = []

    def request(self, agent: SurvivorAgent) -> None:
        """Record an agent's need for this resource this week."""
        self.queue.append(agent)
        self._arrivals_this_week += 1

    def allocate(self, rng: np.random.Generator) -> list[SurvivorAgent]:
        """Serve as much of the queue as capacity allows. Returns unmet agents."""
        demand = len(self.queue)
        self.demand.append(demand)
        self.arrivals.append(self._arrivals_this_week)
        self._arrivals_this_week = 0

        if demand == 0:
            self.served.append(0)
            self.unmet.append(0)
            self.utilization.append(0.0)
            return []

        if self.priority == "acuity":
            # Stable sort, so equal-acuity agents keep arrival order — which is
            # what makes an all-zero acuity vector behave as FIFO rather than
            # scrambling the queue.
            self.queue.sort(key=lambda a: -a.acuity)
        elif self.priority == "random":
            rng.shuffle(self.queue)
        # "fifo": already in arrival order.

        served = self.queue[: self.capacity]
        unmet = self.queue[self.capacity :]

        self.served.append(len(served))
        self.unmet.append(len(unmet))
        self.utilization.append(min(len(served) / max(self.capacity, 1), 1.0))
        self.carryover = unmet
        return unmet

    def summary(self) -> dict:
        """One row of the resource result table."""
        return {
            "resource": self.name,
            "capacity": self.capacity,
            "total_demand": sum(self.demand),
            "total_served": sum(self.served),
            "total_unmet": sum(self.unmet),
            "total_new_requests": sum(self.arrivals),
            "total_dropped": sum(self.dropped),
            "queue_at_end": len(self.carryover),
            "mean_utilization": float(np.mean(self.utilization)) if self.utilization else 0.0,
            "peak_demand": max(self.demand) if self.demand else 0,
            "weeks_over_capacity": sum(1 for d in self.demand if d > self.capacity),
        }


def run_abm(
    agents: list[SurvivorAgent],
    drop_threshold: int = 8,
    seed: int = config.RANDOM_SEED,
    resources: list[config.Resource] | None = None,
    verbose: bool = False,
) -> dict:
    """Simulate 52 weeks of constrained delivery.

    Parameters
    ----------
    agents:
        Built by level3.agents.build_agents.
    drop_threshold:
        Weeks an agent waits before abandoning a queue.
    seed:
        Same seed, same result.
    resources:
        Defaults to config.RESOURCES.

    Returns
    -------
    dict
        ``resources`` — one row per resource;
        ``agents`` — one row per agent;
        ``weekly`` — one row per week;
        ``timeseries`` — per-resource weekly arrays.
    """
    pools = {r.name: ResourcePool(r, drop_threshold) for r in (resources or config.RESOURCES)}
    rng = np.random.default_rng(seed)
    weekly = []

    if verbose:
        logger.info(
            "ABM: %d agents, %d resources, %d weeks, drop threshold %d",
            len(agents), len(pools), config.N_WEEKS, drop_threshold,
        )

    for week in range(config.N_WEEKS):
        for pool in pools.values():
            pool.reset_week(rng)

        for agent in agents:
            state = agent.planned[week]
            for pool in pools.values():
                if state in pool.required_by:
                    pool.request(agent)

        total_demand = total_unmet = 0
        for pool in pools.values():
            unmet = pool.allocate(rng)
            total_demand += pool.demand[-1]
            total_unmet += len(unmet)

            for agent in unmet:
                agent.unmet.append(
                    {"week": week + 1, "resource": pool.name, "state": int(agent.planned[week])}
                )
                agent.waits[pool.name] += 1
                if pool.spillover_state is not None:
                    agent.actual[week] = pool.spillover_state

        weekly.append({"week": week + 1, "total_demand": total_demand, "total_unmet": total_unmet})

        if verbose and (week + 1) % 13 == 0:
            logger.info(
                "  week %2d: demand %d, unmet %d (%.1f%%)",
                week + 1, total_demand, total_unmet, total_unmet / max(total_demand, 1) * 100,
            )

    return {
        "resources": pd.DataFrame([p.summary() for p in pools.values()]),
        "agents": summarise_agents(agents),
        "weekly": pd.DataFrame(weekly),
        "timeseries": {
            name: {"demand": p.demand, "served": p.served, "unmet": p.unmet}
            for name, p in pools.items()
        },
    }

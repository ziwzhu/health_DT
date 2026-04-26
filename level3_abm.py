"""
Level 3: Agent-Based Modeling (ABM)
====================================
Layer real-world operational constraints on top of the Level 2 digital twin.

Survivor agents interact with finite resource agents (navigators, imaging
slots, provider capacity) governed by triage and scheduling rules. This
enables "what-if" simulation:
  - What if the survivor population grows 30%?
  - What if we add 2 navigators?
  - What if we change triage priority rules?
  - What if imaging capacity is cut by 20%?

Architecture:
  - SurvivorAgent: one per synthetic patient, carries their phenotype,
    risk factors, and weekly encounter needs from the Markov sequence.
  - ResourcePool: finite capacity for each resource type, with weekly
    reset and queue management.
  - ABMSimulation: orchestrates weekly ticks, matching agents to resources,
    tracking unmet needs, wait times, and downstream effects.

Inputs:  Level 2 synthetic population + sequences
Outputs: Simulation logs, resource utilization metrics, what-if comparisons
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import N_STATES, N_WEEKS, STATE_LABELS, RANDOM_SEED, OUTPUT_DIR


# ============================================================================
# RESOURCE CONFIGURATION
# ============================================================================

@dataclass
class ResourceConfig:
    """
    Define a finite resource in the healthcare system.

    Parameters
    ----------
    name : str
    weekly_capacity : int
        Max patients served per week.
    required_by_states : list of int
        Which encounter states require this resource.
        e.g., Hospital beds needed by state 4 (Hospital Only)
    priority_key : str
        How to prioritize when demand > capacity.
        "acuity" — higher-acuity patients first
        "fifo" — first-come first-served
        "random" — random selection
    spillover_state : int or None
        If patient can't get this resource, what state do they transition to?
        e.g., can't get imaging -> stays in current state (None)
        e.g., can't get hospital bed -> goes to ED (state 4)
    """
    name: str
    weekly_capacity: int
    required_by_states: List[int]
    priority_key: str = "acuity"
    spillover_state: Optional[int] = None
    drop_threshold: int = 8


def default_resource_configs(drop_threshold = 8):
    """
    Default resource configurations matching the 11-state encoding.
    Adjust capacities to match your actual system.

    States (from config.py):
      0=No Encounter, 1=Other Spec Only, 2=MH Only, 3=PCP Only,
      4=Hospital Only, 5=MSK Only, 6=Onc/Hem Only, 7=MH+Other Spec,
      8=Rare/Other, 9=PCP+MH, 10=PCP+MH+Other Spec
    """
    return [
        ResourceConfig(
            name="PCP",
            weekly_capacity=3000,
            required_by_states=[1, 3, 9],  # states involving PCP
            priority_key="acuity",
            drop_threshold = drop_threshold,
        ),
        ResourceConfig(
            name="Other_Specialty",
            weekly_capacity=3300,
            required_by_states=[2, 3, 8, 9],  # states involving other specialty
            priority_key="fifo",
            drop_threshold = drop_threshold,
        ),
        ResourceConfig(
            name="Oncology/Hematology",
            weekly_capacity=400,
            required_by_states=[4],  # states involving Onc/Hem
            priority_key="acuity",
            drop_threshold = drop_threshold,
        ),
        ResourceConfig(
            name="Hospital_Beds",
            weekly_capacity=800,
            required_by_states=[7, 8, 9],  # hospital-based and rare/complex
            priority_key="acuity",
            spillover_state=None,
            drop_threshold = drop_threshold,
        ),
        ResourceConfig(
            name="Musculoskeletal",
            weekly_capacity=650,
            required_by_states=[5],  # MSK Only
            priority_key="fifo",
            drop_threshold = drop_threshold,
        ),
        ResourceConfig(
            name="Mental_Health",
            weekly_capacity=200,
            required_by_states=[6],  # states involving MH
            priority_key="fifo",
            drop_threshold = drop_threshold,
        ),
        ResourceConfig(
            name="Navigator",
            weekly_capacity=2000,
            required_by_states=[7, 8, 9, 10],  # navigators for complex patients
            priority_key="acuity",
            drop_threshold = drop_threshold,
        ),
    ]


# ============================================================================
# SURVIVOR AGENT
# ============================================================================

@dataclass
class SurvivorAgent:
    agent_id: int
    cluster: int
    acuity_score: float  # derived from charlson + stage
    planned_sequence: np.ndarray  # 52-week Markov sequence from Level 2
    actual_sequence: np.ndarray = field(default=None)
    unmet_needs: List = field(default_factory=list)
    wait_weeks: Dict = field(default_factory=lambda: defaultdict(int))

    def __post_init__(self):
        if self.actual_sequence is None:
            self.actual_sequence = self.planned_sequence.copy()


# ============================================================================
# RESOURCE POOL
# ============================================================================

class ResourcePool:
    """Manages a single finite resource with weekly capacity."""

    def __init__(self, config: ResourceConfig):
        self.config = config
        self.name = config.name
        self.base_capacity = config.weekly_capacity
        self.weekly_capacity = config.weekly_capacity
        self.required_by = set(config.required_by_states)
        self.priority_key = config.priority_key
        self.drop_threshold = getattr(config, 'drop_threshold', 8)

        # Tracking
        self.weekly_demand = []
        self.weekly_served = []
        self.weekly_unmet = []
        self.weekly_utilization = []
        self.queue = []
        self.carryover = []
        self.weekly_dropped = []
        self.weekly_arrivals = []
        self._arrivals_this_week = 0

    def reset_week(self, rng):
        """Reset capacity for new week."""
        self.weekly_capacity = self.base_capacity
        # self.queue = list(self.carryover)
        # self.carryover = []
        
        # drop if waited > 8wks or 10% random drop out
        # testing: 8wk, no rand drop
        kept = []
        for a in self.carryover:
            if a.wait_weeks.get(self.name, 0) >= self.drop_threshold:
                continue
            if rng.random() < 0.00:
                continue
            kept.append(a)
        self.weekly_dropped.append(len(self.carryover) - len(kept))
        self.queue = kept
        self.carryover = []
        

    def request(self, agent: SurvivorAgent):
        """Agent requests this resource."""
        self.queue.append(agent)
        self._arrivals_this_week += 1

    def allocate(self, rng):
        """
        Allocate resources to queued agents based on priority.
        Returns (served_agents, unmet_agents).
        """
        demand = len(self.queue)
        self.weekly_demand.append(demand)

        if demand == 0:
            self.weekly_served.append(0)
            self.weekly_unmet.append(0)
            self.weekly_utilization.append(0.0)
            return [], []

        # Sort by priority
        if self.priority_key == "acuity":
            self.queue.sort(key=lambda a: -a.acuity_score)  # highest first
        elif self.priority_key == "fifo":
            pass  # already in order
        elif self.priority_key == "random":
            rng.shuffle(self.queue)

        served = self.queue[:self.weekly_capacity]
        unmet = self.queue[self.weekly_capacity:]

        self.weekly_served.append(len(served))
        self.weekly_unmet.append(len(unmet))
        self.weekly_utilization.append(
            min(len(served) / max(self.base_capacity, 1), 1.0))
        
        self.carryover = unmet

        self.weekly_arrivals.append(self._arrivals_this_week)
        self._arrivals_this_week = 0
        
        return served, unmet

    def set_capacity(self, new_capacity):
        """Modify capacity (for what-if scenarios)."""
        self.base_capacity = new_capacity

    def get_summary(self):
        """Return summary statistics."""
        return {
            "resource": self.name,
            "base_capacity": self.base_capacity,
            "total_demand_with_carryover": sum(self.weekly_demand),
            "total_unmet_with_carryover": sum(self.weekly_unmet),
            "total_served": sum(self.weekly_served),
            "total_new_requests": sum(self.weekly_arrivals),
            "total_dropped": sum(self.weekly_dropped),
            "remaining_in_queue_at_end": len(self.carryover),
            "mean_utilization": np.mean(self.weekly_utilization) if self.weekly_utilization else 0,
            "peak_demand": max(self.weekly_demand) if self.weekly_demand else 0,
            "weeks_over_capacity": sum(1 for d in self.weekly_demand
                                       if d > self.base_capacity),
        }


# ============================================================================
# ABM SIMULATION ENGINE
# ============================================================================

class ABMSimulation:
    """
    Main simulation engine. Runs weekly ticks for N_WEEKS, matching
    survivor agents to resource pools.
    """

    def __init__(self, agents, resource_configs=None, seed=RANDOM_SEED):
        self.agents = agents
        self.resource_configs = resource_configs or default_resource_configs()
        self.resources = {rc.name: ResourcePool(rc)
                          for rc in self.resource_configs}
        self.rng = np.random.default_rng(seed)
        self.week_logs = []

    def run(self, n_weeks=N_WEEKS, verbose=True):
        """Run the full simulation."""
        if verbose:
            print(f"\n[Level 3] Running ABM: {len(self.agents)} agents, "
                  f"{len(self.resources)} resources, {n_weeks} weeks")

        for week in range(n_weeks):
            week_log = self._tick(week)
            self.week_logs.append(week_log)

            if verbose and (week + 1) % 13 == 0:  # quarterly report
                total_unmet = week_log["total_unmet"]
                total_demand = week_log["total_demand"]
                print(f"  Week {week+1:3d}: demand={total_demand}, "
                      f"unmet={total_unmet} "
                      f"({total_unmet/max(total_demand,1)*100:.1f}%)")

        if verbose:
            print(f"[Level 3] Simulation complete.")

        return self.get_results()

    def _tick(self, week):
        """Execute one week of simulation."""
        # Reset all resource pools
        for rp in self.resources.values():
            rp.reset_week(self.rng)

        # Each agent declares their need for this week
        for agent in self.agents:
            planned_state = agent.planned_sequence[week]

            # Check which resources this state requires
            for rp in self.resources.values():
                if planned_state in rp.required_by:
                    rp.request(agent)

        # Allocate resources
        week_unmet_total = 0
        week_demand_total = 0

        for rp in self.resources.values():
            served, unmet = rp.allocate(self.rng)
            week_demand_total += len(served) + len(unmet)
            week_unmet_total += len(unmet)

            # Track unmet needs per agent
            for agent in unmet:
                agent.unmet_needs.append({
                    "week": week + 1,
                    "resource": rp.name,
                    "planned_state": agent.planned_sequence[week],
                })
                agent.wait_weeks[rp.name] += 1

                # Apply spillover: if resource unavailable, modify actual sequence
                cfg = rp.config
                if cfg.spillover_state is not None:
                    agent.actual_sequence[week] = cfg.spillover_state
                # If no spillover defined, patient keeps planned state but
                # the encounter is "unmet" (logged but not fulfilled)

        return {
            "week": week + 1,
            "total_demand": week_demand_total,
            "total_unmet": week_unmet_total,
        }

    def get_results(self):
        """Compile simulation results."""
        # Resource summaries
        resource_summaries = []
        for rp in self.resources.values():
            resource_summaries.append(rp.get_summary())
        df_resources = pd.DataFrame(resource_summaries)

        # Agent-level unmet needs
        agent_records = []
        for agent in self.agents:
            agent_records.append({
                "agent_id": agent.agent_id,
                "cluster": agent.cluster,
                "acuity_score": agent.acuity_score,
                "total_unmet": len(agent.unmet_needs),
                "planned_hospital_weeks": (agent.planned_sequence == 4).sum(),
                "actual_hospital_weeks": (agent.actual_sequence == 4).sum(),
                **{f"wait_{k}": v for k, v in agent.wait_weeks.items()},
            })
        df_agents = pd.DataFrame(agent_records)

        # Weekly time series
        df_weekly = pd.DataFrame(self.week_logs)

        # Resource utilization time series
        resource_ts = {}
        for name, rp in self.resources.items():
            resource_ts[name] = {
                "demand": rp.weekly_demand,
                "served": rp.weekly_served,
                "unmet": rp.weekly_unmet,
                "utilization": rp.weekly_utilization,
            }

        return {
            "df_resources": df_resources,
            "df_agents": df_agents,
            "df_weekly": df_weekly,
            "resource_timeseries": resource_ts,
        }


# ============================================================================
# WHAT-IF SCENARIO RUNNER
# ============================================================================

def run_scenario(level2_results, scenario_name, modifications,
                 resource_configs=None, seed=RANDOM_SEED):
    """
    Run a what-if scenario by modifying population size, resource capacity,
    or triage rules.

    Parameters
    ----------
    level2_results : dict from run_level2
    scenario_name : str
    modifications : dict with optional keys:
        "population_scale" : float — multiply population by this factor
        "resource_changes" : dict of {resource_name: new_capacity}
        "extra_patients" : int — add this many additional patients

    Returns
    -------
    scenario_results : dict from ABM simulation
    """
    print(f"\n{'=' * 65}")
    print(f"WHAT-IF SCENARIO: {scenario_name}")
    print(f"{'=' * 65}")

    df_synth = level2_results["df_synthetic"].copy()
    seq_synth = level2_results["sequences_synthetic"].copy()
    cluster_trans = level2_results["cluster_transitions"]
    init_dist = level2_results["initial_distributions"]

    rng = np.random.default_rng(seed + hash(scenario_name) % 10000)

    # Apply population scaling
    pop_scale = modifications.get("population_scale", 1.0)
    extra = modifications.get("extra_patients", 0)

    if pop_scale != 1.0 or extra > 0:
        n_original = len(df_synth)
        n_target = int(n_original * pop_scale) + extra

        if n_target > n_original:
            # Resample with replacement to grow population
            extra_needed = n_target - n_original
            sample_idx = rng.choice(n_original, size=extra_needed, replace=True)
            df_extra = df_synth.iloc[sample_idx].copy()
            df_extra["patient_id"] = np.arange(n_original + 1,
                                                n_original + extra_needed + 1)
            # Regenerate sequences for extra patients
            from level2_initialization import simulate_sequences
            seq_extra = simulate_sequences(df_extra, cluster_trans, init_dist,
                                            seed=seed + 999)
            df_synth = pd.concat([df_synth, df_extra], ignore_index=True)
            seq_synth = np.vstack([seq_synth, seq_extra])
        elif n_target < n_original:
            keep = rng.choice(n_original, size=n_target, replace=False)
            df_synth = df_synth.iloc[keep].reset_index(drop=True)
            seq_synth = seq_synth[keep]

        print(f"  Population: {n_original} -> {len(df_synth)}")

    # Build agents
    agents = _build_agents(df_synth, seq_synth)

    # Apply resource changes
    configs = resource_configs or default_resource_configs()
    resource_scale = modifications.get("resource_scale", {}) #if defining scale changes
    resource_abs = modifications.get("resource_changes", {}) # if defining absolute changes
    for cfg in configs:
        if cfg.name in resource_scale:
            old_cap = cfg.weekly_capacity
            cfg.weekly_capacity = int(old_cap * resource_scale[cfg.name])
            print(f"  {cfg.name}: {old_cap} -> {cfg.weekly_capacity}")
        elif cfg.name in resource_abs:
            old_cap = cfg.weekly_capacity
            cfg.weekly_capacity = resource_abs[cfg.name]
            print(f"  {cfg.name}: {old_cap} -> {cfg.weekly_capacity}")

    # Run simulation
    sim = ABMSimulation(agents, configs, seed=seed)
    results = sim.run(verbose=True)

    return results


def _build_agents(df_synth, seq_synth, desired_count = None, seed = RANDOM_SEED):
    """Convert synthetic population to SurvivorAgents."""
    agents = []
    
    # get sim number
    # NOTE (future todo): Legacy global RNG — should be default_rng(seed) for
    # isolation, but changing would alter sampled indices. See DEV_NOTES.md §2.

    n = len(df_synth) if desired_count is None else min(desired_count, len(df_synth))
    np.random.seed(seed)
    sampled_idx = np.random.choice(len(df_synth), size = n, replace = False)
    
    for i in sampled_idx:
        row = df_synth.iloc[i]

        # Compute acuity score from available features
        acuity = 0.0
        if "charlson_approx" in row.index:
            acuity += row["charlson_approx"]
        elif "charlson_index" in row.index:
            acuity += row["charlson_index"]
        if "SUMMARY_STAGE_2000_1998_2017" in row.index:
            try:
                acuity += float(row["SUMMARY_STAGE_2000_1998_2017"])
            except (ValueError, TypeError):
                pass
        elif "stage" in row.index:
            stage_map = {"I": 0, "II": 1, "III": 2, "IV": 4}
            acuity += stage_map.get(row["stage"], 0)
        if "n_hospital_weeks" in row.index:
            acuity += row["n_hospital_weeks"] * 0.5

        agent = SurvivorAgent(
            agent_id=row.get("patient_id", i),
            cluster=row["cluster"],
            acuity_score=acuity,
            planned_sequence=seq_synth[i],
        )
        agents.append(agent)

    return agents


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_abm_results(baseline_results, scenario_results=None,
                     scenario_name="Scenario"):
    """
    Plot resource utilization and unmet demand.
    """
    import matplotlib.pyplot as plt

    # Resource utilization bar chart
    df_res = baseline_results["df_resources"]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Utilization rates
    x = range(len(df_res))
    axes[0].bar(x, df_res["mean_utilization"] * 100, alpha=0.7,
                label="Baseline", color="steelblue")
    if scenario_results:
        df_scen = scenario_results["df_resources"]
        axes[0].bar([i + 0.35 for i in x],
                     df_scen["mean_utilization"] * 100, alpha=0.7,
                     width=0.35, label=scenario_name, color="coral")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df_res["resource"], rotation=30, ha="right")
    axes[0].set_ylabel("Mean Utilization (%)")
    axes[0].set_title("Resource Utilization")
    axes[0].legend()
    axes[0].set_ylim(0, 120)
    axes[0].axhline(y=100, color="red", linestyle="--", alpha=0.3)

    # Unmet demand
    axes[1].bar(x, df_res["total_unmet"], alpha=0.7,
                label="Baseline", color="steelblue")
    if scenario_results:
        axes[1].bar([i + 0.35 for i in x],
                     df_scen["total_unmet"], alpha=0.7,
                     width=0.35, label=scenario_name, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df_res["resource"], rotation=30, ha="right")
    axes[1].set_ylabel("Total Unmet Encounters (52 weeks)")
    axes[1].set_title("Unmet Demand")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "abm_results.png"), dpi=150)
    plt.close()
    print(f"  Saved: abm_results.png")

    # Weekly demand time series for top resources
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    ts = baseline_results["resource_timeseries"]

    for name, data in ts.items():
        axes[0].plot(range(1, len(data["demand"]) + 1),
                      data["demand"], label=name, alpha=0.7)
    axes[0].set_ylabel("Weekly Demand")
    axes[0].set_title("Baseline: Weekly Resource Demand")
    axes[0].legend(fontsize=8)

    for name, data in ts.items():
        axes[1].plot(range(1, len(data["unmet"]) + 1),
                      data["unmet"], label=name, alpha=0.7)
    axes[1].set_xlabel("Week")
    axes[1].set_ylabel("Weekly Unmet")
    axes[1].set_title("Baseline: Weekly Unmet Demand")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "abm_timeseries.png"), dpi=150)
    plt.close()
    print(f"  Saved: abm_timeseries.png")


# def plot_abm_multi_scenario(baseline_results, scenarios):
#     """
#     Plot resource utilization and unmet demand for multiple scenarios vs. baseline.
#     """
#     import matplotlib.pyplot as plt

#     # Resource utilization bar chart
#     df_res = baseline_results["df_resources"]
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))

#     # Utilization rates
#     x = range(len(df_res))
#     axes[0].bar(x, df_res["mean_utilization"] * 100, alpha=0.7,
#                 label="Baseline", color="steelblue")
#     if scenario_results:
#         df_scen = scenario_results["df_resources"]
#         axes[0].bar([i + 0.35 for i in x],
#                      df_scen["mean_utilization"] * 100, alpha=0.7,
#                      width=0.35, label=scenario_name, color="coral")
#     axes[0].set_xticks(x)
#     axes[0].set_xticklabels(df_res["resource"], rotation=30, ha="right")
#     axes[0].set_ylabel("Mean Utilization (%)")
#     axes[0].set_title("Resource Utilization")
#     axes[0].legend()
#     axes[0].set_ylim(0, 120)
#     axes[0].axhline(y=100, color="red", linestyle="--", alpha=0.3)

#     # Unmet demand
#     axes[1].bar(x, df_res["total_unmet"], alpha=0.7,
#                 label="Baseline", color="steelblue")
#     if scenario_results:
#         axes[1].bar([i + 0.35 for i in x],
#                      df_scen["total_unmet"], alpha=0.7,
#                      width=0.35, label=scenario_name, color="coral")
#     axes[1].set_xticks(x)
#     axes[1].set_xticklabels(df_res["resource"], rotation=30, ha="right")
#     axes[1].set_ylabel("Total Unmet Encounters (52 weeks)")
#     axes[1].set_title("Unmet Demand")
#     axes[1].legend()

#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, "abm_results.png"), dpi=150)
#     plt.close()
#     print(f"  Saved: abm_results.png")

#     # Weekly demand time series for top resources
#     fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
#     ts = baseline_results["resource_timeseries"]

#     for name, data in ts.items():
#         axes[0].plot(range(1, len(data["demand"]) + 1),
#                       data["demand"], label=name, alpha=0.7)
#     axes[0].set_ylabel("Weekly Demand")
#     axes[0].set_title("Baseline: Weekly Resource Demand")
#     axes[0].legend(fontsize=8)

#     for name, data in ts.items():
#         axes[1].plot(range(1, len(data["unmet"]) + 1),
#                       data["unmet"], label=name, alpha=0.7)
#     axes[1].set_xlabel("Week")
#     axes[1].set_ylabel("Weekly Unmet")
#     axes[1].set_title("Baseline: Weekly Unmet Demand")
#     axes[1].legend(fontsize=8)

#     plt.tight_layout()
#     plt.savefig(os.path.join(OUTPUT_DIR, "abm_timeseries.png"), dpi=150)
#     plt.close()
#     print(f"  Saved: abm_timeseries.png")
    

# ============================================================================
# RUN LEVEL 3
# ============================================================================

def run_level3(level2_results, resource_configs=None, scenarios=None,
               seed=RANDOM_SEED):
    """
    Execute Level 3: baseline ABM + optional what-if scenarios.

    Parameters
    ----------
    level2_results : dict from run_level2
    resource_configs : list of ResourceConfig (default: auto)
    scenarios : dict of {name: modifications} for what-if comparisons

    Returns
    -------
    dict with baseline results and scenario comparisons
    """
    print(f"\n{'=' * 65}")
    print("LEVEL 3: AGENT-BASED MODELING")
    print(f"{'=' * 65}")

    df_synth = level2_results["df_synthetic"]
    seq_synth = level2_results["sequences_synthetic"]

    # Build agents
    agents = _build_agents(df_synth, seq_synth)

    # Baseline simulation
    print(f"\n--- Baseline Simulation ---")
    configs = resource_configs or default_resource_configs()
    sim = ABMSimulation(agents, configs, seed=seed)
    baseline = sim.run(verbose=True)

    # Print resource summary
    print(f"\n  Resource Summary:")
    print(baseline["df_resources"].to_string(index=False))

    # Agent summary by cluster
    df_ag = baseline["df_agents"]
    print(f"\n  Unmet Needs by Cluster:")
    for c in sorted(df_ag["cluster"].unique()):
        sub = df_ag[df_ag["cluster"] == c]
        print(f"    Cluster {c}: mean_unmet={sub['total_unmet'].mean():.1f}, "
              f"max_unmet={sub['total_unmet'].max()}, n={len(sub)}")

    # Save baseline
    #baseline["df_resources"].to_csv(
        #os.path.join(OUTPUT_DIR, "abm_resource_summary.csv"), index=False)
    #baseline["df_agents"].to_csv(
        #os.path.join(OUTPUT_DIR, "abm_agent_summary.csv"), index=False)

    # Run what-if scenarios
    scenario_results = {}
    if scenarios:
        for name, mods in scenarios.items():
            result = run_scenario(level2_results, name, mods,
                                   seed=seed)
            scenario_results[name] = result

            print(f"\n  {name} Resource Summary:")
            print(result["df_resources"].to_string(index=False))

    # Plot
    first_scenario = None
    first_name = ""
    if scenario_results:
        first_name = list(scenario_results.keys())[0]
        first_scenario = scenario_results[first_name]
    plot_abm_results(baseline, first_scenario, first_name)

    # Comparison table if scenarios exist
    if scenario_results:
        print(f"\n{'=' * 65}")
        print("SCENARIO COMPARISON")
        print(f"{'=' * 65}")
        print(f"\n  {'Metric':<30} {'Baseline':>12}", end="")
        for name in scenario_results:
            print(f" {name:>15}", end="")
        print()
        print(f"  {'-' * (45 + 15 * len(scenario_results))}")

        for _, row in baseline["df_resources"].iterrows():
            res_name = row["resource"]
            print(f"  {res_name + ' unmet':<30} {row['total_unmet']:>12}", end="")
            for name, sr in scenario_results.items():
                scen_row = sr["df_resources"][
                    sr["df_resources"]["resource"] == res_name]
                if len(scen_row) > 0:
                    print(f" {scen_row.iloc[0]['total_unmet']:>15}", end="")
                else:
                    print(f" {'N/A':>15}", end="")
            print()

    print(f"\n{'=' * 65}")
    print("LEVEL 3 COMPLETE")
    print(f"{'=' * 65}")

    return {
        "baseline": baseline,
        "scenarios": scenario_results,
    }
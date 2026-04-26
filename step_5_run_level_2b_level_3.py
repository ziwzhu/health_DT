# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:50:58 2026

@author: ziz4012

# Simulate sequences and run - for level 2B and level 3
"""
import numpy as np
import pandas as pd
import os, sys

sys.path.insert(0, os.getcwd())
from config import N_STATES, N_WEEKS, STATE_LABELS, RANDOM_SEED, OUTPUT_DIR
from level2_initialization import simulate_sequences
from level3_abm import ABMSimulation, _build_agents, default_resource_configs, plot_abm_results#, plot_abm_multi_scenario

# Load transitions + initdist from .npz

# NOTE (future todo): Cluster index remapping — npz keys are 1-indexed
# (from R: cluster_1, cluster_2, cluster_3) while synthetic data uses
# 0-indexed clusters (from sklearn classifier). The `- 1` aligns them.
# Validated by Frobenius norms < 0.04. See DEV_NOTES.md §1.

DATA = OUTPUT_DIR

ct_raw = np.load(os.path.join(DATA, "cluster_transitions.npz"))
cluster_trans = {int(k.split("_")[1].split(".")[0]) - 1: ct_raw[k] for k in ct_raw.files}
#print(cluster_trans)

id_raw = np.load(os.path.join(DATA, "initial_distributions.npz"))
init_dist = {int(k.split("_")[1].split(".")[0]) - 1: id_raw[k] for k in id_raw.files}
#print(init_dist)

# Scenario CSVs - simulated files
FILES = {
    "orig_age": "synthpop_patients_orig_age_clustered.csv",
    "Age_5yr_Younger": "synthpop_patients_ageshift_minus_5_clustered.csv",
    "Age_5yr_Older": "synthpop_patients_ageshift_plus_5_clustered.csv",
    "1.5x_Stage_1":  "synthpop_patients_1.5x_stage1_clustered.csv",
    "2x_Stage_7": "synthpop_patients_2x_stage7_clustered.csv",
    }

# Level 2B - Simulate sequence for each synthesized sample based on cluster transition/initialization matrix
# repeat for 50 repetitions
#print(RANDOM_SEED)
N = 50
seeds = np.random.default_rng(RANDOM_SEED).integers(0, 100000, size = N)
pops = {}
for s in seeds:
    for i, (key, fname) in enumerate(FILES.items()):
        df = pd.read_csv(os.path.join(DATA, fname))
        seed = int(s) + i
        seq = simulate_sequences(df, cluster_trans, init_dist, seed = seed)
        np.save(os.path.join(DATA, f"sequences_{key}_seed{s:05d}.npy"), seq)
        pops[key] = (df, seq)
    print(f"Seed {s:05d} done.")
    
# 51 to 100
seeds = np.random.default_rng(RANDOM_SEED).integers(0, 100000, size = 100)[50:]
pops = {}
for s in seeds:
    for i, (key, fname) in enumerate(FILES.items()):
        df = pd.read_csv(os.path.join(DATA, fname))
        seed = int(s) + i
        seq = simulate_sequences(df, cluster_trans, init_dist, seed = seed)
        np.save(os.path.join(DATA, f"sequences_{key}_seed{s:05d}.npy"), seq)
        pops[key] = (df, seq)
    print(f"Seed {s:05d} done.")

# Level 3: ABM for each
def run_abm(df, seq, seed, desired_count = None, drop_threshold = 8, verbose = True):
    agents = _build_agents(df, seq, desired_count, seed = seed)
    sim = ABMSimulation(agents, default_resource_configs(drop_threshold), seed = seed)
    return sim.run(verbose=verbose)

baseline = run_abm(*pops["orig_age"], RANDOM_SEED)

scenarios = {}
for key in FILES:
    if key == "orig_age":
        continue
    scenarios[key] = run_abm(*pops[key], RANDOM_SEED + hash(key) % 10000)

# print out some utilization matrix
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
print(f"Utilization Matrix for Baseline")
print(baseline["df_resources"])
for key in FILES:
    if key == "orig_age":
        continue
    print(f"Utilization Matrix for {key}")
    print(scenarios[key]["df_resources"])
# plot
#plot_abm_multi_scenario(baseline, scenarios)

##### Update: level 3 summarized results from 100 simulations
# loop through all files for simulation
import glob

records = []

for key, fname in FILES.items():
    files = sorted(glob.glob(os.path.join(DATA, f"sequences_{key}_seed*.npy")))
    df_synth = pd.read_csv(os.path.join(DATA, fname))
    print(f"{key}: {len(files)} sequence runs found")
    
    for f in files:
        seed_str = f.split("seed")[-1].replace(".npy", "")
        seq = np.load(f)
        
        for threshold in [4, 8, 12]:
            result = run_abm(df_synth, seq, RANDOM_SEED, 
                             desired_count = None, drop_threshold = threshold, 
                             verbose = False)
            dfr = result["df_resources"].copy()
            dfr["key"] = key
            dfr["lvl2_seq_seed"] = seed_str
            dfr["drop_threshold"] = threshold
            records.append(dfr)
            
df_runs = pd.concat(records, ignore_index = True)
df_runs.to_csv(os.path.join(DATA, "abm_runs_all_results.csv"), index = False)

# summary
summary = (df_runs.groupby(["key", "drop_threshold", "resource"])
           .agg (n_runs = ("lvl2_seq_seed", "count"),
                 **{col + "_mean": (col, "mean")
                    for col in df_runs.columns
                    if col not in ["key", "drop_threshold", "resource", "lvl2_seq_seed"]},
                 **{col + "_std": (col, "std")
                    for col in df_runs.columns
                    if col not in ["key", "drop_threshold", "resource", "lvl2_seq_seed"]},
                 )
           ).reset_index()
print(summary)
summary.to_csv(os.path.join(DATA, "abm_runs_results_summary.csv"), index = False)

            
    


"""
module5_transitions.py — Extract per-cluster transition matrices.

These are the direct inputs to Level 2 (Markov simulation).

Inputs:  sequences, labels
Outputs: dict of cluster_id → N_STATES×N_STATES transition probability matrices
"""

import numpy as np
import pandas as pd
import os

from config import N_STATES, STATE_LABELS, OUTPUT_DIR


def extract_cluster_transitions(sequences, labels):
    """
    Compute aggregate state transition matrix per cluster.

    Parameters
    ----------
    sequences : np.ndarray, shape (n_patients, n_weeks)
    labels : np.ndarray, shape (n_patients,)

    Returns
    -------
    cluster_trans : dict of {cluster_id: np.ndarray (N_STATES × N_STATES)}
    """
    cluster_trans = {}

    for c in sorted(np.unique(labels)):
        mask = labels == c
        cluster_seqs = sequences[mask]

        trans_mat = np.zeros((N_STATES, N_STATES))
        for seq in cluster_seqs:
            for t in range(len(seq) - 1):
                trans_mat[seq[t], seq[t + 1]] += 1

        row_sums = trans_mat.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cluster_trans[c] = trans_mat / row_sums

    _print_transitions(cluster_trans)
    _save_transitions(cluster_trans)

    return cluster_trans


def extract_initial_state_distribution(sequences, labels):
    """
    Compute the distribution of starting states per cluster.
    Needed for Level 2 to initialize new synthetic patients.

    Returns
    -------
    init_dist : dict of {cluster_id: np.ndarray (N_STATES,)}
    """
    init_dist = {}
    for c in sorted(np.unique(labels)):
        mask = labels == c
        first_states = sequences[mask, 0]
        dist = np.zeros(N_STATES)
        for s in range(N_STATES):
            dist[s] = (first_states == s).sum() / len(first_states)
        init_dist[c] = dist
    return init_dist


def _print_transitions(cluster_trans):
    """Pretty-print transition matrices."""
    print(f"\n[Module 5] Cluster Transition Matrices:")
    short_labels = [STATE_LABELS.get(s, str(s))[:10] for s in range(N_STATES)]

    for c, mat in cluster_trans.items():
        print(f"\n  Cluster {c}:")
        header = "  " + " " * 12 + "".join([f"{l:>12}" for l in short_labels])
        print(header)
        for s_from in range(N_STATES):
            row_str = f"  {short_labels[s_from]:<12}"
            row_str += "".join([f"{mat[s_from, s_to]:>12.3f}"
                                for s_to in range(N_STATES)])
            print(row_str)


def _save_transitions(cluster_trans):
    """Save transition matrices as CSV files (one per cluster)."""
    for c, mat in cluster_trans.items():
        df = pd.DataFrame(
            mat,
            index=[STATE_LABELS.get(s, str(s)) for s in range(N_STATES)],
            columns=[STATE_LABELS.get(s, str(s)) for s in range(N_STATES)],
        )
        path = os.path.join(OUTPUT_DIR, f"transition_cluster_{c}.csv")
        df.to_csv(path)

    # Also save as single numpy file
    np.savez(os.path.join(OUTPUT_DIR, "cluster_transitions.npz"),
             **{f"cluster_{c}": mat for c, mat in cluster_trans.items()})

    print(f"  Saved: transition_cluster_*.csv and cluster_transitions.npz")
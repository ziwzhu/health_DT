"""Tests for the things a refactor breaks: conventions, shapes, determinism.

These run on generated fixture data — no SEER-Medicare access needed. They are
not tests of scientific conclusions.

    pytest
"""

import numpy as np
import pandas as pd
import pytest
from health_dt.demo_data import make_cohort

from health_dt import config
from health_dt.level1.preprocess import binarise_comorbidities, encode_targets
from health_dt.level1.sequences import derive_features, encode_sequences, normalise_labels
from health_dt.level1.transitions import (
    extract_initial_distributions,
    extract_transitions,
    load_transitions,
    save_transitions,
    transition_matrix,
)
from health_dt.level2.simulate import replication_seeds, simulate_sequences
from health_dt.level3.abm import run_abm
from health_dt.level3.agents import build_agents, compute_acuity


@pytest.fixture(scope="module")
def cohort():
    """A small fixture cohort: (raw frame, 0-indexed labels, encoded sequences)."""
    raw, labels = make_cohort(200, seed=7)
    return raw, normalise_labels(labels["cluster"].to_numpy()), encode_sequences(raw)


# --- Data conventions --------------------------------------------------------


def test_state_encoding_covers_all_states():
    assert set(config.PATTERN_TO_STATE.values()) | {config.RARE_STATE} == set(range(config.N_STATES))


def test_encoded_sequences_are_valid(cohort):
    _, _, sequences = cohort
    assert sequences.shape[1] == config.N_WEEKS
    assert sequences.min() >= 0 and sequences.max() < config.N_STATES


def test_unknown_pattern_becomes_rare():
    frame = pd.DataFrame({f"event{i}": ["999999"] for i in range(1, config.N_WEEKS + 1)})
    assert (encode_sequences(frame) == config.RARE_STATE).all()


def test_comorbidities_use_code_3_not_nonzero():
    """Using > 0 would count unknown-status patients as having the condition."""
    frame = pd.DataFrame({col: [0, 1, 2, 3] for col in config.COMORBIDITY_COLUMNS})
    out = binarise_comorbidities(frame)
    assert out[config.COMORBIDITY_COLUMNS[0]].tolist() == [0, 0, 0, 1]
    assert out["comorbid_count"].tolist() == [0, 0, 0, len(config.COMORBIDITY_COLUMNS)]


def test_labels_normalise_to_zero_indexed():
    """R emits 1-indexed labels; everything in Python is 0-indexed."""
    assert normalise_labels(np.array([1, 2, 3, 1])).tolist() == [0, 1, 2, 0]
    frame = pd.DataFrame({config.LABEL_COLUMN: [3, 1, 2, 1]})
    out, maps = encode_targets(frame)
    assert maps["target"] == {1: 0, 2: 1, 3: 2}


def test_sequence_features_match_hospital_states(cohort):
    _, _, sequences = cohort
    features = derive_features(sequences)
    expected = np.isin(sequences, config.HOSPITAL_STATES).sum(axis=1)
    assert (features["n_hospital_weeks"].to_numpy() == expected).all()


# --- Transition matrices -----------------------------------------------------


def test_transition_rows_are_stochastic(cohort):
    _, labels, sequences = cohort
    for matrix in extract_transitions(sequences, labels).values():
        row_sums = matrix.sum(axis=1)
        assert np.all(np.isclose(row_sums, 1.0) | np.isclose(row_sums, 0.0))


def test_transition_matrix_matches_naive_loop(cohort):
    """The vectorised counter must equal the original nested-loop version."""
    _, _, sequences = cohort
    subset = sequences[:40]
    naive = np.zeros((config.N_STATES, config.N_STATES))
    for row in subset:
        for t in range(len(row) - 1):
            naive[row[t], row[t + 1]] += 1
    rows = naive.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1
    np.testing.assert_allclose(transition_matrix(subset), naive / rows)


def test_transition_roundtrip_and_cluster_id_check(cohort, tmp_path):
    """A cluster-ID mismatch must raise, not silently simulate the wrong phenotype."""
    _, labels, sequences = cohort
    matrices = extract_transitions(sequences, labels)
    path = tmp_path / "t.npz"
    save_transitions(matrices, path, write_csv=False)

    loaded = load_transitions(path, expected_clusters=sorted(set(labels.tolist())))
    assert set(loaded) == set(matrices)

    with pytest.raises(ValueError, match="cluster ID mismatch"):
        load_transitions(path, expected_clusters=[97, 98])


# --- Simulation --------------------------------------------------------------


def test_simulation_is_deterministic(cohort):
    _, labels, sequences = cohort
    matrices = extract_transitions(sequences, labels)
    initial = extract_initial_distributions(sequences, labels)
    population = pd.DataFrame({"cluster": labels})

    first = simulate_sequences(population, matrices, initial, seed=1)
    second = simulate_sequences(population, matrices, initial, seed=1)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (len(population), config.N_WEEKS)


def test_simulation_rejects_unknown_cluster(cohort):
    _, labels, sequences = cohort
    matrices = extract_transitions(sequences, labels)
    initial = extract_initial_distributions(sequences, labels)
    with pytest.raises(KeyError, match="no transition matrix"):
        simulate_sequences(pd.DataFrame({"cluster": [99]}), matrices, initial)


def test_replication_seeds_extend_rather_than_replace():
    """Adding replications later must not invalidate earlier ones."""
    np.testing.assert_array_equal(replication_seeds(5), replication_seeds(20)[:5])


# --- ABM ---------------------------------------------------------------------


def test_acuity_uses_first_available_column():
    assert compute_acuity({"charlson_approx": 4.0}) == 4.0
    assert compute_acuity({"charlson_approx": 4.0, "charlson_index": 9.0}) == 4.0
    assert compute_acuity({"n_hospital_weeks": 6}) == 3.0
    assert compute_acuity({}) == 0.0


def test_abm_conserves_requests(cohort):
    """Served plus unmet must equal total demand for every resource."""
    _, labels, sequences = cohort
    population = pd.DataFrame({"cluster": labels, "patient_id": np.arange(len(labels))})
    result = run_abm(build_agents(population, sequences), drop_threshold=8)

    for _, row in result["resources"].iterrows():
        assert row["total_served"] + row["total_unmet"] == row["total_demand"]
    assert len(result["weekly"]) == config.N_WEEKS


def test_abm_is_deterministic(cohort):
    _, labels, sequences = cohort
    population = pd.DataFrame({"cluster": labels, "patient_id": np.arange(len(labels))})

    def once():
        return run_abm(build_agents(population, sequences), drop_threshold=8)["resources"]

    pd.testing.assert_frame_equal(once(), once())


def test_unlimited_capacity_leaves_nothing_unmet(cohort):
    _, labels, sequences = cohort
    population = pd.DataFrame({"cluster": labels, "patient_id": np.arange(len(labels))})
    generous = [
        config.Resource(r.name, 10**6, r.required_by_states, r.priority) for r in config.RESOURCES
    ]
    result = run_abm(build_agents(population, sequences), resources=generous)
    assert result["resources"]["total_unmet"].sum() == 0

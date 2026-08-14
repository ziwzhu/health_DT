"""Level 1 — phenotype discovery, classification, and transition extraction.

The clustering itself runs externally (Longest Common Subsequence distance in R,
k=3 merged from k=4); this level consumes those labels.

    preprocess   regroup categoricals, build the modelling matrices
    sequences    decode weekly event strings into state codes
    classify     train and select the phenotype classifier
    transitions  per-phenotype Markov matrices (the handoff to Level 2)
    profile      Tables 1 and 2 — characteristics and outcomes by phenotype

Outputs consumed by Level 2: ``best_model.pkl``, ``cluster_transitions.npz``,
``initial_distributions.npz``, ``processed_synth_raw_data.csv``.
"""

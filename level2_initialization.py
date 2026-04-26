"""
Level 2: Digital Twin Initialization
=====================================
Generate a synthetic survivor population that matches the real population's
feature distributions and phenotype-specific utilization patterns.

Two sub-steps:
  2A. Population Synthesis — sample synthetic patients from real feature
      distributions (demographics, comorbidities, cancer features), then
      assign each a phenotype cluster based on a trained classifier.
  2B. Utilization Simulation — for each synthetic patient, generate a
      52-week state sequence using the cluster-specific Markov transition
      matrices extracted in Level 1 (Module 5).

Inputs:  Level 1 results (cluster transitions, initial distributions,
         patient data with cluster labels)
Outputs: Synthetic population DataFrame + simulated sequences
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import N_STATES, N_WEEKS, STATE_LABELS, RANDOM_SEED, OUTPUT_DIR


# ============================================================================
# 2A. POPULATION SYNTHESIS
# ============================================================================

def learn_feature_distributions(df_patients):
    """
    Extract the marginal distributions of key features from real patients.
    Returns a dict of {column: distribution_params} for sampling.

    For numeric columns: store (mean, std, min, max).
    For categorical columns: store value counts as probabilities.
    """
    distributions = {}

    # Skip internal columns
    skip = {"patient_id", "cluster", "latent_profile"}

    for col in df_patients.columns:
        if col in skip:
            continue
        if pd.api.types.is_numeric_dtype(df_patients[col]):
            distributions[col] = {
                "type": "numeric",
                "mean": df_patients[col].mean(),
                "std": df_patients[col].std(),
                "min": df_patients[col].min(),
                "max": df_patients[col].max(),
                "is_integer": (df_patients[col] == df_patients[col].round()).all(),
            }
        else:
            vc = df_patients[col].value_counts(normalize=True)
            distributions[col] = {
                "type": "categorical",
                "values": vc.index.tolist(),
                "probs": vc.values.tolist(),
            }

    print(f"[Level 2A] Learned distributions for {len(distributions)} features")
    return distributions


def synthesize_population(distributions, n_synthetic, seed=RANDOM_SEED):
    """
    Generate n_synthetic patients by sampling from learned distributions.

    Returns
    -------
    df_synthetic : pd.DataFrame with same columns as real data
    """
    rng = np.random.default_rng(seed)
    data = {}

    data["patient_id"] = np.arange(1, n_synthetic + 1)

    for col, params in distributions.items():
        if params["type"] == "numeric":
            vals = rng.normal(params["mean"], max(params["std"], 0.01),
                              n_synthetic)
            vals = np.clip(vals, params["min"], params["max"])
            if params["is_integer"]:
                vals = np.round(vals).astype(int)
            data[col] = vals
        else:
            data[col] = rng.choice(params["values"], size=n_synthetic,
                                    p=params["probs"])

    df = pd.DataFrame(data)
    print(f"[Level 2A] Synthesized {n_synthetic} patients")
    return df


def train_phenotype_classifier(df_patients, feature_cols=None):
    """
    Train a classifier to predict phenotype cluster from patient features.
    This lets us assign clusters to synthetic patients.

    Parameters
    ----------
    df_patients : must have a 'cluster' column
    feature_cols : list of str, columns to use as predictors.
                   If None, auto-detected (all numeric + encoded categorical).

    Returns
    -------
    clf : fitted classifier
    feature_cols : list of columns used
    accuracy : cross-val accuracy
    """
    df = df_patients.copy()

    if feature_cols is None:
        # Auto-detect: use numeric columns + one-hot encode categorical
        skip = {"patient_id", "cluster", "latent_profile"}
        feature_cols = []
        for col in df.columns:
            if col in skip:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                feature_cols.append(col)

        # One-hot encode categorical columns
        cat_cols = [c for c in df.columns
                    if c not in skip and not pd.api.types.is_numeric_dtype(df[c])]
        if cat_cols:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=False)
            new_cols = [c for c in df.columns
                        if c not in skip and c not in feature_cols and c != "cluster"
                        and pd.api.types.is_numeric_dtype(df[c])]
            feature_cols.extend(new_cols)

    X = df[feature_cols].values.astype(float)
    y = df["cluster"].values

    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED,
                                  max_depth=10, n_jobs=-1)

    # Cross-validate
    scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
    print(f"[Level 2A] Phenotype classifier CV accuracy: "
          f"{scores.mean():.3f} ± {scores.std():.3f}")

    # Fit on full data
    clf.fit(X, y)

    return clf, feature_cols, scores.mean()


def assign_phenotypes(df_synthetic, clf, feature_cols, df_real_for_encoding):
    """
    Predict phenotype cluster for each synthetic patient.

    Handles categorical encoding to match the training encoding.
    """
    df = df_synthetic.copy()

    # One-hot encode using same columns as training
    cat_cols = [c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c])
                and c != "patient_id"]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

    # Align columns with training set
    missing_cols = list(set(feature_cols) - set(df.columns))
    if missing_cols:
        df = pd.concat([df, pd.DataFrame(0, index=df.index, columns=missing_cols)], axis=1)
    extra_cols = set(df.columns) - set(feature_cols) - {"patient_id", "cluster"}

    X = df[feature_cols].values.astype(float)
    X = np.nan_to_num(X, nan=0.0)

    clusters = clf.predict(X)
    cluster_probs = clf.predict_proba(X)

    df_synthetic = df_synthetic.copy()
    df_synthetic["cluster"] = clusters

    # Distribution of assigned clusters
    vc = pd.Series(clusters).value_counts(normalize=True).sort_index()
    print(f"[Level 2A] Assigned phenotypes: {dict(vc.round(3))}")

    return df_synthetic


# ============================================================================
# 2B. UTILIZATION SIMULATION (Markov Chain)
# ============================================================================

def simulate_sequences(df_synthetic, cluster_transitions, initial_distributions,
                       seed=RANDOM_SEED):
    """
    For each synthetic patient, generate a 52-week utilization sequence
    using the Markov transition matrix for their assigned cluster.

    Parameters
    ----------
    df_synthetic : must have 'cluster' column
    cluster_transitions : dict {cluster_id: np.ndarray (N_STATES × N_STATES)}
    initial_distributions : dict {cluster_id: np.ndarray (N_STATES,)}

    Returns
    -------
    sequences : np.ndarray (n_patients × N_WEEKS)
    """
    rng = np.random.default_rng(seed)
    n = len(df_synthetic)
    sequences = np.zeros((n, N_WEEKS), dtype=int)

    clusters = df_synthetic["cluster"].values

    for i in range(n):
        c = clusters[i]
        trans = cluster_transitions[c]
        init = initial_distributions[c]

        # Starting state
        state = rng.choice(N_STATES, p=init)
        sequences[i, 0] = state

        for w in range(1, N_WEEKS):
            state = rng.choice(N_STATES, p=trans[state])
            sequences[i, w] = state

    print(f"[Level 2B] Simulated {n} sequences of {N_WEEKS} weeks")

    return sequences


# ============================================================================
# VALIDATION: Compare real vs synthetic distributions
# ============================================================================

def validate_twin(df_real, seq_real, df_synth, seq_synth):
    """
    Compare real and synthetic populations on key metrics.
    """
    print(f"\n{'=' * 65}")
    print("DIGITAL TWIN VALIDATION: Real vs Synthetic")
    print(f"{'=' * 65}")
    print(f"  Population sizes: Real={len(df_real)}, Synthetic={len(df_synth)}")

    # Feature distribution comparison
    skip = {"patient_id", "cluster", "latent_profile"}
    numeric_cols = [c for c in df_real.columns
                    if c not in skip and pd.api.types.is_numeric_dtype(df_real[c])]

    print(f"\n  Feature Distributions (mean ± std):")
    print(f"  {'Feature':<30} {'Real':>20} {'Synthetic':>20}")
    print(f"  {'-'*70}")
    for col in numeric_cols[:10]:  # top 10
        r_mean, r_std = df_real[col].mean(), df_real[col].std()
        s_mean, s_std = df_synth[col].mean(), df_synth[col].std()
        print(f"  {col:<30} {r_mean:>8.2f} ± {r_std:<8.2f} "
              f"{s_mean:>8.2f} ± {s_std:<8.2f}")
    
    # Categorical distribution comparison
    cat_cols = [c for c in df_real.columns
                    if c not in skip and c in df_synth.columns
                    and not pd.api.types.is_numeric_dtype(df_real[c])]
    if cat_cols:
        print(f"\n Categorical Distributions:")
        for col in cat_cols[:5]:
            print(f"\n {col}:")
            r_vc = df_real[col].value_counts(normalize=True).head(5)
            s_vc = df_synth[col].value_counts(normalize=True).head(5)
            for val in r_vc.index:
                r_p = r_vc.get(val, 0)
                s_p = s_vc.get(val, 0) if val in s_vc.index else 0
                print(f"{val}: Real = {r_p:.1%}, Synth = {s_p:.1%}")

    # State frequency comparison
    print(f"\n  State Frequencies:")
    print(f"  {'State':<30} {'Real':>10} {'Synthetic':>10}")
    print(f"  {'-'*50}")
    for s in range(N_STATES):
        r_freq = (seq_real == s).mean() * 100
        s_freq = (seq_synth == s).mean() * 100
        label = STATE_LABELS.get(s, f"State {s}")
        print(f"  {label:<30} {r_freq:>8.1f}% {s_freq:>8.1f}%")
    
    # Transition matrix fidelity per cluster
    print(f"\n Transition Matrix Fidelity (Frobenius norm, lower=better:")
    for c in sorted(set(df_real["cluster"].unique()) & set(df_synth["cluster"].unique())):
        def _tmat(seqs):
            mat = np.zeros((N_STATES, N_STATES))
            for seq in seqs:
                for t in range(len(seq)-1):
                    mat[seq[t], seq[t+1]] += 1
            rs = mat.sum(axis=1, keepdims=True); rs[rs==0] = 1
            return mat/rs
        r_mat = _tmat(seq_real[df_real["cluster"].values==c])
        s_mat = _tmat(seq_synth[df_synth["cluster"].values==c])
        frob = np.linalg.norm(r_mat - s_mat, "fro")
        print(f" Cluster{c}: {frob:.4f}")

    # Cluster distribution comparison
    print(f"\n  Cluster Distribution:")
    r_clust = df_real["cluster"].value_counts(normalize=True).sort_index()
    s_clust = df_synth["cluster"].value_counts(normalize=True).sort_index()
    for c in sorted(set(r_clust.index) | set(s_clust.index)):
        r = r_clust.get(c, 0)
        s = s_clust.get(c, 0)
        print(f"  Cluster {c}: Real={r:.1%}, Synthetic={s:.1%}")
    
    # Discriminator accuracy (pMSE equivalent)
    num_cols = [c for c in df_real.columns
                    if c not in skip and c in df_synth.columns 
                    and pd.api.types.is_numeric_dtype(df_real[c])]
    
    if num_cols:
        # from sklearn.ensemble import RandomForestClassifier
        # from sklearn.model_selection import cross_val_score
        X = np.vstack([df_real[num_cols].fillna(0).values,
                       df_synth[num_cols].fillna(0).values])
        y = np.array([0]*len(df_real) + [1]*len(df_synth))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        lr = LogisticRegression(max_iter=1000, solver="lbfgs")
        lr.fit(X_scaled, y)
        p_hat = lr.predict_proba(X_scaled)[:, 1]
        
        #expected value for null
        c = len(df_synth)/len(y)
        pMSE = np.mean((p_hat-c)**2)
        
        k = X_scaled.shape[1] + 1
        N = len(y)
        
        null_pMSE = (k-1)*(1-c)**2*c/N
        pMSE_ratio = pMSE/null_pMSE
        #acc = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42), X, y, cv=5, scoring="accuracy").mean()
        #print(f"\n Discriminator Accuracy or pMSE: {acc:.3f} (ideal = 0.500)")
        print(f"\n pMSE: {pMSE:.6f}")
        print(f"\n pMSE ratio: {pMSE_ratio:.3f} (ideal ~ 1.0)")

    return True


# ============================================================================
# RUN LEVEL 2
# ============================================================================

def run_level2(level1_results, n_synthetic=None, seed=RANDOM_SEED):
    """
    Execute the full Level 2 pipeline.

    Parameters
    ----------
    level1_results : dict from run_level1
    n_synthetic : int, number of synthetic patients (default: same as real)

    Returns
    -------
    dict with synthetic population, sequences, and models
    """
    print(f"\n{'=' * 65}")
    print("LEVEL 2: DIGITAL TWIN INITIALIZATION")
    print(f"{'=' * 65}")

    df_real = level1_results["df_patients"]
    seq_real = level1_results["sequences"]
    labels = level1_results["labels"]
    cluster_trans = level1_results["cluster_transitions"]
    init_dist = level1_results["initial_distributions"]

    if "cluster" not in df_real.columns:
        df_real = df_real.copy()
        df_real["cluster"] = labels

    n_synthetic = n_synthetic or len(df_real)
    
    # Check for pre-generated synthpop output (from manual R run)
    synthpop_path = os.path.join(OUTPUT_DIR, "synthpop_patients.csv")
    if os.path.isfile(synthpop_path):
        print(f"[Level 2A] Found pre-generated R synthpop: {synthpop_path}")
        df_synth = pd.read_csv(synthpop_path)
        if "cluster" in df_synth.columns:
            df_synth["cluster"] = df_synth["cluster"].astype(int)
        print(f"[Level 2A] Loaded {len(df_synth)} patients")
        
        # 2B: Simulate utilization sequences
        print(f"\n--- 2B: Utilization Simulation ---")
        seq_synth = simulate_sequences(df_synth, cluster_trans, init_dist,seed=seed)
        # Validate
        validate_twin(df_real, seq_real, df_synth, seq_synth)
        # Save
        df_synth.to_csv(os.path.join(OUTPUT_DIR, "synthetic_patients.csv"),
                         index=False)
        np.save(os.path.join(OUTPUT_DIR, "synthetic_sequences.npy"), seq_synth)
        print(f"\n  Saved: synthetic_patients.csv, synthetic_sequences.npy")
    
        return {
            "df_synthetic": df_synth,
            "sequences_synthetic": seq_synth,
            "synthesis_method": "synthpop_R",
            "cluster_transitions": cluster_trans,
            "initial_distributions": init_dist,
        }
        
        
    # 2A: Learn distributions and synthesize population
    print(f"\n--- 2A: Population Synthesis ---")
    distributions = learn_feature_distributions(df_real)
    df_synth = synthesize_population(distributions, n_synthetic, seed=seed)

    # Train phenotype classifier and assign clusters
    clf, feat_cols, acc = train_phenotype_classifier(df_real)
    df_synth = assign_phenotypes(df_synth, clf, feat_cols, df_real)

    # 2B: Simulate utilization sequences
    print(f"\n--- 2B: Utilization Simulation ---")
    seq_synth = simulate_sequences(df_synth, cluster_trans, init_dist,
                                    seed=seed)

    # Validate
    validate_twin(df_real, seq_real, df_synth, seq_synth)

    # Save
    df_synth.to_csv(os.path.join(OUTPUT_DIR, "synthetic_patients.csv"),
                     index=False)
    np.save(os.path.join(OUTPUT_DIR, "synthetic_sequences.npy"), seq_synth)
    print(f"\n  Saved: synthetic_patients.csv, synthetic_sequences.npy")

    return {
        "df_synthetic": df_synth,
        "sequences_synthetic": seq_synth,
        "classifier": clf,
        "feature_cols": feat_cols,
        "distributions": distributions,
        "cluster_transitions": cluster_trans,
        "initial_distributions": init_dist,
    }
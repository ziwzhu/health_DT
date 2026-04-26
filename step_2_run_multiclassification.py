# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:25:47 2026

@author: ziz4012

Run models for level 1 to predict clustering based on characteristics
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score

random_state = 42
TARGET = "target" # alt: target, target_cluster_lcs_4clus, target_cluster_pca_3clus, target_cluster_pca_5clus

# 1. Load processed data
df = pd.read_csv("processed_data.csv")
target_cols = [c for c in df.columns if c.startswith("target")]
X = df.drop(columns = target_cols)
y = df[TARGET]
print(f"Features: {X.shape[1]}, Samples: {X.shape[0]}\n")

# 2. Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state=random_state, stratify=y
    )
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")
print(f"Train class distribution: \n{y_train.value_counts().sort_index()}")
print(f"Test class distribution: \n{y_test.value_counts().sort_index()}\n")

# 3. Define modules
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            multi_class = "multinomial", max_iter = 1000,
            class_weight = "balanced", random_state = random_state
            ))
        ]),
    "Random Forest": RandomForestClassifier(
            n_estimators = 200, class_weight = "balanced", random_state = random_state, n_jobs = -1
        ),
    "Gradient Boosting": GradientBoostingClassifier(
            n_estimators = 200, random_state = random_state
        ), 
    }

# 4. Train and evaluate
results = {}

for name, model in models.items():
    print("=" * 60)
    print("Model Comparison (Macro F1)")
    print("=" * 60)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    macro_f1 = f1_score(y_test, y_pred, average = "macro")
    results[name] = macro_f1
    
    print(classification_report(y_test, y_pred, digits = 3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\n Marco Fq: {macro_f1:.4f}\n")

# 5. summary - skip, just record best model;
print("=" * 60)
print("Model Comparison (Macro F1)")
print("=" * 60)
for name, score in sorted(results.items(), key = lambda x:x[1], reverse = True):
    print(f"{name:25s} {score:.4f}")
    
best_name = max(results, key = results.get)
print(f"\nBest model: {best_name}\n")

# 6. Feature importance from best model (if tree-based)
best_model = models[best_name]

if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "named_steps") and hasattr(best_model.named_steps.get("clf", None), "coef_"):
    # for logistic regression, use mean absolute coefficient across classes
    importances = np.mean(np.abs(best_model.named_steps["clf"].coef_), axis = 0)
else:
    importances = None
    
if importances is not None:
    feat_imp = pd.Series(importances, index = X.columns).sort_values(ascending=False)
    print("Top 15 features:")
    print(feat_imp.head(15).to_string())

# save best model for future use
import joblib
joblib.dump(best_model, "best_model.pkl")

# load:
# import joblib
# import pandas as pd
# model = joblib.load("best_model.pkl")
# new_data = pd.read_csv("new_processed_data.csv")
# predictions = model.predict(new_data)

    
    
    
    
    
    
    
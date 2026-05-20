from pathlib import Path

import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

INPUT = Path("data/processed/descriptors.csv")
OUT = Path("results/tables/classification_metrics.csv")

OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

X = df.drop(columns=["SMILES", "Ki (nM)", "pKi"])

thresholds = {
    "Ki < 1000 nM": 1000,
    "Ki < 100 nM": 100,
}

models = {
    "Logistic regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=5000, class_weight="balanced"))
    ]),
    "Random forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    ),
    "Gradient boosting": GradientBoostingClassifier(
        random_state=42
    )
}

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

rows = []

for threshold_name, threshold_value in thresholds.items():
    y = (df["Ki (nM)"] < threshold_value).astype(int)

    class_counts = y.value_counts().to_dict()

    print(f"\nThreshold: {threshold_name}")
    print("Class counts:", class_counts)

    for model_name, model in models.items():
        print(f"Running {model_name}...")

        scores = cross_validate(
            model,
            X,
            y,
            cv=cv,
            scoring={
                "auc": "roc_auc",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1"
            },
            n_jobs=-1
        )

        rows.append({
            "Threshold": threshold_name,
            "Model": model_name,
            "Inactive_n": class_counts.get(0, 0),
            "Active_n": class_counts.get(1, 0),
            "AUC_mean": scores["test_auc"].mean(),
            "AUC_sd": scores["test_auc"].std(),
            "Precision_mean": scores["test_precision"].mean(),
            "Precision_sd": scores["test_precision"].std(),
            "Recall_mean": scores["test_recall"].mean(),
            "Recall_sd": scores["test_recall"].std(),
            "F1_mean": scores["test_f1"].mean(),
            "F1_sd": scores["test_f1"].std(),
        })

results = pd.DataFrame(rows)
results.to_csv(OUT, index=False)

print("\nClassification results:")
print(results)

print(f"\nSaved results to: {OUT}")

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

INPUT = Path("data/processed/descriptors.csv")
OUT_FIG = Path("results/figures/roc_curves_classification.png")
OUT_TABLE = Path("results/tables/roc_auc_test_results.csv")

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

rows = []

plt.figure(figsize=(7, 6))

for threshold_name, threshold_value in thresholds.items():
    y = (df["Ki (nM)"] < threshold_value).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    for model_name, model in models.items():
        print(f"Training {model_name} for {threshold_name}...")

        model.fit(X_train, y_train)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)

        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)

        label = f"{model_name}, {threshold_name} (AUC={roc_auc:.3f})"
        plt.plot(fpr, tpr, label=label)

        rows.append({
            "Threshold": threshold_name,
            "Model": model_name,
            "AUC": roc_auc
        })

plt.plot([0, 1], [0, 1], linestyle="--")

plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves for classification models")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()

results = pd.DataFrame(rows)
results.to_csv(OUT_TABLE, index=False)

print(results)
print(f"Saved ROC figure to: {OUT_FIG}")
print(f"Saved AUC table to: {OUT_TABLE}")

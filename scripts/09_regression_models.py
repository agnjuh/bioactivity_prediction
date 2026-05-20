from pathlib import Path

import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

INPUT = Path("data/processed/descriptors.csv")
OUT = Path("results/tables/regression_metrics.csv")

OUT.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

X = df.drop(columns=["SMILES", "Ki (nM)", "pKi"])
y = df["pKi"]

cv = KFold(n_splits=10, shuffle=True, random_state=42)

models = {
    "Ridge regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),
    "Random forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "Gradient boosting": GradientBoostingRegressor(
        random_state=42
    )
}

rows = []

for name, model in models.items():
    print(f"Running {name}...")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring={
            "r2": "r2",
            "mse": "neg_mean_squared_error"
        },
        n_jobs=-1
    )

    rows.append({
        "Model": name,
        "R2_mean": scores["test_r2"].mean(),
        "R2_sd": scores["test_r2"].std(),
        "MSE_mean": -scores["test_mse"].mean(),
        "MSE_sd": scores["test_mse"].std()
    })

results = pd.DataFrame(rows)
results.to_csv(OUT, index=False)

print("\nRegression results:")
print(results)

print(f"\nSaved results to: {OUT}")

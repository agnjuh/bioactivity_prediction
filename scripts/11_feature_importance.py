from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

INPUT = Path("data/processed/descriptors.csv")
OUT_TABLE = Path("results/tables/random_forest_feature_importance.csv")
OUT_FIG = Path("results/figures/random_forest_feature_importance.png")

OUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

X = df.drop(columns=["SMILES", "Ki (nM)", "pKi"])
y = df["pKi"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest regressor...")
model.fit(X_train, y_train)

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values("Importance", ascending=False)
importance.to_csv(OUT_TABLE, index=False)

top = importance.head(20)

plt.figure(figsize=(8, 6))
plt.barh(top["Feature"][::-1], top["Importance"][::-1])
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Top 20 Random Forest feature importances")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()

print("\nTop 20 features:")
print(top)

print(f"\nSaved table to: {OUT_TABLE}")
print(f"Saved figure to: {OUT_FIG}")

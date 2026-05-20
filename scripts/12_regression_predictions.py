from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

INPUT = Path("data/processed/descriptors.csv")
OUT_FIG = Path("results/figures/random_forest_predicted_vs_observed.png")
OUT_TABLE = Path("results/tables/random_forest_test_predictions.csv")

df = pd.read_csv(INPUT)

X = df.drop(columns=["SMILES", "Ki (nM)", "pKi"])
y = df["pKi"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

print("Training Random Forest regressor...")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

pred_df = pd.DataFrame({
    "Observed_pKi": y_test.values,
    "Predicted_pKi": y_pred
})

pred_df.to_csv(OUT_TABLE, index=False)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, s=12, alpha=0.7)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel("Observed pKi")
plt.ylabel("Predicted pKi")
plt.title(f"Random Forest regression\nR² = {r2:.3f}, MSE = {mse:.3f}")
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()

print(f"R2: {r2:.3f}")
print(f"MSE: {mse:.3f}")
print(f"Saved figure to: {OUT_FIG}")
print(f"Saved predictions to: {OUT_TABLE}")

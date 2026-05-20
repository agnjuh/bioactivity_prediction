from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold

INPUT = Path("data/processed/descriptors.csv")
OUT_REG = Path("results/tables/random_forest_regression_tuning.csv")
OUT_CLS = Path("results/tables/random_forest_classification_tuning.csv")

df = pd.read_csv(INPUT)

# Subsample for tuning to reduce runtime
df_tune = df.sample(n=3000, random_state=42)

X = df_tune.drop(columns=["SMILES", "Ki (nM)", "pKi"])
y_reg = df_tune["pKi"]

param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [None, 20],
    "min_samples_leaf": [1, 3],
}

print("Running Random Forest regression tuning...")

reg_search = GridSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid=param_grid,
    scoring="r2",
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=1
)

reg_search.fit(X, y_reg)

reg_results = pd.DataFrame(reg_search.cv_results_).sort_values("rank_test_score")
reg_results[[
    "rank_test_score",
    "mean_test_score",
    "std_test_score",
    "param_n_estimators",
    "param_max_depth",
    "param_min_samples_leaf"
]].to_csv(OUT_REG, index=False)

print("Best regression parameters:", reg_search.best_params_)
print("Best regression CV R2:", reg_search.best_score_)

thresholds = {
    "Ki < 1000 nM": 1000,
    "Ki < 100 nM": 100,
}

rows = []

for threshold_name, threshold_value in thresholds.items():
    print(f"Running Random Forest classification tuning for {threshold_name}...")

    y_cls = (df_tune["Ki (nM)"] < threshold_value).astype(int)

    cls_search = GridSearchCV(
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"
        ),
        param_grid=param_grid,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs=-1,
        verbose=1
    )

    cls_search.fit(X, y_cls)

    rows.append({
        "Threshold": threshold_name,
        "Best_AUC": cls_search.best_score_,
        "Best_n_estimators": cls_search.best_params_["n_estimators"],
        "Best_max_depth": cls_search.best_params_["max_depth"],
        "Best_min_samples_leaf": cls_search.best_params_["min_samples_leaf"],
    })

    print("Best classification parameters:", cls_search.best_params_)
    print("Best classification CV AUC:", cls_search.best_score_)

pd.DataFrame(rows).to_csv(OUT_CLS, index=False)

print(f"Saved regression tuning results to: {OUT_REG}")
print(f"Saved classification tuning results to: {OUT_CLS}")

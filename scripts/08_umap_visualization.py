from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import umap

INPUT = Path("data/processed/descriptors.csv")

FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

X = df.drop(columns=["SMILES", "Ki (nM)", "pKi"])
y = df["pKi"]

X_scaled = StandardScaler().fit_transform(X)

reducer = umap.UMAP(
    n_components=2,
    n_neighbors=30,
    min_dist=0.1,
    metric="euclidean",
    random_state=42
)

coords = reducer.fit_transform(X_scaled)

umap_df = pd.DataFrame({
    "UMAP1": coords[:, 0],
    "UMAP2": coords[:, 1],
    "pKi": y
})

umap_df.to_csv(TABLE_DIR / "umap_coordinates.csv", index=False)

plt.figure(figsize=(7, 5))

scatter = plt.scatter(
    umap_df["UMAP1"],
    umap_df["UMAP2"],
    c=umap_df["pKi"],
    s=10,
    alpha=0.7
)

plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.title("UMAP of molecular descriptor space")
plt.colorbar(scatter, label="pKi")

plt.tight_layout()
plt.savefig(FIG_DIR / "umap_descriptor_space.png", dpi=300)
plt.close()

print("Saved UMAP visualization.")

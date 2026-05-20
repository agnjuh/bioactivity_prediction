from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

INPUT = Path("data/processed/descriptors.csv")
FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

X = df.drop(columns=["SMILES", "pKi"])
y = df["pKi"]

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": coords[:, 0],
    "PC2": coords[:, 1],
    "pKi": y
})

pca_df.to_csv(TABLE_DIR / "pca_coordinates.csv", index=False)

plt.figure(figsize=(7, 5))
scatter = plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    c=pca_df["pKi"],
    s=10,
    alpha=0.7
)
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)")
plt.title("PCA of molecular descriptor space")
plt.colorbar(scatter, label="pKi")
plt.tight_layout()
plt.savefig(FIG_DIR / "pca_descriptor_space.png", dpi=300)
plt.close()

explained = pd.DataFrame({
    "Component": ["PC1", "PC2"],
    "Explained variance ratio": pca.explained_variance_ratio_
})

explained.to_csv(TABLE_DIR / "pca_explained_variance.csv", index=False)

print(explained)
print("Saved PCA figure and coordinates.")

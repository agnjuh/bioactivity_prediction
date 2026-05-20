from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

INPUT = Path("data/processed/descriptors.csv")

FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

X = df.drop(columns=["SMILES", "Ki (nM)", "pKi"])
y = df["pKi"]

# Scale descriptors
X_scaled = StandardScaler().fit_transform(X)

# t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    init='pca',
    random_state=42
)

coords = tsne.fit_transform(X_scaled)

tsne_df = pd.DataFrame({
    "tSNE1": coords[:, 0],
    "tSNE2": coords[:, 1],
    "pKi": y
})

tsne_df.to_csv(TABLE_DIR / "tsne_coordinates.csv", index=False)

# Plot
plt.figure(figsize=(7, 5))

scatter = plt.scatter(
    tsne_df["tSNE1"],
    tsne_df["tSNE2"],
    c=tsne_df["pKi"],
    s=10,
    alpha=0.7
)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("t-SNE of molecular descriptor space")

plt.colorbar(scatter, label="pKi")

plt.tight_layout()
plt.savefig(FIG_DIR / "tsne_descriptor_space.png", dpi=300)

plt.close()

print("Saved t-SNE visualization.")

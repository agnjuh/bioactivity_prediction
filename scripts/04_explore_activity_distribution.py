from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

INPUT = Path("data/processed/bioactivity_filtered.csv")

FIG_DIR = Path("results/figures")
TABLE_DIR = Path("results/tables")

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

summary = df[["Ki (nM)", "pKi"]].describe()
summary.to_csv(TABLE_DIR / "activity_summary.csv")

# Raw Ki distribution
plt.figure(figsize=(7, 5))
plt.hist(df["Ki (nM)"], bins=80)
plt.xlabel("Ki (nM)")
plt.ylabel("Frequency")
plt.title("Distribution of Ki values")
plt.tight_layout()
plt.savefig(FIG_DIR / "ki_distribution_raw.png", dpi=300)
plt.close()

# pKi distribution
plt.figure(figsize=(7, 5))
plt.hist(df["pKi"], bins=50)
plt.xlabel("pKi")
plt.ylabel("Frequency")
plt.title("Distribution of pKi values")
plt.tight_layout()
plt.savefig(FIG_DIR / "pki_distribution.png", dpi=300)
plt.close()

print(summary)

print("\nSaved figures:")
print(FIG_DIR / "ki_distribution_raw.png")
print(FIG_DIR / "pki_distribution.png")

import numpy as np
import pandas as pd
from pathlib import Path

RAW_DATA = Path("data/raw/bioactivity_dataset.csv")
OUT_DATA = Path("data/processed/bioactivity_processed.csv")

OUT_DATA.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW_DATA)

print("Original shape:", df.shape)
print("Columns:", df.columns.tolist())

# Dataset column names
smiles_col = "SMILES"
ki_col = "Ki (nM)"

# Keep only required columns
df = df[[smiles_col, ki_col]].copy()

# Remove missing values
df = df.dropna()

# Convert Ki values to numeric
df[ki_col] = pd.to_numeric(df[ki_col], errors="coerce")
df = df.dropna(subset=[ki_col])

# Keep only positive Ki values
df = df[df[ki_col] > 0].copy()

# Convert Ki from nM to M
df["Ki_M"] = df[ki_col] * 1e-9

# Calculate pKi
df["pKi"] = -np.log10(df["Ki_M"])

print("\nProcessed shape:", df.shape)

print("\npKi summary:")
print(df["pKi"].describe())

# Save processed dataset
df.to_csv(OUT_DATA, index=False)

print(f"\nSaved processed dataset to: {OUT_DATA}")

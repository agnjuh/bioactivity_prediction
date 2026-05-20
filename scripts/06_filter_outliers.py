from pathlib import Path

import pandas as pd

INPUT = Path("data/processed/bioactivity_processed.csv")
OUTPUT = Path("data/processed/bioactivity_filtered.csv")

df = pd.read_csv(INPUT)

print("Original shape:", df.shape)

df = df[(df["pKi"] >= 4) & (df["pKi"] <= 11)].copy()

print("Filtered shape:", df.shape)

df.to_csv(OUTPUT, index=False)

print(f"Saved filtered dataset to: {OUTPUT}")

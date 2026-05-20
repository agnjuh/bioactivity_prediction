from pathlib import Path

import pandas as pd

INPUT = Path("data/raw/bioactivity_dataset.csv")

df = pd.read_csv(INPUT)

print("Dataset shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst rows:")
print(df.head())

print("\nMissing values:")
print(df.isna().sum())

print("\nSummary statistics:")
print(df.describe())

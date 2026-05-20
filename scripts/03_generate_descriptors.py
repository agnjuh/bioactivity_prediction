from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

INPUT = Path("data/processed/bioactivity_filtered.csv")
OUTPUT = Path("data/processed/descriptors.csv")

df = pd.read_csv(INPUT)

print("Loaded filtered dataset:", df.shape)

fp_generator = GetMorganGenerator(radius=2, fpSize=2048)

descriptor_rows = []

for _, row in df.iterrows():
    smiles = row["SMILES"]
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        continue

    rdkit_desc = {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBA": Descriptors.NumHAcceptors(mol),
        "HBD": Descriptors.NumHDonors(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
        "RingCount": Descriptors.RingCount(mol),
    }

    fp = fp_generator.GetFingerprint(mol)
    fp_bits = list(fp)

    fp_dict = {f"FP_{j}": bit for j, bit in enumerate(fp_bits)}

    combined = {
        "SMILES": smiles,
        "Ki (nM)": row["Ki (nM)"],
        "pKi": row["pKi"],
        **rdkit_desc,
        **fp_dict,
    }

    descriptor_rows.append(combined)

descriptor_df = pd.DataFrame(descriptor_rows)

print("Descriptor matrix shape:", descriptor_df.shape)

descriptor_df.to_csv(OUTPUT, index=False)

print(f"Saved descriptors to: {OUTPUT}")

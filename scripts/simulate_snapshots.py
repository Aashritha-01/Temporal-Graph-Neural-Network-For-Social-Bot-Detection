import pandas as pd
import numpy as np
import os

EDGE_FILE = "data/raw/edges.csv"
OUT_DIR = "data/snapshots"
NUM_SNAPSHOTS = 8   # weekly (8 weeks) | use 12 for months

os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(EDGE_FILE)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

chunk = len(df) // NUM_SNAPSHOTS

for i in range(1, NUM_SNAPSHOTS + 1):
    snap_df = df.iloc[: i * chunk]
    out = f"{OUT_DIR}/snapshot_{i}.csv"
    snap_df.to_csv(out, index=False)
    print(f"✔ Snapshot {i} created | edges = {len(snap_df)}")

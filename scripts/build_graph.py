import json
import pandas as pd
import os
from tqdm import tqdm

DATA_DIR = "data/raw/twibot20"
OUT_FILE = "data/raw/edges.csv"

edges = []

for split in ["train.json", "dev.json", "test.json"]:
    path = os.path.join(DATA_DIR, split)
    print("Reading:", path)

    with open(path, "r", encoding="utf-8") as f:
        users = json.load(f)

    for u in tqdm(users):
        uid = u.get("ID")

        nbr = u.get("neighbor")

        # 🔴 CRITICAL FIX
        if nbr is None:
            continue

        following = nbr.get("following", [])
        follower = nbr.get("follower", [])

        for v in following:
            edges.append([uid, v])

        for v in follower:
            edges.append([v, uid])

df = pd.DataFrame(edges, columns=["src", "dst"])
df.drop_duplicates(inplace=True)

os.makedirs("data/raw", exist_ok=True)
df.to_csv(OUT_FILE, index=False)

print("✔ Static graph built successfully")
print("Total edges:", len(df))

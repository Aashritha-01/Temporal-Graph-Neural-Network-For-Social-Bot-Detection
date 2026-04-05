import json
import os

DATA_DIR = "data/raw/twibot20"
OUT_FILE = "data/node_ids.json"

node_ids = []

for split in ["train.json", "dev.json", "test.json"]:
    with open(os.path.join(DATA_DIR, split), "r", encoding="utf-8") as f:
        data = json.load(f)
        for u in data:
            uid = str(u["ID"])
            if uid not in node_ids:
                node_ids.append(uid)

os.makedirs("data", exist_ok=True)

with open(OUT_FILE, "w") as f:
    json.dump(node_ids, f)

print(f"✔ node_ids.json created with {len(node_ids)} users")

import torch
import json
import numpy as np
import os
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score

# -------- BASE PROJECT DIRECTORY --------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------- PATHS --------
EMB_DIR = os.path.join(BASE_DIR, "data", "embeddings")
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "twibot20")
NODE_MAP_FILE = os.path.join(BASE_DIR, "data", "node_ids.json")

# -------- LOAD NODE ID MAP --------
with open(NODE_MAP_FILE, "r", encoding="utf-8") as f:
    node_ids = json.load(f)   # index → user_id

# -------- LOAD LABELS --------
labels = {}
for split in ["train.json", "dev.json", "test.json"]:
    split_path = os.path.join(DATA_DIR, split)
    if not os.path.exists(split_path):
        continue

    with open(split_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for u in data:
            labels[str(u["ID"])] = int(u["label"])

# -------- CHECK EMBEDDINGS --------
if not os.path.exists(EMB_DIR):
    raise RuntimeError(f"❌ Embedding directory not found: {EMB_DIR}")

# -------- LOAD EMBEDDINGS --------
snapshots = sorted([f for f in os.listdir(EMB_DIR) if f.endswith(".pt")])
node_scores = defaultdict(list)

for snap in snapshots:
    emb_path = os.path.join(EMB_DIR, snap)

    print("Loading:", snap)

    emb = torch.load(
        emb_path,
        map_location=torch.device('cpu')
    )

    # 🔥 Convert logits → probabilities → predictions
    probs = torch.softmax(emb, dim=1)
    preds = probs.argmax(dim=1).cpu().numpy()

    num_nodes = min(len(preds), len(node_ids))

    for i in range(num_nodes):
        user_id = str(node_ids[i])
        node_scores[user_id].append(preds[i])

# -------- FINAL PREDICTION (MAJORITY VOTING) --------
predicted = {}

for user_id, scores in node_scores.items():
    # 🔥 average prediction across snapshots
    predicted[user_id] = int(sum(scores) / len(scores) > 0.7)

# -------- BOT COUNT SUMMARY --------
total_users = len(predicted)
bot_count = sum(predicted.values())
human_count = total_users - bot_count
bot_percentage = (bot_count / total_users) * 100

print("\n📊 BOT DETECTION SUMMARY")
print(f"Total users analyzed: {total_users}")
print(f"Detected bots: {bot_count}")
print(f"Detected humans: {human_count}")
print(f"Bot percentage: {bot_percentage:.2f}%")

# -------- EVALUATION --------
y_true, y_pred = [], []

for user_id, pred in predicted.items():
    if user_id in labels:
        y_true.append(labels[user_id])
        y_pred.append(pred)

print(f"\nSamples evaluated (with ground truth): {len(y_true)}")

if len(y_true) == 0:
    raise RuntimeError("❌ No overlapping users between predictions and labels")

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n✔ Bot Detection Completed")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
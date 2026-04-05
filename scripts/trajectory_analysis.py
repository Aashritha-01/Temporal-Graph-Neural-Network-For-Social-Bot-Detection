import torch
import numpy as np
import os
import matplotlib.pyplot as plt

emb_dir = "data/embeddings"

# Only load .pt files
files = sorted([f for f in os.listdir(emb_dir) if f.endswith(".pt")])

bot_scores = []

for f in files:
    emb = torch.load(
        os.path.join(emb_dir, f),
        map_location=torch.device('cpu')
    )

    # Convert logits → probabilities
    probs = torch.softmax(emb, dim=1)

    # Take bot class probability (index 1)
    bot_prob = probs[:, 1]

    avg_bot_score = bot_prob.mean().item()
    bot_scores.append(avg_bot_score)

    print(f"{f} → Avg bot probability: {avg_bot_score:.4f}")

print("Trajectory analysis completed.")

# -------- PLOT --------
plt.figure()

plt.plot(bot_scores, marker='o')
plt.title("Bot Behavior Trajectory Over Time")
plt.xlabel("Snapshots")
plt.ylabel("Average Bot Probability")
plt.grid()

# Save plot
os.makedirs("data/plots", exist_ok=True)
plt.savefig("data/plots/trajectory_plot.png")

plt.show()
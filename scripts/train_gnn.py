import torch
import pandas as pd
import os
import json
from torch_geometric.nn import SAGEConv

# -------- MODEL --------
class TGNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = SAGEConv(3, 32)   # 🔥 3 features (in/out/total degree)
        self.conv2 = SAGEConv(32, 16)
        self.lin = torch.nn.Linear(16, 2)  # 🔥 classifier

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return self.lin(x)

# -------- PATHS --------
snap_dir = "data/snapshots"
snapshots = sorted(os.listdir(snap_dir))

# -------- LOAD GLOBAL NODE IDS --------
with open("data/node_ids.json", "r", encoding="utf-8") as f:
    node_ids = json.load(f)

node_map = {str(n): i for i, n in enumerate(node_ids)}
num_nodes = len(node_ids)

# -------- LOAD LABELS --------
label_map = {}
for file in ["data/raw/twibot20/train.json",
             "data/raw/twibot20/dev.json",
             "data/raw/twibot20/test.json"]:

    with open(file, "r", encoding="utf-8") as f:
        users = json.load(f)

    for u in users:
        label_map[str(u["ID"])] = int(u["label"])

node_labels = torch.tensor([label_map.get(str(n), 0) for n in node_ids])

# -------- MODEL SETUP --------
model = TGNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# balance bot vs human
num_bots = (node_labels == 1).sum().item()
num_humans = (node_labels == 0).sum().item()

total = num_humans + num_bots
weight = torch.tensor([
    total / (2 * num_humans),
    total / (2 * num_bots)
])

criterion = torch.nn.CrossEntropyLoss(weight=weight)

os.makedirs("data/embeddings", exist_ok=True)

# -------- TRAIN ON EACH SNAPSHOT --------
for snap in snapshots:
    print(f"\n📌 Training on {snap}")

    df = pd.read_csv(os.path.join(snap_dir, snap))

    # -------- BUILD EDGE INDEX --------
    edges = []
    for s, d in zip(df.src, df.dst):
        if str(s) in node_map and str(d) in node_map:
            edges.append([node_map[str(s)], node_map[str(d)]])

    if len(edges) == 0:
        print("⚠️ No valid edges, skipping snapshot")
        continue

    edge_index = torch.tensor(edges, dtype=torch.long).t()

#graph visualization
    import networkx as nx
    import matplotlib.pyplot as plt

# Take subset for clarity
    edge_index_small = edge_index[:, :300].cpu().numpy()

    G = nx.Graph()
    edges = list(zip(edge_index_small[0], edge_index_small[1]))
    G.add_edges_from(edges)

# Assign colors based on labels
    colors = []
    for node in G.nodes():
        if node_labels[node] == 1:
           colors.append("red")   # bot
        else:
            colors.append("blue")  # human

    plt.figure(figsize=(7,7))
    pos = nx.spring_layout(G, k=0.2)

    nx.draw(
        G,
        pos,
        node_color=colors,
        node_size=25,
        edge_color='gray',
        alpha=0.7
    )

    plt.title(f"User Interaction Graph: {snap}")

# Save image
    os.makedirs("data/plots", exist_ok=True)
    plt.savefig(f"data/plots/{snap}_ghimg.png")
    plt.close()

    print(f"📊 Graph image saved: {snap}")

    # -------- BUILD NODE FEATURES --------
    x = torch.zeros((num_nodes, 3))

    for s, d in zip(df.src, df.dst):
        if str(s) in node_map:
            x[node_map[str(s)], 0] += 1   # out-degree
        if str(d) in node_map:
            x[node_map[str(d)], 1] += 1   # in-degree

    x[:, 2] = x[:, 0] + x[:, 1]  # total degree

    print("Nodes:", num_nodes, "Edges:", edge_index.shape[1])

    # -------- TRAINING LOOP --------
    for epoch in range(70):   # 🔥 more training
        optimizer.zero_grad()

        out = model(x, edge_index)
        loss = criterion(out, node_labels)

        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            pred = out.argmax(dim=1)
            acc = (pred == node_labels).sum().item() / len(node_labels)
            print(f"Epoch {epoch}: Loss={loss.item():.4f}, Acc={acc:.4f}")

    # -------- SAVE EMBEDDINGS --------
    emb = out.detach().cpu()
    torch.save(emb, f"data/embeddings/{snap}.pt")

    print(f"✔ Finished training on {snap}")
import networkx as nx
import torch
import pandas as pd
import os

SNAP_DIR = "data/snapshots"
EMB_DIR = "outputs/embeddings"
OUT_DIR = "outputs/scores"
os.makedirs(OUT_DIR, exist_ok=True)

for snap in os.listdir(SNAP_DIR):
    df = pd.read_csv(os.path.join(SNAP_DIR, snap))
    G = nx.from_pandas_edgelist(df, "user_id", "target_id")

    degree = nx.degree_centrality(G)
    kcore = nx.core_number(G)
    emb = torch.load(os.path.join(EMB_DIR, snap + ".pt"))

    rows = []
    for i, node in enumerate(G.nodes()):
        rows.append({
            "node": node,
            "degree": degree[node],
            "kcore": kcore[node],
            "embedding_norm": emb[i].norm().item()
        })

    pd.DataFrame(rows).to_csv(
        f"{OUT_DIR}/{snap}", index=False
    )

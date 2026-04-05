import pandas as pd

df = pd.read_csv("outputs/trajectories/node_trajectories.csv")

alerts = []

for node, g in df.groupby("node"):
    if g["kcore"].diff().max() >= 3:
        alerts.append(node)

pd.DataFrame(alerts, columns=["Detected_Node"]) \
  .to_csv("outputs/detected_nodes.csv", index=False)

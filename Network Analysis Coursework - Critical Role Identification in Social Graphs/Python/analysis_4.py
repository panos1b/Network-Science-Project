import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from igraph import Graph
import leidenalg


nodes_df = pd.read_csv("Nodes.csv")
edges_df = pd.read_csv("Edges.csv")
vertex_names = nodes_df["Id"].astype(str).tolist()

g = Graph(directed=True)
g.add_vertices(vertex_names)

id_to_label = dict(zip(nodes_df["Id"].astype(str), nodes_df["Label"]))

for v in g.vs:
    v["Label"] = id_to_label[v["name"]]

edge_list = list(zip(edges_df["Source"].astype(str), edges_df["Target"].astype(str)))
g.add_edges(edge_list)

partition = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition)
memberships = partition.membership

g.vs["community"] = memberships

name_to_community = {v["name"]: v["community"] for v in g.vs}
nodes_df["community"] = nodes_df["Id"].astype(str).map(name_to_community)

nodes_df.to_csv("Nodes_updated.csv", index=False)
print("Nodes_updated.csv has been saved with community assignments.")

sns.set_theme(style="white")

# Compute a layout for the graph; here we use Fruchterman-Reingold.
layout = g.layout("fr")
coords = np.array(layout.coords)

communities = sorted(set(memberships))
colors = sns.color_palette("tab20", len(communities))
community_color = {comm: colors[i] for i, comm in enumerate(communities)}


plt.figure(figsize=(10, 8))
for i, v in enumerate(g.vs):
    # Scatter each node using its community's color.
    plt.scatter(coords[i, 0], coords[i, 1],
                color=community_color[v["community"]],
                s=100, zorder=2)
    # Instead of the Id, display the node's Label from the CSV.
    plt.text(coords[i, 0], coords[i, 1], v["Label"],
             fontsize=9, ha='center', va='center', zorder=3)

plt.title("Leiden Community Detection")
plt.axis("off")
plt.tight_layout()

plt.savefig("cluster_distribution.png", bbox_inches='tight')
print("The cluster distribution chart has been saved as cluster_distribution.png.")

plt.show()

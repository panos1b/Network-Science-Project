import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from community import community_louvain
import networkx.algorithms.community as nx_comm

nodes_df = pd.read_csv('Nodes.csv')
edges_df = pd.read_csv('Edges.csv')
G = nx.DiGraph()
G.add_nodes_from(nodes_df['Id'])
G.add_edges_from(edges_df[['Source', 'Target']].values)

# Clustering Coefficient Distribution
clustering_coeffs = nx.clustering(G)
plt.figure(figsize=(20, 10))
sns.histplot(list(clustering_coeffs.values()), bins=30, kde=False)
plt.title('Clustering Coefficient Distribution')
plt.xlabel('Clustering Coef')
plt.ylabel('Count')
plt.savefig('clustering_distribution.png')
plt.close()

# Triadic Closure
undir_G = G.to_undirected()
transitivity = nx.transitivity(undir_G)
print(f"Triadic Closure/Transitivity: {transitivity:.6f}")

# Count triangles
triangle_counts = nx.triangles(undir_G)
total_triangles = sum(triangle_counts.values()) // 3  # Each triangle is counted three times

print(f"Total number of triangles in the network: {total_triangles}")


# Bridges
bridges = list(nx.bridges(undir_G))
print(f"\nNumber of bridges: {len(bridges)}")
bridge_ids = [(u, v) for u, v in bridges]
pd.DataFrame(bridge_ids, columns=['Source', 'Target']).to_csv('bridges.csv', index=False)

# Local Bridges
local_bridges = []
for u, v in undir_G.edges():
    common_neighbors = set(undir_G.neighbors(u)) & set(undir_G.neighbors(v))
    common_neighbors.discard(u)
    common_neighbors.discard(v)
    if not common_neighbors:
        local_bridges.append((u, v))
        
print(f"Number of local bridges: {len(local_bridges)}")
pd.DataFrame(local_bridges, columns=['Source', 'Target']).to_csv('local_bridges.csv', index=False)

# Modularity
partition = community_louvain.best_partition(undir_G)
modularity = community_louvain.modularity(partition, undir_G)
print(f"\nModularity: {modularity:.4f}")


import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community.quality import modularity
import itertools
import numpy as np
from tqdm import tqdm

# Plot results
def plot_metric(column, title, ylabel):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=nodes_df['Label'], y=nodes_df[column], palette='viridis')
    plt.title(title)
    plt.xlabel('Node Label')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{column}.png')
    plt.close()

nodes_df = pd.read_csv('Nodes.csv')
edges_df = pd.read_csv('Edges.csv')
nodes_df['Id'] = nodes_df['Id'].astype(str)
edges_df['Source'] = edges_df['Source'].astype(str)
edges_df['Target'] = edges_df['Target'].astype(str)

# Ensure betweenness centrality column exists
if 'betweenesscentrality' not in nodes_df.columns:
    raise ValueError("The column 'betweenesscentrality' is missing from Nodes.csv")

# Convert to dictionary for mapping
betweenness = nodes_df.set_index('Id')['betweenesscentrality'].to_dict()

G = nx.Graph()
G.add_nodes_from(nodes_df['Id'])
G.add_edges_from(zip(edges_df['Source'], edges_df['Target']))

# Compute degrees once
degrees = dict(G.degree())
inv_degrees = {node: 1.0 / degrees[node] if degrees[node] > 0 else 0.0 for node in G.nodes()}

# Calculate Bridging Coefficient using vectorized approach
bridging_coeff = {}
for node in tqdm(G.nodes(), desc="Calculating Bridging Coefficients"):
    bridging_coeff[node] = inv_degrees[node] / sum(inv_degrees[neighbor] for neighbor in G.neighbors(node)) if degrees[node] > 0 else 0.0

bridging_centrality = {node: betweenness.get(node, 0) * bridging_coeff[node] for node in tqdm(G.nodes(), desc="Calculating Bridging Centrality")}

nodes_df['BridgingCoefficient'] = nodes_df['Id'].map(bridging_coeff)
nodes_df['BridgingCentrality'] = nodes_df['Id'].map(bridging_centrality)

nodes_df.to_csv('Nodes_Bridging_Centrality.csv', index=False)


plot_metric('BridgingCentrality', 'Bridging Centrality', 'Centrality Score')

# Girvan-Newman
def get_best_partition(graph):
    communities_gen = girvan_newman(graph)
    best_comm = max(tqdm(itertools.islice(communities_gen, 50), desc="Evaluating Partitions"), key=lambda comm: modularity(graph, comm), default=None)
    return best_comm

best_partition = get_best_partition(G)
cluster_dict = {node: i for i, cluster in enumerate(best_partition) for node in cluster} if best_partition else {}

nodes_df['GirvanNewmanCluster'] = nodes_df['Id'].map(cluster_dict).fillna(-1).astype(int)

nodes_df.to_csv('Nodes_updated_2.csv', index=False)

plot_metric('GirvanNewmanCluster', 'Girvan-Newman Clusters', 'Cluster ID')

#!/usr/bin/env python3
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

edges_df = pd.read_csv('Edges.csv')
nodes_df = pd.read_csv('Nodes.csv')
G = nx.DiGraph()
node_labels = {}  # Store labels for later use

for _, row in nodes_df.iterrows():
    node_id = row['Id']
    label = row['Label']
    description = str(row.get('Description', '')).lower()  # Ensure description is lowercase
    G.add_node(node_id, label=label, description=description)

    if pd.notna(label) and label.strip():  # Only store labels if they exist
        node_labels[node_id] = label

# Add edges
for _, row in edges_df.iterrows():
    source = row['Source']
    target = row['Target']
    G.add_edge(source, target)

wcc = list(nx.weakly_connected_components(G))
num_wcc = len(wcc)

scc = list(nx.strongly_connected_components(G))
num_scc = len(scc)
scc_more_than_one = [comp for comp in scc if len(comp) > 1]
num_scc_more_than_one = len(scc_more_than_one)

try:
    katz_centrality = nx.katz_centrality_numpy(G)
except Exception as e:
    print("Error computing Katz centrality:", e)
    katz_centrality = {}

pagerank = nx.pagerank(G)

keywords = [
    "pronouns", "just stop oil", "donate", "mental health", "communist", "nature", "vegan",
    "lefty", "queer", "free palestine", "lesbian", "trans", "bi", "union", "climate change",
    "activist", "feminist", "equal non_lefts", "blm", "lgbt", "progressive", "democratic socialist",
    "they/them", "he/him", "she/them", "he/them", "she/her", "they/he", "liberal", "climate",
    "palestinians", "gaza", "antifascism", "socialist", "leninist", "anti-imperialist", "universal healthcare",
    "chinese buddhism", "#freepalestine", "luigi did nothing wrong", "refugee", "anti-terf", "genocide"
]

# Assign left-leaning (1) or non-left-leaning (0) for labeled nodes
left_leaning_nodes = {}
for node in node_labels:
    description = G.nodes[node].get('description', '').lower()  # Case-insensitive check
    is_left = any(keyword in description for keyword in keywords)
    left_leaning_nodes[node] = int(is_left)
    G.nodes[node]['left_leaning'] = int(is_left)

# Compute counts
left_count = sum(left_leaning_nodes.values())
non_left_count = len(left_leaning_nodes) - left_count
total_labeled_nodes = len(left_leaning_nodes)

# Compute Homophily Index (only for edges between labeled nodes)
labeled_edges = [
    (u, v) for u, v in G.edges()
    if u in left_leaning_nodes and v in left_leaning_nodes
]
same_category_edges = sum(
    1 for u, v in labeled_edges
    if left_leaning_nodes[u] == left_leaning_nodes[v]
)
homophily_index = same_category_edges / len(labeled_edges)

print(f"Assortativity Coefficient: {nx.assortativity.attribute_assortativity_coefficient(G, 'left_leaning'):.4f}")
print(f"Homophily Index: {homophily_index:.4f}")
print(f"Actual amount of cross-gender edges: {len(labeled_edges) - same_category_edges:.0f}")
print(f"Expected amount of cross-gender edges: "
      f"{2*(left_count/total_labeled_nodes)*(non_left_count/total_labeled_nodes)*len(labeled_edges):.0f}")

fig, ax = plt.subplots(figsize=(12, 6))
metrics = ['Weakly Connected Components', 'Strongly Connected Components', 'SCC (size > 1)']
values = [num_wcc, num_scc, num_scc_more_than_one]

ax.bar(metrics, values, color=['skyblue', 'lightgreen', 'salmon'])
ax.set_title('Graph Connectivity Components')
ax.set_ylabel('Count')
ax.set_xticklabels(metrics, rotation=30, fontsize=10)  # Smaller labels with angle
for i, v in enumerate(values):
    ax.text(i, v + 0.05 * max(values), str(v), ha='center', fontsize=10)

plt.show()

node_data = pd.DataFrame({
    'Label': [node_labels.get(node, str(node)) for node in G.nodes() if node in node_labels],
    'Katz Centrality': [katz_centrality.get(node, 0) for node in G.nodes() if node in node_labels],
    'PageRank': [pagerank.get(node, 0) for node in G.nodes() if node in node_labels]
})

# Sort by PageRank and keep top 20
node_data = node_data.sort_values(by='PageRank', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(node_data['Label'], node_data['PageRank'], color='red', alpha=0.7, label='PageRank')
ax.barh(node_data['Label'], node_data['Katz Centrality'], color='blue', alpha=0.7, label='Katz Centrality')
ax.set_yticks(range(len(node_data['Label'])))
ax.set_yticklabels(node_data['Label'], fontsize=9)  # Smaller font size for readability
ax.set_title('Top 20 Nodes by Centrality Measures')
ax.set_xlabel('Centrality Score')
ax.legend()

plt.gca().invert_yaxis()  # Invert to show highest rank at top
plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ax.pie([left_count, non_left_count], labels=['Left-Leaning', 'Non-Left-Leaning'], autopct='%1.1f%%',
       colors=['lightblue', 'pink'], startangle=90)
ax.set_title(f'Network Political Composition (Homophily Index: {homophily_index:.2f})')

plt.show()

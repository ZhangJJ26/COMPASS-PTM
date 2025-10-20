import pandas as pd
import numpy as np
import math
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def read_crosstalk_file(file_path):
    return pd.read_csv(file_path)

def create_nodes(df):
    node1_list = [f"{row['PTM1']}@{row['Residue1'][0]}" for _, row in df.iterrows()]
    node2_list = [f"{row['PTM2']}@{row['Residue2'][0]}" for _, row in df.iterrows()]
    
    all_nodes = sorted(list(set(node1_list + node2_list)))
    return all_nodes, node1_list, node2_list

def calculate_npmi_matrix(all_nodes, node1_list, node2_list):
    n = len(all_nodes)
    npmi_matrix = np.zeros((n, n))
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    total_pairs = len(node1_list)
    all_node_occurrences = Counter(node1_list + node2_list)
    node_prob = {node: count / (2 * total_pairs) for node, count in all_node_occurrences.items()}
    pair_counts = Counter(zip(node1_list, node2_list))

    for i, node_i in enumerate(all_nodes):
        for j, node_j in enumerate(all_nodes):
            if i == j:
                npmi_matrix[i, j] = 0
                continue
            pair_count = pair_counts.get((node_i, node_j), 0) + pair_counts.get((node_j, node_i), 0)
            
            if pair_count == 0:
                npmi_matrix[i, j] = 0
            else:
                p_xy = pair_count / total_pairs
                p_x = node_prob[node_i]
                p_y = node_prob[node_j]
                
                pmi = math.log2(p_xy / (p_x * p_y))
                npmi = pmi / (-math.log2(p_xy))
                npmi_matrix[i, j] = npmi
    
    return npmi_matrix, all_nodes


def visualize_npmi_matrix(npmi_matrix, all_nodes, output_file=None):
    plt.figure(figsize=(12, 10))
    sns.heatmap(npmi_matrix, xticklabels=all_nodes, yticklabels=all_nodes, 
                cmap='coolwarm', center=0, annot=False)
    plt.title('Normalized PMI Matrix for PTM-Residue Pairs')
    plt.tight_layout()
    if output_file:
        plt.savefig(output_file, dpi=300)
    plt.show()


def main(crosstalk_file_path, output_matrix_path=None, output_viz_path=None):
    df = read_crosstalk_file(crosstalk_file_path)
    all_nodes, node1_list, node2_list = create_nodes(df)
    npmi_matrix, all_nodes = calculate_npmi_matrix(all_nodes, node1_list, node2_list)
    npmi_df = pd.DataFrame(npmi_matrix, index=all_nodes, columns=all_nodes)
    if output_matrix_path:
        npmi_df.to_csv(output_matrix_path)
    visualize_npmi_matrix(npmi_matrix, all_nodes, output_viz_path)
    
    return npmi_df

import pandas as pd
import numpy as np

# Define the PTM types we want to keep
keep_ptms = [
    "acetylation",
    "methylation",
    "phosphorylation", 
    "succinylation",
    "ubiquitination",
    "O-linked glycosylation",
    "N-linked glycosylation"
]

def transform_matrix(file):
    matrix_df = pd.read_csv(file)
    all_cols = list(matrix_df.columns)
    matrix_df = matrix_df.set_index(matrix_df.columns[0])
    all_rows = list(matrix_df.index)
    
    # Create dictionaries to store which columns/rows to keep and which to group
    keep_cols = []
    rare_cols = []
    keep_rows = []
    rare_rows = []
    
    # Categorize columns
    for col in all_cols:
        parts = col.split('@')
        if len(parts) == 2:
            ptm_type, residue = parts
            if any(keep_ptm.lower() in ptm_type.lower() for keep_ptm in keep_ptms):
                keep_cols.append(col)
            else:
                rare_cols.append(col)
    
    # Categorize rows
    for row in all_rows:
        row = str(row)
        print("row", row)
        parts = row.split('@')
        if len(parts) == 2:
            ptm_type, residue = parts
            if any(keep_ptm.lower() in ptm_type.lower() for keep_ptm in keep_ptms):
                keep_rows.append(row)
            else:
                rare_rows.append(row)
    
    # Create a new DataFrame with the columns we want to keep
    new_df = matrix_df.copy()
    
    # For columns that should be grouped as "Rare"
    if rare_cols:
        # Calculate average of rare PTM columns for each row
        rare_values = new_df[rare_cols].mean(axis=1)
        # Drop the original rare columns
        new_df = new_df.drop(columns=rare_cols)
        # Add the new "Rare" column
        new_df["Rare_PTM"] = rare_values
    
    # For rows that should be grouped as "Rare"
    if rare_rows:
        # Calculate average of rare PTM rows for each column
        rare_values_rows = new_df.loc[rare_rows].mean(axis=0)
        # Drop the original rare rows
        new_df = new_df.drop(index=rare_rows)
        # Add the new "Rare" row
        new_df.loc["Rare_PTM"] = rare_values_rows
    
    new_df.to_csv('transformed_matrix.csv')
    
    return new_df


def matrix_to_embedding(matrix):
    matrix = pd.read_csv(matrix)
    matrix = matrix.set_index(matrix.columns[0])
    matrix = matrix.values
    torch_tensor = torch.tensor(matrix, dtype=torch.float32)
    torch.save(torch_tensor, 'matrix.pt')

    embedding = torch.load('matrix.pt')
    print(embedding)
    print(embedding.shape)
    
    return embedding


if __name__ == "__main__":
    transform_matrix('npmi_matrix.csv')
    matrix_to_embedding('transformed_matrix.csv')
    print("Done!")

import os
import re
import networkx as nx
import matplotlib.pyplot as plt
import json

def create_pattern_dict(FILE_PATH):
    """
    Create a dictionary from a list of patterns.

    Args:
        patterns (list): A list of patterns.

    Returns:
        dict: A dictionary with patterns as keys and True as values.
    """

    print(f"Looking for file at:\n{FILE_PATH}\n")

    # === STEP 1: Prepare regex pattern for transaction lines ===
    pattern = re.compile(
        r"^(\S+\s+\S+),"              # Timestamp
        r"(\S+),"                     # From Bank
        r"(\S+),"                     # From Account
        r"(\S+),"                     # To Bank
        r"(\S+),"                     # To Account
        r"([\d.]+),"                  # Amount Received
        r"([^,]+),"                   # Receiving Currency
        r"([\d.]+),"                  # Amount Paid
        r"([^,]+),"                   # Payment Currency
        r"([^,]+),"                   # Payment Format
        r"(\d+)$"                     # Is Laundering
    )

    # === STEP 2: Parse the file into dictionary ===
    pattern_dict = {}
    current_key = None
    current_values = []
    all_keys = {}

    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Detect beginning of a new laundering attempt section
                if line.startswith("BEGIN LAUNDERING ATTEMPT -"):
                    current_key = line.split("-", 1)[1].strip()
                    if current_key not in all_keys.keys():
                        all_keys[current_key] = 1
                    else:   
                        all_keys[current_key] += 1
                        current_key = f"{current_key}_dup{all_keys[current_key]}"
                        print(f"Warning: Duplicate key found: {current_key}")
                    
                    current_values = []
                    continue

                # Detect end of section
                if line.startswith("END LAUNDERING ATTEMPT -"):
                    if current_key is not None:
                        # Save all collected transactions at the end
                        pattern_dict[current_key] = current_values
                    current_key = None
                    current_values = []
                    continue

                # Process transaction line (only inside a section)
                if current_key:
                    m = pattern.match(line)
                    if m:
                        current_values.append(m.groups())
                    else:
                        print(f"Line did NOT match regex: {line}")

    else:
        print("File not found!")

    return pattern_dict

def show_pattern_dict(pattern_dict):
    """
    Display the contents of a pattern dictionary.

    Args:
        pattern_dict (dict): A dictionary with patterns as keys.
    """
    for key, vals in pattern_dict.items():
        print(f"\n=== {key} ===")
        print(f"Transactions: {len(vals)}")
        for row in vals:  # show only first 2 per section
            print(row)


def filter_dict(d: dict, substring: str):
    list_in_list =[value for key, value in d.items() if substring in str(key)]
    return [item for sublist in list_in_list for item in sublist]

import pandas as pd

def gather_suspicious_network(df, start_node, max_depth=None):
    """
    Recursively gather all suspicious nodes (Is Laundering == 1)
    connected to a given node.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataset with columns:
        ['From_Node', 'To_Node', 'Is Laundering']
    start_node : str
        Node ID to start from (e.g., 'BANK1_12345')
    max_depth : int or None
        Optional depth limit for traversal.

    Returns
    -------
    connected_df : pd.DataFrame
        Subset of df with only suspicious connected transactions.
    connected_nodes : set
        All suspicious connected nodes found.
    """

    # Filter only suspicious transactions
    suspicious_df = df.copy()
    # suspicious_df = df[df["Is Laundering"] == 1].copy()
    # print(suspicious_df.head())
    if start_node not in set(suspicious_df["From_Node"]) | set(suspicious_df["To_Node"]):
        raise ValueError(f"Node '{start_node}' not found among suspicious nodes.")

    connected_nodes = {start_node}
    frontier = {start_node}
    depth = 0

    while frontier and (max_depth is None or depth < max_depth):
        # Find all transactions involving current frontier nodes
        mask = suspicious_df["From_Node"].isin(frontier) | suspicious_df["To_Node"].isin(frontier)
        subset = suspicious_df[mask]

        # Gather all suspicious nodes connected to these
        new_nodes = set(subset["From_Node"]) | set(subset["To_Node"])
        new_nodes -= connected_nodes

        if not new_nodes:
            break

        connected_nodes |= new_nodes
        frontier = new_nodes
        depth += 1

    connected_df = suspicious_df[
        suspicious_df["From_Node"].isin(connected_nodes)
        | suspicious_df["To_Node"].isin(connected_nodes)
    ]

    return connected_df, connected_nodes

def create_frequency_dict(df, save = False, node_column="From_Node"):
    """
    Create a frequency dictionary counting occurrences of each node.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing transaction data.
    node_column : str
        Column name to count frequencies from.

    Returns
    -------
    freq_dict : dict
        Dictionary with node IDs as keys and their counts as values.
    """
    freq_series = df[node_column].value_counts()
    freq_dict = freq_series.to_dict()

    if save == True:
        with open(f"{df}_{node_column}.txt", "w") as f:
            json.dump(freq_dict, f, indent=4)

    return freq_dict
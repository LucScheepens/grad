import igraph as ig
import os
import pandas as pd
from collections import deque

def preprocess_df():
    """
    Preprocess the IBM Hi-Small Transactions dataset.
    Returns:
        pd.DataFrame: Preprocessed DataFrame with additional columns.
    """

    current_dir = os.path.dirname(os.getcwd())
    FILE_PATH = os.path.join(current_dir, "data", "IBM", "HI-SmallTransactions.txt")

    current_dir = os.path.dirname(os.getcwd())
    CSV_PATH = os.path.join(current_dir, "data", "IBM", "Hi-Small_Trans.csv")
    df_full = pd.read_csv(CSV_PATH, dtype=str)
    df_full["Is Laundering"] = pd.to_numeric(df_full["Is Laundering"], errors="coerce").fillna(2).astype(int)

    df_full["From_Node"] = df_full["From Bank"].astype(str) + "_" + df_full["Account"].astype(str)
    df_full["To_Node"] = df_full["To Bank"].astype(str) + "_" + df_full["Account.1"].astype(str)
    df_full = df_full.rename(columns={"Account.1": "To Account"})

    
    keys = df_full["To Bank"].astype(str) + "|" + df_full["To Account"]
    keys2 = df_full["From Bank"].astype(str) + "|" + df_full["Account"]

    all_keys = pd.concat([keys, keys2])

    codes, uniques = pd.factorize(all_keys)
    mapping = dict(zip(uniques, range(len(uniques))))

    df_full["To_Account_int"]   = keys.map(mapping)
    df_full["From_Account_int"] = keys2.map(mapping)

    return df_full


def build_igraph_from_df(df):
    g = ig.Graph.DataFrame(
        df[["From_Node", "To_Node"]],
        directed=True
    )
    return g

def extract_laundering_networks_igraph(
    df,
    max_depth=5,
    max_networks=10,
    collapse_threshold=10
):
    """
    Extract laundering-centered networks with surrounding non-laundering nodes
    using igraph for fast traversal.
    Returns a list of dictionaries (not graphs).
    """

    # --- Build full graph ---
    g = ig.Graph.DataFrame(
        df[["From_Account_int", "To_Account_int"]],
        directed=True,
        use_vids=True
    )

    # --- Identify laundering nodes ---
    laundering_nodes = set(
        df.loc[df["Is Laundering"] == 1, "From_Account_int"]
    ).union(
        df.loc[df["Is Laundering"] == 1, "To_Account_int"]
    )

    # --- Subgraph of laundering nodes only ---
    laundering_subgraph = g.subgraph(laundering_nodes)
    laundering_components = laundering_subgraph.components(mode="weak")

    networks = []
    seen_laundering_nodes = set()

    # --- Process each laundering component ---
    for comp in laundering_components:

        core_nodes = set(comp)

        # Skip if this laundering component was already processed
        if core_nodes & seen_laundering_nodes:
            continue

        # --- BFS expansion from all core laundering nodes ---
        visited = {}
        collapsed_nodes = set()

        queue = deque((v, 0) for v in core_nodes)

        while queue:
            node, depth = queue.popleft()

            if node in visited:
                continue

            visited[node] = depth

            neighbors = set(g.neighbors(node, mode="all"))

            # Collapse highly connected nodes
            if len(neighbors) > collapse_threshold:
                collapsed_nodes.add(node)
                continue

            # Depth control
            if depth >= max_depth:
                continue

            for nbr in neighbors:
                if nbr not in visited:
                    queue.append((nbr, depth + 1))

        component_nodes = set(visited.keys())

        # --- Extract transactions inside this network ---
        transactions = df[
            df["From_Account_int"].isin(component_nodes) &
            df["To_Account_int"].isin(component_nodes)
        ].copy()

        # --- Track laundering nodes in this component ---
        laundering_in_component = core_nodes

        start_node = next(iter(core_nodes))

        network = {
            "start_node": start_node,
            "nodes": component_nodes,
            "laundering_nodes": laundering_in_component,
            "collapsed_nodes": collapsed_nodes,
            "node_depths": visited,
            "transactions": transactions
        }

        networks.append(network)

        seen_laundering_nodes.update(core_nodes)

        print(
            f"Laundering network from {start_node}: "
            f"{len(component_nodes)} nodes, "
            f"{len(transactions)} transactions, "
            f"{len(collapsed_nodes)} collapsed hubs"
        )

        if len(networks) >= max_networks:
            break

    return networks

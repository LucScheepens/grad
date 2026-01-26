import random
from collections import deque

def crop_network(network, crop_ratio=0.7, random_seed=None):
    """
    Crop a network to a connected subgraph.
    crop_ratio = fraction of nodes to keep.
    """

    if random_seed is not None:
        random.seed(random_seed)

    nodes = list(network["nodes"])
    target_size = max(2, int(len(nodes) * crop_ratio))

    start_node = random.choice(nodes)

    visited = set()
    queue = deque([start_node])

    # BFS until we hit target size
    while queue and len(visited) < target_size:
        node = queue.popleft()
        if node in visited:
            continue

        visited.add(node)

        neighbors = set(
            network["transactions"]
            .loc[
                (network["transactions"]["From_Account_int"] == node) |
                (network["transactions"]["To_Account_int"] == node),
                ["From_Account_int", "To_Account_int"]
            ]
            .values
            .ravel()
        )

        neighbors &= set(nodes)

        for nbr in neighbors:
            if nbr not in visited:
                queue.append(nbr)

    cropped_nodes = visited

    # Filter transactions
    cropped_tx = network["transactions"][
        network["transactions"]["From_Account_int"].isin(cropped_nodes) &
        network["transactions"]["To_Account_int"].isin(cropped_nodes)
    ].copy()

    return {
        **network,
        "start_node": start_node,
        "nodes": cropped_nodes,
        "laundering_nodes": network["laundering_nodes"] & cropped_nodes,
        "collapsed_nodes": network["collapsed_nodes"] & cropped_nodes,
        "node_depths": {n: d for n, d in network["node_depths"].items() if n in cropped_nodes},
        "transactions": cropped_tx
    }


def delete_random_edges(network, delete_frac=0.15, random_seed=None):
    """
    Randomly delete edges without splitting the network.
    delete_frac = fraction of edges to attempt to delete.
    """

    if random_seed is not None:
        random.seed(random_seed)

    tx = network["transactions"].copy()
    nodes = list(network["nodes"])

    if len(tx) < 2:
        return network  # nothing to delete safely

    # Build igraph
    g = ig.Graph.DataFrame(
        tx[["From_Account_int", "To_Account_int"]],
        directed=False,
        use_vids=False
    )

    target_deletions = int(len(tx) * delete_frac)
    edge_indices = list(range(g.ecount()))
    random.shuffle(edge_indices)

    deleted = 0
    kept_edges = set(range(g.ecount()))

    for ei in edge_indices:
        if deleted >= target_deletions:
            break

        g_test = g.copy()
        g_test.delete_edges([ei])

        # Check connectivity
        if g_test.is_connected():
            kept_edges.remove(ei)
            g = g_test
            deleted += 1

    # Rebuild transactions from kept edges
    kept_edge_list = list(kept_edges)
    kept_edges_df = tx.iloc[kept_edge_list].copy()

    return {
        **network,
        "transactions": kept_edges_df
    }


def add_nodes_to_network(
    network,
    full_df,
    max_new_nodes=10,
    max_depth=2,
    collapse_threshold=10,
    random_seed=None
):
    """
    Expand a network by adding new nodes from the full graph.
    whilst respecting hubs.
    """

    if random_seed is not None:
        random.seed(random_seed)

    current_nodes = set(network["nodes"])
    new_nodes = set()
    collapsed_nodes = set(network["collapsed_nodes"])

    # Build adjacency from full_df
    adj = {}
    for _, row in full_df.iterrows():
        u = row["From_Account_int"]
        v = row["To_Account_int"]
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)

    boundary = list(current_nodes)
    random.shuffle(boundary)

    queue = deque((n, 0) for n in boundary)

    while queue and len(new_nodes) < max_new_nodes:
        node, depth = queue.popleft()

        if depth >= max_depth:
            continue

        neighbors = adj.get(node, set())

        # Collapse hubs
        if len(neighbors) > collapse_threshold:
            collapsed_nodes.add(node)
            continue

        for nbr in neighbors:
            if nbr not in current_nodes and nbr not in new_nodes:
                new_nodes.add(nbr)
                queue.append((nbr, depth + 1))

                if len(new_nodes) >= max_new_nodes:
                    break

    augmented_nodes = current_nodes | new_nodes

    # Filter transactions
    augmented_tx = full_df[
        full_df["From_Account_int"].isin(augmented_nodes) &
        full_df["To_Account_int"].isin(augmented_nodes)
    ].copy()

    return {
        **network,
        "nodes": augmented_nodes,
        "collapsed_nodes": collapsed_nodes,
        "node_depths": {
            **network["node_depths"],
            **{n: None for n in new_nodes}  # unknown depth for added nodes
        },
        "transactions": augmented_tx
    }

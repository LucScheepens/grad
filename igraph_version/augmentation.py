import random
from collections import deque
import igraph as ig 
import pandas as pd
import time
import copy


def build_igraph_from_transactions(tx_df):
    """
    Build an undirected igraph graph from transactions dataframe.
    """
    g = ig.Graph.DataFrame(
        tx_df[["From_Account_int", "To_Account_int"]],
        directed=False,
        use_vids=False
    )
    return g


def crop_network(network, crop_ratio=0.8, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    g = network["graph"]

    # --- only nodes that exist in the graph ---
    graph_nodes = set(v["name"] for v in g.vs)
    nodes = list(set(network["nodes"]) & graph_nodes)

    if len(nodes) < 2:
        return network

    target_size = max(2, int(len(nodes) * crop_ratio))

    start_node = random.choice(nodes)

    try:
        start_vid = g.vs.find(name=start_node).index
    except ValueError:
        # graph is inconsistent → skip crop
        return network

    # --- BFS using igraph ---
    order = g.bfs(start_vid)[0]
    bfs_nodes = [g.vs[v]["name"] for v in order if v != -1]

    cropped_nodes = set(bfs_nodes[:target_size])

    # --- filter transactions ---
    tx = network["transactions"]
    cropped_tx = tx[
        tx["From_Account_int"].isin(cropped_nodes) &
        tx["To_Account_int"].isin(cropped_nodes)
    ].copy()

    # --- rebuild graph from cropped_tx ---
    g_sub = ig.Graph.DataFrame(
        cropped_tx[["From_Account_int", "To_Account_int"]],
        directed=False,
        use_vids=False
    )

    return {
        **network,
        "start_node": start_node,
        "nodes": cropped_nodes,
        "laundering_nodes": network["laundering_nodes"] & cropped_nodes,
        "collapsed_nodes": network["collapsed_nodes"] & cropped_nodes,
        "node_depths": {
            n: d for n, d in network["node_depths"].items()
            if n in cropped_nodes
        },
        "transactions": cropped_tx,
        "graph": g_sub
    }



def delete_random_edges_bridge_safe(network, delete_frac=0.15, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    g = network["graph"].copy()
    tx = network["transactions"].copy()

    if g.ecount() < 2:
        return network

    bridges = set(g.bridges())
    non_bridges = [i for i in range(g.ecount()) if i not in bridges]

    if not non_bridges:
        return network

    target_deletions = int(len(non_bridges) * delete_frac)
    random.shuffle(non_bridges)

    delete_eids = non_bridges[:target_deletions]

    # Drop from graph
    g.delete_edges(delete_eids)

    # Drop same rows from transactions (assumes same order)
    tx = tx.drop(tx.index[delete_eids]).reset_index(drop=True)

    return {
        **network,
        "transactions": tx,
        "graph": g
    }


def add_nodes_to_network_incremental(
    network,
    full_graph,
    max_new_nodes=10,
    max_depth=2,
    collapse_threshold=10,
    random_seed=None
):
    if random_seed is not None:
        random.seed(random_seed)

    g = network["graph"].copy()

    current_nodes = set(network["nodes"])
    collapsed_nodes = set(network["collapsed_nodes"])
    new_nodes = set()

    # ---- frontier BFS ----
    boundary = list(current_nodes)
    random.shuffle(boundary)

    from collections import deque
    queue = deque((n, 0) for n in boundary)

    while queue and len(new_nodes) < max_new_nodes:
        node, depth = queue.popleft()
        if depth >= max_depth:
            continue

        try:
            vid = full_graph.vs.find(name=node).index
        except ValueError:
            continue

        neighbors_vids = full_graph.neighbors(vid)
        neighbors = {full_graph.vs[v]["name"] for v in neighbors_vids}

        if len(neighbors) > collapse_threshold:
            collapsed_nodes.add(node)
            continue

        for nbr in neighbors:
            if nbr not in current_nodes and nbr not in new_nodes:
                new_nodes.add(nbr)
                queue.append((nbr, depth + 1))
                if len(new_nodes) >= max_new_nodes:
                    break

    if not new_nodes:
        return network

    # ---- add vertices ----
    for n in new_nodes:
        if n not in g.vs["name"]:
            g.add_vertex(name=n)

    # ---- name → index mapping ----
    name_to_vid = {v["name"]: v.index for v in g.vs}

    # ---- add edges safely ----
    edges_to_add = []

    for n in new_nodes:
        try:
            vid = full_graph.vs.find(name=n).index
        except ValueError:
            continue

        for nbr_vid in full_graph.neighbors(vid):
            nbr = full_graph.vs[nbr_vid]["name"]
            if nbr in current_nodes or nbr in new_nodes:
                if n in name_to_vid and nbr in name_to_vid:
                    edges_to_add.append(
                        (name_to_vid[n], name_to_vid[nbr])
                    )

    if edges_to_add:
        g.add_edges(edges_to_add)

    # ---- update transactions ----
    tx = network["transactions"].copy()
    new_tx = [
        {"From_Account_int": g.vs[e[0]]["name"],
         "To_Account_int": g.vs[e[1]]["name"]}
        for e in edges_to_add
    ]

    if new_tx:
        tx = pd.concat([tx, pd.DataFrame(new_tx)], ignore_index=True)

    return {
        **network,
        "nodes": current_nodes | new_nodes,
        "collapsed_nodes": collapsed_nodes,
        "node_depths": {
            **network["node_depths"],
            **{n: None for n in new_nodes}
        },
        "transactions": tx,
        "graph": g
    }



def augment_network_view_fast(
    network,
    full_graph,
    p_crop=0.6,
    p_edge_drop=0.2,
    p_node_add=0.2,
    crop_ratio_range=(0.6, 0.9),
    edge_drop_range=(0.05, 0.2),
    max_new_nodes=10,
    random_seed=None
):
    if random_seed is not None:
        random.seed(random_seed)

    # 🔥 DO NOT MUTATE ORIGINAL
    aug_net = copy.deepcopy(network)

    # --- Crop ---
    if random.random() < p_crop:
        ratio = random.uniform(*crop_ratio_range)
        aug_net = crop_network(aug_net, crop_ratio=ratio)

    # --- Edge drop ---
    if random.random() < p_edge_drop:
        frac = random.uniform(*edge_drop_range)
        aug_net = delete_random_edges_bridge_safe(aug_net, delete_frac=frac)

    # --- Node add ---
    if random.random() < p_node_add:
        aug_net = add_nodes_to_network_incremental(
            aug_net,
            full_graph=full_graph,
            max_new_nodes=max_new_nodes
        )

    return aug_net



import networkx as nx

def bipartite_stacked_layout(left_nodes, right_nodes, x_gap=4.0, y_gap=2.0):
    pos = {}

    for i, node in enumerate(left_nodes):
        pos[node] = (0.0, -i * y_gap)

    for i, node in enumerate(right_nodes):
        pos[node] = (x_gap, -i * y_gap)

    return pos

def stacked_layout(left_nodes, middle_nodes, right_nodes, x_gap=4.0, y_gap=2.0):
    pos = {}
    y_gap = 2.0

    for i, node in enumerate(left_nodes):
        pos[node] = (0.0, -i * y_gap)

    for i, node in enumerate(middle_nodes):
        pos[node] = (4.0, -i * y_gap)

    for i, node in enumerate(right_nodes):
        pos[node] = (8.0, -i * y_gap)

def generate_pattern(pattern_type, n_left=2, n_right=3, laudering=1):
    if pattern_type == "Bipartite":
        left_nodes = [f"BANK1_{10001+i}" for i in range(n_left)]
        right_nodes = [f"BANK2_{20001+i}" for i in range(n_right)]

        transactions = [
            (l, r, laudering)
            for l in left_nodes
            for r in right_nodes
        ]
        pos = bipartite_stacked_layout(left_nodes, right_nodes)

    elif pattern_type == "Stacked":
        left_nodes = [f"BANK1_{10001+i}" for i in range(n_left)]
        middle_nodes = [f"BANK2_{20001+i}" for i in range(n_right)]
        right_nodes = [f"BANK3_{30001+i}" for i in range(n_left)]

        transactions = [
            (l, m, laudering)
            for l in left_nodes
            for m in middle_nodes
        ] + [
            (m, r, laudering)
            for m in middle_nodes
            for r in right_nodes
        ]

        # Create a custom layout for stacked pattern
        pos = stacked_layout(left_nodes, middle_nodes, right_nodes)

    G = nx.DiGraph()
    G.add_nodes_from(left_nodes, bipartite=0)
    G.add_nodes_from(right_nodes, bipartite=1)

    for u, v, is_laundering in transactions:
        G.add_edge(u, v, laundering=is_laundering)

    return G, pos
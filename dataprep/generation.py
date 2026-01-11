import networkx as nx
import random
import json

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

def generate_bank_id():
    # length of the hex block (8 or 9, based on your samples)
    with open("Bank_dict.txt", "r") as f:
        bank_dict = f.read()
    bank_dict = json.loads(bank_dict)

    prefix = random.choices(list(bank_dict.keys()), weights=list(bank_dict.values()))[0]

    length = random.choice([8, 9])
    
    # valid characters: 0–9 and A–F
    hex_chars = "0123456789ABCDEF"
    
    # generate random hex characters except the last one
    body = "".join(random.choice(hex_chars) for _ in range(length - 1))
    
    # force last character to be '0'
    return f"{prefix}_{body}0"

def generate_pattern(pattern_type, n_left=2, n_right=3, laudering=1):
    if pattern_type == "Bipartite":
        left_nodes = [f"{generate_bank_id()}" for i in range(n_left)]
        right_nodes = [f"{generate_bank_id()}" for i in range(n_right)]

        transactions = [
            (l, r, laudering)
            for l in left_nodes
            for r in right_nodes
        ]
        pos = bipartite_stacked_layout(left_nodes, right_nodes)

    elif pattern_type == "Stacked":
        left_nodes = [f"{generate_bank_id()}" for i in range(n_left)]
        middle_nodes = [f"{generate_bank_id()}" for i in range(n_right)]
        right_nodes = [f"{generate_bank_id()}" for i in range(n_left)]

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
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

    elif pattern_type == "fan_out":
        left_nodes = [f"{generate_bank_id()}" for i in range(1)]
        right_nodes = [f"{generate_bank_id()}" for i in range(n_right)]

        transactions = [
            (left_nodes[0], r, laudering)
            for r in right_nodes
        ]
        pos = bipartite_stacked_layout(left_nodes, right_nodes)
    
    elif pattern_type == "fan_in":
        left_nodes = [f"{generate_bank_id()}" for i in range(n_right)]
        right_nodes = [f"{generate_bank_id()}" for i in range(1)]

        transactions = [
            (l, right_nodes[0], laudering)
            for l in left_nodes
        ]
        pos = bipartite_stacked_layout(left_nodes, right_nodes)

    elif pattern_type == "random":
        total_nodes = n_left + n_right
        nodes = [f"{generate_bank_id()}" for i in range(total_nodes)]

        transactions = []
        for _ in range(n_left * n_right):
            u = random.choice(nodes)
            v = random.choice(nodes)
            if u != v:
                transactions.append((u, v, laudering))

        pos = nx.spring_layout(
            nx.DiGraph(transactions),
            seed=42,
            k=1.0,
            iterations=200,
            scale=3.0
        )

    G = nx.DiGraph()
    G.add_nodes_from(left_nodes, bipartite=0)
    G.add_nodes_from(right_nodes, bipartite=1)

    for u, v, is_laundering in transactions:
        G.add_edge(u, v, laundering=is_laundering)

    return G, pos

# def generate_pattern_in_graph(transactions):
#     pattern_types = ["Bipartite", "Stacked", "fan_out"]
#     pattern_type = random.choice(pattern_types)
    
#     random.randint(0, len(transactions))

#     G, pos = generate_pattern(pattern_type)

#     return G, pos, pattern_type

def generate_pattern_in_graph(G, pattern_type='random'):
    """
    Selects a random edge in G, removes it,
    and replaces it with a laundering pattern
    connecting the same source and destination.
    """

    u, v = random.choice(list(G.edges()))

    G.remove_edge(u, v)
    if pattern_type == 'random':
        pattern_types = ["Bipartite", "Stacked", "fan_out", "fan_in", "random"]
        pattern_type = random.choice(pattern_types)

    pattern_graph, pos = generate_pattern(pattern_type)

    pattern_nodes = list(pattern_graph.nodes())

    in_degrees = dict(pattern_graph.in_degree())
    out_degrees = dict(pattern_graph.out_degree())

    entry_nodes = [n for n in pattern_nodes if in_degrees[n] == 0]
    exit_nodes  = [n for n in pattern_nodes if out_degrees[n] == 0]

    for n in pattern_nodes:
        G.add_node(n)

    for x, y, data in pattern_graph.edges(data=True):
        G.add_edge(x, y, laundering=1)

    for n in entry_nodes:
        G.add_edge(u, n, laundering=1)

    for n in exit_nodes:
        G.add_edge(n, v, laundering=1)

    return G, pos, pattern_type
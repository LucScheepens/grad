import os
import matplotlib.pyplot as plt
import networkx as nx

columns = [
    "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
    "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
    "Payment Format", "Is Laundering"
]

output_dir = os.path.join(os.getcwd(), "pattern_plots")
os.makedirs(output_dir, exist_ok=True)

# === STEP 1: Function to generate graph plots ===
def plot_laundering_pattern(key, transactions):
    """
    Create and save a plot showing the transaction network for a given pattern,
    with improved spacing and reduced overlap.
    """
    import networkx as nx
    import matplotlib.pyplot as plt
    import os

    G = nx.DiGraph()

    # Map full account IDs to short labels (A, B, C...)
    node_map = {}
    next_label = 65   # ASCII A

    for t in transactions:
        data = dict(zip(columns, t))

        src = f"{data['From Bank']}_{data['From Account']}"
        dst = f"{data['To Bank']}_{data['To Account']}"

        if src not in node_map:
            node_map[src] = chr(next_label); next_label += 1
        if dst not in node_map:
            node_map[dst] = chr(next_label); next_label += 1

        G.add_edge(
            node_map[src],
            node_map[dst],
            amount=float(data["Amount Paid"]),
            currency=data["Payment Currency"]
        )

    # Determine edge widths
    amounts = [d["amount"] for _, _, d in G.edges(data=True)]
    if amounts:
        min_amt, max_amt = min(amounts), max(amounts)
        widths = [
            1 + 4 * ((amt - min_amt) / (max_amt - min_amt + 1e-9))
            for amt in amounts
        ]
    else:
        widths = []

    # Try Graphviz layout first (best for avoiding overlap)
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        pos = graphviz_layout(G, prog="dot")  # hierarchical, clean
    except Exception:
        # Fallback: tuned spring layout
        pos = nx.spring_layout(
            G, 
            seed=42,
            k=1.0,        # increase repulsion → nodes spread apart
            iterations=200,
            scale=3.0
        )

    # Resize figure based on graph size
    base_size = 6
    n = max(1, len(G.nodes()))
    fig_size = (base_size + n * 0.3, base_size + n * 0.3)

    plt.figure(figsize=fig_size)

    nx.draw_networkx_nodes(
        G, pos,
        node_size=900,
        node_color="lightblue",
        edgecolors="black",
        linewidths=1.2
    )

    # Use curved edges to prevent arrow overlap
    nx.draw_networkx_edges(
        G, pos,
        width=widths,
        arrowstyle="-|>",
        arrowsize=18,
        connectionstyle="arc3,rad=0.15",
        edge_color="gray"
    )

    # Slight label offset to avoid overlap
    nx.draw_networkx_labels(
        G, pos,
        font_size=10,
        font_weight="bold",
        verticalalignment="center_baseline"
    )

    plt.title(f"Transaction Pattern: {key}", fontsize=12)
    plt.axis("off")

    safe_key = key.replace(":", "").replace(" ", "_").replace("-", "")
    out_path = os.path.join(output_dir, f"{safe_key}.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ Saved plot for '{key}' to: {out_path}")

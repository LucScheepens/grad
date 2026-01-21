import igraph as ig

def build_igraph_from_df(df):
    g = ig.Graph.DataFrame(
        df[["From_Node", "To_Node"]],
        directed=True
    )
    return g
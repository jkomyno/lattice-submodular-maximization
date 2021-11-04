import numpy as np
import itertools
import networkx as nx
from nptyping import NDArray
from typing import Callable, Tuple
from .. import utils


def import_wikilens_ratings(rng: np.random.Generator, basedir: str) -> Callable[[int], Tuple[nx.Graph, NDArray[int]]]:
    """
    Import graph for budget_allocation.
    :return: the bipartite graph G=(V,T;W).
    """
    localpath = 'dataset/konect/wikilens-ratings/out.wikilens-ratings'
    filepath = f'{basedir}/{localpath}'

    edges = []

    #########################
    #  Read entire dataset  #
    #########################

    with open(filepath, 'r') as f:
        # skip the first two lines
        f.readline()
        f.readline()

        while True:
            line = f.readline()
            if not line:
                break

            u_raw, v_raw, w_raw, _ = line.split()
            u, v = map(int, (u_raw, v_raw))
            w = float(w_raw)

            channel_id = u
            customer_id = v
            weight = max(float(w), 0.5)

            edges.append((channel_id, customer_id, weight))

    channel_ids = set()
    customer_ids = set()
    for channel_id, customer_id, _ in edges:
        channel_ids.add(channel_id)
        customer_ids.add(customer_id)

    ###########################
    #  generate entire graph  #
    ###########################

    channel_ids_to_node_tmp = dict(zip(sorted(channel_ids), itertools.count(start=0, step=1)))
    customer_ids_to_node_tmp = dict(zip(sorted(customer_ids), itertools.count(start=len(channel_ids), step=1)))
    edges = list(map(lambda edge: (channel_ids_to_node_tmp[edge[0]], customer_ids_to_node_tmp[edge[1]], edge[2]), edges))

    G_tmp = nx.Graph()

    # add bipartite nodes
    G_tmp.add_nodes_from(channel_ids_to_node_tmp.values(), bipartite=0)   # V
    G_tmp.add_nodes_from(customer_ids_to_node_tmp.values(), bipartite=1)  # T

    for u, v, weight in edges:
        # weights are divided by 5 (the maximum rating) to turn them into influence probabilities
        G_tmp.add_edge(u, v, weight=weight / 5)

    ##############################
    #  trim the graph on demand  #
    ##############################

    def trim_graph(n: int) -> Tuple[nx.Graph, NDArray[int]]:
        nonlocal G_tmp

        # select the largest 20 out-degree channels, remove the customers without edges        
        largest_channels = set(map(utils.fst, sorted(G_tmp.degree, key=utils.snd, reverse=True)[:n]))
        edges = [(u, v, w) for u, v, w in set(G_tmp.edges.data('weight')) if u in largest_channels]
        surviving_customers = set(map(utils.snd, edges))

        channel_ids_to_node = dict(zip(sorted(largest_channels), itertools.count(start=0, step=1)))
        customer_ids_to_node = dict(zip(sorted(surviving_customers), itertools.count(start=len(channel_ids_to_node), step=1)))
        edges = list(map(lambda edge: (channel_ids_to_node[edge[0]], customer_ids_to_node[edge[1]], edge[2]), edges))
        
        G = nx.Graph()

        # add bipartite nodes
        G.add_nodes_from(channel_ids_to_node.values(), bipartite=0)   # V
        G.add_nodes_from(customer_ids_to_node.values(), bipartite=1)  # T

        for u, v, weight in edges:
            G.add_edge(u, v, weight=weight)

        print(f'n channel_ids: {len(channel_ids_to_node)}') 
        print(f'n customer_ids: {len(customer_ids_to_node)}')
        print(f'n edges: {len(edges)}')

        V = [n for n in G.nodes if G.nodes[n]['bipartite'] == 0]
        print(f'V: {V}')

        return G

    return trim_graph

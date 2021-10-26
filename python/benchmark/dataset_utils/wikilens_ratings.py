import numpy as np
import itertools
import networkx as nx
from typing import Tuple
from .. import utils


def import_wikilens_ratings(rng: np.random.Generator, basedir: str) -> Tuple[nx.Graph, int]:
    """
    Import graph for budget_allocation.
    :return: the bipartite graph G=(V,T;W).
    """
    localpath = 'dataset/konect/wikilens-ratings/out.wikilens-ratings'
    filepath = f'{basedir}/{localpath}'

    G = nx.Graph()
    edges = []

    with open(filepath, 'r') as f:
        # skip first line
        f.readline()

        while True:
            line = f.readline()
            if not line:
                break

            u_raw, v_raw, _, _ = line.split()
            u, v = map(int, (u_raw, v_raw))

            channel_id = u
            customer_id = v

            edges.append((channel_id, customer_id))

        edges = list(map(
            utils.snd,
            filter(
                lambda ii: not (
                    ii[0] % 3  == 0 or
                    ii[0] % 5  == 0
                ), enumerate(edges)
            )
        ))
        edges = edges[:2000]
        
        channel_ids = set()
        customer_ids = set()
        for channel_id, customer_id in edges:
            channel_ids.add(channel_id)
            customer_ids.add(customer_id)

        print(f'n channel_ids: {len(channel_ids)}')     # 65
        print(f'n customer_ids: {len(customer_ids)}')   # 1027 
        print(f'n edges: {len(edges)}')                 # 2000

        channel_ids_to_node = dict(zip(sorted(channel_ids), itertools.count(start=0, step=1)))
        customer_ids_to_node = dict(zip(sorted(customer_ids), itertools.count(start=len(channel_ids), step=1)))
        edges = list(map(lambda edge: (channel_ids_to_node[edge[0]], customer_ids_to_node[edge[1]]), edges))

        # add bipartite nodes
        G.add_nodes_from(channel_ids_to_node.values(), bipartite=0)   # V
        G.add_nodes_from(customer_ids_to_node.values(), bipartite=1)  # T

        weights = rng.uniform(low=0.0, high=1.0, size=(len(edges), ))

        for (u, v), weight in zip(edges, weights):
            G.add_edge(u, v, weight=weight)

    return G

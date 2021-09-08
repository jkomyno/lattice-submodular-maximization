import numpy as np
import math
import itertools
import networkx as nx
from typing import Tuple


def import_brunson_revolution(rng: np.random.Generator, basedir: str) -> nx.Graph:
    """
    :return: the bipartite graph G=(V,T;W).
    """
    localpath = 'dataset/konect/brunson-revolution/out.brunson_revolution_revolution'
    filepath = f'{basedir}/{localpath}'

    G = nx.Graph()

    facility_ids = set()
    customer_ids = set()
    edges = []

    with open(filepath, 'r') as f:
        # skip first two lines
        f.readline()
        f.readline()

        while True:
            line = f.readline()
            if not line:
                break

            u_raw, v_raw = line.split()
            u, v = map(int, (u_raw, v_raw))

            facility_id = u
            customer_id = v

            edges.append((facility_id, customer_id))

            facility_ids.add(facility_id)
            customer_ids.add(customer_id)

        print(f'n facility_ids: {len(facility_ids)}')
        print(f'n customer_ids: {len(customer_ids)}')

        facility_ids_to_node = dict(zip(sorted(facility_ids), itertools.count(start=0, step=1)))
        customer_ids_to_node = dict(zip(sorted(customer_ids), itertools.count(start=len(facility_ids), step=1)))
        edges = list(map(lambda edge: (facility_ids_to_node[edge[0]], customer_ids_to_node[edge[1]]), edges))

        # add bipartite nodes
        G.add_nodes_from(facility_ids_to_node.values(), bipartite=0)   # V
        G.add_nodes_from(customer_ids_to_node.values(), bipartite=1)   # T

        weights = rng.integers(low=1, high=100 + 1, size=(len(edges), ))

        for (u, v), weight in zip(edges, weights):
            G.add_edge(u, v, weight=weight)

    return G

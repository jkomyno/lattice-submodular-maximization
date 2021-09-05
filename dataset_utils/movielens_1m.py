import itertools
import networkx as nx
from typing import Tuple


def import_movielens_1m(basedir: str) -> Tuple[nx.Graph, int]:
    """
    :return: the bipartite graph G=(V,T;W).
    """
    localpath = 'dataset/konect/movielens-1m/out.movielens-1m'
    filepath = f'{basedir}/{localpath}'

    G = nx.Graph()

    facility_ids = set()
    customer_ids = set()
    edges = []

    with open(filepath, 'r') as f:
        # skip first line
        f.readline()

        while True:
            line = f.readline()
            if not line:
                break

            u_raw, v_raw, weight_raw, _ = line.split('\s')
            u, v = map(int, (u_raw, v_raw))
            weight = float(weight_raw)

            facility_id = u
            customer_id = v

            edges.append((facility_id, customer_id, weight))

            facility_ids.add(facility_id)
            customer_ids.add(customer_id)

        print(f'n facility_ids: {len(facility_ids)}')
        print(f'n customer_ids: {len(customer_ids)}')

        facility_ids_to_node = dict(zip(sorted(facility_ids), itertools.count(start=0, step=1)))
        customer_ids_to_node = dict(zip(sorted(customer_ids), itertools.count(start=len(facility_ids), step=1)))
        edges = list(map(lambda edge: (facility_ids_to_node[edge[0]], customer_ids_to_node[edge[1]], edge[2]), edges))

        # add bipartite nodes
        G.add_nodes_from(facility_ids_to_node.values(), bipartite=0)   # V
        G.add_nodes_from(customer_ids_to_node.values(), bipartite=1)   # T

        for u, v, weight in edges:
            assert weight >= 0
            G.add_edge(u, v, weight=weight)

    return G

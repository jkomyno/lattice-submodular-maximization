import itertools
import networkx as nx
from typing import Tuple
from .. import utils


def import_movielens_100k(basedir: str) -> Tuple[nx.Graph, int]:
    """
    Import graph for facility_location.
    :return: the bipartite graph G=(V,T;W).
    """
    localpath = 'dataset/konect/movielens-100k_rating/rel.rating'
    filepath = f'{basedir}/{localpath}'

    G = nx.Graph()

    facility_ids = set()
    customer_ids = set()
    edges = []

    with open(filepath, 'r') as f:
        # skip first three lines
        f.readline()
        f.readline()
        f.readline()

        while True:
            line = f.readline()
            if not line:
                break

            u_raw, v_raw, weight_raw, _ = line.split()
            u, v = map(int, (u_raw, v_raw))
            weight = float(weight_raw)

            facility_id = u
            customer_id = v

            edges.append((facility_id, customer_id, weight))

            facility_ids.add(facility_id)
            customer_ids.add(customer_id)

        edges = list(map(
            utils.snd,
            filter(
                lambda ii: not (
                    ii[0] % 3  == 0 or
                    ii[0] % 5  == 0
                ), enumerate(edges)
            )
        ))
        edges = edges[:100]
        
        facility_ids = set()
        customer_ids = set()
        for facility_id, customer_id, _ in edges:
            facility_ids.add(facility_id)
            customer_ids.add(customer_id)

        print(f'n facility_ids: {len(facility_ids)}')   # 75
        print(f'n customer_ids: {len(customer_ids)}')   # 91 
        print(f'n edges: {len(edges)}')                 # 100

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

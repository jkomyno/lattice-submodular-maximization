import itertools
import math
import numpy as np
import networkx as nx
from collections import Counter
from typing import Tuple


def import_yahoo_data(basedir: str) -> Tuple[nx.Graph, int]:
    """
    We map the phrases to channels V and the accounts to customers T, with an
    edge between s and t if a corresponding bid was made.
    A same id could be assigned to both an advertising channel and a customer,
    so we need to be able to distinguish them later.
    We want nodes to start from 0.
    :return: the bipartite graph G=(V,T;W) and the average price of the bids.
    """
    localpath = 'dataset/yahoo-data/ydata-ysm-advertiser-bids-v1_0.txt'
    filepath = f'{basedir}/{localpath}'

    G = nx.Graph()

    phrase_ids = set()
    account_ids = set()
    prices = []
    edges = []

    with open(filepath, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            _, phrase_id_raw, account_id_raw, price_raw, _ = line.split('\t')
            phrase_id, account_id = map(int, (phrase_id_raw, account_id_raw))
            price = float(price_raw)

            prices.append(price)
            edges.append((phrase_id, account_id))

            phrase_ids.add(phrase_id)
            account_ids.add(account_id)

        print(f'n phrase_ids: {len(phrase_ids)}')
        print(f'n account_ids: {len(account_ids)}')
        print(f'price_avg: {np.mean(prices)}')
        print(f'price_max: {np.max(prices)}')
        print(f'price_min: {np.min(prices)}')

        phrase_ids_to_node = dict(zip(sorted(phrase_ids), itertools.count(start=0, step=1)))
        account_ids_to_node = dict(zip(sorted(account_ids), itertools.count(start=len(phrase_ids), step=1)))
        edges = list(map(lambda edge: (phrase_ids_to_node[edge[0]], account_ids_to_node[edge[1]]), edges))
        edge_counter = Counter(edges)

        # add bipartite nodes
        G.add_nodes_from(phrase_ids_to_node.values(), bipartite=0)   # V
        G.add_nodes_from(account_ids_to_node.values(), bipartite=1)  # T

        # w_sum is the sum of the frequencies of the edges
        w_sum = sum((w for _, w in edge_counter.items()))

        for (u, v), w in edge_counter.items():
            # we use the frequency of (u, v) normalized in the [0, 1] interval as
            # the weight of the edge
            weight = w / w_sum
            assert weight >= 0
            assert weight <= 1
            G.add_edge(u, v, weight=weight)

    return G, math.ceil(np.mean(prices))

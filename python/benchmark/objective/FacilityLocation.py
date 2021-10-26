import networkx as nx
import numpy as np
from nptyping import NDArray
from typing import List
from .Objective import Objective


class FacilityLocation(Objective):
    def __init__(self, G: nx.Graph, B: NDArray[int]):
        """
        Generate an integer-lattice smodular, monotone function for the
        facility location problem.
        We are given a set V of facilities, and we aim at deciding how large
        facilities are opened up in order to serve a set of m customers, where we
        represent scale of facilities as integers 0, 1, ..., b ("0" means we do
        not open a facility). The goal is to decide how large each facility should
        be in order to optimally serve a set T of customer.

        http://web.cs.ucla.edu/~baharan//papers/bian17guaranteed_long.pdf (§6, Facility Location)
        """
        V: List[int] = [n for n in G.nodes if G.nodes[n]['bipartite'] == 0]
        T: List[int] = [m for m in G.nodes if G.nodes[m]['bipartite'] == 1]

        super().__init__(V, B)
        self.W = nx.adjacency_matrix(G)

        # list of target customers
        self.T = T

    def value(self, x: NDArray[int]) -> float:
        """
        Value oracle for the facility location problem.
        :param x: scale of all facilities
        """
        super().value(x)

        # W_st is the (|S| * |T|) weight matrix
        W_st = np.array([[self.W[s, t] for s in self.V] for t in self.T])

        # m is the application of p_st to W_st
        M = x * W_st * np.sqrt(1 - x + self.B) / self.B

        return np.sum(np.max(M, axis=1))

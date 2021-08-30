import networkx as nx
from typing import Set
from nptyping import NDArray, Int64
from .Objective import Objective


class FacilityLocation(Objective):
    def __init__(self, G: nx.Graph, b: int):
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
      V, T = nx.bipartite.sets(G)
      super().__init__(list(V), b)
      self.G = G

      # set of target customers
      self.T: Set[int] = T

    def p(self, s: int, t: int, x_s: int) -> float:
        """
        Model the value in service provided to the customer t for the facility
        s of scale x_s.
        It's a monotone, normalized, non-negative function.
        :param s: index of the facility
        :param t: index of the customer
        :param x_s: scale of the facility s
        """
        # w_st is the weight of the edge between the facility s and the
        # target customer t
        w_st = self.G[s][t]['weight']

        return w_st * x_s * ((self.b + 1 - x_s) ** 0.5) / self.b

    def value(self, x: NDArray[Int64]) -> float:
        """
        Value oracle for the facility location problem.
        :param x: scale of all facilities
        """
        super().value(x)
        
        return sum((
            max((
                self.p(s, t, x_s)
                for s, x_s in enumerate(x)
            ))
            for t in self.T
        ))

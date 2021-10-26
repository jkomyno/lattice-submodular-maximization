import networkx as nx
from nptyping import NDArray
from typing import List
from .. import utils
from .Objective import Objective


class BudgetAllocation(Objective):
    def __init__(self, G: nx.Graph, B: NDArray[int]):
      """
      Optimal budget allocation is a special case of the influence maximization
      problem. It can be modeled as a bipartite graph (S, T; W), where S and T
      are collections of advertising channels and customers, respectively. The
      edge weight, p_st ∈ W, represents the influence probability of channel s
      to customer t. The goal is to distribute the budget (e.g., time for a TV
      advertisement, or space of an inline ad) among the source nodes, and to
      maximize the expected influence on the potential customers.
      The total influence of customer t from all channels can be modeled
      by a proper monotone DR-submodular function I_t(x), where x is the
      budget assignment among the advertising channels.
      A concrete application is for search marketing advertiser bidding, in
      which vendors bid for the right to appear alongside the results of
      different search keywords.

      https://arxiv.org/pdf/1606.05615.pdf (§6, Optimal budget allocation with
      continuous assignments)
      """
      V: List[int] = [n for n in G.nodes if G.nodes[n]['bipartite'] == 0]
      T: List[int] = [m for m in G.nodes if G.nodes[m]['bipartite'] == 1]

      super().__init__(V, B)
      self.G = G
      self.W = nx.adjacency_matrix(G)

      # set of advertiser customers
      self.T = T

    def I(self, t: int, x: NDArray[int]) -> float:
        """
        Return the total influence of customer t from all channels.
        :param x: budget assignment among the advertising channels.
        """
        # self.G[s][t]['weight'] models the influence probability of
        # channel s to customer t
        return 1 - utils.prod((
          (1 - self.W[s, t]) ** x[s]
          for (_, s) in self.G.edges(t)
        ))

    def value(self, x: NDArray[int]) -> float:
        """
        Value oracle for the Budget Allocation problem.
        :param x: allotted budget.
        :return: expected number of influenced people
        """
        super().value(x)
        return sum((self.I(t, x) for t in self.T))

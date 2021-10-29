import networkx as nx
from networkx.classes.function import neighbors
from nptyping import NDArray
from typing import List, Tuple
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

      # W[s, t] is the influence probability of channel s to customer t.
      W = nx.adjacency_matrix(G)

      # collect the neighbors s \in S of each t \in T
      neighbors: List[List[int]] = [[s for s in G.neighbors(t)] for t in T]

      # keep track of (1 - p(s, t), s) for each neighbors s \in S of each t \in T
      self.probs_exp_list: List[List[Tuple[float, int]]] = [
        [(1 - W[s, t], s) for s in s_neighbors]
        for s_neighbors, t in zip(neighbors, T)
      ]

    def value(self, x: NDArray[int]) -> float:
        """
        Value oracle for the Budget Allocation problem.
        :param x: allotted budget.
        :return: expected number of influenced people
        """
        super().value(x)
        return sum((
          1 - utils.prod(
            neg_p_st ** x[s]
            for neg_p_st, s in probs_exp
          ) for probs_exp in self.probs_exp_list
        )) 

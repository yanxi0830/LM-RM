"""
LandmarkGraph object based on Fast Downward
"""
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum


class LandmarkNode:
    def __init__(self, node_id=-1, children=None, parent=None, facts=None, disj=False, conj=False,
                 in_goal=False):
        self.id = node_id
        # children, parent is a map from LandmarkNode id to EdgeType
        # i.e. {node0: gn, node1: gn, ...}
        self.children = dict() if children is None else children
        self.parent = dict() if parent is None else parent
        # TODO: sets of FactPair, not really
        self.facts = set() if facts is None else facts
        self.disjunctive = disj
        self.conjunctive = conj
        self.in_goal = in_goal

    def assign_id(self, new_id):
        assert self.id == -1 or new_id == self.id
        self.id = new_id

    def is_goal(self):
        return self.in_goal


class LandmarkGraph:
    """
    This is the actual LandmarkGraph generated from Fast Downward
    """
    def __init__(self, nodes=None):
        # set of LandmarkNodes
        self.nodes = set() if nodes is None else nodes
        self.network = nx.DiGraph()

    def _populate_graph(self):
        """
        TODO: construct the actual LandmarkGraph from node set
        :return:
        """


class EdgeType(Enum):
    """
    Note: The code relies on the fact that larger numbers are
    stronger in the sense that, e.g.,
    - every greedy-necessary ordering is also natural and reasonable
    - every necessary ordering is greedy-necessary, but not vice versa
    """
    necessary = 4
    greedy_necessary = 3
    natural = 2
    reasonable = 1
    obedient_reasonable = 0


class LandmarkStatus(Enum):
    """
    TODO: not sure how to use this yet
    """
    lm_reached = 0
    lm_not_reached = 1
    lm_needed_again = 2


if __name__ == "__main__":
    print("Hello")

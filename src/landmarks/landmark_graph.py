"""
LandmarkGraph object based on Fast Downward
"""
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
from translate.pddl import Atom, NegatedAtom


class LandmarkNode:
    def __init__(self, node_id=-1, children=None, parent=None, facts=None, disj=False, conj=False,
                 in_goal=False):
        self.id = node_id
        # children, parent is a map from LandmarkNode id to EdgeType
        # i.e. {node0: gn, node1: gn, ...}
        self.children = dict() if children is None else children
        self.parent = dict() if parent is None else parent
        # TODO: only support Atom / NegatedAtom for now, need to add Conjunction/Disjunction later
        self.facts = facts
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
        self._populate_network()

    def _populate_network(self):
        """
        Construct the actual LandmarkGraph from node set
        :return:
        """
        for n in self.nodes:
            self.network.add_node(n.id)
            for child_id in n.children.keys():
                self.network.add_edge(n.id, child_id, attr=n.children[child_id])


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


def parse_node(line):
    data = eval(line)
    facts_list = data[3]
    # ['NegatedAtom delivered-coffee()']
    facts = set()
    for fact_str in facts_list:
        space_idx = fact_str.find(' ')
        bracket_idx = fact_str.find('(')
        fact_type = fact_str[:space_idx]
        pred_id = fact_str[space_idx+1:bracket_idx]
        args_list = list(eval(fact_str[bracket_idx:]))

        expression = "{}('{}', {})".format(fact_type, pred_id, args_list)
        literal = eval(expression)
        facts.add(literal)

    return LandmarkNode(data[0], data[1], data[2], facts, data[3], data[4], data[5])


if __name__ == "__main__":
    nodes = set()
    f = open('../../domains/office/landmark.txt')
    lines = [l for l in f]
    f.close()

    for l in lines:
        nodes.add(parse_node(l))

    lm_graph = LandmarkGraph(nodes)

    nx.draw_networkx(lm_graph.network, pos=nx.shell_layout(lm_graph.network), with_labels=True)
    # nx.draw_networkx_edge_labels(lm_graph.network, pos=nx.shell_layout(lm_graph.network))

    plt.show()
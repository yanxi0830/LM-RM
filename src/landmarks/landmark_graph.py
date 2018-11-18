"""
LandmarkGraph object based on Fast Downward
"""
import networkx as nx
import matplotlib.pyplot as plt
from enum import Enum
from translate.pddl import Atom, NegatedAtom


class LandmarkNode:
    def __init__(self, node_id=-1, children=None, parents=None, facts=None, disj=False, conj=False,
                 in_goal=False):
        """
        Initialize LandmarkNode

        :param node_id: unique id per graph
        :param children: map from LandmarkNode id to EdgeType for each children dict{int:int}
                         i.e. {node0: gn, node1: gn, ...}
        :param parents: map from LandmarkNode id to EdgeType for each parents
        :param facts: set of Atom/NegatedAtom (parameterized)
        :param disj: True iff facts are disjunction
        :param conj: True iff facts are conjunction
        :param in_goal: True iff this is a goal Landmark
        """
        self.id = node_id
        self.children = dict() if children is None else children
        self.parents = dict() if parents is None else parents
        # list of Atom / NegatedAtom for now, maybe need Conjunction/Disjunction later
        self.facts = facts
        self.disjunctive = disj
        self.conjunctive = conj
        self.in_goal = in_goal

    def assign_id(self, new_id):
        assert self.id == -1 or new_id == self.id
        self.id = new_id

    def in_init(self):
        """
        Return True iff in the initial state
        :return: boolean
        """
        # check if the node has no parent orderings
        return len(self.parents) == 0

    def __repr__(self):
        return "LandmarkNode({}, {})".format(self.id, self.facts)


class LandmarkGraph:
    """
    This is the actual LandmarkGraph generated from Fast Downward
    """
    def __init__(self, lm_file='landmark.txt'):
        self.lm_file = lm_file          # spec file path
        self.nodes = dict()              # map from LandmarkNode id to LandmarkNode
        self.network = nx.DiGraph()     # easy visual graph

        self._load_landmark_nodes()
        self._populate_network()

    def _populate_network(self):
        """
        Populate nx.DiGraph from sets of nodes
            - nodes: landmark ids
            - edge: EdgeType
        """
        for n_id, n in self.nodes.items():
            self.network.add_node(n.id)
            for child_id in n.children.keys():
                self.network.add_edge(n.id, child_id, attr=n.children[child_id])

    @staticmethod
    def _parse_node(line):
        """
        Return LandmarkNode from a line in landmark.txt
        :param line: str
        :return: LandmarkNode
        """
        data = eval(line)
        facts_list = data[3]
        # ['NegatedAtom delivered-coffee()'] -> set of literals
        facts = set()
        for fact_str in facts_list:
            space_idx = fact_str.find(' ')
            bracket_idx = fact_str.find('(')
            fact_type = fact_str[:space_idx]
            pred_id = fact_str[space_idx + 1:bracket_idx]
            args_list = fact_str[bracket_idx + 1:-1].split(',')
            args_list = [x.strip() for x in args_list]

            expression = "{}('{}', {})".format(fact_type, pred_id, args_list)
            literal = eval(expression)
            facts.add(literal)

        return LandmarkNode(data[0], data[1], data[2], facts, data[4], data[5], data[6])

    def _load_landmark_nodes(self):
        f = open(self.lm_file)
        lines = [l for l in f]
        f.close()

        for l in lines:
            node = self._parse_node(l)
            self.nodes[node.id] = node

    def show_network(self):
        """
        Plot the LandmarkGraph
        """
        nx.draw_networkx(self.network, pos=nx.shell_layout(self.network), with_labels=True)
        plt.show()

    def merge_nodes(self, n1_id, n2_id):
        """
        Update graph with n1 merged with n2 (use n1_id as the final id)
        """
        # update node dict
        n1 = self.nodes[n1_id]
        n2 = self.nodes[n2_id]
        if n1.disjunctive or n2.disjunctive:
            print(n1.disjunctive)
            print(n2.disjunctive)
            raise NotImplementedError("Cannot merge disjunctive nodes")

        new_children = {**n1.children, **n2.children}
        new_parents = {**n1.parents, **n2.parents}
        new_facts = n1.facts.union(n2.facts)
        new_in_goal = n1.in_goal and n2.in_goal
        new_node = LandmarkNode(n1.id, new_children, new_parents, new_facts, disj=False, conj=True, in_goal=new_in_goal)

        # update children dict of parents of new_node
        for p_id in new_node.parents:
            parent = self.nodes[p_id]
            if n2_id in parent.children: parent.children.pop(n2.id)
            parent.children[n1_id] = new_node.parents[p_id]

        # update parent dict of children of new_node
        for c_id in new_node.children:
            child = self.nodes[c_id]
            if n2_id in child.parents: child.parents.pop(n2_id)
            child.parents[n1_id] = new_node.children[c_id]

        # update network
        self.network = nx.contracted_nodes(self.network, n1_id, n2_id)
        return new_node

    def merge_init_nodes(self):
        merge_id = -1
        to_remove = set()
        for n_id, n in self.nodes.items():
            if n.in_init():
                if merge_id == -1:
                    merge_id = n.id
                    continue
                # print("merging {} {}".format(merge_id, n_id))
                new_node = self.merge_nodes(merge_id, n_id)
                to_remove.add(n_id)

        # update self.nodes
        self.nodes[merge_id] = new_node
        for n_id in to_remove:
            self.nodes.pop(n_id)


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
    TODO: not sure if this is useful
    """
    lm_reached = 0
    lm_not_reached = 1
    lm_needed_again = 2


# if __name__ == "__main__":
    # lm_graph = LandmarkGraph('../../domains/office/landmark.txt')
    # lm_graph.show_network()
    # lm_graph.merge_init_nodes()
    # lm_graph.show_network()
    # lm_graph.show_network()

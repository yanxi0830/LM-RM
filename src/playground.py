import networkx as nx
import matplotlib.pyplot as plt


def custom_landmark_graph():
    landmarks = {
        6: "NegatedAtom have-gold()",
        0: "Atom have-gold()",
        1: "NegatedAtom have-bridge()",
        7: "Atom have-bridge()",
        4: "NegatedAtom have-iron()",
        2: "Atom have-iron()",
        5: "NegatedAtom have-wood()",
        3: "Atom have-wood()"
    }

    network = nx.DiGraph()
    for n in landmarks.keys():
        network.add_node(n)

    network.add_edge(6, 0, attr="gn")
    network.add_edge(1, 7, attr="gn")
    network.add_edge(7, 0, attr="gn")
    network.add_edge(4, 2, attr="gn")
    network.add_edge(2, 0, attr="nat")
    network.add_edge(2, 7, attr="gn")
    network.add_edge(5, 3, attr="gn")
    network.add_edge(3, 0, attr="nat")
    network.add_edge(3, 7, attr="gn")

    nx.draw_networkx(network, pos=nx.shell_layout(network), with_labels=True)
    nx.draw_networkx_edge_labels(network, pos=nx.shell_layout(network))

    # new_network = nx.contracted_nodes(network, 1, 6)
    # new_network = nx.contracted_nodes(new_network, 1, 5)
    # new_network = nx.contracted_nodes(new_network, 1, 4)
    #
    # nx.draw_networkx(new_network, pos=nx.shell_layout(new_network), with_labels=True)
    # nx.draw_networkx_edge_labels(new_network, pos=nx.shell_layout(new_network))

    plt.show()


if __name__ == "__main__":
    custom_landmark_graph()

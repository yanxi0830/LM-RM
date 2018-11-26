class PolicyGraph:
    def __init__(self, props, children):
        self.props = props
        self.children = children
        self.state = None

    def add_child(self, policy, new_prop):
        """
        Add a possible policy that can be executed.

        :param policy: (rm_id, state_id, desired_transition)
        :param new_prop: we want to take this action with the policy
        :return: child PolicyGraph
        """
        child = PolicyGraph(self.props + [new_prop], dict())
        self.children[policy] = child
        return child

    def flatten_all_paths(self, paths=None, current_path=None):
        if paths is None:
            paths = []
        if current_path is None:
            current_path = []

        if len(self.children) == 0:
            paths.append(current_path)
        else:
            for c in self.children:
                self.children[c].flatten_all_paths(paths, list(current_path + [c]))
        return paths

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.props) + "\n"
        for child in self.children:
            ret += self.children[child].__str__(level + 1)
        return ret


if __name__ == "__main__":

    root = PolicyGraph([], dict())
    c1 = root.add_child((0, 0), 'e')
    c2 = root.add_child((2, 0), 'e')
    c1.add_child((1, 0), 'f')
    c1.add_child((3, 0), 'f')
    c2.add_child((1, 0), 'f')
    c3.add_child(())

    print(root)
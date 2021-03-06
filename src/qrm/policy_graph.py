import numpy as np
import copy


class PolicyGraph:
    def __init__(self, policy, props, children, parent_state=None, current_state=None, new_task_state=None, action_order=[]):
        self.policy = policy
        self.props = props  # paths of props from root to current node
        self.children = children
        self.cost = 0  # cost from root to this node

        self.action_order = action_order
        # game states
        self.parent_state = copy.deepcopy(parent_state)
        self.current_state = copy.deepcopy(current_state)
        self.new_task_state = new_task_state
        self.parent = None

    def set_parent(self, node):
        self.parent = node

    def add_child(self, policy, new_prop):
        """
        Add a possible policy that can be executed.

        :param policy: (rm_id, state_id, desired_transition)
        :param new_prop: we want to take this action with the policy
        :return: child PolicyGraph
        """
        child = PolicyGraph(policy, self.props + [new_prop], dict(),
                            parent_state=self.current_state, action_order=self.action_order)
        child.set_parent(self)
        self.children[policy] = child
        return child

    def expand_children(self, reward_machines, action_prop):
        """
        Expand and return possible next policies
        Return list of (rm_id, state_id, action_prop) that can be used to execute action_prop
        :param reward_machines: list of reward machines
        :param action_prop: new prop
        :return: child nodes added
        """
        ret = []
        for rm_id, rm in enumerate(reward_machines):
            state_id = rm.get_state_with_transition(action_prop)
            if state_id is not None:
                ret.append(self.add_child((rm_id, state_id, action_prop), action_prop))

        return ret

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

    def get_policy_sequence(self):
        ret = []
        ret.append(self.policy)
        parent = self.parent
        while parent is not None and parent.parent is not None:
            ret.append(parent.policy)
            parent = parent.parent

        ret.reverse()
        return ret

    def update_cost(self, cost):
        self.cost = cost + self.parent.cost

    def save_game_state(self, state, new_task_state):
        self.current_state = copy.deepcopy(state)
        self.new_task_state = new_task_state

    def update_action_orders(self, action_order):
        self.action_order = action_order

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

    print(root.props)

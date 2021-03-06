"""
Utility functions for integrating RewardMachine with LandmarkGraph
"""
from landmarks.landmark_graph import *
from landmarks.planner_utils import *
from reward_machines.reward_machine import RewardMachine
from reward_machines.reward_functions import ConstantRewardFunction
import copy


def compute_rm_from_graph(lm_graph, merge_init_nodes=True):
    """
    Method 1
    - Each non-init landmark corresponds to RM (with terminal state)
    - Edge in each RM corresponds to actions needed to take (ideally only one action for nearest landmark)
    - RM only reflects the necessary orderings, not partially-ordered

    :param lm_graph: LandmarkGraph
    :param merge_init_nodes: bool
    :return: set of RewardMachine
    """
    if merge_init_nodes:
        lm_graph.merge_init_nodes()

    # For each landmark node that is not the initial state, create a RM for it
    reward_machines = set()
    for n_id, n in lm_graph.nodes.items():
        if not n.in_init():
            # initialize empty RewardMachine
            new_rm = RewardMachine()
            # populate the RewardMachine from bottom up
            openlist = list([n])
            while len(openlist) != 0:
                curr_node = openlist.pop(0)
                # add current state
                new_rm.add_state_with_landmarks(n_id, copy.copy(curr_node))

                # look at parent landmarks that must be achieved before current landmark,
                for p_id in curr_node.parents:
                    # add a transition from parent to current
                    reward = 0
                    if curr_node == n:
                        reward = 1
                        new_rm.set_terminal_state(curr_node.id)

                    new_rm.add_transition(p_id, n_id, 'TODO', ConstantRewardFunction(reward))
                    openlist.append(lm_graph.nodes[p_id])

                if len(curr_node.parents) == 0:
                    # this is the initial state
                    new_rm.set_initial_state(curr_node.id)

                if len(curr_node.children) == 0:
                    # this is the terminal state
                    new_rm.set_terminal_state(curr_node.id)

            new_rm.get_txt_representation()
            reward_machines.add(new_rm)

    return reward_machines


def compute_rm_from_graph2(lm_graph, world, merge_init_nodes=True):
    """
    Method 2
    - Iterate each non-init landmark, compute partial-ordered-plan to reach each landmark
    - Convert POP to RewardMachine for each landmark
            - See planner_utils.py
    :param lm_graph: LandmarkGraph
    :param merge_init_nodes: bool
    :return: set of RewardMachine
    """
    if merge_init_nodes:
        lm_graph.merge_init_nodes()

    reward_machines = set()
    for n_id, n, in lm_graph.nodes.items():
        if not n.in_init():
            if n.disjunctive:
                raise NotImplementedError("Disjunctive Facts:", n.facts)

            strict = True if world != "craftworld" else False

            new_rm = get_partial_ordered_rm(lm_graph.file_params, n, world, strict=strict)

            reward_machines.add(new_rm)

    return reward_machines

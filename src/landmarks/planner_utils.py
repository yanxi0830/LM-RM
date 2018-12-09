"""
Compute plans, PDDL-RM port
"""
from translate.pddl.custom_utils import parse_output_ipc
from pop_module.linearizer import linearize
from pop_module.lifter import lift_POP
from pop_module.mip import encode_POP_v2
from translate.pddl_parser.pddl_file import parse_pddl_file
from translate.pddl_parser.parsing_functions import list_to_acc
import networkx as nx
from translate.pddl.custom_utils import write_file
from reward_machines.reward_machine import RewardMachine
from reward_machines.reward_functions import ConstantRewardFunction
import os
from subprocess import call
import matplotlib.pyplot as plt
from landmarks.action_mappings import action_to_prop

CLEANPLAN = "~/git/LM-RM/scripts/cleanplan.sh"


def get_partial_ordered_rm(file_params, lm_node):
    lm_prob_file = construct_problem_file(file_params, lm_node)
    # execute cleanplan.sh to get sequential plan
    # PyCharm bug, running this in PyCharm gives UTF CodecError...run with command line or do some pre-processing
    os.system("{} {}".format(CLEANPLAN, lm_prob_file))

    # use sequential plan, domain, lm_prob to get POP
    lm_plan_file = os.path.splitext(lm_prob_file)[0] + ".plan"
    pop = compute_pop(file_params.domain_file, lm_prob_file, lm_plan_file)
    rm_net = pop_to_rm_network(pop)
    # nx.draw_networkx(rm_net, pos=nx.shell_layout(rm_net), with_labels=False)
    # nx.draw_networkx_edge_labels(rm_net, pos=nx.shell_layout(rm_net))
    # plt.show()
    rm = rm_net_to_reward_machine(rm_net)
    spec = rm.get_txt_representation()
    lm_rm_file = os.path.dirname(file_params.domain_file) + "/lm_reward_machines/" + str(
        lm_node.facts_as_filename()) + ".txt"

    write_file(lm_rm_file, spec)
    # print(lm_rm_file)
    # print(spec)
    return rm


def construct_problem_file(file_params, lm_node, save_path=None):
    """
    Given goal facts, construct a pddl problem file

    :param file_params: FileParams
    :param lm_node: LandmarkNode
    :param save_path: path file to save to, default save as domain file
    :return: problem file path
    """
    task_file = file_params.task_file
    task_pddl = parse_pddl_file("task", task_file)
    goal_facts = lm_node.facts

    new_goal = ['and']
    for fact in goal_facts:
        new_goal.append(fact.as_list())

    task_pddl[5][1] = new_goal
    acc = list_to_acc(task_pddl)

    if save_path is None:
        save_path = file_params.landmark_tasks[lm_node.id]

    write_file(save_path, '\n'.join(acc))

    return save_path


def compute_pop(domain_file, prob_file, plan_file):
    """
    Given path to domain.pddl, task.pddl, task.plan, return POP object

    :param domain_file: str
    :param prob_file: str
    :return: POP
    """
    plan = parse_output_ipc(plan_file)

    seq_pop = lift_POP(domain_file, prob_file, plan, True)
    pop_good = encode_POP_v2(domain_file, prob_file, seq_pop, None, 'pop.txt')

    return pop_good


def compute_linearized_plans(pop):
    plans = linearize(pop)
    # map action to propositions
    plan_props = set()
    for linear_plan in plans:
        # remove init/goal
        plan_actions = linear_plan[1:-1]
        plan_props.add(tuple(map(lambda x: action_to_prop(str(x)), plan_actions)))
    return plan_props


def pop_to_rm_network(pop):
    """
    Return visual network for RewardMachine from pop plan
    """
    network = nx.DiGraph()
    plans = linearize(pop)
    print(plans)

    for linear_plan in plans:
        for i, action in enumerate(linear_plan[:-1]):
            state = frozenset(linear_plan[:i + 1])
            if state not in network:
                network.add_node(state)
            if i != 0:
                prev_state = frozenset(linear_plan[:i])
                network.add_edge(prev_state, state, attr=action)

    return network


def rm_net_to_reward_machine(rm_net):
    rm = RewardMachine()
    node2id = dict()
    for i, node in enumerate(rm_net.nodes()):
        rm.add_state(i)
        node2id[node] = i

    for node in rm_net.nodes():
        # no parent, initial state
        if len(list(rm_net.predecessors(node))) == 0:
            rm.set_initial_state(node2id[node])

        selfloop = []
        for child in rm_net.successors(node):
            action = rm_net.get_edge_data(node, child)['attr']
            event_prop = action_to_prop(str(action))
            if event_prop in selfloop:
                selfloop.pop(selfloop.index(event_prop))
            else:
                selfloop.append('!' + str(event_prop))
            reward = 0
            if len(list(rm_net.successors(child))) == 0:
                # child is terminal, get reward 1
                reward = 1
            rm.add_transition(node2id[node], node2id[child], event_prop, ConstantRewardFunction(reward))

        # add self loop
        if len(list(rm_net.successors(node))) == 0:
            # no children, terminal state
            rm.set_terminal_state(node2id[node])
        else:
            rm.add_transition(node2id[node], node2id[node], '&'.join(selfloop), ConstantRewardFunction(0))

    return rm


def compute_and_save_rm_spec(domain_file, prob_file, plan_file, rm_file_dest, render=False):
    pop = compute_pop(domain_file, prob_file, plan_file)
    task_rm_net = pop_to_rm_network(pop)
    if render:
        nx.draw_networkx(task_rm_net, pos=nx.shell_layout(task_rm_net), with_labels=False)
        nx.draw_networkx_edge_labels(task_rm_net, pos=nx.shell_layout(task_rm_net))
        plt.show()

    task_rm = rm_net_to_reward_machine(task_rm_net)
    spec = task_rm.get_txt_representation()

    write_file(rm_file_dest, spec)

    return pop


def save_sequential_rm_spec(domain_file, prob_file, plan_file, rm_file_dest, render=False):
    plan = parse_output_ipc(plan_file)
    seq_pop = lift_POP(domain_file, prob_file, plan, True)

    task_rm_net = pop_to_rm_network(seq_pop)

    if render:
        nx.draw_networkx(task_rm_net, pos=nx.shell_layout(task_rm_net), with_labels=False)
        nx.draw_networkx_edge_labels(task_rm_net, pos=nx.shell_layout(task_rm_net))
        plt.show()

    task_rm = rm_net_to_reward_machine(task_rm_net)
    spec = task_rm.get_txt_representation()

    write_file(rm_file_dest, spec)

    return seq_pop
# if __name__ == "__main__":
#     domain_file = "../../domains/office/domain.pddl"
#     prob_file = "../../domains/office/t3.pddl"
#     plan_file = "../../domains/office/t3.plan"
#     rm_file_dest = "../../experiments/office/reward_machines/new_task1.txt"
#
#     compute_and_save_rm_spec(domain_file, prob_file, plan_file, rm_file_dest, render=False)

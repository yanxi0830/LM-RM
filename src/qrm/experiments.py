import random, time
import tensorflow as tf
from worlds.game import *
from qrm.policy_bank_dqn import PolicyBankDQN
from common.replay_buffer import create_experience_replay_buffer
from tester.saver import Saver
from reward_machines.reward_machine import RewardMachine
from qrm.qrm import *
from qrm.policy_graph import PolicyGraph
from worlds.game_objects import Actions
import logging
import copy

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run_qrm_save_model(alg_name, tester, curriculum, num_times, show_print):
    learning_params = tester.learning_params

    time_init = time.time()
    for n in range(num_times):
        random.seed(n)
        sess = tf.Session()

        curriculum.restart()
        # Creating the experience replay buffer
        prioritized_replay_beta_iters = learning_params.prioritized_replay_beta_iters
        if learning_params.prioritized_replay_beta_iters is None:
            prioritized_replay_beta_iters = curriculum.total_steps

        replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size,
                                                                       learning_params.prioritized_replay,
                                                                       learning_params.prioritized_replay_alpha,
                                                                       learning_params.prioritized_replay_beta0,
                                                                       prioritized_replay_beta_iters)

        # Creating policy bank
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions = len(task_aux.get_actions())

        policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, tester.get_reward_machines())

        # Task loop
        while not curriculum.stop_learning():
            rm_file = curriculum.get_next_task()
            run_qrm_task(sess, rm_file, policy_bank, tester, curriculum, replay_buffer, beta_schedule, show_print)

        # Save session
        if task_aux.params.game_type == "craftworld":
            save_model_path = '../model/' + str(task_aux.params.game_type) + '/' + task_aux.game.get_map_id() + '/' + str(alg_name)
        else:
            save_model_path = '../model/' + str(task_aux.params.game_type) + '/' + str(alg_name)

        print("Saving model to {} ...".format(save_model_path))
        saver = tf.train.Saver()
        saver.save(sess, save_model_path)

    print("Time:", "%0.2f" % ((time.time() - time_init) / 60), "mins")


def build_policy_graph(prop_order, reward_machines):
    """
    Given linearized plan sequence, build the graph of the possible policy
    sequence that can be chosen
    :param prop_order:
    :param reward_machines:
    :return: PolicyGraph
    """
    root = PolicyGraph([], dict())
    children = [root]
    for p in prop_order:
        next_level_children = []
        for rm_id, rm in enumerate(reward_machines):
            state_id = rm.get_state_with_transition(p)
            if state_id is not None:
                for c in children:
                    next_level_child = c.add_child((rm_id, state_id, p), p)
                    next_level_children.append(next_level_child)
        children = next_level_children

    return root


def search_policy(prop_order, tester, curriculum, new_task_rm, reward_machines, policy_bank, bound=np.inf):
    """
    Given a linearized plan sequence, do a exhaustive search over the sequence of RM policies
    to execute that achieves most optimal cost.
    :param prop_order: sequence of high-level actions
    :return: cost, sequence of policies (RM-id, state_id)
    """
    policy_graph = build_policy_graph(prop_order, reward_machines)
    print(policy_graph)
    # Follow the graph during execution to calculate the cost
    all_policies = policy_graph.flatten_all_paths([])
    min_costs = np.full(len(all_policies), np.inf)
    for j, p in enumerate(copy.deepcopy(all_policies)):
        task = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task.get_features())

        new_task_u1 = new_task_rm.get_initial_state()
        s1, s1_features = task.get_state_and_features()
        curr_policy = None
        for t in range(tester.testing_params.num_steps):
            if curr_policy is None:
                curr_policy = p.pop(0)
            curr_policy_rm = reward_machines[curr_policy[0]]
            a = policy_bank.get_best_action(curr_policy[0], curr_policy[1],
                                            s1_features.reshape((1, num_features)),
                                            add_noise=False)
            task.execute_action(a)
            s2, s2_features = task.get_state_and_features()
            curr_policy_u2 = curr_policy_rm.get_next_state(curr_policy[1], task.get_true_propositions())
            new_task_u2 = new_task_rm.get_next_state(new_task_u1, task.get_true_propositions())

            desired_next_state = curr_policy_rm.get_next_state(curr_policy[1], curr_policy[2])
            if curr_policy_u2 == desired_next_state:
                logger.info("EXECUTED ACTION {}, SWITCHING POLICIES".format(curr_policy[2]))
                curr_policy = None
            elif curr_policy_u2 == curr_policy[1]:
                logger.info("STILL FOLLOWING CURRENT POLICY {}, DON'T SWITCH".format(curr_policy[2]))
            else:
                logger.info("OOPS, WRONG WAY, PRUNE THIS OPTION")
                break

            if task.is_env_game_over() or t + 1 >= np.min(min_costs) or t + 1 >= bound:
                break

            if new_task_rm.is_terminal_state(new_task_u2):
                print("NEW COMPOSED TASK FINISHED WITH {}".format(all_policies[j]))
                print("STEPS:", t + 1)
                min_costs[j] = t + 1
                min_idx = np.argmin(min_costs)
                # TEMPORARY WORKAROUND FOR keyboardworld, stop searching since policies are the same..
                # return min_costs[min_idx], all_policies[min_idx]
                break
            else:
                s1, s1_features = s2, s2_features
                new_task_u1 = new_task_u2

    min_idx = np.argmin(min_costs)
    return min_costs[min_idx], all_policies[min_idx]


def dfs_search_policy():
    """
    TODO:
    :return:
    """
    raise NotImplementedError()


def load_model_and_test_composition(alg_name, tester, curriculum, num_times, new_task, show_print):
    for n in range(num_times):
        random.seed(n)
        sess = tf.Session()

        curriculum.restart()

        # Initialize a policy_bank graph to be loaded with saved model
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions = len(task_aux.get_actions())
        policy_bank = PolicyBankDQN(sess, num_actions, num_features,
                                    tester.learning_params, tester.get_reward_machines())

        # Load the model
        saver = tf.train.Saver()

        # Get path
        if task_aux.params.game_type == "craftworld":
            save_model_path = '../model/' + str(
                task_aux.params.game_type) + '/' + task_aux.game.get_map_id()
        else:
            save_model_path = '../model/' + str(task_aux.params.game_type)

        saver.restore(sess, tf.train.latest_checkpoint(save_model_path))

        reward_machines = tester.get_reward_machines()
        print("Loaded {} policies (RMs)".format(len(reward_machines)))

        # partial-ordered RM of new task
        new_task_rm = RewardMachine(new_task.rm_file)
        linearized_plans = new_task.get_linearized_plan()
        print(linearized_plans)
        least_cost = float('inf')
        best_policy = []  # list of (rm_id, state_id) corresponding to each action

        for i, curr_plan in enumerate(linearized_plans):
            # Get the least cost path for the current linearized plan
            cost, switching_seq = search_policy(curr_plan, tester, curriculum, new_task_rm, reward_machines, policy_bank, bound=least_cost)
            if cost < tester.testing_params.num_steps:
                print(cost, switching_seq)
            if cost < least_cost:
                least_cost = cost
                best_policy = switching_seq

        # Execute the best policy
        print("Executing Best Policy...{} ({} steps)".format(best_policy, least_cost))
        task = Game(tester.get_task_params(curriculum.get_current_task()))
        new_task_u1 = new_task_rm.get_initial_state()
        s1, s1_features = task.get_state_and_features()
        r_total = 0
        curr_policy = None
        for t in range(int(least_cost)):
            task.render()
            if curr_policy is None:
                curr_policy = best_policy.pop(0)
            curr_policy_rm = reward_machines[curr_policy[0]]

            a = policy_bank.get_best_action(curr_policy[0], curr_policy[1],
                                            s1_features.reshape((1, num_features)),
                                            add_noise=False)
            print("Action:", Actions(a))
            task.execute_action(a)

            s2, s2_features = task.get_state_and_features()
            new_task_u2 = new_task_rm.get_next_state(new_task_u1, task.get_true_propositions())

            curr_policy_u2 = curr_policy_rm.get_next_state(curr_policy[1], task.get_true_propositions())
            desired_next_state = curr_policy_rm.get_next_state(curr_policy[1], curr_policy[2])
            if curr_policy_u2 == desired_next_state:
                logger.info("EXECUTED ACTION {}, SWITCHING POLICIES".format(curr_policy[2]))
                curr_policy = None

            r = new_task_rm.get_reward(new_task_u1, new_task_u2, s1, a, s2)
            r_total += r * tester.learning_params.gamma ** t

            s1, s1_features = s2, s2_features
            new_task_u1 = new_task_u2
        task.render()
        print("Rewards:", r_total)

###################################################################################################################
#
# Quick testing functions, need cleanup
#
###################################################################################################################
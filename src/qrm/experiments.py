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
from worlds.mouse_world import Colors
import logging
import copy
import pygame

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def run_qrm_save_model(alg_name, tester, curriculum, num_times, show_print):
    learning_params = tester.learning_params
    json_saver = Saver(alg_name, tester, curriculum)

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
            save_model_path = '../model/' + str(
                task_aux.params.game_type) + '/' + task_aux.game.get_map_id() + '/' + str(alg_name)
        else:
            save_model_path = '../model/' + str(task_aux.params.game_type) + '/' + str(alg_name)

        print("Saving model to {} ...".format(save_model_path))
        saver = tf.train.Saver()
        saver.save(sess, save_model_path)

        tf.reset_default_graph()
        sess.close()

        # Backing up the results
        json_saver.save_results()

    tester.show_results()
    print("Time:", "%0.2f" % ((time.time() - time_init) / 60), "mins")


def dfs_search_policy(prop_order, tester, curriculum, new_task_rm, reward_machines, policy_bank, bound=np.inf):
    """
    Given a linearized plan sequence, do dfs with pruning over the sequence of RM policies
    to execute that achieves most optimal cost.
    :param prop_order: sequence of high-level actions
    :return: cost, sequence of policies (RM-id, state_id)
    """
    game = Game(tester.get_task_params(curriculum.get_current_task()))

    root = PolicyGraph(tuple(), [], dict(), current_state=game, new_task_state=new_task_rm.get_initial_state())
    open_list = [root]
    action_idx = 0
    least_cost = bound
    least_cost_path = []

    action_order = list(prop_order)

    while len(open_list) != 0 and action_idx < len(action_order):
        curr_node = open_list.pop()
        action_idx = 0 if len(action_order) == 1 else len(curr_node.props)

        if len(curr_node.props) == 0:  # cost for root is 0
            p = action_order[action_idx]  # next level
            next_level_children = curr_node.expand_children(reward_machines, p)
            open_list.extend(next_level_children)
            continue

        # don't expand
        if curr_node != root and (curr_node.cost == np.inf or curr_node.cost > least_cost):
            continue

        # execute the current policy to complete action
        cost, game_state, new_task_u2, r, bonus_events = execute_policy_and_get_cost(curr_node, reward_machines,
                                                                                     policy_bank, tester, new_task_rm,
                                                                                     curr_node.parent.new_task_state,
                                                                                     least_cost)

        for b in bonus_events:
            if b in action_order:
                action_order.remove(b)

        # cost to execute action from parent state
        curr_node.save_game_state(game_state, new_task_u2)
        curr_node.update_cost(cost)

        if cost == np.inf or curr_node.cost > least_cost:
            continue

        if new_task_rm.is_terminal_state(new_task_u2) and r > 0:
            if curr_node.cost < least_cost:
                least_cost = curr_node.cost
                least_cost_path = curr_node.get_policy_sequence()

        if action_idx < len(action_order):
            p = action_order[action_idx]  # next level
            next_level_children = curr_node.expand_children(reward_machines, p)
            open_list.extend(next_level_children)

    return least_cost, least_cost_path


def execute_policy_and_get_cost(curr_node, reward_machines, policy_bank, tester, new_task_rm, new_task_u1,
                                bound=np.inf):
    """
    Explore on the environment under current policy to complete curr_action to get actual cost
    :param curr_policy: PolicyGraph node
    :param curr_action:
    :param reward_machines:
    :param policy_bank:
    :param bound:
    :return: cost, game_state, new_task_state
    """
    game = copy.deepcopy(curr_node.parent_state)
    num_features = len(game.get_features())
    s1, s1_features = game.get_state_and_features()
    curr_policy = curr_node.policy
    curr_policy_rm = reward_machines[curr_policy[0]]

    bonus = []
    for t in range(tester.testing_params.num_steps):
        a = policy_bank.get_best_action(curr_policy[0], curr_policy[1],
                                        s1_features.reshape((1, num_features)), add_noise=False)
        game.execute_action(a)
        # game.render()
        s2, s2_features = game.get_state_and_features()
        curr_policy_u2 = curr_policy_rm.get_next_state(curr_policy[1], game.get_true_propositions())
        new_task_u2 = new_task_rm.get_next_state(new_task_u1, game.get_true_propositions())

        desired_next_state = curr_policy_rm.get_next_state(curr_policy[1], curr_policy[2])

        r = new_task_rm.get_reward(new_task_u1, new_task_u2, s1, a, s2)
        if curr_policy_u2 == desired_next_state:
            logger.info("EXECUTED ACTION {}, CAN GO TO NEXT LEVEL".format(curr_policy[2]))
            return t + 1, game, new_task_u2, r, bonus
        elif curr_policy_u2 == curr_policy[1]:
            logger.info("STILL FOLLOWING CURRENT POLICY {}, CONTINUE".format(curr_policy[2]))
            if new_task_u2 != new_task_u1:
                logger.info(
                    "ENCOUNTERED EVENT {} WHILE FOLLOWING {}".format(game.get_true_propositions(), curr_policy[2]))
                bonus.append(game.get_true_propositions())
        else:
            logger.info("OOPS, WRONG WAY, PRUNE THIS OPTION")
            return np.inf, game, new_task_u1, r, bonus

        if game.is_env_game_over() or t + 1 >= bound:
            return np.inf, game, new_task_u2, r, bonus

        s1, s1_features = s2, s2_features
        new_task_u1 = new_task_u2

    return np.inf, game, new_task_u1, 0, bonus


def get_qrm_generalization_performance(alg_name, tester, curriculum, num_times, new_tasks, show_print):
    """
    Testing all the tasks in new_tasks and return the success rate and cumulative reward
    """
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
    if task_aux.params.game_type == "farmworld":
        save_model_path = '../model/' + str(task_aux.params.game_type) + '/' + task_aux.game.get_map_id()
    else:
        save_model_path = '../model/' + str(task_aux.params.game_type)
    saver.restore(sess, tf.train.latest_checkpoint(save_model_path))

    reward_machines = tester.get_reward_machines()
    print("Loaded {} policies (RMs)".format(len(reward_machines)))

    success_count = 0
    all_task_rewards = []

    for new_task in new_tasks:
        # partial-ordered RM of new task
        new_task_rm = RewardMachine(new_task.rm_file)
        linearized_plans = new_task.get_linearized_plan()
        print("There are {} possible linearized plans: {}".format(len(linearized_plans), linearized_plans))
        least_cost = float('inf')
        best_policy = []  # list of (rm_id, state_id) corresponding to each action

        for i, curr_plan in enumerate(linearized_plans):
            # Get the least cost path for the current linearized plan
            cost, switching_seq = dfs_search_policy(curr_plan, tester, curriculum, new_task_rm, reward_machines,
                                                    policy_bank, bound=least_cost)
            if cost < least_cost:
                print(cost, switching_seq)
                least_cost = cost
                best_policy = switching_seq
                # finding optimal takes too long, end early if find a solution
                break

        # Couldn't solve the task
        if least_cost == np.inf:
            print("Failed to execute this task: {}".format(new_task))
            r_total = 0.0
            all_task_rewards.append(r_total)
            continue

        # Execute the best policy
        print("Executing Best Policy...{} ({} steps)".format(best_policy, least_cost))
        task = Game(tester.get_task_params(curriculum.get_current_task()))
        new_task_u1 = new_task_rm.get_initial_state()
        s1, s1_features = task.get_state_and_features()
        r_total = 0
        curr_policy = None

        for t in range(int(least_cost)):
            if show_print:
                task.render()
            if curr_policy is None:
                curr_policy = best_policy.pop(0)
            curr_policy_rm = reward_machines[curr_policy[0]]

            a = policy_bank.get_best_action(curr_policy[0], curr_policy[1],
                                            s1_features.reshape((1, num_features)),
                                            add_noise=False)
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
        if show_print:
            task.render()
        print("Rewards:", r_total)

        all_task_rewards.append(r_total)
        if r_total > 0:
            success_count += 1

    success_rate = float(success_count) / len(new_tasks)
    acc_reward = sum(all_task_rewards)
    print(all_task_rewards)
    return success_rate, acc_reward


def load_model_and_test_composition(alg_name, tester, curriculum, num_times, new_task, show_print):
    """
    Testing a single task (see run_new_task.py)
    TODO: refactor with get_qrm_generalization_performance
    """
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
            save_model_path = '../model/' + str(task_aux.params.game_type) + '/' + task_aux.game.get_map_id()
        else:
            save_model_path = '../model/' + str(task_aux.params.game_type)

        saver.restore(sess, tf.train.latest_checkpoint(save_model_path))

        reward_machines = tester.get_reward_machines()
        print("Loaded {} policies (RMs)".format(len(reward_machines)))

        # partial-ordered RM of new task
        new_task_rm = RewardMachine(new_task.rm_file)
        linearized_plans = new_task.get_linearized_plan()
        print("There are {} possible linearized plans: {}".format(len(linearized_plans), linearized_plans))
        least_cost = float('inf')
        best_policy = []  # list of (rm_id, state_id) corresponding to each action

        for i, curr_plan in enumerate(linearized_plans):
            # Get the least cost path for the current linearized plan
            # cost, switching_seq = search_policy(curr_plan, tester, curriculum, new_task_rm, reward_machines,
            #                                     policy_bank, bound=least_cost)
            cost, switching_seq = dfs_search_policy(curr_plan, tester, curriculum, new_task_rm, reward_machines,
                                                    policy_bank, bound=least_cost)
            if cost < least_cost:
                print(cost, switching_seq)
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
            if show_print:
                task.render()
            if curr_policy is None:
                curr_policy = best_policy.pop(0)
            curr_policy_rm = reward_machines[curr_policy[0]]

            a = policy_bank.get_best_action(curr_policy[0], curr_policy[1],
                                            s1_features.reshape((1, num_features)),
                                            add_noise=False)
            if show_print: print("Action:", Actions(a))
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
        if show_print:
            task.render()
        print("Rewards:", r_total)

        return r_total


###################################################################################################################
#
# Quick testing functions, need cleanup
#
###################################################################################################################


def build_policy_graph(prop_order, reward_machines):
    """
    Given linearized plan sequence, build the graph of the possible policy
    sequence that can be chosen
    :param prop_order:
    :param reward_machines:
    :return: PolicyGraph
    """
    root = PolicyGraph(tuple(), [], dict())
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

            r = new_task_rm.get_reward(new_task_u1, new_task_u2, s1, a, s2)
            if new_task_rm.is_terminal_state(new_task_u2) and r > 0:
                print("NEW COMPOSED TASK FINISHED WITH {}".format(all_policies[j]))
                print("STEPS:", t + 1)
                min_costs[j] = t + 1
                min_idx = np.argmin(min_costs)
                # TEMPORARY WORKAROUND FOR keyboardworld, stop searching since policies are the same
                # return min_costs[min_idx], all_policies[min_idx]
                break
            elif new_task_rm.is_terminal_state(new_task_u2) and r == 0:
                # dead-end
                break
            else:
                s1, s1_features = s2, s2_features
                new_task_u1 = new_task_u2

    min_idx = np.argmin(min_costs)
    return min_costs[min_idx], all_policies[min_idx]


def render_mouseworld_task(alg_name, tester, curriculum, num_times, new_task, show_print):
    random.seed(0)
    sess = tf.Session()

    curriculum.restart()

    task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
    num_features = len(task_aux.get_features())
    num_actions = len(task_aux.get_actions())
    policy_bank = PolicyBankDQN(sess, num_actions, num_features,
                                tester.learning_params, tester.get_reward_machines())

    # Load the model
    saver = tf.train.Saver()

    # Get path
    if task_aux.params.game_type == "craftworld":
        save_model_path = '../model/' + str(task_aux.params.game_type) + '/' + task_aux.game.get_map_id()
    else:
        save_model_path = '../model/' + str(task_aux.params.game_type)

    saver.restore(sess, tf.train.latest_checkpoint(save_model_path))

    reward_machines = tester.get_reward_machines()
    print("Loaded {} policies (RMs)".format(policy_bank.get_number_of_policies()))

    # partial-ordered RM of new task
    new_task_rm = RewardMachine(new_task.rm_file)
    linearized_plans = new_task.get_linearized_plan()
    print("There are {} possible linearized plans: {}".format(len(linearized_plans), linearized_plans))
    least_cost = float('inf')
    best_policy = []  # list of (rm_id, state_id) corresponding to each action

    for i, curr_plan in enumerate(linearized_plans):
        # Get the least cost path for the current linearized plan
        # cost, switching_seq = search_policy(curr_plan, tester, curriculum, new_task_rm, reward_machines,
        #                                     policy_bank, bound=least_cost)
        cost, switching_seq = dfs_search_policy(curr_plan, tester, curriculum, new_task_rm, reward_machines,
                                                policy_bank, bound=least_cost)
        if cost < least_cost:
            print(cost, switching_seq)
            least_cost = cost
            best_policy = switching_seq

    # Execute the best policy
    print("Executing Best Policy...{} ({} steps)".format(best_policy, least_cost))
    task = Game(tester.get_task_params(curriculum.get_current_task()))
    new_task_u1 = new_task_rm.get_initial_state()
    s1, s1_features = task.get_state_and_features()
    r_total = 0
    curr_policy = None

    pygame.init()
    gameDisplay = pygame.display.set_mode((task.game.params.max_x, task.game.params.max_y))
    pygame.display.set_caption('Fake Keyboard')
    clock = pygame.time.Clock()

    for t in range(int(least_cost)):
        if curr_policy is None:
            curr_policy = best_policy.pop(0)
        curr_policy_rm = reward_machines[curr_policy[0]]

        a = policy_bank.get_best_action(curr_policy[0], curr_policy[1],
                                        s1_features.reshape((1, num_features)),
                                        add_noise=False)
        print("Action:", Actions(a))
        task.execute_action(a)

        gameDisplay.fill(Colors.WHITE.value)
        task.game.agent.draw_on_display(gameDisplay)
        # for k in task.game.keyboard_keys:
        #     k.draw_on_display(gameDisplay, letters=False)
        task.game.draw_current_text_on_display(gameDisplay)
        pygame.display.update()
        clock.tick(30)

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

    print("Rewards:", r_total)

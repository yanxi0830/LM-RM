import random, time
import tensorflow as tf
from worlds.game import *
from qrm.policy_bank_dqn import PolicyBankDQN
from common.replay_buffer import create_experience_replay_buffer
from tester.saver import Saver
from reward_machines.reward_machine import RewardMachine
from qrm.qrm import *
from qrm.policy_graph import PolicyGraph

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
        save_model_path = '../model/' + str(task_aux.params.game_type) + '/' + str(alg_name)
        print("Saving model to {} ...".format(save_model_path))
        saver = tf.train.Saver()
        saver.save(sess, save_model_path)

    print("Time:", "%0.2f" % ((time.time() - time_init) / 60), "mins")


def load_model_and_test_composition(alg_name, tester, curriculum, num_times, show_print):
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
        saver.restore(sess, tf.train.latest_checkpoint('../model/' + str(task_aux.params.game_type)))

        reward_machines = tester.get_reward_machines()
        print("Loaded {} policies (RMs)".format(len(reward_machines)))

        # partial-ordered RM of new task
        # TODO: integrate with planner_utils.py
        new_task_rm = RewardMachine('../experiments/office/reward_machines/new_task.txt')
        linearized_plans = [['e', 'f', 'g'], ['f', 'e', 'g']]
        least_cost = float('inf')
        best_policy = []  # list of (rm_id, state_id) corresponding to each action

        def build_policy_graph(prop_order):
            """
            Given linearized plan sequence, build the graph of the possible policy
            sequence that can be chosen
            :param prop_order:
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

        def search_policy(prop_order):
            """
            Given a linearized plan sequence, do a (DFS) search over the sequence of RM policies
            to execute that achieves most optimal cost
            :param prop_order:
            :return: cost, sequence of policies (RM-id, state_id)
            """
            policy_graph = build_policy_graph(prop_order)
            print(policy_graph)
            # Follow the graph during execution to calculate the cost
            all_policies = policy_graph.flatten_all_paths([])
            min_costs = [float('inf')] * len(all_policies)
            for p in all_policies.copy():
                task = Game(tester.get_task_params(curriculum.get_current_task()))
                new_task_u1 = new_task_rm.get_initial_state()
                s1, s1_features = task.get_state_and_features()
                for t in range(tester.testing_params.num_steps):
                    curr_policy = p.pop(0)
                    a = policy_bank.get_best_action(curr_policy[0], curr_policy[1],
                                                    s1_features.reshape((1, num_features)),
                                                    add_noise=False)
                    task.execute_action(a)
                    s2, s2_features = task.get_state_and_features()
                    curr_policy_u2 = curr_policy.get_next_state(curr_policy[1], task.get_true_propositions())
                    new_task_u2 = new_task_rm.get_next_state(new_task_u1, task.get_true_propositions())

                    # desired_next_state = curr_policy.get_next_state(curr_policy[1], )


            return 0, []

        for i, curr_plan in enumerate(linearized_plans):
            # Get the least cost path for the current linearized plan
            cost, switching_seq = search_policy(curr_plan)
            if cost < least_cost:
                least_cost = cost
                best_policy = switching_seq

        # Execute the best policy




        # simulator for running new task
        # task = Game(tester.get_task_params(curriculum.get_current_task()))
        # s1, s1_features = task.get_state_and_features()







###################################################################################################################
#
# Quick testing functions, need cleanup
#
###################################################################################################################


# def train_and_save_qrm(alg_name, tester, curriculum, num_times, show_print):
#     learning_params = tester.learning_params
#
#     time_init = time.time()
#     for n in range(num_times):
#         random.seed(n)
#         sess = tf.Session()
#
#         curriculum.restart()
#
#         # Creating the experience replay buffer
#         replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size,
#                                                                        learning_params.prioritized_replay,
#                                                                        learning_params.prioritized_replay_alpha,
#                                                                        learning_params.prioritized_replay_beta0,
#                                                                        curriculum.total_steps if learning_params.prioritized_replay_beta_iters is None else learning_params.prioritized_replay_beta_iters)
#
#         # Creating policy bank
#         task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
#         num_features = len(task_aux.get_features())
#         num_actions = len(task_aux.get_actions())
#
#         policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, tester.get_reward_machines())
#
#         # Task loop
#         while not curriculum.stop_learning():
#             rm_file = curriculum.get_next_task()
#             run_qrm_task(sess, rm_file, policy_bank, tester, curriculum, replay_buffer, beta_schedule, show_print)
#
#         # Save session
#         print("Saving model...")
#         saver = tf.train.Saver()
#         saver.save(sess, '../model/my-model')
#
#         reward_machines = tester.get_reward_machines()
#
#         task = Game(tester.get_task_params('PLACEHOLDER'))
#         # task.game.agent = (7, 3)
#         s1, s1_features = task.get_state_and_features()
#
#         for t in range(tester.testing_params.num_steps):
#             task.render()
#             a = policy_bank.get_best_action(0, 0, s1_features.reshape((1, num_features)), add_noise=False)
#             print("Action:", a)
#             # Executing the action
#             task.execute_action(a)
#             s2, s2_features = task.get_state_and_features()
#
#             s1, s1_features = s2, s2_features
#
#
# def load_and_test(tester, curriculum, num_times, show_print):
#     learning_params = tester.learning_params
#
#     tf.reset_default_graph()
#     curriculum.restart()
#
#     with tf.Session() as sess:
#         # saver = tf.train.import_meta_graph('../model/my-model.meta')
#
#         # Creating the experience replay buffer
#         replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size,
#                                                                        learning_params.prioritized_replay,
#                                                                        learning_params.prioritized_replay_alpha,
#                                                                        learning_params.prioritized_replay_beta0,
#                                                                        curriculum.total_steps if learning_params.prioritized_replay_beta_iters is None else learning_params.prioritized_replay_beta_iters)
#
#         # Creating policy bank
#         task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
#         num_features = len(task_aux.get_features())
#         num_actions = len(task_aux.get_actions())
#
#         policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, tester.get_reward_machines())
#
#         saver = tf.train.Saver()
#         saver.restore(sess, tf.train.latest_checkpoint('../model'))
#
#         task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
#
#         reward_machines = tester.get_reward_machines()
#         policy_b = reward_machines[5]
#
#         task = Game(tester.get_task_params('PLACEHOLDER'))
#         task.game.agent = (1, 7)
#         s1, s1_features = task.get_state_and_features()
#
#         for t in range(tester.testing_params.num_steps):
#             task.render()
#             a = policy_bank.get_best_action(1, 0, s1_features.reshape((1, num_features)), add_noise=False)
#
#             print("Action:", a)
#             # Executing the action
#             task.execute_action(a)
#             s2, s2_features = task.get_state_and_features()
#
#             s1, s1_features = s2, s2_features
#
#
# def run_qrm_and_save_policy(alg_name, tester, curriculum, num_times, show_print):
#     learning_params = tester.learning_params
#
#     # Running the tasks 'num_times'
#     for t in range(num_times):
#         # Setting the random seed to 't'
#         random.seed(t)
#         sess = tf.Session()
#
#         # Reseting default values
#         curriculum.restart()
#
#         # Creating the experience replay buffer
#         replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size,
#                                                                        learning_params.prioritized_replay,
#                                                                        learning_params.prioritized_replay_alpha,
#                                                                        learning_params.prioritized_replay_beta0,
#                                                                        curriculum.total_steps if learning_params.prioritized_replay_beta_iters is None else learning_params.prioritized_replay_beta_iters)
#
#         # Creating policy bank
#         task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
#         num_features = len(task_aux.get_features())
#         num_actions = len(task_aux.get_actions())
#
#         policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, tester.get_reward_machines())
#
#         # Task loop
#         while not curriculum.stop_learning():
#             rm_file = curriculum.get_next_task()
#             # Running 'task_rm_id' for one episode
#             run_qrm_task(sess, rm_file, policy_bank, tester, curriculum, replay_buffer, beta_schedule, show_print)
#
#         # I think this can be used to test on a new task
#         # Starting interaction with the environment
#         # TODO: run the new task i.e. get reward of 1 when reaching goal propositions
#         print("Done! Computed {} policies (RMs)".format(len(tester.get_reward_machines())))
#         reward_machines = tester.get_reward_machines()
#         rm1 = reward_machines[1]  # task of delivering mail
#         rm3 = reward_machines[3]  # task of delivering coffee
#         landmark_rms = [1, 3]  # task_rm_id we are following
#
#         task = Game(tester.get_task_params('PLACEHOLDER'))
#         s1, s1_features = task.get_state_and_features()
#
#         # the initial state of all tasks
#         u1s = [reward_machines[rm].get_initial_state() for rm in landmark_rms]
#
#         curr_task = landmark_rms.pop(0)  # current RM we are following
#         curr_u1 = u1s.pop(0)
#         r_total = 0
#         for t in range(tester.testing_params.num_steps):
#             # Choosing an action using the right policy from current RM
#             a = policy_bank.get_best_action(curr_task, curr_u1, s1_features.reshape((1, num_features)), add_noise=False)
#
#             # Executing the action
#             task.execute_action(a)
#             s2, s2_features = task.get_state_and_features()
#             curr_u2 = reward_machines[curr_task].get_next_state(curr_u1, task.get_true_propositions())
#             u2s = []  # remaining RM states
#             for i, rm_id in enumerate(landmark_rms):
#                 u2s.append(reward_machines[rm_id].get_next_state(u1s[i], task.get_true_propositions()))
#
#             r = reward_machines[curr_task].get_reward(curr_u1, curr_u2, s1, a, s2)
#
#             other_rewards = []
#             for i, rm_id in enumerate(landmark_rms):
#                 other_rewards.append(reward_machines[rm_id].get_reward(u1s[i], u2s[i], s1, a, s2))
#
#             r_total += r * learning_params.gamma ** t
#
#             # Check game over
#             if task.is_env_game_over():
#                 break
#
#             # Check if current landmark is reached
#             if reward_machines[curr_task].is_terminal_state(curr_u2):
#                 if len(landmark_rms) == 0:
#                     print("New Composed Task Finished!")
#                     print("Steps:", t + 1)
#                     print("Rewards:", r_total)
#                     break
#
#                 # Go to next landmark (no dead-end so terminal state == goal)
#                 curr_task = landmark_rms.pop(0)
#                 curr_u1 = u1s.pop()
#             else:
#                 # Moving to the next state
#                 s1, s1_features, curr_u1 = s2, s2_features, curr_u2
#
#         print("ALL FINISHED")
#
#
def parallel_composition_test(alg_name, tester, curriculum, num_times, show_print):
    learning_params = tester.learning_params

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
        saver.restore(sess, tf.train.latest_checkpoint('../model/' + str(task_aux.params.game_type)))

        reward_machines = tester.get_reward_machines()
        print("Loaded {} policies (RMs)".format(len(reward_machines)))

        # partial-ordered RM of new task
        new_task_rm = RewardMachine('../experiments/office/reward_machines/new_task.txt')
        new_task_u1 = new_task_rm.get_initial_state()

        # simulator for running new task
        task = Game(tester.get_task_params(curriculum.get_current_task()))
        s1, s1_features = task.get_state_and_features()

        # the initial state of all learned policies
        u1s = [rm.get_initial_state() for rm in reward_machines]
        curr_policy = None  # current RM we are following
        r_total = 0
        old_high_level_prop = None

        for t in range(tester.testing_params.num_steps):
            task.render()
            # Get the current high-level action we want to execute
            if curr_policy is None:
                high_level_prop = new_task_rm.get_random_next_prop(new_task_u1)
                old_high_level_prop = high_level_prop
            else:
                high_level_prop = old_high_level_prop

            # From pool of learned RMs, pick one that **possibly** executes the high-level action
            # TODO: might not execute another path of be stuck infinite loop
            if curr_policy is None:
                for i, rm in enumerate(reward_machines):
                    next_state = rm.get_next_state(u1s[i], high_level_prop)
                    if next_state != rm.u_broken and next_state != u1s[i]:
                        print("Got a RM to follow because {} have transition {}".format(i, high_level_prop))
                        curr_policy = rm
                        # break

            # WE'RE STUCK
            if curr_policy is None:
                print("STUCK! Want to take " + high_level_prop)
                print(u1s)
                # Option 1: Update other RM states after taking each action
                #           - i.e. break apart the policy [coffee->office] into [coffee] and [office]
                # Option 2: Re-plan

            # Follow this current policy until reaches the desired next state
            curr_policy_id = reward_machines.index(curr_policy)
            a = policy_bank.get_best_action(curr_policy_id, u1s[curr_policy_id],
                                            s1_features.reshape((1, num_features)),
                                            add_noise=False)

            task.execute_action(a)
            s2, s2_features = task.get_state_and_features()

            # u2s = []    # all learned policies new states
            # for i, rm in enumerate(reward_machines):
            #     u2s.append(rm.get_next_state(u1s[i], task.get_true_propositions()))

            curr_policy_u2 = curr_policy.get_next_state(u1s[curr_policy_id], task.get_true_propositions())
            new_task_u2 = new_task_rm.get_next_state(new_task_u1, task.get_true_propositions())

            # check if current policy successfully executed the selected action
            desired_next_state = curr_policy.get_next_state(u1s[curr_policy_id], high_level_prop)
            if curr_policy_u2 == desired_next_state:
                print("EXECUTED ACTION {}, SWITCHING POLICIES".format(high_level_prop))
                # If a small policy reaches terminal, reset it to the initial state to be re-used
                if reward_machines[curr_policy_id]._is_terminal(curr_policy_u2):
                    curr_policy_u2 = reward_machines[curr_policy_id].u0
                curr_policy = None
            elif curr_policy_u2 == u1s[curr_policy_id]:
                print("Follow current policy, don't switch")
            else:
                print("Oops, wrong way")

            r = new_task_rm.get_reward(new_task_u1, new_task_u2, s1, a, s2)

            r_total += r * learning_params.gamma ** t

            if task.is_env_game_over():
                break

            if new_task_rm.is_terminal_state(new_task_u2):
                print("New Composed Task Finished!")
                print("Steps:", t + 1)
                print("Rewards:", r_total)
                break

            else:
                s1, s1_features = s2, s2_features
                new_task_u1 = new_task_u2
                u1s[curr_policy_id] = curr_policy_u2

        print("ALL FINISHED")

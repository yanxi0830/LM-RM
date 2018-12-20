import numpy as np
import random, time
import tensorflow as tf
from worlds.game import *
from qrm.policy_bank_dqn import PolicyBankDQN
from common.schedules import LinearSchedule
from common.replay_buffer import create_experience_replay_buffer
from tester.saver import Saver
from os import listdir
from os.path import isfile, join
from reward_machines.reward_machine import RewardMachine
from baselines.hrl import MetaController
from qrm.experiments import dfs_search_policy
from worlds.game_objects import Actions


def run_hrl_baseline(sess, rm_file, meta_controllers, options, policy_bank, tester, curriculum, replay_buffer,
                     beta_schedule, show_print):
    """
    Strategy:
        - I'll learn a tabular metacontroller over the posible subpolicies
        - Initialice a regular policy bank with eventually subpolicies (e.g. Fa, Fb, Fc, Fb, Fd)
        - Learn as usual
        - Pick actions using the sequence
    """

    # Initializing parameters
    learning_params = tester.learning_params
    testing_params = tester.testing_params
    reward_machines = tester.get_reward_machines()
    rm_id = tester.get_reward_machine_id_from_file(rm_file)
    task_params = tester.get_task_params(rm_file)
    task = Game(task_params)
    actions = task.get_actions()
    num_features = len(task.get_features())
    meta_controller = meta_controllers[rm_id]
    rm = reward_machines[rm_id]
    num_steps = learning_params.max_timesteps_per_task
    training_reward = 0

    # Starting interaction with the environment
    if show_print: print("Executing", num_steps, "actions...")
    t = 0
    curriculum_stop = False

    # Getting the initial state of the environment and the reward machine
    s1, s1_features = task.get_state_and_features()
    u1 = rm.get_initial_state()

    while t < learning_params.max_timesteps_per_task and not curriculum_stop:
        # selecting a macro action from the meta controller
        mc_s1, mc_s1_features, mc_u1 = s1, s1_features, u1
        mc_r = []
        mc_a = meta_controller.get_action_epsilon_greedy(mc_s1_features, mc_u1)
        mc_option = meta_controller.get_option(mc_a)  # tuple <rm_id,u_0>
        mc_done = False
        if show_print: print(mc_option)

        # The selected option must be executed at least one step (i.e. len(mc_r) == 0)
        while len(mc_r) == 0 or not meta_controller.finish_option(mc_a, task.get_true_propositions()):

            # Choosing an action to perform
            if random.random() < 0.1:
                a = random.choice(actions)
            else:
                a = policy_bank.get_best_action(mc_option[0], mc_option[1], s1_features.reshape((1, num_features)))

            # updating the curriculum
            curriculum.add_step()

            # Executing the action
            task.execute_action(a)
            s2, s2_features = task.get_state_and_features()
            events = task.get_true_propositions()
            u2 = rm.get_next_state(u1, events)
            reward = rm.get_reward(u1, u2, s1, a, s2)
            training_reward += reward

            # updating the reward for the meta controller
            mc_r.append(reward)

            # Getting rewards and next states for each option
            rewards, next_states = [], []
            for j in range(len(options)):
                j_rewards, j_next_states = options[j].get_rewards_and_next_states(s1, a, s2, events)
                rewards.append(j_rewards)
                next_states.append(j_next_states)
            # Mapping rewards and next states to specific policies in the policy bank
            rewards = policy_bank.select_rewards(rewards)
            next_policies = policy_bank.select_next_policies(next_states)

            # Adding this experience to the experience replay buffer
            replay_buffer.add(s1_features, a, s2_features, rewards, next_policies)

            # Learning
            if curriculum.get_current_step() > learning_params.learning_starts and curriculum.get_current_step() % learning_params.train_freq == 0:
                if learning_params.prioritized_replay:
                    experience = replay_buffer.sample(learning_params.batch_size,
                                                      beta=beta_schedule.value(curriculum.get_current_step()))
                    S1, A, S2, Rs, NPs, weights, batch_idxes = experience
                else:
                    S1, A, S2, Rs, NPs = replay_buffer.sample(learning_params.batch_size)
                    weights, batch_idxes = None, None
                abs_td_errors = policy_bank.learn(S1, A, S2, Rs, NPs, weights)  # returns the absolute td_error
                if learning_params.prioritized_replay:
                    new_priorities = abs_td_errors + learning_params.prioritized_replay_eps
                    replay_buffer.update_priorities(batch_idxes, new_priorities)

            # Updating the target network
            if curriculum.get_current_step() > learning_params.learning_starts and curriculum.get_current_step() % learning_params.target_network_update_freq == 0:
                policy_bank.update_target_network()

            # Printing
            if show_print and (t + 1) % learning_params.print_freq == 0:
                print("Step:", t + 1, "\tTotal reward:", training_reward)

            # Testing
            if testing_params.test and curriculum.get_current_step() % testing_params.test_freq == 0:
                tester.run_test(curriculum.get_current_step(), sess, run_hrl_baseline_test, meta_controllers,
                                policy_bank, num_features)

            # Restarting the environment (Game Over)
            if task.is_env_game_over() or rm.is_terminal_state(u2):
                # Restarting the game
                task = Game(task_params)
                s2, s2_features = task.get_state_and_features()
                u2 = rm.get_initial_state()

                mc_done = True

                if curriculum.stop_task(t):
                    curriculum_stop = True

            # checking the steps time-out
            if curriculum.stop_learning():
                curriculum_stop = True

            # Moving to the next state
            s1, s1_features, u1 = s2, s2_features, u2

            t += 1
            if t == learning_params.max_timesteps_per_task or curriculum_stop or mc_done:
                break

        # learning on the meta controller
        mc_s2, mc_s2_features, mc_u2 = s1, s1_features, u1
        mc_reward = _get_discounted_reward(mc_r, learning_params.gamma)
        mc_steps = len(mc_r)

        meta_controller.learn(mc_s1_features, mc_u1, mc_a, mc_reward, mc_s2_features, mc_u2, mc_done, mc_steps)

    # meta_controller.show()
    # input()


def _get_discounted_reward(r_all, gamma):
    dictounted_r = 0
    for r in r_all[::-1]:
        dictounted_r = r + gamma * dictounted_r
    return dictounted_r


def run_hrl_baseline_test(sess, reward_machines, task_params, rm_id, learning_params, testing_params, meta_controllers,
                          policy_bank, num_features):
    # Initializing parameters
    meta_controller = meta_controllers[rm_id]
    task = Game(task_params)
    rm = reward_machines[rm_id]
    s1, s1_features = task.get_state_and_features()
    u1 = rm.get_initial_state()

    # Starting interaction with the environment
    r_total = 0
    t = 0
    while t < testing_params.num_steps:
        # selecting a macro action from the meta controller
        mc_s1, mc_s1_features, mc_u1 = s1, s1_features, u1
        mc_a = meta_controller.get_best_action(mc_s1_features, mc_u1)
        mc_option = meta_controller.get_option(mc_a)  # tuple <rm_id,u_0>

        # The selected option must be executed at least one step
        first = True
        while first or not meta_controller.finish_option(mc_a, task.get_true_propositions()):
            first = False

            # Choosing an action to perform
            a = policy_bank.get_best_action(mc_option[0], mc_option[1], s1_features.reshape((1, num_features)))

            # Executing the action
            task.execute_action(a)
            s2, s2_features = task.get_state_and_features()
            events = task.get_true_propositions()
            u2 = rm.get_next_state(u1, events)
            reward = rm.get_reward(u1, u2, s1, a, s2)
            r_total += reward * learning_params.gamma ** t

            # Moving to the next state
            s1, s1_features, u1 = s2, s2_features, u2

            t += 1
            # Restarting the environment (Game Over)
            if task.is_env_game_over() or rm.is_terminal_state(u2) or t == testing_params.num_steps:
                return r_total

    return 0


def _get_option_files(folder):
    return [f.replace(".txt", "") for f in listdir(folder) if isfile(join(folder, f))]


def run_hrl_save_model(alg_name, tester, curriculum, num_times, show_print, use_rm):
    """
        NOTE: To implement this baseline, we encode each option as a reward machine with one transition
        - use_rm: Indicates whether to prune options using the reward machine
    """

    # Setting up the saver
    json_saver = Saver(alg_name, tester, curriculum)
    learning_params = tester.learning_params

    # Running the tasks 'num_times'
    time_init = time.time()
    for t in range(num_times):

        # Setting the random seed to 't'
        random.seed(t)
        sess = tf.Session()

        # Reseting default values
        curriculum.restart()

        # Creating the experience replay buffer
        replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size,
                                                                       learning_params.prioritized_replay,
                                                                       learning_params.prioritized_replay_alpha,
                                                                       learning_params.prioritized_replay_beta0,
                                                                       curriculum.total_steps if learning_params.prioritized_replay_beta_iters is None else learning_params.prioritized_replay_beta_iters)

        # Loading options for this experiment
        option_folder = "../experiments/%s/options/" % tester.get_world_name()

        options = []  # NOTE: The policy bank also uses this list (in the same order)
        option2file = []
        for option_file in _get_option_files(
                option_folder):  # NOTE: The option id indicates what the option does (e.g. "a&!n")
            option = RewardMachine(join(option_folder, option_file + ".txt"))
            options.append(option)
            option2file.append(option_file)

        # getting num inputs and outputs net
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions = len(task_aux.get_actions())

        # initializing the meta controllers (one metacontroller per task)
        meta_controllers = []
        reward_machines = tester.get_reward_machines()
        for i in range(len(reward_machines)):
            rm = reward_machines[i]
            num_states = len(rm.get_states())
            policy_name = "Reward_Machine_%d" % i
            mc = MetaController(sess, policy_name, options, option2file, rm, use_rm, learning_params, num_features,
                                num_states, show_print)
            meta_controllers.append(mc)

        # initializing the bank of policies with one policy per option
        policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, options)
        # Task loop
        while not curriculum.stop_learning():
            if show_print: print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
            rm_file = curriculum.get_next_task()

            # Running 'rm_file' for one episode
            run_hrl_baseline(sess, rm_file, meta_controllers, options, policy_bank, tester, curriculum, replay_buffer,
                             beta_schedule, show_print)

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

    # Showing results
    tester.show_results()
    print("Time:", "%0.2f" % ((time.time() - time_init) / 60), "mins")


def load_hrl_and_test_composition(alg_name, tester, curriculum, num_times, new_task, show_print):
    learning_params = tester.learning_params

    for n in range(num_times):
        random.seed(n)
        sess = tf.Session()

        curriculum.restart()

        # Initialize a policy_bank and meta-controller graph to be loaded with saved model

        # Loading options for this experiment
        option_folder = "../experiments/%s/options/" % tester.get_world_name()

        options = []  # NOTE: The policy bank also uses this list (in the same order)
        option2file = []
        for option_file in _get_option_files(
                option_folder):  # NOTE: The option id indicates what the option does (e.g. "a&!n")
            option = RewardMachine(join(option_folder, option_file + ".txt"))
            options.append(option)
            option2file.append(option_file)

        # getting num inputs and outputs net
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions = len(task_aux.get_actions())

        # initializing the meta controllers (one metacontroller per task)
        meta_controllers = []
        reward_machines = tester.get_reward_machines()
        for i in range(len(reward_machines)):
            rm = reward_machines[i]
            num_states = len(rm.get_states())
            policy_name = "Reward_Machine_%d" % i
            mc = MetaController(sess, policy_name, options, option2file, rm, True, learning_params, num_features,
                                num_states, show_print)
            meta_controllers.append(mc)

        # initializing the bank of policies with one policy per option
        policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, options)

        saver = tf.train.Saver()
        # Get path
        if task_aux.params.game_type == "craftworld":
            save_model_path = '../model/' + str(task_aux.params.game_type) + '/' + task_aux.game.get_map_id()
        else:
            save_model_path = '../model/' + str(task_aux.params.game_type)

        saver.restore(sess, tf.train.latest_checkpoint(save_model_path))

        print("Loaded {} policies (RMs)".format(policy_bank.policies))

        # partial-ordered RM of new task
        new_task_rm = RewardMachine(new_task.rm_file)
        linearized_plans = new_task.get_linearized_plan()
        print("There are {} possible linearized plans: {}".format(len(linearized_plans), linearized_plans))
        least_cost = float('inf')
        best_policy = []  # list of (rm_id, state_id) corresponding to each action

        for i, curr_plan in enumerate(linearized_plans):
            # Get the least cost path for the current linearized plan
            cost, switching_seq = execute_option_sequence(curr_plan, tester, curriculum, new_task_rm, options,
                                                          option2file, policy_bank, bound=least_cost)
            if cost < least_cost:
                print(cost, switching_seq)
                least_cost = cost
                best_policy = switching_seq

        # Execute the best policy
        print("Executing Best Policy...{} ({} steps)".format(best_policy, least_cost))
        # task = Game(tester.get_task_params(curriculum.get_current_task()))
        # new_task_u1 = new_task_rm.get_initial_state()
        # s1, s1_features = task.get_state_and_features()
        # r_total = 0
        # curr_policy = None
        # for t in range(int(least_cost)):
        #     task.render()
        #     if curr_policy is None:
        #         curr_policy = best_policy.pop(0)
        #     curr_policy_rm = reward_machines[curr_policy[0]]
        #
        #     a = policy_bank.get_best_action(curr_policy[0], curr_policy[1],
        #                                     s1_features.reshape((1, num_features)),
        #                                     add_noise=False)
        #     print("Action:", Actions(a))
        #     task.execute_action(a)
        #
        #     s2, s2_features = task.get_state_and_features()
        #     new_task_u2 = new_task_rm.get_next_state(new_task_u1, task.get_true_propositions())
        #
        #     curr_policy_u2 = curr_policy_rm.get_next_state(curr_policy[1], task.get_true_propositions())
        #     desired_next_state = curr_policy_rm.get_next_state(curr_policy[1], curr_policy[2])
        #     if curr_policy_u2 == desired_next_state:
        #         curr_policy = None
        #
        #     r = new_task_rm.get_reward(new_task_u1, new_task_u2, s1, a, s2)
        #     r_total += r * tester.learning_params.gamma ** t
        #
        #     s1, s1_features = s2, s2_features
        #     new_task_u1 = new_task_u2
        # task.render()
        # print("Rewards:", r_total)
        #
        # return r_total


def execute_option_sequence(curr_plan, tester, curriculum, new_task_rm, options, option2file, policy_bank, bound=np.inf):
    cost = np.inf
    switch_seq = []

    game = Game(tester.get_task_params(curriculum.get_current_task()))
    num_features = len(game.get_features())
    new_task_u1 = new_task_rm.get_initial_state()
    s1, s1_features = game.get_state_and_features()
    r_total = 0

    curr_op = 0
    for t in range(tester.testing_params.num_steps):
        p = curr_plan[curr_op]
        option_id = option2file.index(p)
        curr_rm = options[option_id]
        print(option2file)
        # print(curr_rm.get_txt_representation())

        a = policy_bank.get_best_action(3, curr_rm.get_initial_state(),
                                        s1_features.reshape((1, num_features)))

        print(Actions(a))
        game.execute_action(a)
        game.render()
        s2, s2_features = game.get_state_and_features()
        new_task_u2 = new_task_rm.get_next_state(new_task_u1, game.get_true_propositions())
        curr_policy_u2 = curr_rm.get_next_state(curr_rm.get_initial_state(), game.get_true_propositions())

        if game.get_true_propositions() == p:
            curr_op += 1
            switch_seq.append((option_id, curr_rm.get_initial_state(), p))
            if curr_op >= len(curr_plan):
                break

        s1, s1_features = s2, s2_features
        new_task_u1 = new_task_u2

    return t, switch_seq


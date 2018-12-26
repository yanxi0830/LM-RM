"""
Train from Options for generalization
"""
import random, time
import tensorflow as tf
from reward_machines.reward_machine import RewardMachine
from worlds.game import *
from worlds.game import *
from os import listdir
from os.path import isfile, join
from qrm.policy_bank_dqn import PolicyBankDQN
from common.replay_buffer import create_experience_replay_buffer
from tester.saver import Saver
import numpy as np

def _get_option_files(folder):
    return [f.replace(".txt", "") for f in listdir(folder) if isfile(join(folder, f))]


def get_options_rm(tester):
    # Loading options for this experiment
    option_folder = "../experiments/%s/options/" % tester.get_world_name()

    options = []  # NOTE: The policy bank also uses this list (in the same order)
    option2file = []
    for option_file in _get_option_files(
            option_folder):  # NOTE: The option id indicates what the option does (e.g. "a&!n")
        option = RewardMachine(join(option_folder, option_file + ".txt"))
        options.append(option)
        option2file.append(option_file)

    return options, option2file


def run_options_task(sess, rm_file, curr_option_id, options, option2file, policy_bank, tester, curriculum, replay_buffer, beta_schedule, show_print):
    """
    This code runs one training episode.
        - curr_option_id: It is the path towards the option RM to solve on this episode
        - options: list of option RewardMachines
    """
    # Initializing parameters and the game
    learning_params = tester.learning_params
    testing_params = tester.testing_params
    task_params = tester.get_task_params(rm_file)
    task = Game(task_params)
    actions = task.get_actions()
    num_features = len(task.get_features())
    num_steps = learning_params.max_timesteps_per_task
    rm = options[curr_option_id]
    training_reward = 0

    s1, s1_features = task.get_state_and_features()
    u1 = rm.get_initial_state()

    for t in range(num_steps):
        if random.random() < 0.3:   # more exploration improve performance?
            a = random.choice(actions)
        else:
            a = policy_bank.get_best_action(curr_option_id, u1, s1_features.reshape((1, num_features)))

        curriculum.add_step()

        # Executing the action
        task.execute_action(a)
        s2, s2_features = task.get_state_and_features()
        events = task.get_true_propositions()
        u2 = rm.get_next_state(u1, events)
        reward = rm.get_reward(u1, u2, s1, a, s2)
        if show_print and reward > 0:
            print("REWARD {} from {} (Step {})".format(reward, events, t+1))
        training_reward += reward

        # Getting rewards and next states for each reward machine
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
            tester.run_test(curriculum.get_current_step(), sess, run_options_test, options, option2file, policy_bank, num_features)

        # Restarting the environment (Game Over)
        if task.is_env_game_over() or rm.is_terminal_state(u2):
            # Restarting the game
            task = Game(task_params)
            s2, s2_features = task.get_state_and_features()
            u2 = rm.get_initial_state()

            if curriculum.stop_task(t):
                break

        # checking the steps time-out
        if curriculum.stop_learning():
            break

        # Moving to the next state
        s1, s1_features, u1 = s2, s2_features, u2


def run_options_test(sess, reward_machines, task_params, task_rm_id, learning_params, testing_params, options, option2file, policy_bank, num_features):
    # print("POLICIES ", policy_bank.policies)
    # print(len(options))
    # print(option2file)

    task = Game(task_params)
    rm = reward_machines[task_rm_id]
    s1, s1_features = task.get_state_and_features()
    u1 = rm.get_initial_state()

    r_total = 0
    for t in range(testing_params.num_steps):
        # Choose an option to execute by one step look-ahead
        macro_action = rm.get_random_next_prop(u1)
        op_id = option2file.index(macro_action)

        # Choosing an action using the right policy
        a = policy_bank.get_best_action(op_id, options[op_id].get_initial_state(), s1_features.reshape((1, num_features)), add_noise=False)

        task.execute_action(a)
        s2, s2_features = task.get_state_and_features()
        u2 = rm.get_next_state(u1, task.get_true_propositions())
        r = rm.get_reward(u1, u2, s1, a, s2)

        r_total += r * learning_params.gamma ** t

        # Restarting the environment (Game Over)
        if task.is_env_game_over() or rm.is_terminal_state(u2):
            break

        # Moving to the next state
        s1, s1_features, u1 = s2, s2_features, u2

    return r_total


def run_options_save_model(alg_name, tester, curriculum, num_times, show_print):
    # Setting up the saver
    json_saver = Saver(alg_name, tester, curriculum)
    learning_params = tester.learning_params

    # Running the tasks num_times
    time_init = time.time()
    for t in range(num_times):
        random.seed(t)
        sess = tf.Session()

        curriculum.restart()

        # Creating the experience replay buffer
        replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size,
                                                                       learning_params.prioritized_replay,
                                                                       learning_params.prioritized_replay_alpha,
                                                                       learning_params.prioritized_replay_beta0,
                                                                       curriculum.total_steps if learning_params.prioritized_replay_beta_iters is None else learning_params.prioritized_replay_beta_iters)

        options, option2file = get_options_rm(tester)
        curr_option_id = 0
        # getting num inputs and outputs net
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions = len(task_aux.get_actions())

        # initializing the bank of policies with one policy per option
        policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, options)

        # Task loop
        while not curriculum.stop_learning():
            if show_print:
                print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)

            rm_file = curriculum.get_next_task()
            # Running 'curr_option' for one episode
            curr_option_id = (curr_option_id + 1) % len(options)

            run_options_task(sess, rm_file, curr_option_id, options, option2file, policy_bank, tester, curriculum, replay_buffer, beta_schedule, show_print)

        # Save session
        if task_aux.params.game_type != "officeworld":
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


def load_options_model_test_composition(alg_name, tester, curriculum, num_times, new_task, show_print):
    learning_params = tester.learning_params

    for n in range(num_times):
        random.seed(n)
        sess = tf.Session()
        curriculum.restart()

        options, option2file = get_options_rm(tester)
        curr_option_id = 0
        # getting num inputs and outputs net
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions = len(task_aux.get_actions())

        # initializing the bank of policies with one policy per option
        policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, options)

        # Load the model
        saver = tf.train.Saver()

        # Get path
        if task_aux.params.game_type != "officeworld":
            save_model_path = '../model/' + str(task_aux.params.game_type) + '/' + task_aux.game.get_map_id()
        else:
            save_model_path = '../model/' + str(task_aux.params.game_type)
        saver.restore(sess, tf.train.latest_checkpoint(save_model_path))

        reward_machines = tester.get_reward_machines()
        print("Loaded {} policies (options)".format(policy_bank.get_number_of_policies()))

        new_task_rm = RewardMachine(new_task.rm_file)
        linearized_plans = new_task.get_linearized_plan()
        print("There are {} possible linearized plans: {}".format(len(linearized_plans), linearized_plans))

        least_cost = float('inf')
        best_policy = []  # linearized plan
        best_reward = 0
        for i, curr_plan in enumerate(linearized_plans):
            cost, r_total = execute_plan_get_cost(curr_plan, tester, curriculum, options, option2file, policy_bank,
                                                  new_task_rm)
            if cost < least_cost:
                least_cost = cost
                best_policy = curr_plan
                best_reward = r_total

        print("Rewards", best_reward)
        print("Steps", least_cost)
        print(best_policy)


def execute_plan_get_cost(curr_plan, tester, curriculum, options, option2file, policy_bank, new_task_rm):
    task = Game(tester.get_task_params(curriculum.get_current_task()))
    num_features = len(task.get_features())
    u1 = new_task_rm.get_initial_state()
    s1, s1_features = task.get_state_and_features()

    prop_idx = 0
    r_total = 0
    for t in range(tester.testing_params.num_steps):
        macro_action = curr_plan[prop_idx]
        op_id = option2file.index(macro_action)

        # Choosing an action using the right policy
        a = policy_bank.get_best_action(op_id, options[op_id].get_initial_state(),
                                        s1_features.reshape((1, num_features)), add_noise=False)

        task.execute_action(a)
        # task.render()
        # print(Actions(a))
        s2, s2_features = task.get_state_and_features()
        events = task.get_true_propositions()
        u2 = new_task_rm.get_next_state(u1, events)
        r = new_task_rm.get_reward(u1, u2, s1, a, s2)

        r_total += r * tester.learning_params.gamma ** t

        # Restarting the environment (Game Over)
        if task.is_env_game_over():
            return np.inf, 0

        if r > 0 and new_task_rm.is_terminal_state(u2):
            return t+1, r_total

        if u2 != u1:
            prop_idx += 1

        # Moving to the next state
        s1, s1_features, u1 = s2, s2_features, u2

    return np.inf, 0


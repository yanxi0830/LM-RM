import numpy as np
import random, time
import tensorflow as tf
from worlds.game import *
from qrm.policy_bank_dqn import PolicyBankDQN
from common.schedules import LinearSchedule
from common.replay_buffer import create_experience_replay_buffer
from tester.saver import Saver
from reward_machines.reward_machine import RewardMachine


def run_qrm_task(sess, rm_file, policy_bank, tester, curriculum, replay_buffer, beta_schedule, show_print):
    """
    This code runs one training episode. 
        - rm_file: It is the path towards the RM machine to solve on this episode
    """
    # Initializing parameters and the game
    learning_params = tester.learning_params
    testing_params = tester.testing_params
    reward_machines = tester.get_reward_machines()
    task_rm_id = tester.get_reward_machine_id_from_file(rm_file)
    task_params = tester.get_task_params(rm_file)
    task = Game(task_params)
    actions = task.get_actions()
    num_features = len(task.get_features())
    num_steps = learning_params.max_timesteps_per_task
    rm = reward_machines[task_rm_id]
    training_reward = 0
    # Getting the initial state of the environment and the reward machine
    s1, s1_features = task.get_state_and_features()
    u1 = rm.get_initial_state()

    # Starting interaction with the environment
    # if show_print: print("Executing", num_steps)
    for t in range(num_steps):

        # Choosing an action to perform (more exploration)
        if random.random() < 0.1:
            a = random.choice(actions)
        else:
            a = policy_bank.get_best_action(task_rm_id, u1, s1_features.reshape((1, num_features)))

        # updating the curriculum
        curriculum.add_step()

        # Executing the action
        task.execute_action(a)
        s2, s2_features = task.get_state_and_features()
        events = task.get_true_propositions()
        u2 = rm.get_next_state(u1, events)
        reward = rm.get_reward(u1, u2, s1, a, s2)
        training_reward += reward

        # Getting rewards and next states for each reward machine
        rewards, next_states = [], []
        for j in range(len(reward_machines)):
            j_rewards, j_next_states = reward_machines[j].get_rewards_and_next_states(s1, a, s2, events)
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
            if training_reward == 1:
                print("REWARD=1")
            # print("Step:", t + 1, "\tTotal reward:", training_reward)

        # Testing
        if testing_params.test and curriculum.get_current_step() % testing_params.test_freq == 0:
            tester.run_test(curriculum.get_current_step(), sess, run_qrm_test, policy_bank, num_features)

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

    # if show_print: print("Done! Total reward:", training_reward)


def run_qrm_test(sess, reward_machines, task_params, task_rm_id, learning_params, testing_params, policy_bank,
                 num_features):
    # Initializing parameters
    task = Game(task_params)
    rm = reward_machines[task_rm_id]
    s1, s1_features = task.get_state_and_features()
    u1 = rm.get_initial_state()

    # Starting interaction with the environment
    r_total = 0
    for t in range(testing_params.num_steps):
        # Choosing an action using the right policy
        a = policy_bank.get_best_action(task_rm_id, u1, s1_features.reshape((1, num_features)), add_noise=False)

        # Executing the action
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


def run_qrm_experiments(alg_name, tester, curriculum, num_times, show_print):
    # Setting up the saver
    saver = Saver(alg_name, tester, curriculum)
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

        # Creating policy bank
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions = len(task_aux.get_actions())

        policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, tester.get_reward_machines())

        # Task loop
        while not curriculum.stop_learning():
            # if show_print: print("Current step:", curriculum.get_current_step(), "from", curriculum.total_steps)
            rm_file = curriculum.get_next_task()
            # Running 'task_rm_id' for one episode
            run_qrm_task(sess, rm_file, policy_bank, tester, curriculum, replay_buffer, beta_schedule, show_print)
        tf.reset_default_graph()
        sess.close()

        # Backing up the results
        saver.save_results()

    # Showing results
    tester.show_results()
    print("Time:", "%0.2f" % ((time.time() - time_init) / 60), "mins")


def run_qrm_and_save_policy(alg_name, tester, curriculum, num_times, show_print):
    learning_params = tester.learning_params

    # Running the tasks 'num_times'
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

        # Creating policy bank
        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions = len(task_aux.get_actions())

        policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, tester.get_reward_machines())

        # Task loop
        while not curriculum.stop_learning():
            rm_file = curriculum.get_next_task()
            # Running 'task_rm_id' for one episode
            run_qrm_task(sess, rm_file, policy_bank, tester, curriculum, replay_buffer, beta_schedule, show_print)

        # I think this can be used to test on a new task
        # Starting interaction with the environment
        # TODO: run the new task i.e. get reward of 1 when reaching goal propositions
        print("Done! Computed {} policies (RMs)".format(len(tester.get_reward_machines())))
        reward_machines = tester.get_reward_machines()
        rm1 = reward_machines[1]    # task of delivering mail
        rm3 = reward_machines[3]    # task of delivering coffee
        landmark_rms = [1, 3]   # task_rm_id we are following

        task = Game(tester.get_task_params('PLACEHOLDER'))
        s1, s1_features = task.get_state_and_features()

        # the initial state of all tasks
        u1s = [reward_machines[rm].get_initial_state() for rm in landmark_rms]

        curr_task = landmark_rms.pop(0)     # current RM we are following
        curr_u1 = u1s.pop(0)
        r_total = 0
        for t in range(tester.testing_params.num_steps):
            # Choosing an action using the right policy from current RM
            a = policy_bank.get_best_action(curr_task, curr_u1, s1_features.reshape((1, num_features)), add_noise=False)

            # Executing the action
            task.execute_action(a)
            s2, s2_features = task.get_state_and_features()
            curr_u2 = reward_machines[curr_task].get_next_state(curr_u1, task.get_true_propositions())
            u2s = []    # remaining RM states
            for i, rm_id in enumerate(landmark_rms):
                u2s.append(reward_machines[rm_id].get_next_state(u1s[i], task.get_true_propositions()))

            r = reward_machines[curr_task].get_reward(curr_u1, curr_u2, s1, a, s2)

            other_rewards = []
            for i, rm_id in enumerate(landmark_rms):
                other_rewards.append(reward_machines[rm_id].get_reward(u1s[i], u2s[i], s1, a, s2))

            r_total += r * learning_params.gamma ** t

            # Check game over
            if task.is_env_game_over():
                break

            # Check if current landmark is reached
            if reward_machines[curr_task].is_terminal_state(curr_u2):
                if len(landmark_rms) == 0:
                    print("New Composed Task Finished!")
                    print("Steps:", t+1)
                    print("Rewards:", r_total)
                    break

                # Go to next landmark (no dead-end so terminal state == goal)
                curr_task = landmark_rms.pop(0)
                curr_u1 = u1s.pop()
            else:
                # Moving to the next state
                s1, s1_features, curr_u1 = s2, s2_features, curr_u2

        print("ALL FINISHED")


def parallel_composition_test(alg_name, tester, curriculum, num_times, show_print):
    learning_params = tester.learning_params

    for t in range(num_times):
        random.seed(t)
        sess = tf.Session()

        curriculum.restart()

        replay_buffer, beta_schedule = create_experience_replay_buffer(learning_params.buffer_size,
                                                                       learning_params.prioritized_replay,
                                                                       learning_params.prioritized_replay_alpha,
                                                                       learning_params.prioritized_replay_beta0,
                                                                       curriculum.total_steps if learning_params.prioritized_replay_beta_iters is None else learning_params.prioritized_replay_beta_iters)

        task_aux = Game(tester.get_task_params(curriculum.get_current_task()))
        num_features = len(task_aux.get_features())
        num_actions = len(task_aux.get_actions())

        policy_bank = PolicyBankDQN(sess, num_actions, num_features, learning_params, tester.get_reward_machines())

        while not curriculum.stop_learning():
            rm_file = curriculum.get_next_task()
            run_qrm_task(sess, rm_file, policy_bank, tester, curriculum, replay_buffer, beta_schedule, show_print)

        print("Done! Computed {} policies (RMs)".format(len(tester.get_reward_machines())))
        reward_machines = tester.get_reward_machines()

        # partial ordered RM for task of getting mail and coffee
        new_task_rm = RewardMachine('../experiments/office/reward_machines/new_task.txt')
        new_task_u1 = new_task_rm.get_initial_state()

        task = Game(tester.get_task_params('PLACEHOLDER'))
        s1, s1_features = task.get_state_and_features()

        # the initial state of all learned policies
        u1s = [rm.get_initial_state() for rm in reward_machines]
        curr_policy = None      # current RM we are following
        r_total = 0
        old_high_level_prop = None

        for t in range(tester.testing_params.num_steps):
            # task.render()
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
                print("Steps:", t+1)
                print("Rewards:", r_total)
                break

            else:
                s1, s1_features = s2, s2_features
                new_task_u1 = new_task_u2
                u1s[curr_policy_id] = curr_policy_u2


        print("ALL FINISHED")
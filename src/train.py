import argparse
from argparser import add_training_args
from tester.tester import Tester
from tester.tester_params import TestingParameters
from common.curriculum import CurriculumLearner
from qrm.learning_params import LearningParameters
from qrm.experiments import run_qrm_save_model
from baselines.run_hrl import run_hrl_save_model
from baselines.run_dqn import run_dqn_save_model


def get_params_craft_world(experiment):
    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 10 * step_unit
    testing_params.num_steps = 1000

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.9
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.tabular_case = True
    learning_params.max_timesteps_per_task = testing_params.num_steps

    # This are the parameters that tabular q-learning would use to work as 'tabular q-learning'
    learning_params.lr = 1
    learning_params.batch_size = 1
    learning_params.learning_starts = 1
    learning_params.buffer_size = 1

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps
    curriculum.total_steps = 1000 * step_unit
    curriculum.min_steps = 1

    print("Craft World ----------")
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("num_steps:", testing_params.num_steps)
    print("total_steps:", curriculum.total_steps)

    return testing_params, learning_params, tester, curriculum


def get_params_office_world(experiment):
    step_unit = 500

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 1 * step_unit
    testing_params.num_steps = 1000

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.9
    learning_params.tabular_case = True
    learning_params.max_timesteps_per_task = testing_params.num_steps

    # This are the parameters that tabular q-learning would use to work as 'tabular q-learning'
    learning_params.lr = 1
    learning_params.batch_size = 1
    learning_params.learning_starts = 1
    learning_params.buffer_size = 1

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = 1000  # 1000
    curriculum.total_steps = 1000 * step_unit
    curriculum.min_steps = 1

    print("Office World ----------")
    print("TRAIN gamma:", learning_params.gamma)
    print("tabular_case:", learning_params.tabular_case)
    print("num_steps:", testing_params.num_steps)
    print("total_steps:", curriculum.total_steps)

    return testing_params, learning_params, tester, curriculum


def get_params_keyboard_world(experiment):
    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 10 * step_unit
    testing_params.num_steps = 1000

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.gamma = 0.9
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.tabular_case = True
    learning_params.max_timesteps_per_task = testing_params.num_steps

    # This are the parameters that tabular q-learning would use to work as 'tabular q-learning'
    learning_params.lr = 1
    learning_params.batch_size = 1
    learning_params.learning_starts = 1
    learning_params.buffer_size = 1

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = testing_params.num_steps
    curriculum.total_steps = 1000 * step_unit
    curriculum.min_steps = 1

    print("Keyboard World ----------")
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("num_steps:", testing_params.num_steps)
    print("total_steps:", curriculum.total_steps)

    return testing_params, learning_params, tester, curriculum


def get_params_mouse_world(experiment):
    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 10 * step_unit
    testing_params.num_steps = 3000  # I'm giving one minute to the agent to solve the task

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.lr = 1e-5  # 5e-5 seems to be better than 1e-4
    learning_params.gamma = 0.9
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.buffer_size = 50000
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.batch_size = 64
    learning_params.target_network_update_freq = 500  # obs: 500 makes learning more stable, but slower
    learning_params.learning_starts = 1000

    # Tabular case
    learning_params.tabular_case = False  # it is not possible to use tabular RL
    learning_params.use_random_maps = False
    learning_params.use_double_dqn = True
    learning_params.prioritized_replay = True
    learning_params.num_hidden_layers = 6
    learning_params.num_neurons = 64

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = 300
    curriculum.total_steps = 3000 * step_unit
    curriculum.min_steps = 1

    print("Mouse World ----------")
    print("lr:", learning_params.lr)
    print("batch_size:", learning_params.batch_size)
    print("num_hidden_layers:", learning_params.num_hidden_layers)
    print("target_network_update_freq:", learning_params.target_network_update_freq)
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("use_double_dqn:", learning_params.use_double_dqn)
    print("prioritized_replay:", learning_params.prioritized_replay)
    print("use_random_maps:", learning_params.use_random_maps)

    return testing_params, learning_params, tester, curriculum


def get_params_farm_world(experiment):
    step_unit = 1000

    # configuration of testing params
    testing_params = TestingParameters()
    testing_params.test = True
    testing_params.test_freq = 10 * step_unit
    testing_params.num_steps = 600  # I'm giving one minute to the agent to solve the task

    # configuration of learning params
    learning_params = LearningParameters()
    learning_params.lr = 1e-5  # 5e-5 seems to be better than 1e-4
    learning_params.gamma = 0.9
    learning_params.max_timesteps_per_task = testing_params.num_steps
    learning_params.buffer_size = 50000
    learning_params.print_freq = step_unit
    learning_params.train_freq = 1
    learning_params.batch_size = 32
    learning_params.target_network_update_freq = 100  # obs: 500 makes learning more stable, but slower
    learning_params.learning_starts = 1000

    # Tabular case
    learning_params.tabular_case = False  # it is not possible to use tabular RL in the water world
    learning_params.use_random_maps = False
    learning_params.use_double_dqn = True
    learning_params.prioritized_replay = True
    learning_params.num_hidden_layers = 6
    learning_params.num_neurons = 64

    # Setting the experiment
    tester = Tester(learning_params, testing_params, experiment)

    # Setting the curriculum learner
    curriculum = CurriculumLearner(tester.get_task_rms())
    curriculum.num_steps = 300
    curriculum.total_steps = 2000 * step_unit
    curriculum.min_steps = 1

    print("Water World ----------")
    print("lr:", learning_params.lr)
    print("batch_size:", learning_params.batch_size)
    print("num_hidden_layers:", learning_params.num_hidden_layers)
    print("target_network_update_freq:", learning_params.target_network_update_freq)
    print("TRAIN gamma:", learning_params.gamma)
    print("Total steps:", curriculum.total_steps)
    print("tabular_case:", learning_params.tabular_case)
    print("use_double_dqn:", learning_params.use_double_dqn)
    print("prioritized_replay:", learning_params.prioritized_replay)
    print("use_random_maps:", learning_params.use_random_maps)

    return testing_params, learning_params, tester, curriculum


def train_policy(world, alg_name, experiment, num_times, show_print):
    if world == 'officeworld':
        testing_params, learning_params, tester, curriculum = get_params_office_world(experiment)
    if world == 'craftworld':
        testing_params, learning_params, tester, curriculum = get_params_craft_world(experiment)
    if world == 'keyboardworld':
        testing_params, learning_params, tester, curriculum = get_params_keyboard_world(experiment)
    if world == 'mouseworld':
        testing_params, learning_params, tester, curriculum = get_params_mouse_world(experiment)
    if world == 'farmworld':
        testing_params, learning_params, tester, curriculum = get_params_farm_world(experiment)

    if alg_name == "qrm":
        run_qrm_save_model(alg_name, tester, curriculum, num_times, show_print)
    if alg_name == "hrl":
        run_hrl_save_model(alg_name, tester, curriculum, num_times, show_print, use_rm=False)
    if alg_name == "hrl-rm":
        run_hrl_save_model(alg_name, tester, curriculum, num_times, show_print, use_rm=True)
    if alg_name == "dqn":
        run_dqn_save_model(alg_name, tester, curriculum, num_times, show_print)


if __name__ == "__main__":
    # EXAMPLE: python3 train.py --algorithm="qrm" --world="craft" --map=0
    parser = argparse.ArgumentParser(prog="train_policy",
                                     description="Trains landmark policies with multi-task RL on a particular domain.")
    parser = add_training_args(parser)

    args = parser.parse_args()

    alg_name = args.algorithm
    world = args.world
    map_id = args.map
    num_times = args.num_times
    show_print = args.verbosity is not None

    if world == "office":
        experiment = "../experiments/office/tests/office.txt"
    else:
        experiment = "../experiments/%s/tests/%s_%d.txt" % (world, world, map_id)
    world += "world"

    print("world: " + world, "alg_name: " + alg_name, "experiment: " + experiment, "num_times: " + str(num_times),
          show_print)
    train_policy(world, alg_name, experiment, num_times, show_print)

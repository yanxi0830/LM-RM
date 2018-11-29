import argparse
from argparser import add_training_args
from tester.tester import Tester
from tester.tester_params import TestingParameters
from common.curriculum import CurriculumLearner
from qrm.learning_params import LearningParameters
from qrm.experiments import run_qrm_save_model


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
    step_unit = 1000

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


def train_policy(world, alg_name, experiment, num_times, show_print):
    if world == 'officeworld':
        testing_params, learning_params, tester, curriculum = get_params_office_world(experiment)
    if world == 'craftworld':
        testing_params, learning_params, tester, curriculum = get_params_craft_world(experiment)

    if alg_name == "qrm":
        run_qrm_save_model(alg_name, tester, curriculum, num_times, show_print)


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

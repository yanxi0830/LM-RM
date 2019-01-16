import argparse
from argparser import *
from qrm.experiments import load_model_and_test_composition
from baselines.run_hrl import load_hrl_model_test_composition
from baselines.run_options import load_options_model_test_composition
from train import *
from reward_machines.task import Task
import time


def run_task(world, alg_name, experiment, num_times, new_task, show_print):
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

    r = 0
    if alg_name == "qrm":
        r = load_model_and_test_composition(alg_name, tester, curriculum, num_times, new_task, show_print)
    elif alg_name == "hrl-rm":
        r = load_hrl_model_test_composition(alg_name, tester, curriculum, num_times, new_task, show_print, use_rm=True)
    elif alg_name == "hrl":
        r = load_hrl_model_test_composition(alg_name, tester, curriculum, num_times, new_task, show_print, use_rm=False)
    elif alg_name == "options":
        r = load_options_model_test_composition(alg_name, tester, curriculum, num_times, new_task, show_print)
    return r


if __name__ == "__main__":
    # EXAMPLE: python3 run_new_task.py  --algorithm="qrm" --world="craft" --map=0
    #                                   --domain_file=domain.pddl --prob_file=prob.pddl
    #                                   --plan_file=prob.plan --rm_file_dest=task.txt

    parser = argparse.ArgumentParser(prog="run_new_task",
                                     description="Zero-shot execution of new task given high-level model")
    parser = add_run_args(parser)
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

    domain_file = args.domain_file
    prob_file = args.prob_file
    plan_file = args.plan_file
    rm_file_dest = args.rm_file_dest

    strict = False
    if world == "mouseworld" or world == "keyboardworld" or world == "farmworld":
        strict = True

    new_task = Task(domain_file, prob_file, plan_file, rm_file_dest, world, use_partial_order=False, strict=strict)
    print("world: " + world, "alg_name: " + alg_name, "experiment: " + experiment, "num_times: " + str(num_times),
          show_print)

    print("world: {}, alg_name: {}, experiment: {}, num_times: {}, "
          "domain_file: {}, prob_file: {}, plan_file: {}".format(world, alg_name, experiment, str(num_times),
                                                                 domain_file, prob_file, plan_file))

    init_t = time.time()
    run_task(world, alg_name, experiment, num_times, new_task, show_print)
    print("Time: ", time.time() - init_t, "s")

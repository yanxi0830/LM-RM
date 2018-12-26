"""
domains/{world-name}/tasks contains all goal directed tasks and their sequential plans
    - generate these using some scripts (in generators)
"""
import argparse
from os import listdir
from os.path import isfile, join
from argparser import add_run_args
from reward_machines.task import Task
from run_new_task import run_task
import argparse
from argparser import *
from qrm.experiments import load_model_and_test_composition
from baselines.run_hrl import load_hrl_model_test_composition
from train import *
from reward_machines.task import Task
import time
from qrm.experiments import get_qrm_generalization_performance
from baselines.run_hrl import get_hrl_generalization_performance


def get_generalization_performance(alg_name, tester, curriculum, num_times, new_tasks, show_print):
    if alg_name == "qrm":
        return get_qrm_generalization_performance(alg_name, tester, curriculum, num_times, new_tasks, show_print)
    if alg_name == "hrl-rm":
        return get_hrl_generalization_performance(alg_name, tester, curriculum, num_times, new_tasks, show_print, True)
    if alg_name == "hrl":
        return get_hrl_generalization_performance(alg_name, tester, curriculum, num_times, new_tasks, show_print, False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog="run_new_task",
                                     description="Zero-shot execution of new task given high-level model")
    parser = add_run_args(parser)
    args = parser.parse_args()

    alg_name = args.algorithm
    world = args.world
    map_id = args.map
    num_times = args.num_times
    show_print = args.verbosity is not None

    folder = "../domains/{}/tasks".format(world)
    tasks = [f for f in listdir(folder) if
             isfile(join(folder, f)) and not f.endswith('.plan') and 'domain' not in f]
    domain_file = "../domains/{}/domain.pddl".format(world)

    if world == "office":
        experiment = "../experiments/office/tests/office.txt"
    else:
        experiment = "../experiments/%s/tests/%s_%d.txt" % (world, world, map_id)
    world += "world"

    success_count = 0
    new_tasks = []
    for i, prob_file in enumerate(tasks):
        prob_file = "{}/{}".format(folder, prob_file)
        plan_file = prob_file.replace("pddl", "plan")
        rm_file_dest = "../experiments/{}/reward_machines/new_{}.txt".format(world[:-5], i)
        new_task = Task(domain_file, prob_file, plan_file, rm_file_dest, world, use_partial_order=False)
        new_tasks.append(new_task)

    testing_params, learning_params, tester, curriculum = get_params_office_world(experiment)
    success_rate, acc_reward = get_generalization_performance(alg_name, tester, curriculum, num_times, new_tasks, show_print)

    print("=====================")
    print("Algorithm:", alg_name)
    print("Number of Tasks:", len(new_tasks))
    print("Generalization Performance:", success_rate, "Cumulative Reward:", acc_reward)

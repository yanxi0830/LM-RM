import argparse
from argparser import add_landmark_args
from landmarks.landmark_utils import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="generate_reward_machines_from_landmarks",
                                     description="Generates reward machines for each landmarks given a task")
    parser = add_landmark_args(parser)

    args = parser.parse_args()
    domain_file = args.domain_file
    prob_file = args.prob_file

    lm_graph = LandmarkGraph(FileParams(domain_file, prob_file))
    compute_rm_from_graph2(lm_graph)

"""
FarmWorld PDDL generator

GOALS:
    (have-pig)
    (have-cow)
    (have-chicken)
    (have-pork)
    (have-beef)
    (have-wings)
    (have-egg)
    (have-milk)
    (have-dessert)
"""
import argparse
import itertools
import random

parser = argparse.ArgumentParser(
    description='generate OfficeWorld PDDL tasks'
)

OBJECT_TYPES = ["pig", "cow", "chicken", "pork", "beef", "wings",
                "egg", "milk", "dessert"]
GOALS = ["(have-{})".format(obj) for obj in OBJECT_TYPES]


def write_file(file_name, contents):
    """Write the contents of a provided list or string to a file"""
    f = open(file_name, 'w')
    if contents.__class__.__name__ == 'list':
        f.write("\n".join(contents))
    else:
        f.write(contents)
    f.close()


def format_problem(name, goals_list):
    acc = list(['(define (problem {})'.format(name), '\t(:domain farmworld)'])
    acc.append('\t(:objects')
    acc.append('\t)')

    acc.append('\t(:init')
    acc.append('\t)')

    acc.append("\t(:goal (and")
    for g in goals_list:
        acc.append('\t\t{}'.format(g))
    acc.append('\t\t)')
    acc.append('\t)')
    acc.append(')')

    return '\n'.join(acc)


if __name__ == "__main__":
    NUM_TASKS = 300
    LO = 1
    HI = 9
    problem_bank = []

    for i in range(LO, HI+1):
        problem_bank += list(itertools.combinations(GOALS, i))

    random.shuffle(problem_bank)

    for j in range(NUM_TASKS):
        task = list(problem_bank[j])
        pddl = format_problem(str(j), task)
        write_file('../../domains/farm/tasks/{}.pddl'.format(j), pddl)


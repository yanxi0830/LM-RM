"""
OfficeWorld PDDL generator

GOALS:
    (visited-A)
    (visited-B)
    (visited-C)
    (visited-D)
    (visited-mail)
    (visited-coffee)
    (delivered-coffee)
    (delivered-mail)
"""
import argparse

parser = argparse.ArgumentParser(
    description='generate OfficeWorld PDDL tasks'
)

LOCATIONS = ["A", "B", "C", "D", "mail", "coffee"]
GOALS = ['(visited-{})'.format(loc) for loc in LOCATIONS] + ['(delivered-coffee)', '(delivered-mail)']


def write_file(file_name, contents):
    """Write the contents of a provided list or string to a file"""
    f = open(file_name, 'w')
    if contents.__class__.__name__ == 'list':
        f.write("\n".join(contents))
    else:
        f.write(contents)
    f.close()


def format_problem(name, goals_list):
    acc = list(['(define (problem {})'.format(name), '\t(:domain officeworld)'])
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

    # for i in range(len(GOALS)):
    #     # Get all possible combinations with i sub-goals
    #     # TODO
    # pddl = format_problem("1", GOALS)
    # print(pddl)
    for i in range(len(GOALS)-2):
        task = GOALS[i:i+2]
        pddl = format_problem(str(i), task)
        write_file('../../domains/office/tasks/{}.pddl'.format(i), pddl)

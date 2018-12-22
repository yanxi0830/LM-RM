"""
CraftWorld PDDL generator

GOALS:
    (have-wood)
    (have-grass)
    (have-iron)
    (have-plank)
    (have-stick)
    (have-cloth)
    (have-rope)
    (have-bridge)
    (have-bed)
    (have-axe)
    (have-shears)
    (have-gold)
    (have-gem)
    (have-goldware)
    (have-ring)
    (have-saw)
    (have-bow)
"""
import argparse

parser = argparse.ArgumentParser(
    description='generate OfficeWorld PDDL tasks'
)

OBJECT_TYPES = ["wood", "grass", "iron", "plank", "stick", "cloth",
                "rope", "bridge", "bed", "axe", "shears", "gold",
                "gem", "goldware", "ring", "saw", "bow"]
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
    acc = list(['(define (problem {})'.format(name), '\t(:domain craftworld)'])
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
    for i in range(len(GOALS)-3):
        task = GOALS[i:i+3]
        pddl = format_problem(str(i), task)
        write_file('../../domains/craft/tasks/{}.pddl'.format(i), pddl)

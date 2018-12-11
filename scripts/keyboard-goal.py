"""
Text to PDDL specification.
"""
import argparse

parser = argparse.ArgumentParser(
    description='generate a keyboard problem with text'
)
parser.add_argument('--text', type=str, help='text to type')

args = parser.parse_args()

chars = "abcdefghijklmnopqrstuvwxyz"


def get_chars(prefix):
    acc = []
    for c in chars:
        acc.append("{}-{}".format(prefix, c))
    return ' '.join(acc)


def get_letter(ch):
    if ch.isupper():
        return 'up-{}'.format(ch.lower())
    if ch.islower():
        return 'lo-{}'.format(ch)


def write_file(file_name, contents):
    """Write the contents of a provided list or string to a file"""
    f = open(file_name, 'w')
    if contents.__class__.__name__ == 'list':
        f.write("\n".join(contents))
    else:
        f.write(contents)
    f.close()


def format_problem(text):
    acc = list(['(define (problem t1)', '\t(:domain keyboardworld)'])

    # OBJECTS
    acc.append('\t(:objects')
    cursors = ['cursor-{}'.format(i) for i in range(len(text) + 1)]
    acc.append('\t\t' + ' '.join(cursors + ['-', 'cursor-position']))
    acc.append('\t)')

    # INIT
    acc.append('\t(:init')
    acc.append('\t\t(caps-off)')
    acc.append('\t\t(current-position cursor-0)')

    for c in chars:
        acc.append("\t\t(maps-to {} lo-{})".format(c, c))
        acc.append("\t\t(maps-to {} up-{})".format(c, c))

    for i in range(len(text)):
        acc.append("\t\t(position-predecessor {} {})".format(cursors[i], cursors[i + 1]))

    acc.append('\t)')

    # GOAL
    acc.append("\t(:goal (and")

    for i, ch in enumerate(text):
        acc.append('\t\t(char-at {} {})'.format(get_letter(ch), cursors[i]))

    acc.append('\t\t)')
    acc.append('\t)')
    acc.append(')')

    return '\n'.join(acc)


if __name__ == "__main__":
    pddl = format_problem(args.text)
    write_file('../domains/keyboard/t1.pddl', pddl)
    # write_file('../domains/keyboard/new_goal.pddl', pddl)

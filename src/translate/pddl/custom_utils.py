class Plan:
    def __init__(self, actions=None):
        self.actions = actions or []

    def add_actions(self, new_actions):
        self.actions.extend(new_actions)

    def write_to_file(self, file_name):
        # Write the script to a file
        f = open(file_name, 'w')

        step = 0
        for action in self.actions:
            line = str(step) + ': (' + action.operator + ' ' + ' '.join(action.arguments) + ") [1]\n"
            f.write(line)
            step += 1

        f.close()

    def get_action_sequence(self):
        return [item.operator for item in self.actions]


class GroundAction:
    def __init__(self, line):
        self.operator = line.split(' ')[0]
        self.arguments = line.split(' ')[1:]

    def compact_rep(self):
        toReturn = self.operator
        for arg in self.arguments:
            toReturn += "\\n" + str(arg)
        return toReturn

    def __str__(self):
        return "(" + ' '.join([self.operator] + self.arguments) + ")"

    def __repr__(self):
        return self.__str__()


def parse_output_ipc(file_name):
    # Check for the failed solution
    # if match_value(file_name, '.* No plan will solve it.*'):
    #    print "No solution."
    #    return None

    # Get the plan
    action_list = read_file(file_name)

    actions = [GroundAction(line[1:-1].strip(' ').lower()) for line in action_list]

    return Plan(actions)


def read_file(file_name):
    """Return a list of the lines of a file."""
    f = open(file_name, 'r')
    file_lines = [line.rstrip("\n") for line in f.readlines()]
    f.close()
    return file_lines


def write_file(file_name, contents):
    """Write the contents of a provided list or string to a file"""
    f = open(file_name, 'w')
    if contents.__class__.__name__ == 'list':
        f.write("\n".join(contents))
    else:
        f.write(contents)
    f.close()


def append_file(file_name, contents):
    """Append the contents of a provided list or string to a file"""
    f = open(file_name, 'a')
    if contents.__class__.__name__ == 'list':
        f.write("\n".join(contents))
    else:
        f.write(contents)
    f.close()


def get_opts():
    import sys
    argv = sys.argv

    opts = {}
    flags = []

    while argv:
        if argv[0][0] == '-':  # find "-name value" pairs
            opts[argv[0]] = argv[1]  # dict key is "-name" arg
            argv = argv[2:]
        else:
            flags.append(argv[0])
            argv = argv[1:]
    return opts, flags

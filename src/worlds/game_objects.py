from enum import Enum
import random

"""
The following classes are the types of objects that we are currently supporting 
"""


class Entity:
    def __init__(self, i, j):  # row and column
        self.i = i
        self.j = j

    def change_position(self, i, j):
        self.i = i

        self.j = j

    def idem_position(self, i, j):
        return self.i == i and self.j == j

    def interact(self, agent):
        return True


class Agent(Entity):
    def __init__(self, i, j, actions):
        super().__init__(i, j)
        self.actions = actions

    def get_actions(self):
        return self.actions

    def __str__(self):
        return "A"

    def __repr__(self):
        return "Agent({}, {})".format(self.i, self.j)


class Obstacle(Entity):
    def __init__(self, i, j):
        super().__init__(i, j)

    def interact(self, agent):
        return False

    def __str__(self):
        return "X"


class Empty(Entity):
    def __init__(self, i, j, label=" "):
        super().__init__(i, j)
        self.label = label

    def __str__(self):
        return self.label


class Actions(Enum):
    """
    Enum with the actions that the agent can execute
    """
    up = 0  # move up
    right = 1  # move right
    down = 2  # move down
    left = 3  # move left
    jump = 4  # jump or click
    none = 5  # none
    # drop = 6

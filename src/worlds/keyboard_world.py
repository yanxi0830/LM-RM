from worlds.game_objects import *
import numpy as np
import os


class KeyboardWorldParams:
    def __init__(self, file_map, use_tabular_representation):
        self.file_map = file_map
        self.use_tabular_representation = use_tabular_representation


class KeyboardWorld:

    def __init__(self, params):
        self.params = params
        self._load_map(params.file_map)
        self.env_game_over = False
        self.last_action = -1

    def get_map_id(self):
        map_id = os.path.basename(self.params.file_map)
        return os.path.splitext(map_id)[0]

    def execute_action(self, a):
        """
        We execute 'action' in the game
        """
        action = Actions(a)
        agent = self.agent

        # Getting new position after executing action
        ni, nj = self._get_next_position(action)

        # Interacting with the objects that is in the next position (this doesn't include monsters)
        action_succeeded = self.map_array[ni][nj].interact(agent)

        # So far, an action can only fail if the new position is a wall
        if action_succeeded:
            agent.change_position(ni, nj)
        self.last_action = action

    def _get_next_position(self, action):
        """
        Returns the position where the agent would be if we execute action
        """
        agent = self.agent
        ni, nj = agent.i, agent.j

        # without jumping
        direction = action

        # OBS: Invalid actions behave as NO-OP
        if direction == Actions.up: ni -= 1
        if direction == Actions.down: ni += 1
        if direction == Actions.left: nj -= 1
        if direction == Actions.right: nj += 1

        return ni, nj

    def get_state(self):
        return None  # we are only using "simple reward machines" for the craft domain

    def get_actions(self):
        """
        Returns the list with the actions that the agent can perform
        """
        return self.agent.get_actions()

    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        if self.last_action == Actions.jump:
            ret = str(self.map_array[self.agent.i][self.agent.j]).strip()
        else:
            ret = ''
        return ret

    # The following methods return different feature representations of the map ------------
    def get_features(self):
        if self.params.use_tabular_representation:
            return self._get_features_one_hot_representation()
        return self._get_features_manhattan_distance()

    def _get_features_manhattan_distance(self):
        # map from object classes to numbers
        class_ids = self.class_ids  # {"a":0,"b":1}
        N, M = self.map_height, self.map_width
        ret = []
        for i in range(N):
            for j in range(M):
                obj = self.map_array[i][j]
                if str(obj) in class_ids:
                    ret.append(self._manhattan_distance(obj))

        return np.array(ret, dtype=np.float64)

    def _manhattan_distance(self, obj):
        """
        Returns the Manhattan distance between 'obj' and the agent
        """
        return abs(obj.i - self.agent.i) + abs(obj.j - self.agent.j)

    def _get_features_one_hot_representation(self):
        """
        Returns a one-hot representation of the state (useful for the tabular case)
        """
        N, M = self.map_height, self.map_width
        ret = np.zeros((N, M), dtype=np.float64)
        ret[self.agent.i, self.agent.j] = 1
        return ret.ravel()  # from 3D to 1D (use a.flatten() is you want to copy the array)

    def show_map(self):
        """
        Prints the current map
        """
        print(self.__str__())

    def show(self):
        self.show_map()

    def __str__(self):
        r = ""
        for i in range(self.map_height):
            s = ""
            for j in range(self.map_width):
                if self.agent.idem_position(i, j):
                    s += str(self.agent)
                else:
                    s += str(self.map_array[i][j])
            if i > 0:
                r += "\n"
            r += s
        return r

    def _load_map(self, file_map):
        actions = [Actions.up.value, Actions.right.value, Actions.down.value, Actions.left.value, Actions.jump.value]

        self.map_array = []
        self.class_ids = {}
        f = open(file_map)
        i, j = 0, 0
        for l in f:
            if len(l.rstrip()) == 0:
                continue

            row = []
            j = 0
            for e in l.rstrip():
                if e in "1234567890qwertyuiopasdfghjklzxcvbnmC":
                    entity = Empty(i, j, label=e)
                    if e not in self.class_ids:
                        self.class_ids[e] = len(self.class_ids)
                if e in " A":
                    entity = Empty(i, j)
                if e == "X":
                    entity = Obstacle(i, j)
                if e == "A":
                    self.agent = Agent(i, j, actions)
                row.append(entity)
                j += 1
            self.map_array.append(row)
            i += 1
        f.close()

        self.map_height, self.map_width = len(self.map_array), len(self.map_array[0])


def play(params, task, max_time):
    from reward_machines.reward_machine import RewardMachine

    # commands
    str_to_action = {"w": Actions.up.value, "d": Actions.right.value, "s": Actions.down.value, "a": Actions.left.value,
                     "b": Actions.jump.value}
    # play the game!
    game = KeyboardWorld(params)
    rm = RewardMachine(task)
    s1 = game.get_state()
    u1 = rm.get_initial_state()
    for t in range(max_time):
        # Showing game
        game.show_map()
        print("Events:", game.get_true_propositions())
        print("Features:", game.get_features())
        acts = game.get_actions()
        # Getting action
        print("\nAction? ", end="")
        a = input()
        print()
        # Executing action
        if a in str_to_action and str_to_action[a] in acts:
            game.execute_action(str_to_action[a])

            s2 = game.get_state()
            events = game.get_true_propositions()
            u2 = rm.get_next_state(u1, events)
            reward = rm.get_reward(u1, u2, s1, a, s2)
            print("Reward: ", reward)

            if game.env_game_over or rm.is_terminal_state(u2):  # Game Over
                print("Game Over")
                break

            s1, u1 = s2, u2
        else:
            print("Forbidden action")
    game.show_map()
    return reward


if __name__ == '__main__':
    map = "../../experiments/keyboard/maps/map_0.map"
    tasks = ["../../experiments/keyboard/reward_machines/t%d.txt" % i for i in [1, 2]]
    max_time = 100
    use_tabular_representation = True

    for task in tasks:
        while True:
            params = KeyboardWorldParams(map, use_tabular_representation)
            if play(params, task, max_time) > 0:
                break

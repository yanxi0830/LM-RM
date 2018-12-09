from worlds.game import GameParams
from worlds.mouse_world import MouseWorldParams


class TesterMouseWorld:
    def __init__(self, experiment, data=None):
        if data is None:
            # Reading the file
            self.experiment = experiment
            f = open(experiment)
            lines = [l.rstrip() for l in f]
            f.close()
            # setting the test attributes
            self.tasks = eval(lines[1])

        else:
            self.experiment = data["experiment"]
            self.tasks = data["tasks"]

    def get_dictionary(self):
        d = {}
        d["experiment"] = self.experiment
        d["tasks"] = self.tasks
        return d

    def get_reward_machine_files(self):
        return self.tasks

    def get_task_specifications(self):
        return self.tasks

    def get_task_params(self, task_specification):
        params = MouseWorldParams()
        return GameParams("mouseworld", params)

    def get_task_rm_file(self, task_specification):
        return task_specification

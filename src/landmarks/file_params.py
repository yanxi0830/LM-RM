import os


class FileParams:
    def __init__(self, domain_file, task_file):
        self.domain_file = domain_file
        self.task_file = task_file
        self.landmark_file = os.path.splitext(task_file)[0] + ".landmark"
        self.plan_file = os.path.splitext(task_file)[0] + ".plan"
        self.landmark_tasks = dict()    # node_id to landmark problem file

    def create_lm_path(self, n_id):
        self.landmark_tasks[n_id] = os.path.splitext(self.task_file)[0] + "-" + str(n_id) + ".pddl"

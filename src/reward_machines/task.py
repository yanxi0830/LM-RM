"""
A Task object including helpful info defined for the task.
- PDDL task/domain file
- RewardMachine for the task
- Linearized/Partial-Ordered Plans
"""
from landmarks.planner_utils import compute_and_save_rm_spec, compute_linearized_plans, save_sequential_rm_spec


class Task:
    def __init__(self, domain_file, task_file, plan_file, rm_file, game_type, use_partial_order=False):
        self.domain_file = domain_file
        self.task_file = task_file
        self.plan_file = plan_file      # pre-process using ./cleanplan.sh
        self.game_type = game_type

        # compute and save reward machine file
        self.rm_file = rm_file

        # use sequential plans for keyboardworld/mouseworld for now, need full gurobipy license..
        if game_type == 'keyboardworld' or game_type == "mouseworld" or not use_partial_order:
            self.pop = save_sequential_rm_spec(self.domain_file, self.task_file, self.plan_file, self.rm_file,
                                               self.game_type)
        else:
            self.pop = compute_and_save_rm_spec(self.domain_file, self.task_file, self.plan_file, self.rm_file,
                                                self.game_type)

    def get_linearized_plan(self):
        return compute_linearized_plans(self.pop, self.game_type)

    def __str__(self):
        return self.task_file

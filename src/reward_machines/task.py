"""
A Task object including helpful info defined for the task.
- PDDL task/domain file
- RewardMachine for the task
- Linearized/Partial-Ordered Plans
"""
from landmarks.planner_utils import compute_and_save_rm_spec, compute_linearized_plans


class Task:
    def __init__(self, domain_file, task_file, plan_file, rm_file):
        self.domain_file = domain_file
        self.task_file = task_file
        self.plan_file = plan_file      # pre-process using ./cleanplan.sh

        # compute and save reward machine file
        self.rm_file = rm_file
        self.pop = compute_and_save_rm_spec(self.domain_file, self.task_file, self.plan_file, self.rm_file)

    def get_linearized_plan(self):
        return compute_linearized_plans(self.pop)
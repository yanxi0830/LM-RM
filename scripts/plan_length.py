"""
Get plan lengths
"""
import os
import shutil
from translate.pddl.custom_utils import read_file


TASK_PATH="../domains/farm/tasks"

directory = os.fsencode(TASK_PATH)

lengths = []

for i, file in enumerate(os.listdir(directory)):
    filename = os.fsdecode(file)
    if filename.endswith(".plan"):
        lines = read_file(TASK_PATH + "/" + filename)
        lengths.append(len(lines))
        if len(lines) == 0:
            print(filename)

print("Average:", sum(lengths) / float(len(lengths)))
print("Max:", max(lengths))
print("Min:", min(lengths))
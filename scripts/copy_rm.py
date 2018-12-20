"""
Copy over all files in /PATH/TO/lm_reward_machines for training...
"""
import os
import shutil

SRC_PATH="../domains/craft/lm_reward_machines"
DEST_PATH="../experiments/craft/reward_machines"

directory = os.fsencode(SRC_PATH)

for i, file in enumerate(os.listdir(directory)):
    filename = os.fsdecode(file)
    shutil.copy(SRC_PATH+"/"+filename, DEST_PATH+"/t{}.txt".format(i+1))
    print("Copied {} to {}".format(SRC_PATH+"/"+filename, DEST_PATH+"/t{}.txt".format(i+1)))

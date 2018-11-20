# LM-RM
Generating Reward Machines from Landmarks

### Work in Progress

##### Generate Reward Machines for each landmark given task
```bash
# generate landmark graph for given task
cd scripts
./plan.sh ../domains/office/t1.pddl

# reward machine spec files save to domains/office/lm_reward_machines
cd src
python -m landmarks.landmark_utils
```

##### Cleanup intermediate files
```bash
cd scripts
./cleanup.sh
```
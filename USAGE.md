# Running Experiments

### 1. Generate Reward Machines from PDDL

`domains/[ENV]/lm_reward_machines` includes the set of reward machines corresponding to all landmarks in the domain used for training. Use `generate_lm_rm.py` to generate them for a single task.

```bash
cd scripts
./plan.sh ../domains/craft/t1.pddl

cd src
python generate_lm_rm.py --world="craft" --domain_file="../domains/craft/domain.pddl" --prob_file="../domains/craft/t1.pddl"
```

### 2. Pre-train Model
`model/[ENV]/` includes pre-trained Tensorflow model trained using various algorithms. To train and save them:
```bash
python train.py --world="craft" --map=0 --algorithm="qrm"
```

### 3. Test Generalization Performance
To test the generalization performance on a set of new tasks. We first generate a set of random tasks in PDDL saved inside `domains/[ENV]/tasks`. This is done using `craft_generator.py`/`farm_generator.py`. The task complexity and number of tasks can be set using the `NUM_TASKS`/`HI`/`LO` flag inside the main block. 
```bash
python craft_generator.py
```

Running `ENV_geneartor.py` will output random tasks inside `domains/[ENV]/tasks`. Use the script `plan_all.sh` to compute plans for all the tasks.
```bash
cd scripts
# The second argument is path to the directory where the random tasks are saved
./plan_all.sh ../domains/craft/tasks
```

Now, modify `model/ENV/checkpoint` and use `generalization_test.py` to evaluate the generalization performance on the set of random tasks for each algorithm. For example, to evaluate the performance using "QRM" on "craft" tasks:

- Make sure the [checkpoint](model/craftworld/map_0/checkpoint) file points to "qrm" (or "hrl", "hrl-rm"). 
```
model_checkpoint_path: "qrm"
all_model_checkpoint_paths: "qrm"
```

- `generalization_test.py` takes in an optional argument to choose whether or not partial-ordered plans should be used for execution:
```bash
# Use sequential plans
python generalization_tests.py --world="craft" --map=0 --algorithm="qrm"

# Use partial-ordered plans
python generalization_tests.py --world="craft" --map=0 --algorithm="qrm" --use_partial=1
```

### Cleanup
```bash
cd scripts
./cleanup.sh
```

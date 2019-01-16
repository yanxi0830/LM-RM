# LM-RM
This projects studies the use of high-level models and plans for RL to solve tasks without training specifically for them. 

This repo is based on code from [QRM](https://bitbucket.org/RToroIcarte/qrm/overview), [POPGEN](https://bitbucket.org/haz/pop-gen/), and [FastDownward](http://www.fast-downward.org/). 

### Installation
The following scripts assumes you have a `~/git/` directory
```bash
cd git
git clone https://github.com/yanxi0830/LM-RM
```

##### Fast Downward Dependency
This repo depends on a modified version of [FastDownward](http://www.fast-downward.org/) for landmark graphs extraction. 
```bash
cd git
git clone https://github.com/yanxi0830/downward
```

##### Other Dependencies
The code requires: Python3, numpy, tensorflow, networkx, and gurobipy
```bash
conda create --name lmrm python=3.6
conda activate lmrm
conda install tensorflow 
pip install matplotlib
pip install networkx

# Gurobi
# Need to get license..
conda config --add channels http://conda.anaconda.org/gurobi
conda install gurobi
grbgetkey LICENSE-CODE
```

### Running Experiments
See [usage](USAGE.md) for detailed steps on running experiments. 

### Running Examples

#### Generating Reward Machines from Landmarks
Given a high-level domain model and a specific task, find its ordered landmarks and builds a reward machine for each individual landmarks. The resulting reward machines can be used to train an agent to learn the optimal policies for achieving the landmarks. 
```bash
# generate landmark graph for given task
cd scripts
./plan.sh ../domains/craft/t1.pddl

# reward machine spec files are saved to PATH/TO/DOMAIN/lm_reward_machines
cd src
python generate_lm_rm.py --world="craft" --domain_file="../domains/craft/domain.pddl" --prob_file="../domains/craft/t1.pddl"
```

#### Training Landmark Policies
The repo includes a pre-trained Tensorflow model for the landmark policies from a collection of simple tasks for each domain. To train them:
```bash
python train.py --world="craft" --algorithm="qrm"
```

#### Execution of New Tasks
Given a new task presented in terms of the high-level model, we compute a partial ordered plan and use it to compose the landmark policies for execution. 
```bash
# compute a sequential plan to be refined into partial-order
./cleanplan.sh ../domains/craft/new.pddl

python run_new_tasks.py --world="craft" --domain_file="../domains/craft/domain.pddl" --prob_file "../domains/craft/new.pddl" --plan_file="../domains/office/new.plan"
```
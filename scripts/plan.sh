#!/bin/bash

# Sample Usage: ./plan.sh /path/to/problem/t1.pddl
# Sequential plan saved to /path/to/problem/t1.plan
# Landmark file save dto /path/to/problem/t1.landmark

PLANNER=~/git/downward
TASK=$(cd "$(dirname "$1")"; pwd)/$(basename "$1")
TASK_DIR=$(cd "$(dirname "$1")"; pwd)
TASK_NAME="${TASK##*/}"
TASK_NAME=${TASK_NAME%.*}

echo 'Task PDDL: ' $TASK
echo $TASK_DIR
echo $PLANNER
echo $TASK_NAME

cd $PLANNER
./fast-downward.py --cleanup
./fast-downward.py --translate $TASK
./fast-downward.py --alias seq-sat-lama-2011 output.sas

ghead -n -1 ./sas_plan.1 > $TASK_DIR/$TASK_NAME.plan
cat landmark.txt > $TASK_DIR/$TASK_NAME.landmark

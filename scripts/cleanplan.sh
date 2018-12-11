#!/bin/bash

# Sample Usage: ./plan.sh /path/to/problem/t1.pddl
# Sequential plan saved to /path/to/problem/t1.plan
# Landmark file save dto /path/to/problem/t1.landmark

PLANNER=${BASEDIR:-$(dirname $0)/../..}/downward
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

(type ghead &> /dev/null && HEAD=ghead) || HEAD=head

if [ -e ./sas_plan.2 ]; then
    $HEAD -n -1 ./sas_plan.2 > $TASK_DIR/$TASK_NAME.plan
else
    $HEAD -n -1 ./sas_plan.1 > $TASK_DIR/$TASK_NAME.plan
fi

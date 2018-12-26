#!/bin/bash

TASK_DIR=$1

for pddl in `ls $TASK_DIR`; do
    if [[ $pddl != *"domain"* ]]; then
        ./cleanplan.sh $TASK_DIR/$pddl
    fi
done
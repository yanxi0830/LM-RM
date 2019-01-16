#!/bin/bash

cd ..
set -x
find . -type f -name '*gurobi.log' -exec rm {} +
find . -type f -name '*pop.txt' -exec rm {} +
find . -type f -name '*.landmark' -exec rm {} +
find . -type f -name '*-*.pddl' -exec rm {} +
find . -type f -name '*.plan' -exec rm {} +
find . -type f -name 'new_*.txt' -exec rm {} +

rm ./domains/craft/tasks/*
cp ./domains/craft/domain.pddl ./domains/craft/tasks/domain.pddl

rm ./domains/farm/tasks/*
cp ./domains/farm/domain.pddl ./domains/farm/tasks/domain.pddl
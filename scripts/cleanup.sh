#!/bin/bash

cd ..
set -x
find . -type f -name 'gurobi.log' -exec rm {} +
find . -type f -name 'pop.txt' -exec rm {} +
find . -type f -name '*.landmark' -exec rm {} +
find . -type f -name '*-*.pddl' -exec rm {} +
find . -type f -name '*.plan' -exec rm {} +

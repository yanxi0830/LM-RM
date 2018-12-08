#!/bin/bash


for char in a b c d e f g h i j k l m n o p q r s t u v w x y z; do
    python keyboard-goal.py --text "$char"
    ./plan.sh ../domains/keyboard/t1.pddl
    cd ../src
    python generate_lm_rm.py --domain_file="../domains/keyboard/domain.pddl" --prob_file="../domains/keyboard/t1.pddl"
    cd ../scripts
done

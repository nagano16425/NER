#!/bin/sh
#
# HOW TO USE
# Command:bash basic2-1_train.sh (end)
#
python3 basic2-1.py Train Dev $1 > log_train_$1.txt
# python3 basic2-1.py Train Test $1 > log_train_$1.txt

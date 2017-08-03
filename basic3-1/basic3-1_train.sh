#!/bin/sh
#
# HOW TO USE
# Command:bash basic3-1_train.sh (end)
#
python3 basic3-1.py Train Dev $1 > log_train_$1.txt
# python3 basic3-1.py Train Test $1 > log_train_$1.txt

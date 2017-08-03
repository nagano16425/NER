#!/bin/sh
#
# HOW TO USE
# Command:bash basic_train.sh (end)
#
python3 basic.py Train Dev $1 > log_train_$1.txt
# python3 basic.py Train Test $1 > log_train_$1.txt

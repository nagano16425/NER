#!/bin/sh
#
# HOW TO USE
# Command:bash Proposed2_train.sh (end)
#
# python3 preprocess.py Train Dev
python3 preprocess.py Train Test
# python3 train.py Train Dev $1 > log_train_$1.txt
python3 train.py Train Test $1 > log_train_$1.txt

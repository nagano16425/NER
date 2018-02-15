#!/bin/sh
#
# HOW TO USE
# Command:bash Proposed2_train_cv.sh (end)
#
python3 preprocess.py 12 34
python3 train.py 12 34 $1 > log_train_Traffic34_$1.txt
python3 preprocess.py 34 12
python3 train.py 34 12 $1 > log_train_Traffic12_$1.txt

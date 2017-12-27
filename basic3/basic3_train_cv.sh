#!/bin/sh
#
# HOW TO USE
# Command:bash basic3_train_cv.sh (end)
#
python3 basic3.py 12 34 $1 > log_train_Traffic34_$1.txt
python3 basic3.py 34 12 $1 > log_train_Traffic12_$1.txt

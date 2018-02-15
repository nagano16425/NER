#!/bin/sh
#
# HOW TO USE
# Command:bash Proposed2_cv.sh
#
# bash Proposed2_train_cv.sh 10
bash Proposed2_train_cv.sh 1000
bash Proposed2_pre_cv.sh 1 10 1
bash Proposed2_pre_cv.sh 20 100 10
bash Proposed2_pre_cv.sh 200 1000 100

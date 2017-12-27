#!/bin/sh
#
# HOW TO USE
# Command:bash Proposed1_cv.sh
#
# bash Proposed1_train_cv.sh 10
bash Proposed1_train_cv.sh 1000
bash Proposed1_pre_cv.sh 1 10 1
bash Proposed1_pre_cv.sh 20 100 10
bash Proposed1_pre_cv.sh 200 1000 100

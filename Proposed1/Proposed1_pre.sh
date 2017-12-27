#!/bin/sh
#
# HOW TO USE
# Command:bash Proposed1_pre.sh (start) (end) (counter)
#
for (( i = $1 ; i <= $2 ; i+=$3 ))
do
python3 test.py Train Dev $i > log_pre_Dev_$i.txt
# python3 test.py Train Test $i > log_pre_Test_$i.txt
done
for (( i = $1 ; i <= $2 ; i+=$3 ))
do
    perl evalIOB2.pl TrafficDev.iob2 EviRNNDev_$i.iob2 > eval_EviDev_$i.iob2
    # perl evalIOB2.pl TrafficTest.iob2 EviRNNTest_$i.iob2 > eval_EviTest_$i.iob2
done

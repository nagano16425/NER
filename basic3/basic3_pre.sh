#!/bin/sh
#
# HOW TO USE
# Command:bash basic3_pre.sh (start) (end) (counter)
#
for (( i = $1 ; i <= $2 ; i+=$3 ))
do
python3 basic3.py Train Dev $i > log_pre_Dev_$i.txt
# python3 basic3.py Train Test $i > log_pre_Test_$i.txt
done
for (( i = $1 ; i <= $2 ; i+=$3 ))
do
    perl evalIOB2.pl TrafficDev.iob2 EviDev_$i.iob2 > eval_EviDev_$i.iob2
    # perl evalIOB2.pl TrafficTest.iob2 EviTest_$i.iob2 > eval_EviTest_$i.iob2
done

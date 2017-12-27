#!/bin/sh
#
# HOW TO USE
# Command:bash basic_pre_cv.sh (start) (end) (counter)
#
for (( i = $1 ; i <= $2 ; i+=$3 ))
do
python3 basic.py 12 34 $i > log_pre_Traffic34_$i.txt
python3 basic.py 34 12 $i > log_pre_Traffic12_$i.txt
done
for (( i = $1 ; i <= $2 ; i+=$3 ))
do
    perl evalIOB2.pl Traffic12.iob2 Evi12_$i.iob2 > eval_Evi12_$i.iob2
    perl evalIOB2.pl Traffic34.iob2 Evi34_$i.iob2 > eval_Evi34_$i.iob2
    cat Evi12_$i.iob2 Evi34_$i.iob2 > result_$i.iob2
    perl evalIOB2.pl Traffic.iob2 result_$i.iob2 > result_eval_$i.iob2
done

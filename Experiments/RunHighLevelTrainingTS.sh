#!/bin/sh

#./RunHighLevelTrainingTS.sh 2000
pathname="./HighlevelTrainingWithFS.py"

for VARIABLE in 0 1 2 3 4 5 6 7 8 9
do
python3 $pathname --container_N $1 --batch_choice $(($VARIABLE))
done

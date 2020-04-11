#!/bin/sh

#./TrainSubScheduler.sh
#cd ..
pathname="./TrainSubSchedulerWith27Nodes.py"

for VARIABLE in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
do
python3 $pathname --start_sample $(($VARIABLE*10))
done

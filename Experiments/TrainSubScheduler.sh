#!/bin/sh

#./TrainSubScheduler.sh
pathname="./TrainSubSchedulerWith27Nodes.py"

for VARIABLE in 0 1 2 3 4 5 6 7 8 9
do
python3 $pathname --start_sample $(($VARIABLE*20))
done
